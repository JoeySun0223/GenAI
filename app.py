import os
import gradio as gr
import numpy as np
from PIL import Image
import vtracer
import tempfile
import zipfile
import cairosvg
import sys
import imageio
import os.path as osp
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import torch
from scipy.ndimage import binary_dilation
import json
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from xml.dom import minidom
import re

# 创建输出目录
output_dir = "MultiView"
os.makedirs(output_dir, exist_ok=True)
os.makedirs("memory_output_parts", exist_ok=True)
os.makedirs("SVG_OUTPUT", exist_ok=True)

# 初始化Trellis pipeline
pipeline = TrellisImageTo3DPipeline.from_pretrained("TRELLIS-image-large")
pipeline.cuda()

def calculate_path_area(path_data):
    """计算SVG路径的面积"""
    # 使用Shoelace公式计算多边形面积
    def shoelace_area(points):
        x = points[:, 0]
        y = points[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    # 提取路径中的所有点
    points = []
    current_point = np.array([0, 0])
    
    # 解析路径数据
    commands = re.findall(r'([MLHVCSQTAZmlhvcsqtaz])([^MLHVCSQTAZmlhvcsqtaz]*)', path_data)
    
    for cmd, params in commands:
        params = [float(p) for p in re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', params)]
        
        if cmd in 'Mm':  # 移动
            if len(params) >= 2:
                if cmd == 'M':
                    current_point = np.array([params[0], params[1]])
                else:  # m
                    current_point = current_point + np.array([params[0], params[1]])
                points.append(current_point)
        
        elif cmd in 'Ll':  # 线段
            if len(params) >= 2:
                if cmd == 'L':
                    current_point = np.array([params[0], params[1]])
                else:  # l
                    current_point = current_point + np.array([params[0], params[1]])
                points.append(current_point)
        
        elif cmd in 'Hh':  # 水平线
            if params:
                if cmd == 'H':
                    current_point[0] = params[0]
                else:  # h
                    current_point[0] += params[0]
                points.append(current_point.copy())
        
        elif cmd in 'Vv':  # 垂直线
            if params:
                if cmd == 'V':
                    current_point[1] = params[0]
                else:  # v
                    current_point[1] += params[0]
                points.append(current_point.copy())
        
        elif cmd in 'Cc':  # 三次贝塞尔曲线
            if len(params) >= 6:
                # 这里简化处理，只取终点
                if cmd == 'C':
                    current_point = np.array([params[4], params[5]])
                else:  # c
                    current_point = current_point + np.array([params[4], params[5]])
                points.append(current_point)
        
        elif cmd == 'Z' or cmd == 'z':  # 闭合路径
            if points and not np.array_equal(points[0], current_point):
                points.append(points[0].copy())
    
    if len(points) < 3:
        return 0
    
    # 转换为numpy数组并计算面积
    points = np.array(points)
    return shoelace_area(points)

def merge_svg_strings(svg_strings, output_path, width=512, height=512):
    """合并多个SVG字符串为一个SVG文件，按路径面积排序（面积大的在底层）"""
    root = ET.Element("svg")
    root.set("xmlns", "http://www.w3.org/2000/svg")
    root.set("width", str(width))
    root.set("height", str(height))
    root.set("viewBox", f"0 0 {width} {height}")
    root.set("style", "background-color: transparent;")
    all_paths = []
    for svg_str in svg_strings:
        try:
            tree = ET.ElementTree(ET.fromstring(svg_str))
            svg_root = tree.getroot()
            for elem in svg_root.findall(".//{http://www.w3.org/2000/svg}path"):
                path_data = elem.get('d', '')
                if path_data:
                    area = calculate_path_area(path_data)
                    all_paths.append((elem, area))
        except Exception as e:
            print(f"处理SVG字符串时出错: {str(e)}")
            continue
    all_paths.sort(key=lambda x: x[1], reverse=True)
    for elem, area in all_paths:
        root.append(elem)
    try:
        rough_string = ET.tostring(root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_string = reparsed.toprettyxml(indent="  ")
        pretty_string = '\n'.join([line for line in pretty_string.split('\n') if line.strip()])
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(pretty_string)
        print(f"成功合并SVG文件到: {output_path}")
        return True
    except Exception as e:
        print(f"保存合并后的SVG文件时出错: {str(e)}")
        return False

class HybridMemoryApproach:
    """混合方案：结合Memory Bank思想和后处理优化"""
    
    def __init__(self, checkpoint_path, model_config_path, memory_size=5, use_memory_filtering=True):
        self.checkpoint = checkpoint_path
        self.model_cfg = model_config_path
        self.model = build_sam2(model_config_path, checkpoint_path)
        self.model.cuda(device=0)  # 明确指定使用cuda:0
        self.memory_size = memory_size
        self.use_memory_filtering = use_memory_filtering
        
        # 使用更宽松的参数，接近test.py的设置
        self.mask_generator = SAM2AutomaticMaskGenerator(
            self.model,
            points_per_side=64,
            points_per_batch=128,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.92,
            stability_score_offset=0.7,
            crop_n_layers=1,
            box_nms_thresh=0.7,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=25,
            use_m2m=True
        )
        
        # Memory Bank状态
        self.memory_bank = {
            'masks': deque(maxlen=memory_size),
            'features': deque(maxlen=memory_size),
            'angles': deque(maxlen=memory_size),
            'consistency_scores': deque(maxlen=memory_size)
        }
        
    def extract_mask_features(self, mask, image):
        """提取掩码特征用于memory bank"""
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        area = cv2.contourArea(contours[0])
        perimeter = cv2.arcLength(contours[0], True)
        x, y, w, h = cv2.boundingRect(contours[0])
        aspect_ratio = w / h if h > 0 else 0
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        masked_region = image[mask > 0]
        if len(masked_region) > 0:
            mean_color = np.mean(masked_region, axis=0)
            color_std = np.std(masked_region, axis=0)
        else:
            mean_color = np.zeros(3)
            color_std = np.zeros(3)
        
        if len(contours[0]) >= 8:
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contours[0], epsilon, True)
            if len(approx) >= 8:
                fourier_desc = self.compute_fourier_descriptors(approx.reshape(-1, 2))
                fourier_features = np.abs(fourier_desc[:8])
            else:
                fourier_features = np.zeros(8)
        else:
            fourier_features = np.zeros(8)
        
        features = np.concatenate([
            [area / (mask.shape[0] * mask.shape[1])],
            [perimeter / (mask.shape[0] + mask.shape[1])],
            [aspect_ratio],
            [circularity],
            mean_color / 255,
            color_std / 255,
            fourier_features
        ])
        
        return features
    
    def compute_fourier_descriptors(self, contour_points):
        """计算傅里叶描述子"""
        complex_coords = contour_points[:, 0] + 1j * contour_points[:, 1]
        fourier_coeffs = np.fft.fft(complex_coords)
        if len(fourier_coeffs) > 1 and abs(fourier_coeffs[1]) > 1e-6:
            fourier_coeffs = fourier_coeffs / fourier_coeffs[1]
        return fourier_coeffs
    
    def calculate_temporal_consistency(self, current_mask, current_features, current_angle):
        """计算与memory bank中掩码的时间一致性"""
        if len(self.memory_bank['masks']) == 0:
            return 1.0
        
        consistency_scores = []
        
        for i, (stored_mask, stored_features, stored_angle) in enumerate(
            zip(self.memory_bank['masks'], self.memory_bank['features'], self.memory_bank['angles'])
        ):
            if stored_features is not None and current_features is not None:
                feature_similarity = cosine_similarity([stored_features], [current_features])[0][0]
            else:
                feature_similarity = 0
            
            shape_similarity = self.calculate_shape_similarity(current_mask, stored_mask)
            angle_weight = self.calculate_3d_rotation_weight(current_angle, stored_angle)
            time_weight = 0.3 + 0.7 * ((i + 1) / len(self.memory_bank['masks']))
            
            consistency = (0.3 * feature_similarity + 0.4 * shape_similarity) * angle_weight * time_weight
            consistency_scores.append(consistency)
        
        return max(consistency_scores) if consistency_scores else 0.0
    
    def calculate_3d_rotation_weight(self, current_angle, stored_angle):
        """基于3D旋转几何计算角度权重"""
        angle_diff = abs(current_angle - stored_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        if angle_diff <= 30:
            return 1.0
        elif angle_diff <= 60:
            return 0.7
        elif angle_diff <= 90:
            return 0.4
        elif angle_diff <= 120:
            return 0.2
        else:
            return 0.1
    
    def calculate_shape_similarity(self, mask1, mask2):
        """计算形状相似度"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        iou = intersection / union if union > 0 else 0
        
        contours1, _ = cv2.findContours(mask1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(mask2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours1 and contours2:
            shape_similarity = cv2.matchShapes(contours1[0], contours2[0], cv2.CONTOURS_MATCH_I2, 0)
            shape_similarity = 1 / (1 + shape_similarity)
        else:
            shape_similarity = 0
        
        return 0.7 * iou + 0.3 * shape_similarity
    
    def update_memory_bank(self, masks, features, angle, consistency_scores):
        """更新memory bank"""
        for mask, feature, score in zip(masks, features, consistency_scores):
            self.memory_bank['masks'].append(mask)
            self.memory_bank['features'].append(feature)
            self.memory_bank['angles'].append(angle)
            self.memory_bank['consistency_scores'].append(score)
    
    def filter_masks_by_memory_consistency(self, masks, image, angle):
        """基于memory bank一致性过滤掩码"""
        filtered_masks = []
        mask_features = []
        consistency_scores = []
        
        for mask_dict in masks:
            mask = mask_dict["segmentation"]
            features = self.extract_mask_features(mask, image)
            consistency = self.calculate_temporal_consistency(mask, features, angle)
            threshold = self.get_adaptive_threshold(angle)
            
            if consistency > threshold:
                mask_dict["temporal_consistency"] = consistency
                filtered_masks.append(mask_dict)
                mask_features.append(features)
                consistency_scores.append(consistency)
        
        return filtered_masks, mask_features, consistency_scores
    
    def get_adaptive_threshold(self, angle):
        """根据角度自适应调整一致性阈值"""
        if len(self.memory_bank['masks']) == 0:
            return 0.01
        
        if len(self.memory_bank['angles']) > 0:
            last_angle = self.memory_bank['angles'][-1]
            angle_diff = abs(angle - last_angle)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            if angle_diff <= 30:
                return 0.01
            elif angle_diff <= 60:
                return 0.008
            elif angle_diff <= 90:
                return 0.005
            else:
                return 0.002
        
        return 0.01
    
    def merge_masks_with_memory_guidance(self, masks, similarity_threshold=0.8):
        """使用memory bank指导合并掩码"""
        if len(masks) <= 1:
            return masks
        
        merged_masks = []
        used_indices = set()
        
        for i, mask1 in enumerate(masks):
            if i in used_indices:
                continue
                
            similar_masks = [mask1]
            used_indices.add(i)
            
            for j, mask2 in enumerate(masks[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                similarity = self.calculate_mask_similarity(mask1["segmentation"], mask2["segmentation"])
                
                if similarity > similarity_threshold:
                    similar_masks.append(mask2)
                    used_indices.add(j)
            
            if len(similar_masks) > 1:
                merged_mask = self.merge_mask_list(similar_masks)
                merged_masks.append(merged_mask)
            else:
                merged_masks.append(mask1)
        
        return merged_masks
    
    def calculate_mask_similarity(self, mask1, mask2):
        """计算掩码相似度"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        iou = intersection / union if union > 0 else 0
        
        contours1, _ = cv2.findContours(mask1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(mask2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours1 and contours2:
            shape_similarity = cv2.matchShapes(contours1[0], contours2[0], cv2.CONTOURS_MATCH_I2, 0)
            shape_similarity = 1 / (1 + shape_similarity)
        else:
            shape_similarity = 0
        
        return 0.7 * iou + 0.3 * shape_similarity
    
    def merge_mask_list(self, mask_list):
        """合并掩码列表"""
        base_mask = max(mask_list, key=lambda x: x.get("predicted_iou", 0))
        merged_segmentation = np.zeros_like(base_mask["segmentation"])
        for mask in mask_list:
            merged_segmentation = np.logical_or(merged_segmentation, mask["segmentation"])
        
        merged_mask = base_mask.copy()
        merged_mask["segmentation"] = merged_segmentation
        merged_mask["area"] = merged_segmentation.sum()
        
        y_indices, x_indices = np.where(merged_segmentation)
        if len(y_indices) > 0 and len(x_indices) > 0:
            merged_mask["bbox"] = [
                int(x_indices.min()),
                int(y_indices.min()),
                int(x_indices.max() - x_indices.min()),
                int(y_indices.max() - y_indices.min())
            ]
        
        return merged_mask
    
    def process_multi_angle_with_memory(self, input_dir, output_dir):
        """使用memory bank处理多角度图像"""
        print("开始使用Memory Bank处理多角度图像...")
        
        os.makedirs(output_dir, exist_ok=True)
        all_results = {}
        
        image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()
        
        for img_file in image_files:
            angle = self.extract_angle_from_filename(img_file)
            print(f"处理角度 {angle}°: {img_file}")
            
            img_path = os.path.join(input_dir, img_file)
            orig_img = Image.open(img_path)
            
            if orig_img.mode == 'RGBA':
                alpha_channel = np.array(orig_img.split()[-1])
                transparent_mask = alpha_channel > 0
                image = np.array(orig_img.convert("RGB"))
            else:
                image = np.array(orig_img.convert("RGB"))
                transparent_mask = np.ones(image.shape[:2], dtype=bool)
            
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                masks = self.mask_generator.generate(image)
            
            masks = self.filter_transparent_background_masks(masks, transparent_mask)
            
            if self.use_memory_filtering:
                filtered_masks, mask_features, consistency_scores = self.filter_masks_by_memory_consistency(
                    masks, image, angle
                )
            else:
                filtered_masks = self.simple_filter_masks(masks)
                mask_features = [self.extract_mask_features(m["segmentation"], image) for m in filtered_masks]
                consistency_scores = [1.0] * len(filtered_masks)
            
            if self.use_memory_filtering:
                merged_masks = self.merge_masks_with_memory_guidance(filtered_masks)
            else:
                merged_masks = filtered_masks
            
            if self.use_memory_filtering:
                self.update_memory_bank(
                    [m["segmentation"] for m in merged_masks],
                    mask_features,
                    angle,
                    consistency_scores
                )
            
            angle_dir = os.path.join(output_dir, f"angle_{int(angle):03d}deg")
            os.makedirs(angle_dir, exist_ok=True)
            
            saved_count = 0
            for i, mask_dict in enumerate(merged_masks):
                mask = mask_dict["segmentation"]
                mask = np.logical_and(mask, transparent_mask)
                
                rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
                rgba[..., :3][mask] = image[mask]
                rgba[..., 3][mask] = 255
                
                mask_img = Image.fromarray(rgba, mode='RGBA')
                consistency = mask_dict.get("temporal_consistency", 0)
                mask_filename = f"memory_part_{saved_count+1}_cons{consistency:.3f}.png"
                mask_img.save(os.path.join(angle_dir, mask_filename))
                saved_count += 1
            
            all_results[angle] = {
                "original_count": len(masks),
                "filtered_count": len(filtered_masks),
                "final_count": len(merged_masks),
                "avg_consistency": np.mean(consistency_scores) if consistency_scores else 0,
                "masks": merged_masks
            }
            
            print(f"  原始掩码: {len(masks)}, 过滤后: {len(filtered_masks)}, 最终: {len(merged_masks)}")
            print(f"  平均一致性: {all_results[angle]['avg_consistency']:.3f}")
        
        self.generate_memory_report(all_results, output_dir)
        print(f"Memory Bank处理完成！结果保存在 {output_dir}")
        return all_results
    
    def extract_angle_from_filename(self, filename):
        """从文件名提取角度信息"""
        if 'deg' in filename:
            try:
                deg_index = filename.find('deg')
                if deg_index > 0:
                    start_index = deg_index - 1
                    while start_index >= 0 and filename[start_index].isdigit():
                        start_index -= 1
                    start_index += 1
                    angle_str = filename[start_index:deg_index]
                    return float(angle_str)
            except:
                pass
        return 0.0
    
    def generate_memory_report(self, all_results, output_dir):
        """生成memory bank处理报告"""
        total_original = sum(result["original_count"] for result in all_results.values())
        total_filtered = sum(result["filtered_count"] for result in all_results.values())
        total_final = sum(result["final_count"] for result in all_results.values())
        avg_consistency = np.mean([result["avg_consistency"] for result in all_results.values()])
        
        report = {
            "total_angles": len(all_results),
            "total_original_masks": total_original,
            "total_filtered_masks": total_filtered,
            "total_final_masks": total_final,
            "reduction_percentage": ((total_original - total_final) / total_original * 100) if total_original > 0 else 0,
            "average_consistency": float(avg_consistency),
            "memory_size": self.memory_size,
            "angle_details": all_results
        }
        
        with open(os.path.join(output_dir, "memory_bank_report.json"), 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nMemory Bank报告：")
        print(f"  处理角度数: {report['total_angles']}")
        print(f"  原始掩码总数: {report['total_original_masks']}")
        print(f"  过滤后掩码总数: {report['total_filtered_masks']}")
        print(f"  最终掩码总数: {report['total_final_masks']}")
        print(f"  减少比例: {report['reduction_percentage']:.1f}%")
        print(f"  平均一致性: {report['average_consistency']:.3f}")
        print(f"  Memory Bank大小: {report['memory_size']}")
    
    def filter_transparent_background_masks(self, masks, transparent_mask):
        """过滤掉透明背景区域的掩码"""
        filtered_masks = []
        
        for mask_dict in masks:
            mask = mask_dict["segmentation"]
            mask_in_valid_region = np.logical_and(mask, transparent_mask)
            valid_ratio = np.sum(mask_in_valid_region) / np.sum(mask) if np.sum(mask) > 0 else 0
            
            if valid_ratio > 0.1:
                mask_dict["segmentation"] = mask_in_valid_region
                mask_dict["area"] = np.sum(mask_in_valid_region)
                filtered_masks.append(mask_dict)
        
        return filtered_masks
    
    def simple_filter_masks(self, masks):
        """使用简单过滤策略"""
        filtered_masks = []
        for mask_dict in masks:
            mask = mask_dict["segmentation"]
            if all(self.mask_iou(mask, m["segmentation"]) < 0.8 for m in filtered_masks):
                filtered_masks.append(mask_dict)
        return filtered_masks
    
    def mask_iou(self, mask1, mask2):
        """计算两个掩码的IoU"""
        inter = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return inter / union if union > 0 else 0

def generate_3d_images(input_image, progress=gr.Progress()):
    try:
        progress(0.2, desc="正在生成3D模型...")
        
        # 清理MultiView目录中的旧文件
        for old_file in os.listdir(output_dir):
            if old_file.endswith('.png'):
                os.remove(os.path.join(output_dir, old_file))
        
        # 清理之前的处理结果
        if os.path.exists("memory_output_parts"):
            import shutil
            shutil.rmtree("memory_output_parts")
        if os.path.exists("SVG_OUTPUT"):
            import shutil
            shutil.rmtree("SVG_OUTPUT")
        
        # 重新创建目录
        os.makedirs("memory_output_parts", exist_ok=True)
        os.makedirs("SVG_OUTPUT", exist_ok=True)
        
        # 使用3D_Gen的逻辑生成图像
        outputs = pipeline.run(input_image, seed=1)
        
        num_images = 12
        angles_degrees = [i * 30 for i in range(12)]
        yaws = [angle * np.pi / 180 for angle in angles_degrees]
        pitch = [0.0] * num_images
        
        progress(0.4, desc="正在准备渲染...")
        extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, rs=2, fovs=40)
        
        progress(0.6, desc="正在渲染图像...")
        render_results = render_utils.render_frames(
            outputs['gaussian'][0],
            extrinsics,
            intrinsics,
            {'resolution': 512, 'bg_color': None}
        )
        
        progress(0.8, desc="正在保存图像到MultiView...")
        images = []
        base_filename = "generated"
        
        for i, img_data in enumerate(render_results['color']):
            angle = angles_degrees[i]
            
            if 'alpha' in render_results:
                rgb = img_data.astype(np.uint8)
                alpha = render_results['alpha'][i]
                if alpha.shape[-1] == 1:
                    alpha = alpha.squeeze(-1)
                alpha_binary = np.zeros_like(alpha, dtype=np.uint8)
                alpha_binary[alpha > 0.1] = 255
                rgba = np.dstack([rgb, alpha_binary])
                img = Image.fromarray(rgba, 'RGBA')
            else:
                img = Image.fromarray(img_data.astype(np.uint8))
            
            # 保存到MultiView目录
            img_path = osp.join(output_dir, f"{base_filename}_{angle:03d}deg.png")
            img.save(img_path)
            images.append(img_path)
            
            progress(0.8 + (i + 1) * 0.2 / num_images, desc=f"正在保存第 {i + 1}/{num_images} 张图片...")
        
        return images
    except Exception as e:
        raise gr.Error(f"生成3D图像时出错: {str(e)}")

def process_with_hybrid_memory(progress=gr.Progress()):
    """使用hybrid_memory_approach处理MultiView中的图像"""
    try:
        progress(0.1, desc="正在初始化SAM2模型...")
        
        # 初始化hybrid memory approach
        checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        
        processor = HybridMemoryApproach(checkpoint, model_cfg, memory_size=5, use_memory_filtering=True)
        
        progress(0.3, desc="正在使用Memory Bank处理图像...")
        
        # 处理MultiView中的图像，输出到memory_output_parts目录
        output_dir_hybrid = "memory_output_parts"
        results = processor.process_multi_angle_with_memory("MultiView", output_dir_hybrid)
        
        progress(0.6, desc="正在收集处理后的图像...")
        
        # 收集所有处理后的图像路径
        processed_images = []
        for angle_dir in os.listdir(output_dir_hybrid):
            if angle_dir.startswith("angle_"):
                angle_path = os.path.join(output_dir_hybrid, angle_dir)
                if os.path.isdir(angle_path):
                    for file in os.listdir(angle_path):
                        if file.endswith('.png'):
                            processed_images.append(os.path.join(angle_path, file))
        
        progress(1.0, desc="Memory Bank处理完成")
        
        return processed_images
    except Exception as e:
        raise gr.Error(f"Memory Bank处理时出错: {str(e)}")

def svg_to_png(svg_path, png_path):
    cairosvg.svg2png(url=svg_path, write_to=png_path)

def convert_to_svg(image_paths, progress=gr.Progress()):
    try:
        if not image_paths:
            raise gr.Error("没有找到需要转换的图片！")
        
        # 检查是否已经有处理过的结果
        if os.path.exists("memory_output_parts") and os.listdir("memory_output_parts"):
            progress(0.1, desc="发现已有处理结果，跳过Memory Bank处理...")
            processed_images = []
            for angle_dir in os.listdir("memory_output_parts"):
                if angle_dir.startswith("angle_"):
                    angle_path = os.path.join("memory_output_parts", angle_dir)
                    if os.path.isdir(angle_path):
                        for file in os.listdir(angle_path):
                            if file.endswith('.png'):
                                processed_images.append(os.path.join(angle_path, file))
        else:
            # 首先使用hybrid_memory_approach处理
            progress(0.1, desc="正在使用Memory Bank处理图像...")
            processed_images = process_with_hybrid_memory(progress)
        
        if not processed_images:
            raise gr.Error("Memory Bank处理失败！")
        
        progress(0.3, desc="正在转换为SVG...")
        
        # 使用Tracer_multi的逻辑处理memory_output_parts目录
        params = {
            'colormode': 'color',
            'hierarchical': 'stacked',
            'mode': 'spline',
            'filter_speckle': 4,
            'color_precision': 6,
            'layer_difference': 16,
            'corner_threshold': 60,
            'length_threshold': 4.0,
            'max_iterations': 10,
            'splice_threshold': 45,
            'path_precision': 3
        }
        
        # 处理memory_output_parts目录中的子目录
        svg_files = []
        png_previews = []
        
        subdirs = [d for d in os.listdir("memory_output_parts") if os.path.isdir(os.path.join("memory_output_parts", d))]
        if not subdirs:
            raise gr.Error("在memory_output_parts中没有找到子目录！")
        
        total_subdirs = len(subdirs)
        for i, subdir in enumerate(subdirs):
            progress(0.3 + (i / total_subdirs) * 0.7, desc=f"正在处理角度 {subdir}...")
            
            input_subdir = os.path.join("memory_output_parts", subdir)
            png_files = [f for f in os.listdir(input_subdir) if f.endswith('.png')]
            
            if not png_files:
                continue
            
            print(f"处理子目录 {subdir}，找到 {len(png_files)} 个PNG文件")
            svg_strings = []
            
            for png_file in png_files:
                try:
                    png_path = os.path.join(input_subdir, png_file)
                    img = Image.open(png_path).convert('RGBA')
                    pixels = list(img.getdata())
                    size = img.size
                    svg_str = vtracer.convert_pixels_to_svg(pixels, size=size, **params)
                    if svg_str:
                        svg_strings.append(svg_str)
                except Exception as e:
                    print(f"转换文件 {png_file} 时出错: {str(e)}")
                    continue
            
            if svg_strings:
                # 合并该角度的所有SVG
                merged_svg = os.path.join("SVG_OUTPUT", f"{subdir}_merged.svg")
                if merge_svg_strings(svg_strings, merged_svg):
                    svg_files.append(merged_svg)
                    
                    # 生成预览图
                    preview_png = os.path.join("SVG_OUTPUT", f"{subdir}_preview.png")
                    svg_to_png(merged_svg, preview_png)
                    png_previews.append(preview_png)
                    
                    print(f"成功合并 {len(svg_strings)} 个SVG为 {merged_svg}")
        
        if not svg_files:
            raise gr.Error("所有文件转换都失败了！")
        
        progress(1.0, desc="转换完成")
        return [], [svg_files, [], png_previews], png_previews
    except Exception as e:
        raise gr.Error(f"转换SVG时出错: {str(e)}")

def get_svg_paths_from_state(svg_state):
    svg_files, _, _ = svg_state
    return svg_files

def zip_files(file_list, zip_name):
    zip_path = os.path.join(tempfile.gettempdir(), zip_name)
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in file_list:
            if isinstance(file, tuple):
                file = file[0]
            if file and os.path.exists(file):
                zipf.write(file, os.path.basename(file))
    return zip_path

with gr.Blocks(title="3D图像生成与SVG转换") as demo:
    gr.Markdown("# 3D图像生成与SVG转换工具")
    gr.Markdown("上传一张图片，生成3D视图，并一键批量转换为SVG")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                type="pil",
                label="上传图片",
                image_mode="RGBA",
                height=320
            )
            with gr.Row():
                generate_btn = gr.Button("生成3D图像", variant="secondary", size="sm")
                convert_btn = gr.Button("转换为SVG", variant="secondary", size="sm")
        with gr.Column():
            output_gallery = gr.Gallery(
                label="生成的3D图像",
                show_label=True,
                columns=3,
                height=320,
                object_fit="contain",
                interactive=False
            )
    with gr.Row():
        svg_gallery = gr.Gallery(
            label="SVG PNG预览",
            show_label=True,
            columns=3,
            height=320,
            object_fit="contain"
        )
        svg_state = gr.State([[], [], []])
    with gr.Row():
        svg_zip_btn = gr.Button("下载SVG", variant="secondary", size="sm")
        png_zip_btn = gr.Button("下载PNG", variant="secondary", size="sm")
    generate_btn.click(
        fn=generate_3d_images,
        inputs=input_image,
        outputs=output_gallery
    )
    convert_btn.click(
        fn=convert_to_svg,
        inputs=output_gallery,
        outputs=[gr.State(), svg_state, svg_gallery]
    )
    svg_zip_btn.click(
        fn=lambda svg_state: zip_files(get_svg_paths_from_state(svg_state), "all_svg.zip"),
        inputs=svg_state,
        outputs=gr.File(label="SVG打包下载")
    )
    png_zip_btn.click(
        fn=lambda files: zip_files(files, "all_png.zip"),
        inputs=output_gallery,
        outputs=gr.File(label="PNG打包下载")
    )
if __name__ == "__main__":
    print("启动Gradio应用...")
    print("请在本地浏览器中访问显示的URL")
    demo.launch(share=False, server_name="0.0.0.0", server_port=7861, show_error=True) 