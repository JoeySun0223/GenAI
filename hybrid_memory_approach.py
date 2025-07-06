import os
import torch
import numpy as np
from PIL import Image
import json
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from collections import defaultdict, deque
import matplotlib.pyplot as plt

class HybridMemoryApproach:
    """混合方案：结合Memory Bank思想和后处理优化"""
    
    def __init__(self, checkpoint_path, model_config_path, memory_size=5, use_memory_filtering=True):
        self.checkpoint = checkpoint_path
        self.model_cfg = model_config_path
        self.model = build_sam2(model_config_path, checkpoint_path)
        self.memory_size = memory_size
        self.use_memory_filtering = use_memory_filtering  # 新增：是否使用memory bank过滤
        
        # 使用更宽松的参数，接近test.py的设置
        self.mask_generator = SAM2AutomaticMaskGenerator(
            self.model,
            points_per_side=64,  # 从32增加到64
            points_per_batch=128,  # 从64增加到128
            pred_iou_thresh=0.7,  # 从0.8降低到0.7
            stability_score_thresh=0.92,  # 从0.95降低到0.92
            stability_score_offset=0.7,  # 从0.8降低到0.7
            crop_n_layers=1,
            box_nms_thresh=0.7,  # 从0.8降低到0.7
            crop_n_points_downscale_factor=2,
            min_mask_region_area=25,  # 从100降低到25
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
        # 几何特征
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        # 基本几何特征
        area = cv2.contourArea(contours[0])
        perimeter = cv2.arcLength(contours[0], True)
        x, y, w, h = cv2.boundingRect(contours[0])
        aspect_ratio = w / h if h > 0 else 0
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # 颜色特征
        masked_region = image[mask > 0]
        if len(masked_region) > 0:
            mean_color = np.mean(masked_region, axis=0)
            color_std = np.std(masked_region, axis=0)
        else:
            mean_color = np.zeros(3)
            color_std = np.zeros(3)
        
        # 形状特征（傅里叶描述子）
        if len(contours[0]) >= 8:
            # 简化轮廓点
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contours[0], epsilon, True)
            if len(approx) >= 8:
                # 计算傅里叶描述子
                fourier_desc = self.compute_fourier_descriptors(approx.reshape(-1, 2))
                fourier_features = np.abs(fourier_desc[:8])  # 取前8个系数
            else:
                fourier_features = np.zeros(8)
        else:
            fourier_features = np.zeros(8)
        
        # 组合特征向量
        features = np.concatenate([
            [area / (mask.shape[0] * mask.shape[1])],  # 归一化面积
            [perimeter / (mask.shape[0] + mask.shape[1])],  # 归一化周长
            [aspect_ratio],
            [circularity],
            mean_color / 255,  # 归一化颜色
            color_std / 255,   # 颜色标准差
            fourier_features   # 傅里叶描述子
        ])
        
        return features
    
    def compute_fourier_descriptors(self, contour_points):
        """计算傅里叶描述子"""
        # 将轮廓点转换为复数
        complex_coords = contour_points[:, 0] + 1j * contour_points[:, 1]
        
        # 计算傅里叶变换
        fourier_coeffs = np.fft.fft(complex_coords)
        
        # 归一化（除以第一个非零系数）
        if len(fourier_coeffs) > 1 and abs(fourier_coeffs[1]) > 1e-6:
            fourier_coeffs = fourier_coeffs / fourier_coeffs[1]
        
        return fourier_coeffs
    
    def calculate_temporal_consistency(self, current_mask, current_features, current_angle):
        """计算与memory bank中掩码的时间一致性，基于3D旋转角度优化"""
        if len(self.memory_bank['masks']) == 0:
            return 1.0  # 第一个掩码，一致性为1
        
        consistency_scores = []
        
        for i, (stored_mask, stored_features, stored_angle) in enumerate(
            zip(self.memory_bank['masks'], self.memory_bank['features'], self.memory_bank['angles'])
        ):
            # 1. 特征相似度
            if stored_features is not None and current_features is not None:
                feature_similarity = cosine_similarity([stored_features], [current_features])[0][0]
            else:
                feature_similarity = 0
            
            # 2. 形状相似度
            shape_similarity = self.calculate_shape_similarity(current_mask, stored_mask)
            
            # 3. 基于3D旋转的角度权重（关键改进）
            angle_weight = self.calculate_3d_rotation_weight(current_angle, stored_angle)
            
            # 4. 时间权重（越近的帧权重越高，但降低影响）
            time_weight = 0.3 + 0.7 * ((i + 1) / len(self.memory_bank['masks']))
            
            # 5. 综合一致性分数，更注重几何一致性
            consistency = (0.3 * feature_similarity + 0.4 * shape_similarity) * angle_weight * time_weight
            consistency_scores.append(consistency)
        
        return max(consistency_scores) if consistency_scores else 0.0
    
    def calculate_3d_rotation_weight(self, current_angle, stored_angle):
        """基于3D旋转几何计算角度权重，适用于非对称模型"""
        angle_diff = abs(current_angle - stored_angle)
        
        # 处理角度环绕（0°和360°是同一个角度）
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        # 1. 相邻角度（30°间隔）最高权重
        if angle_diff <= 30:
            return 1.0
        
        # 2. 60°间隔，中等权重
        elif angle_diff <= 60:
            return 0.7
        
        # 3. 90°间隔，较低权重
        elif angle_diff <= 90:
            return 0.4
        
        # 4. 120°间隔，很低权重
        elif angle_diff <= 120:
            return 0.2
        
        # 5. 其他角度，极低权重
        else:
            return 0.1
    
    def calculate_shape_similarity(self, mask1, mask2):
        """计算形状相似度"""
        # IoU
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        iou = intersection / union if union > 0 else 0
        
        # 轮廓相似度
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
        """基于memory bank一致性过滤掩码，使用更宽松的阈值"""
        filtered_masks = []
        mask_features = []
        consistency_scores = []
        
        for mask_dict in masks:
            mask = mask_dict["segmentation"]
            
            # 提取特征
            features = self.extract_mask_features(mask, image)
            
            # 计算时间一致性
            consistency = self.calculate_temporal_consistency(mask, features, angle)
            
            # 使用更宽松的阈值，并考虑角度因素
            threshold = self.get_adaptive_threshold(angle)
            
            if consistency > threshold:
                mask_dict["temporal_consistency"] = consistency
                filtered_masks.append(mask_dict)
                mask_features.append(features)
                consistency_scores.append(consistency)
        
        return filtered_masks, mask_features, consistency_scores
    
    def get_adaptive_threshold(self, angle):
        """根据角度自适应调整一致性阈值，适用于非对称模型"""
        # 第一个角度（0°）使用较低阈值
        if len(self.memory_bank['masks']) == 0:
            return 0.01  # 更宽松的初始阈值
        
        # 根据角度间隔调整阈值
        if len(self.memory_bank['angles']) > 0:
            last_angle = self.memory_bank['angles'][-1]
            angle_diff = abs(angle - last_angle)
            
            # 处理角度环绕
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            # 相邻角度（30°）使用较低阈值
            if angle_diff <= 30:
                return 0.01
            # 60°间隔
            elif angle_diff <= 60:
                return 0.008
            # 90°间隔
            elif angle_diff <= 90:
                return 0.005
            # 大角度跳跃使用极低阈值
            else:
                return 0.002
        
        return 0.01  # 默认阈值
    
    def merge_masks_with_memory_guidance(self, masks, similarity_threshold=0.8):
        """使用memory bank指导合并掩码"""
        if len(masks) <= 1:
            return masks
        
        # 使用memory bank中的信息来指导合并
        memory_guidance = self.get_memory_guidance()
        
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
                    
                # 计算相似度
                similarity = self.calculate_mask_similarity(mask1["segmentation"], mask2["segmentation"])
                
                # 如果有memory guidance，调整相似度
                if memory_guidance:
                    similarity = self.adjust_similarity_with_memory(similarity, mask1, mask2, memory_guidance)
                
                if similarity > similarity_threshold:
                    similar_masks.append(mask2)
                    used_indices.add(j)
            
            # 合并相似掩码
            if len(similar_masks) > 1:
                merged_mask = self.merge_mask_list(similar_masks)
                merged_masks.append(merged_mask)
            else:
                merged_masks.append(mask1)
        
        return merged_masks
    
    def get_memory_guidance(self):
        """从memory bank获取指导信息"""
        if len(self.memory_bank['masks']) == 0:
            return None
        
        # 分析memory bank中的模式
        guidance = {
            'common_shapes': [],
            'typical_sizes': [],
            'color_patterns': []
        }
        
        # 这里可以添加更复杂的模式分析
        return guidance
    
    def adjust_similarity_with_memory(self, base_similarity, mask1, mask2, memory_guidance):
        """使用memory bank调整相似度"""
        # 简单的调整策略，可以根据需要扩展
        adjustment = 0.0
        
        # 如果两个掩码都与memory bank中的模式一致，提高相似度
        if memory_guidance:
            # 这里可以添加更复杂的逻辑
            pass
        
        return min(base_similarity + adjustment, 1.0)
    
    def calculate_mask_similarity(self, mask1, mask2):
        """计算掩码相似度"""
        # IoU
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        iou = intersection / union if union > 0 else 0
        
        # 形状相似度
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
        # 使用最大IoU的掩码作为基础
        base_mask = max(mask_list, key=lambda x: x.get("predicted_iou", 0))
        
        # 合并所有掩码
        merged_segmentation = np.zeros_like(base_mask["segmentation"])
        for mask in mask_list:
            merged_segmentation = np.logical_or(merged_segmentation, mask["segmentation"])
        
        # 更新掩码信息
        merged_mask = base_mask.copy()
        merged_mask["segmentation"] = merged_segmentation
        merged_mask["area"] = merged_segmentation.sum()
        
        # 重新计算边界框
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
        
        # 获取所有角度图像并排序
        image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()
        
        for img_file in image_files:
            # 解析角度信息
            angle = self.extract_angle_from_filename(img_file)
            
            print(f"处理角度 {angle}°: {img_file}")
            
            # 加载图像
            img_path = os.path.join(input_dir, img_file)
            orig_img = Image.open(img_path)
            
            # 检查是否有alpha通道
            if orig_img.mode == 'RGBA':
                # 提取alpha通道作为透明区域掩码
                alpha_channel = np.array(orig_img.split()[-1])
                transparent_mask = alpha_channel > 0  # 非透明区域
                
                # 转换为RGB用于SAM2处理
                image = np.array(orig_img.convert("RGB"))
            else:
                # 没有alpha通道，假设整个图像都是有效的
                image = np.array(orig_img.convert("RGB"))
                transparent_mask = np.ones(image.shape[:2], dtype=bool)
            
            # 1. 生成初始掩码
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                masks = self.mask_generator.generate(image)
            
            # 2. 过滤掉透明背景区域的掩码
            masks = self.filter_transparent_background_masks(masks, transparent_mask)
            
            # 3. 基于memory bank过滤掩码
            if self.use_memory_filtering:
                filtered_masks, mask_features, consistency_scores = self.filter_masks_by_memory_consistency(
                    masks, image, angle
                )
            else:
                # 使用类似test.py的简单过滤策略
                filtered_masks = self.simple_filter_masks(masks)
                mask_features = [self.extract_mask_features(m["segmentation"], image) for m in filtered_masks]
                consistency_scores = [1.0] * len(filtered_masks)  # 默认一致性为1
            
            # 4. 使用memory bank指导合并掩码
            if self.use_memory_filtering:
                merged_masks = self.merge_masks_with_memory_guidance(filtered_masks)
            else:
                merged_masks = filtered_masks  # 不进行合并
            
            # 5. 更新memory bank
            if self.use_memory_filtering:
                self.update_memory_bank(
                    [m["segmentation"] for m in merged_masks],
                    mask_features,
                    angle,
                    consistency_scores
                )
            
            # 6. 保存结果
            angle_dir = os.path.join(output_dir, f"angle_{int(angle):03d}deg")
            os.makedirs(angle_dir, exist_ok=True)
            
            saved_count = 0
            for i, mask_dict in enumerate(merged_masks):
                mask = mask_dict["segmentation"]
                
                # 确保掩码只在非透明区域有效
                mask = np.logical_and(mask, transparent_mask)
                
                # 创建RGBA图像
                rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
                rgba[..., :3][mask] = image[mask]
                rgba[..., 3][mask] = 255
                
                # 保存掩码
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
        
        # 生成总结报告
        self.generate_memory_report(all_results, output_dir)
        
        print(f"Memory Bank处理完成！结果保存在 {output_dir}")
        return all_results
    
    def extract_angle_from_filename(self, filename):
        """从文件名提取角度信息"""
        # 文件名格式为 bluecar_XXXdeg.png
        if 'deg' in filename:
            try:
                # 提取deg前的数字部分
                deg_index = filename.find('deg')
                if deg_index > 0:
                    # 从deg位置向前查找数字
                    start_index = deg_index - 1
                    while start_index >= 0 and filename[start_index].isdigit():
                        start_index -= 1
                    start_index += 1
                    
                    angle_str = filename[start_index:deg_index]
                    return float(angle_str)
            except:
                pass
        
        # 如果没有找到角度信息，返回0
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
        
        # 保存报告
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
            
            # 计算掩码在非透明区域的比例
            mask_in_valid_region = np.logical_and(mask, transparent_mask)
            valid_ratio = np.sum(mask_in_valid_region) / np.sum(mask) if np.sum(mask) > 0 else 0
            
            # 更宽松的过滤标准：只要在非透明区域有内容就保留
            if valid_ratio > 0.1:  # 从0.3降低到0.1
                # 更新掩码，只在非透明区域有效
                mask_dict["segmentation"] = mask_in_valid_region
                mask_dict["area"] = np.sum(mask_in_valid_region)
                filtered_masks.append(mask_dict)
        
        return filtered_masks
    
    def simple_filter_masks(self, masks):
        """使用类似test.py的简单过滤策略：去除高度重叠的掩码（IoU>0.8）"""
        filtered_masks = []
        for mask_dict in masks:
            mask = mask_dict["segmentation"]
            # 检查与已保留掩码的重叠度
            if all(self.mask_iou(mask, m["segmentation"]) < 0.8 for m in filtered_masks):
                filtered_masks.append(mask_dict)
        return filtered_masks
    
    def mask_iou(self, mask1, mask2):
        """计算两个掩码的IoU"""
        inter = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return inter / union if union > 0 else 0

# 使用示例
if __name__ == "__main__":
    checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    # 模式1：使用memory bank（更严格的一致性过滤）
    processor_with_memory = HybridMemoryApproach(checkpoint, model_cfg, memory_size=5, use_memory_filtering=True)
    processor_with_memory.process_multi_angle_with_memory("MultiView", "memory_output_parts")
    
    # 模式2：不使用memory bank（更宽松，类似test.py）
    #processor_simple = HybridMemoryApproach(checkpoint, model_cfg, memory_size=5, use_memory_filtering=False)
    #processor_simple.process_multi_angle_with_memory("MultiView", "simple_output_parts") 