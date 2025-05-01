import argparse
import json
import os
import numpy as np
import cv2
from pathlib import Path
from sklearn.cluster import KMeans
from PIL import Image
import xml.dom.minidom as minidom
import re
import math
import glob
import sys

class PNGtoSVGConverter:
    """将PNG图像转换为SVG并使用原始SVG数据进行优化"""
    
    def __init__(self, png_file, original_svg_json=None):
        """初始化转换器"""
        self.png_file = png_file
        self.original_svg_json = original_svg_json
        self.original_svg_data = None
        self.color_layers = {}
        self.svg_paths = []
        self.width = 0
        self.height = 0
        
    def load_original_svg_data(self):
        """加载原始SVG数据"""
        if not self.original_svg_json:
            return False
            
        try:
            with open(self.original_svg_json, 'r', encoding='utf-8') as f:
                self.original_svg_data = json.load(f)
            return True
        except Exception as e:
            print(f"加载原始SVG数据时出错: {e}")
            return False
    
    def convert(self, output_file=None, optimize=True):
        """转换PNG到SVG"""
        if not output_file:
            output_file = Path(self.png_file).with_suffix('.svg')
        
        # 加载原始SVG数据（如果提供）
        if self.original_svg_json:
            self.load_original_svg_data()
        
        # 读取PNG图像
        try:
            img = cv2.imread(self.png_file, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"无法读取图像文件: {self.png_file}")
                
            # 保存图像尺寸
            self.height, self.width = img.shape[:2]
            
            # 保存Alpha通道（如果有）
            if img.shape[2] == 4:
                self.alpha_mask = img[:, :, 3] > 128  # 二值化Alpha通道
                # 只处理RGB部分，保留透明度信息
                img_rgb = img[:, :, :3]
                img = img_rgb  # 不混合白色背景
            
            # 提取颜色层
            self._extract_color_layers(img)
            
            # 为每个颜色层创建路径
            self.svg_paths = []
            
            # 按照颜色层的顺序处理（从下到上）
            for color in self.color_layers_order:
                mask = self.color_layers[color]
                
                # 如果有Alpha通道，将颜色层与Alpha掩码相交
                if hasattr(self, 'alpha_mask'):
                    mask = np.logical_and(mask, self.alpha_mask)
                
                paths = self._create_paths_from_mask(mask)
                for path in paths:
                    self.svg_paths.append({
                        'path': path,
                        'color': color
                    })
            
            # 如果提供了原始SVG数据并启用了优化，则优化路径
            if self.original_svg_data and optimize:
                self._optimize_paths()
            
            # 生成SVG文件
            self._generate_svg(output_file)
            
            print(f"SVG已生成并保存到: {output_file}")
            return True
            
        except Exception as e:
            print(f"转换PNG到SVG时出错: {e}")
            return False
    
    def _extract_color_layers(self, img):
        """从图像中提取颜色层，并确保正确的图层顺序"""
        # 将图像转换为RGB（如果不是）
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 创建掩码，只包括非透明区域
        if hasattr(self, 'alpha_mask'):
            mask = self.alpha_mask
        else:
            # 如果没有Alpha通道，假设整个图像都是非透明的
            mask = np.ones((self.height, self.width), dtype=bool)
        
        # 重塑图像以进行聚类，只包括非透明区域
        pixels = img.reshape(-1, 3)
        valid_pixels = pixels[mask.flatten()]
        
        # 如果没有有效像素，返回
        if len(valid_pixels) == 0:
            return
        
        # 确定颜色数量
        if self.original_svg_data:
            # 从原始SVG中提取唯一颜色
            unique_colors = set()
            for elem in self.original_svg_data.get('elements', []):
                if 'style' in elem and 'fill' in elem['style']:
                    fill_color = elem['style']['fill']
                    if fill_color != 'none':
                        unique_colors.add(fill_color)
                elif 'attributes' in elem and 'fill' in elem['attributes']:
                    fill_color = elem['attributes']['fill']
                    if fill_color != 'none':
                        unique_colors.add(fill_color)
            
            # 将十六进制颜色转换为RGB
            rgb_colors = []
            for color in unique_colors:
                if color.startswith('#'):
                    # 处理#RGB或#RRGGBB格式
                    color = color.lstrip('#')
                    if len(color) == 3:
                        r, g, b = [int(c + c, 16) for c in color]
                    else:
                        r, g, b = [int(color[i:i+2], 16) for i in (0, 2, 4)]
                    rgb_colors.append([r, g, b])
            
            n_colors = max(len(rgb_colors), 2)  # 至少使用2种颜色
            
            # 如果有预定义颜色，使用它们作为初始聚类中心
            if rgb_colors:
                kmeans = KMeans(n_clusters=n_colors, init=np.array(rgb_colors), n_init=1)
            else:
                kmeans = KMeans(n_clusters=n_colors)
        else:
            # 如果没有原始SVG，使用K-means确定主要颜色
            # 使用轮廓系数确定最佳聚类数
            from sklearn.metrics import silhouette_score
            
            # 取样本以加速计算
            sample_size = min(10000, valid_pixels.shape[0])
            pixel_sample = valid_pixels[np.random.choice(valid_pixels.shape[0], sample_size, replace=False)]
            
            best_score = -1
            best_k = 2
            
            # 尝试不同的聚类数
            for k in range(2, min(11, len(pixel_sample))):
                kmeans = KMeans(n_clusters=k, n_init=10)
                cluster_labels = kmeans.fit_predict(pixel_sample)
                
                # 计算轮廓系数
                silhouette_avg = silhouette_score(pixel_sample, cluster_labels)
                
                if silhouette_avg > best_score:
                    best_score = silhouette_avg
                    best_k = k
            
            n_colors = best_k
            kmeans = KMeans(n_clusters=n_colors)
        
        # 拟合K-means，只使用非透明区域的像素
        kmeans.fit(valid_pixels)
        
        # 获取聚类中心（颜色）
        colors = kmeans.cluster_centers_.astype(int)
        
        # 创建一个标签数组，初始化为-1（表示透明区域）
        labels = np.ones(pixels.shape[0], dtype=int) * -1
        
        # 只为非透明区域预测标签
        labels[mask.flatten()] = kmeans.predict(valid_pixels)
        
        # 为每个颜色创建掩码，排除透明区域
        color_layers = {}
        for i in range(n_colors):
            color_mask = (labels == i).reshape(self.height, self.width)
            
            # 将RGB颜色转换为十六进制
            color = '#{:02x}{:02x}{:02x}'.format(colors[i][0], colors[i][1], colors[i][2])
            
            # 计算这个颜色层的平均深度（如果有深度信息）
            brightness = 0.299 * colors[i][0] + 0.587 * colors[i][1] + 0.114 * colors[i][2]
            
            # 存储颜色层及其深度信息
            color_layers[color] = {
                'mask': color_mask,
                'brightness': brightness,
                'area': np.sum(color_mask)  # 计算区域大小
            }
        
        # 根据深度和区域大小对颜色层进行排序
        sorted_colors = sorted(
            color_layers.keys(),
            key=lambda c: (color_layers[c]['brightness'], -color_layers[c]['area'])
        )
        
        # 按排序顺序存储颜色层
        self.color_layers_order = sorted_colors
        self.color_layers = {color: color_layers[color]['mask'] for color in sorted_colors}
    
    def _create_paths_from_mask(self, mask):
        """从二值掩码创建SVG路径（使用曲线拟合）"""
        # 确保掩码是二值图像
        binary_mask = (mask * 255).astype(np.uint8)
        
        # 应用形态学操作来清理掩码
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # 稍微膨胀掩码，使相邻区域有重叠
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        
        # 使用OpenCV查找轮廓
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        svg_paths = []
        for i, contour in enumerate(contours):
            # 跳过太小的轮廓
            if cv2.contourArea(contour) < 10:
                continue
            
            # 跳过图像边界轮廓
            x, y, w, h = cv2.boundingRect(contour)
            if (x <= 1 and y <= 1 and w >= self.width - 3 and h >= self.height - 3):
                continue
            
            # 使用更小的epsilon值进行轮廓简化，保留更多细节
            epsilon = 0.0005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 创建SVG路径数据
            path_data = []
            
            # 移动到第一个点
            path_data.append(f"M {approx[0][0][0]} {approx[0][0][1]}")
            
            # 使用贝塞尔曲线拟合轮廓点
            if len(approx) > 2:
                # 将轮廓点转换为适合贝塞尔曲线拟合的格式
                points = [point[0] for point in approx]
                
                # 闭合轮廓
                points.append(points[0])
                
                # 使用三次贝塞尔曲线拟合
                for i in range(0, len(points) - 1, 3):
                    if i + 3 < len(points):
                        # 完整的贝塞尔曲线段
                        p0 = points[i]
                        p1 = points[i + 1]
                        p2 = points[i + 2]
                        p3 = points[i + 3]
                        
                        # 添加贝塞尔曲线命令
                        path_data.append(f"C {p1[0]} {p1[1]}, {p2[0]} {p2[1]}, {p3[0]} {p3[1]}")
                    else:
                        # 处理剩余的点
                        for j in range(i + 1, len(points)):
                            path_data.append(f"L {points[j][0]} {points[j][1]}")
            else:
                # 对于简单轮廓，使用线段
                for point in approx[1:]:
                    path_data.append(f"L {point[0][0]} {point[0][1]}")
            
            # 闭合路径
            path_data.append("Z")
            
            svg_paths.append(" ".join(path_data))
        
        return svg_paths
    
    def _optimize_paths(self):
        """使用原始SVG数据优化路径"""
        if not self.original_svg_data:
            return
        
        # 提取原始SVG中的路径数据
        original_paths = []
        for elem in self.original_svg_data.get('elements', []):
            if elem.get('type') == 'path' and 'attributes' in elem and 'd' in elem['attributes']:
                path_data = elem['attributes']['d']
                
                # 获取填充颜色
                fill_color = None
                if 'style' in elem and 'fill' in elem['style']:
                    fill_color = elem['style']['fill']
                elif 'fill' in elem['attributes']:
                    fill_color = elem['attributes']['fill']
                
                if fill_color and fill_color != 'none':
                    original_paths.append({
                        'path': path_data,
                        'color': fill_color
                    })
        
        # 如果没有原始路径，返回
        if not original_paths:
            return
        
        # 为每个生成的路径找到最匹配的原始路径
        optimized_paths = []
        for gen_path in self.svg_paths:
            best_match = None
            best_score = float('inf')
            
            for orig_path in original_paths:
                # 如果颜色相似，考虑路径匹配
                if self._colors_are_similar(gen_path['color'], orig_path['color']):
                    # 计算路径相似度分数
                    score = self._path_similarity_score(gen_path['path'], orig_path['path'])
                    if score < best_score:
                        best_score = score
                        best_match = orig_path
            
            # 如果找到匹配，使用原始路径
            if best_match and best_score < 0.5:  # 阈值可调整
                optimized_paths.append({
                    'path': best_match['path'],
                    'color': best_match['color']
                })
            else:
                # 否则保留生成的路径
                optimized_paths.append(gen_path)
        
        # 更新路径
        self.svg_paths = optimized_paths
    
    def _colors_are_similar(self, color1, color2):
        """检查两种颜色是否相似"""
        # 将颜色转换为RGB
        rgb1 = self._hex_to_rgb(color1)
        rgb2 = self._hex_to_rgb(color2)
        
        # 计算欧几里得距离
        distance = sum((c1 - c2) ** 2 for c1, c2 in zip(rgb1, rgb2)) ** 0.5
        
        # 如果距离小于阈值，认为颜色相似
        return distance < 50  # 阈值可调整
    
    def _hex_to_rgb(self, hex_color):
        """将十六进制颜色转换为RGB"""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            return [int(c + c, 16) for c in hex_color]
        return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    
    def _path_similarity_score(self, path1, path2):
        """计算两个SVG路径之间的相似度分数"""
        # 提取路径中的所有点
        points1 = self._extract_points_from_path(path1)
        points2 = self._extract_points_from_path(path2)
        
        # 如果点数差异太大，认为不匹配
        if abs(len(points1) - len(points2)) > min(len(points1), len(points2)) * 0.5:
            return float('inf')
        
        # 计算点集之间的Hausdorff距离
        max_min_dist = 0
        for p1 in points1:
            min_dist = min(self._point_distance(p1, p2) for p2 in points2)
            max_min_dist = max(max_min_dist, min_dist)
        
        return max_min_dist
    
    def _extract_points_from_path(self, path):
        """从SVG路径中提取点"""
        # 使用正则表达式提取所有坐标
        coords = re.findall(r'([MLHVCSQTAZmlhvcsqtaz])([^MLHVCSQTAZmlhvcsqtaz]*)', path)
        
        points = []
        current_point = (0, 0)
        
        for cmd, params in coords:
            # 提取参数
            params = params.strip()
            if params:
                params = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', params)
                params = [float(p) for p in params]
            else:
                params = []
            
            # 根据命令类型处理点
            if cmd in 'Mm':  # 移动
                if len(params) >= 2:
                    if cmd == 'M':
                        current_point = (params[0], params[1])
                    else:  # m
                        current_point = (current_point[0] + params[0], current_point[1] + params[1])
                    points.append(current_point)
            
            elif cmd in 'Ll':  # 线段
                if len(params) >= 2:
                    if cmd == 'L':
                        current_point = (params[0], params[1])
                    else:  # l
                        current_point = (current_point[0] + params[0], current_point[1] + params[1])
                    points.append(current_point)
            
            elif cmd in 'Hh':  # 水平线
                if params:
                    if cmd == 'H':
                        current_point = (params[0], current_point[1])
                    else:  # h
                        current_point = (current_point[0] + params[0], current_point[1])
                    points.append(current_point)
            
            elif cmd in 'Vv':  # 垂直线
                if params:
                    if cmd == 'V':
                        current_point = (current_point[0], params[0])
                    else:  # v
                        current_point = (current_point[0], current_point[1] + params[0])
                    points.append(current_point)
            
            elif cmd in 'Cc':  # 三次贝塞尔曲线
                if len(params) >= 6:
                    # 添加控制点和终点
                    if cmd == 'C':
                        points.append((params[0], params[1]))
                        points.append((params[2], params[3]))
                        current_point = (params[4], params[5])
                    else:  # c
                        points.append((current_point[0] + params[0], current_point[1] + params[1]))
                        points.append((current_point[0] + params[2], current_point[1] + params[3]))
                        current_point = (current_point[0] + params[4], current_point[1] + params[5])
                    points.append(current_point)
            
            # 其他命令类型可以类似处理...
        
        return points
    
    def _point_distance(self, p1, p2):
        """计算两点之间的欧几里得距离"""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    
    def _generate_svg(self, output_file):
        """生成SVG文件，确保正确的图层顺序"""
        # 创建SVG文档
        doc = minidom.getDOMImplementation().createDocument(None, "svg", None)
        root = doc.documentElement
        
        # 设置SVG属性
        root.setAttribute("xmlns", "http://www.w3.org/2000/svg")
        root.setAttribute("width", str(self.width))
        root.setAttribute("height", str(self.height))
        root.setAttribute("viewBox", f"0 0 {self.width} {self.height}")
        
        # 添加透明背景设置
        root.setAttribute("style", "background-color: transparent;")
        
        # 如果有原始SVG数据，复制其命名空间和根属性
        if self.original_svg_data:
            # 添加命名空间
            for ns_name, ns_uri in self.original_svg_data.get('namespaces', {}).items():
                if ns_name == 'default':
                    continue  # 已添加默认命名空间
                root.setAttribute(f"xmlns:{ns_name}", ns_uri)
            
            # 添加其他根元素属性
            for attr_name, attr_value in self.original_svg_data.get('root_attributes', {}).items():
                # 跳过已添加的基本属性
                if attr_name in ['width', 'height', 'viewBox', 'style']:
                    continue
                root.setAttribute(attr_name, attr_value)
        
        # 不过滤白色路径，因为我们现在处理透明背景
        # 直接使用所有路径
        filtered_paths = self.svg_paths
        
        # 按照颜色层的顺序添加路径元素
        # 首先创建一个按颜色分组的路径字典
        paths_by_color = {}
        for path_data in filtered_paths:
            color = path_data['color']
            if color not in paths_by_color:
                paths_by_color[color] = []
            paths_by_color[color].append(path_data['path'])
        
        # 按照颜色层的顺序添加路径
        # 注意：我们反转顺序，因为SVG中后添加的元素会显示在上层
        for color in reversed(self.color_layers_order):
            if color in paths_by_color:
                for path in paths_by_color[color]:
                    path_elem = doc.createElement("path")
                    path_elem.setAttribute("d", path)
                    path_elem.setAttribute("fill", color)
                    root.appendChild(path_elem)
        
        # 写入SVG文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(doc.toprettyxml(indent="  "))
        
        # 清理XML格式
        self._clean_svg_file(output_file)

    def _clean_svg_file(self, svg_file):
        """清理SVG文件，移除多余的空白行"""
        with open(svg_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 移除空白行
        lines = [line for line in lines if line.strip()]
        
        with open(svg_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)

def batch_convert_directory(input_dir="OUTPUTS", output_dir="SVG", optimize=True):
    """批量转换目录中的所有PNG图像为SVG"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"将转换 {input_dir} 中的所有PNG图像并保存到 {output_dir}")
    
    # 获取输入目录中的所有PNG文件
    png_files = glob.glob(os.path.join(input_dir, "*.png"))
    
    if not png_files:
        print(f"在 {input_dir} 目录中未找到PNG文件")
        return
    
    print(f"找到 {len(png_files)} 个PNG文件")
    
    # 转换每个PNG文件
    for png_file in png_files:
        # 获取文件名（不含路径和扩展名）
        base_name = os.path.basename(png_file)
        file_name = os.path.splitext(base_name)[0]
        
        # 构建输出SVG文件路径
        output_file = os.path.join(output_dir, f"{file_name}.svg")
        
        print(f"正在转换: {png_file} -> {output_file}")
        
        # 创建转换器并执行转换
        converter = PNGtoSVGConverter(png_file)
        success = converter.convert(output_file, optimize)
        
        if success:
            print(f"成功转换: {output_file}")
        else:
            print(f"转换失败: {png_file}")
    
    print(f"批量转换完成。共处理 {len(png_files)} 个文件")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PNG到SVG转换器')
    parser.add_argument('--batch', action='store_true', help='批量转换OUTPUTS目录中的所有PNG文件')
    parser.add_argument('--input-dir', default='OUTPUTS', help='输入目录路径 (默认: OUTPUTS)')
    parser.add_argument('--output-dir', default='SVG', help='输出目录路径 (默认: SVG)')
    parser.add_argument('png_file', nargs='?', help='要转换的PNG文件路径 (单文件模式)')
    parser.add_argument('-s', '--svg-json', help='原始SVG的JSON数据文件路径')
    parser.add_argument('-o', '--output', help='输出SVG文件路径 (单文件模式)')
    parser.add_argument('--no-optimize', action='store_true', help='禁用使用原始SVG数据进行优化')
    
    args = parser.parse_args()
    
    # 批量转换模式
    if args.batch:
        batch_convert_directory(args.input_dir, args.output_dir, not args.no_optimize)
    # 单文件转换模式
    elif args.png_file:
        converter = PNGtoSVGConverter(args.png_file, args.svg_json)
        converter.convert(args.output, not args.no_optimize)
    else:
        # 如果没有指定PNG文件且不是批量模式，显示帮助
        parser.print_help()

if __name__ == "__main__":
    # 如果直接运行脚本且没有参数，默认执行批量转换
    if len(sys.argv) == 1:
        batch_convert_directory()
    else:
        main() 