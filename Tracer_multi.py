import os
import vtracer
from PIL import Image
import glob
import xml.etree.ElementTree as ET
from xml.dom import minidom
import re
import numpy as np

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

def convert_image_to_svg_in_memory(input_path, params):
    """将单个图片转换为SVG字符串（不保存到文件）"""
    try:
        img = Image.open(input_path).convert('RGBA')
        pixels = list(img.getdata())
        size = img.size
        svg_str = vtracer.convert_pixels_to_svg(
            pixels,
            size=size,
            **params
        )
        return svg_str
    except Exception as e:
        print(f"转换失败 {input_path}: {str(e)}")
        return None

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

def process_directory(input_dir, output_dir, params):
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    if not subdirs:
        print(f"在 {input_dir} 中没有找到子目录！")
        return
    print(f"找到 {len(subdirs)} 个子目录")
    for subdir in subdirs:
        input_subdir = os.path.join(input_dir, subdir)
        png_files = glob.glob(os.path.join(input_subdir, "*.png"))
        if not png_files:
            print(f"在 {input_subdir} 中没有找到PNG文件")
            continue
        print(f"\n处理子目录 {subdir}，找到 {len(png_files)} 个PNG文件")
        svg_strings = []
        for png_file in png_files:
            svg_str = convert_image_to_svg_in_memory(png_file, params)
            if svg_str:
                svg_strings.append(svg_str)
        if svg_strings:
            merged_svg = os.path.join(output_dir, f"{subdir}_merged.svg")
            if merge_svg_strings(svg_strings, merged_svg):
                print(f"成功合并 {len(svg_strings)} 个SVG为 {merged_svg}")
            else:
                print(f"合并SVG失败")

def main():
    os.makedirs("SVG_OUTPUT", exist_ok=True)
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
    process_directory("memory_output_parts", "SVG_OUTPUT", params)

if __name__ == "__main__":
    main() 