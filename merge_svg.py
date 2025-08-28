#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVG合并器
- 批量处理 deal_svg 下所有子文件夹的SVG文件
- 按路径面积排序合并，面积大的在底层
- 输出到 SVG_OUTPUT 目录
"""

import os
import glob
import xml.etree.ElementTree as ET
from xml.dom import minidom
import re
import numpy as np
import shutil

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

def merge_svg_files(svg_files, output_path, width=512, height=512):
    """合并多个SVG文件为一个SVG文件，按路径面积排序（面积大的在底层）"""
    root = ET.Element("svg")
    root.set("xmlns", "http://www.w3.org/2000/svg")
    root.set("width", str(width))
    root.set("height", str(height))
    root.set("viewBox", f"0 0 {width} {height}")
    root.set("style", "background-color: transparent;")
    
    all_paths = []
    for svg_file in svg_files:
        try:
            tree = ET.parse(svg_file)
            svg_root = tree.getroot()
            for elem in svg_root.findall(".//{http://www.w3.org/2000/svg}path"):
                path_data = elem.get('d', '')
                if path_data:
                    area = calculate_path_area(path_data)
                    all_paths.append((elem, area))
        except Exception as e:
            print(f"处理SVG文件时出错 {svg_file}: {str(e)}")
            continue
    
    # 按面积排序，面积大的在底层
    all_paths.sort(key=lambda x: x[1], reverse=True)
    
    # 添加所有路径到根元素
    for elem, area in all_paths:
        root.append(elem)
    
    try:
        rough_string = ET.tostring(root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_string = reparsed.toprettyxml(indent="  ")
        pretty_string = '\n'.join([line for line in pretty_string.split('\n') if line.strip()])
        
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(pretty_string)
        
        print(f"✓ 成功合并 {len(all_paths)} 个路径到: {output_path}")
        return True
    except Exception as e:
        print(f"✗ 保存合并后的SVG文件时出错: {str(e)}")
        return False

def process_directory(input_dir, output_dir):
    """处理目录，合并SVG文件"""
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    if not subdirs:
        print(f"在 {input_dir} 中没有找到子目录！")
        return
    
    print(f"找到 {len(subdirs)} 个子目录")
    
    for subdir in subdirs:
        input_subdir = os.path.join(input_dir, subdir)
        svg_files = glob.glob(os.path.join(input_subdir, "*.svg"))
        
        if not svg_files:
            print(f"在 {input_subdir} 中没有找到SVG文件")
            continue
        
        print(f"\n处理子目录 {subdir}，找到 {len(svg_files)} 个SVG文件")
        
        # 合并SVG文件
        merged_svg = os.path.join(output_dir, f"{subdir}_merged.svg")
        if merge_svg_files(svg_files, merged_svg):
            print(f"  完成: {subdir}_merged.svg")
        else:
            print(f"  失败: {subdir}_merged.svg")

def main():
    # 每次运行都清空输出目录
    if os.path.exists("SVG_OUTPUT"):
        print(f"[INFO] 清空输出目录: SVG_OUTPUT")
        shutil.rmtree("SVG_OUTPUT")
    
    os.makedirs("SVG_OUTPUT", exist_ok=True)
    
    print("开始SVG合并...")
    print(f"输入目录: deal_svg")
    print(f"输出目录: SVG_OUTPUT")
    
    # 检查输入目录是否存在
    if not os.path.exists("deal_svg"):
        print(f"错误: 输入目录 deal_svg 不存在")
        print("请先运行 png_to_svg.py 生成SVG文件，或确保 deal_svg 目录存在")
        return
    
    process_directory("deal_svg", "SVG_OUTPUT")
    
    print(f"\n合并完成！结果保存在: SVG_OUTPUT")

if __name__ == "__main__":
    main()
