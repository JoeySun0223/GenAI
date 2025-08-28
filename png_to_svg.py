#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PNG转SVG转换器
- 批量处理 output_parts 下所有子文件夹的PNG图片
- 转换为SVG并保存到 beforecombine_svg 目录
- 保持原有的目录结构
"""

import os
import vtracer
from PIL import Image
import glob
import shutil

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

def process_directory(input_dir, output_dir, params):
    """处理目录，将PNG转换为SVG"""
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
        
        # 创建对应的输出子目录
        output_subdir = os.path.join(output_dir, subdir)
        os.makedirs(output_subdir, exist_ok=True)
        
        success_count = 0
        for i, png_file in enumerate(png_files):
            svg_str = convert_image_to_svg_in_memory(png_file, params)
            if svg_str:
                # 保存单个SVG文件
                png_name = os.path.splitext(os.path.basename(png_file))[0]
                single_svg_path = os.path.join(output_subdir, f"{png_name}.svg")
                try:
                    with open(single_svg_path, "w", encoding='utf-8') as f:
                        f.write(svg_str)
                    print(f"  ✓ 保存: {single_svg_path}")
                    success_count += 1
                except Exception as e:
                    print(f"  ✗ 保存失败 {single_svg_path}: {str(e)}")
            else:
                print(f"  ✗ 转换失败: {png_file}")
        
        print(f"  完成: {success_count}/{len(png_files)} 个文件成功转换")

def main():
    # 每次运行都清空输出目录
    if os.path.exists("beforecombine_svg"):
        print(f"[INFO] 清空输出目录: beforecombine_svg")
        shutil.rmtree("beforecombine_svg")
    
    os.makedirs("beforecombine_svg", exist_ok=True)
    
    # vtracer参数配置
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
    
    print("开始PNG转SVG转换...")
    print(f"输入目录: output_parts")
    print(f"输出目录: beforecombine_svg")
    
    process_directory("output_parts", "beforecombine_svg", params)
    
    print(f"\n转换完成！结果保存在: beforecombine_svg")

if __name__ == "__main__":
    main()
