#!/usr/bin/env python3
"""
SVG清理工具：将填充改为透明，描边改为黑色并增粗，保留所有路径轮廓
批量处理版本：自动处理SVG_OUTPUT文件夹中的所有SVG文件
"""

import xml.etree.ElementTree as ET
import re
import os
import argparse
from pathlib import Path
import shutil

class SVGCleaner:
    def __init__(self):
        # 需要改为透明的填充相关属性
        self.fill_attributes = [
            'fill', 'fill-opacity', 'fill-rule'
        ]
        
        # 需要改为黑色并增粗的描边相关属性
        self.stroke_attributes = [
            'stroke', 'stroke-width', 'stroke-opacity', 'stroke-linecap', 
            'stroke-linejoin', 'stroke-miterlimit', 'stroke-dasharray', 'stroke-dashoffset'
        ]
        
        self.style_attributes = [
            'style'
        ]
        
        # 需要移除的渐变和滤镜ID（因为改为透明后不再需要）
        self.gradient_ids = set()
        self.filter_ids = set()
        self.pattern_ids = set()
        
    def clean_svg_file(self, input_path, output_path=None):
        """清理单个SVG文件"""
        if output_path is None:
            # 在原文件名后添加_clean后缀
            input_file = Path(input_path)
            output_path = input_file.parent / f"{input_file.stem}_clean{input_file.suffix}"
        
        try:
            # 解析SVG
            tree = ET.parse(input_path)
            root = tree.getroot()
            
            # 清理SVG
            self._clean_svg_element(root)
            
            # 保存清理后的SVG
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            
            print(f"✓ 已清理: {input_path} -> {output_path}")
            return True
            
        except Exception as e:
            print(f"✗ 处理失败 {input_path}: {e}")
            return False
    
    def clean_svg_directory(self, input_dir, output_dir=None):
        """清理目录中的所有SVG文件"""
        input_path = Path(input_dir)
        
        if output_dir is None:
            output_dir = input_path.parent / f"{input_path.name}_cleaned"
        
        output_path = Path(output_dir)
        
        # 清空输出文件夹
        if output_path.exists():
            print(f"清空输出文件夹: {output_path}")
            shutil.rmtree(output_path)
        
        # 重新创建输出文件夹
        output_path.mkdir(exist_ok=True)
        
        # 查找所有SVG文件
        svg_files = list(input_path.glob("*.svg"))
        
        if not svg_files:
            print(f"在 {input_dir} 中没有找到SVG文件")
            return
        
        print(f"找到 {len(svg_files)} 个SVG文件")
        
        success_count = 0
        for svg_file in svg_files:
            output_file = output_path / svg_file.name
            if self.clean_svg_file(svg_file, output_file):
                success_count += 1
        
        print(f"\n清理完成: {success_count}/{len(svg_files)} 个文件成功处理")
        print(f"结果保存在: {output_path}")
    
    def batch_process_svg_output(self):
        """
        批量处理SVG_OUTPUT文件夹中的SVG文件
        输出到SVG_line文件夹
        """
        input_dir = "SVG_OUTPUT"
        output_dir = "SVG_line"
        
        print("开始批量处理SVG文件...")
        print(f"输入文件夹: {input_dir}")
        print(f"输出文件夹: {output_dir}")
        
        # 检查输入文件夹是否存在
        if not os.path.exists(input_dir):
            print(f"错误: 输入文件夹 {input_dir} 不存在")
            return
        
        # 每次运行都清空输出文件夹
        if os.path.exists(output_dir):
            print(f"[INFO] 清空输出目录: {output_dir}")
            shutil.rmtree(output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出文件夹: {output_dir}")
        
        # 查找所有SVG文件
        svg_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith('.svg'):
                    svg_files.append(os.path.join(root, file))
        
        if not svg_files:
            print(f"在 {input_dir} 中没有找到SVG文件")
            return
        
        print(f"找到 {len(svg_files)} 个SVG文件")
        
        # 处理每个SVG文件
        success_count = 0
        for svg_file in svg_files:
            try:
                # 计算相对路径，保持文件夹结构
                rel_path = os.path.relpath(svg_file, input_dir)
                output_file = os.path.join(output_dir, rel_path)
                
                # 确保输出目录存在
                output_dir_path = os.path.dirname(output_file)
                os.makedirs(output_dir_path, exist_ok=True)
                
                # 清理SVG文件
                if self.clean_svg_file(svg_file, output_file):
                    success_count += 1
                
            except Exception as e:
                print(f"处理文件 {svg_file} 时出错: {e}")
        
        print(f"\n批量处理完成: {success_count}/{len(svg_files)} 个文件成功处理")
        print(f"结果保存在: {output_dir}")
    
    def batch_process_beforecombine_svg(self):
        """
        批量处理SVG_Segmented_fixed文件夹中的SVG文件
        输出到SVG_Segmented_line文件夹
        """
        input_dir = "SVG_Segmented_fixed"
        output_dir = "SVG_Segmented_line"
        
        print("开始批量处理SVG_Segmented_fixed文件...")
        print(f"输入文件夹: {input_dir}")
        print(f"输出文件夹: {output_dir}")
        
        # 检查输入文件夹是否存在
        if not os.path.exists(input_dir):
            print(f"错误: 输入文件夹 {input_dir} 不存在")
            return
        
        # 每次运行都清空输出文件夹
        if os.path.exists(output_dir):
            print(f"[INFO] 清空输出目录: {output_dir}")
            shutil.rmtree(output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出文件夹: {output_dir}")
        
        # 查找所有SVG文件
        svg_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith('.svg'):
                    svg_files.append(os.path.join(root, file))
        
        if not svg_files:
            print(f"在 {input_dir} 中没有找到SVG文件")
            return
        
        print(f"找到 {len(svg_files)} 个SVG文件")
        
        # 处理每个SVG文件
        success_count = 0
        for svg_file in svg_files:
            try:
                # 计算相对路径，保持文件夹结构
                rel_path = os.path.relpath(svg_file, input_dir)
                output_file = os.path.join(output_dir, rel_path)
                
                # 确保输出目录存在
                output_dir_path = os.path.dirname(output_file)
                os.makedirs(output_dir_path, exist_ok=True)
                
                # 清理SVG文件
                if self.clean_svg_file(svg_file, output_file):
                    success_count += 1
                
            except Exception as e:
                print(f"处理文件 {svg_file} 时出错: {e}")
        
        print(f"\n批量处理完成: {success_count}/{len(svg_files)} 个文件成功处理")
        print(f"结果保存在: {output_dir}")
    
    def _clean_svg_element(self, element):
        """递归清理SVG元素"""
        # 清理当前元素
        self._remove_color_attributes(element)
        
        # 递归处理子元素
        for child in element:
            self._clean_svg_element(child)
    
    def _remove_color_attributes(self, element):
        """将填充改为透明，描边改为黑色并增粗"""
        # 将填充属性改为透明
        for attr in self.fill_attributes:
            if attr in element.attrib:
                if attr == 'fill':
                    element.attrib[attr] = 'none'  # 透明填充
                elif attr == 'fill-opacity':
                    element.attrib[attr] = '0'     # 完全透明
                # fill-rule 保留，因为它是几何属性
        
        # 将描边属性改为黑色并增粗 - 修复：确保描边可见
        for attr in self.stroke_attributes:
            if attr in element.attrib:
                if attr == 'stroke':
                    element.attrib[attr] = 'black'  # 黑色描边
                elif attr == 'stroke-width':
                    # 修复：确保描边宽度足够大，能清晰显示
                    if element.attrib[attr]:
                        try:
                            current_width = float(element.attrib[attr])
                            # 如果原始宽度太小，设置为更大的值
                            if current_width < 1.0:
                                element.attrib[attr] = '2.0'  # 最小宽度2.0
                            else:
                                element.attrib[attr] = str(current_width * 1.5)
                        except ValueError:
                            element.attrib[attr] = '2.0'  # 默认宽度2.0
                    else:
                        element.attrib[attr] = '2.0'  # 默认宽度2.0
                elif attr == 'stroke-opacity':
                    element.attrib[attr] = '1'     # 完全不透明
                # 其他描边属性保留，因为它们是几何属性
        
        # 修复：确保每个元素都有描边属性
        if 'stroke' not in element.attrib:
            element.attrib['stroke'] = 'black'
        if 'stroke-width' not in element.attrib:
            element.attrib['stroke-width'] = '2.0'
        if 'stroke-opacity' not in element.attrib:
            element.attrib['stroke-opacity'] = '1'
        
        # 处理style属性
        if 'style' in element.attrib:
            style = element.attrib['style']
            # 将颜色相关的CSS样式改为透明填充和黑色描边
            style = self._clean_style_string(style)
            if style.strip():
                element.attrib['style'] = style
            else:
                del element.attrib['style']
        
        # 处理渐变和滤镜引用
        if 'fill' in element.attrib:
            fill_value = element.attrib['fill']
            if fill_value.startswith('url('):
                # 记录需要移除的渐变ID
                match = re.search(r'url\(#([^)]+)\)', fill_value)
                if match:
                    self.gradient_ids.add(match.group(1))
                # 将渐变填充改为透明
                element.attrib['fill'] = 'none'
        
        if 'filter' in element.attrib:
            filter_value = element.attrib['filter']
            if filter_value.startswith('url('):
                # 记录需要移除的滤镜ID
                match = re.search(r'url\(#([^)]+)\)', filter_value)
                if match:
                    self.filter_ids.add(match.group(1))
                # 将滤镜改为无
                element.attrib['filter'] = 'none'
    
    def _clean_style_string(self, style):
        """将CSS样式中的填充改为透明，描边改为黑色并增粗"""
        # 将填充相关的CSS属性改为透明
        fill_properties = [
            (r'fill:[^;]+;?', 'fill:none;'),
            (r'fill-opacity:[^;]+;?', 'fill-opacity:0;'),
            (r'fill-rule:[^;]+;?', ''),  # 移除fill-rule，因为它是几何属性
        ]
        
        # 将描边相关的CSS属性改为黑色并增粗 - 修复：确保描边可见
        stroke_properties = [
            (r'stroke:[^;]+;?', 'stroke:black;'),
            (r'stroke-width:[^;]+;?', 'stroke-width:2.0;'),  # 修复：设置为2.0确保可见
            (r'stroke-opacity:[^;]+;?', 'stroke-opacity:1;'),
            (r'stroke-linecap:[^;]+;?', ''),  # 保留几何属性
            (r'stroke-linejoin:[^;]+;?', ''),  # 保留几何属性
            (r'stroke-miterlimit:[^;]+;?', ''),  # 保留几何属性
            (r'stroke-dasharray:[^;]+;?', ''),  # 保留几何属性
            (r'stroke-dashoffset:[^;]+;?', '')   # 保留几何属性
        ]
        
        # 处理填充属性
        for pattern, replacement in fill_properties:
            if replacement:
                style = re.sub(pattern, replacement, style)
            else:
                style = re.sub(pattern, '', style)
        
        # 处理描边属性
        for pattern, replacement in stroke_properties:
            if replacement:
                style = re.sub(pattern, replacement, style)
            else:
                style = re.sub(pattern, '', style)
        
        # 修复：确保style中包含必要的描边属性
        if 'stroke:' not in style:
            style += 'stroke:black;'
        if 'stroke-width:' not in style:
            style += 'stroke-width:2.0;'
        if 'stroke-opacity:' not in style:
            style += 'stroke-opacity:1;'
        
        # 清理多余的分号和空格
        style = re.sub(r';+', ';', style)
        style = re.sub(r'^\s*;\s*', '', style)
        style = re.sub(r'\s*;\s*$', '', style)
        
        return style.strip()
    
    def _remove_unused_definitions(self, root):
        """移除未使用的渐变、滤镜等定义"""
        # 移除渐变定义
        for gradient in root.findall(".//{*}linearGradient"):
            if 'id' in gradient.attrib and gradient.attrib['id'] in self.gradient_ids:
                gradient.getparent().remove(gradient)
        
        for gradient in root.findall(".//{*}radialGradient"):
            if 'id' in gradient.attrib and gradient.attrib['id'] in self.gradient_ids:
                gradient.getparent().remove(gradient)
        
        # 移除滤镜定义
        for filter_elem in root.findall(".//{*}filter"):
            if 'id' in filter_elem.attrib and filter_elem.attrib['id'] in self.filter_ids:
                filter_elem.getparent().remove(filter_elem)
        
        # 移除图案定义
        for pattern in root.findall(".//{*}pattern"):
            if 'id' in pattern.attrib and pattern.attrib['id'] in self.pattern_ids:
                pattern.getparent().remove(pattern)
    
    def clean_svg_advanced(self, input_path, output_path=None, remove_definitions=True):
        """高级清理：同时移除未使用的定义"""
        if output_path is None:
            input_file = Path(input_path)
            output_path = input_file.parent / f"{input_file.stem}_clean{input_file.suffix}"
        
        try:
            tree = ET.parse(input_path)
            root = tree.getroot()
            
            # 清理SVG元素
            self._clean_svg_element(root)
            
            # 移除未使用的定义
            if remove_definitions:
                self._remove_unused_definitions(root)
            
            # 保存清理后的SVG
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            
            print(f"✓ 已高级清理: {input_path} -> {output_path}")
            return True
            
        except Exception as e:
            print(f"✗ 高级清理失败 {input_path}: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="SVG清理工具：将填充改为透明，描边改为黑色并增粗，保留所有路径轮廓")
    parser.add_argument("input", nargs='?', help="输入SVG文件或目录（可选，默认使用批量处理模式）")
    parser.add_argument("-o", "--output", help="输出路径")
    parser.add_argument("-r", "--recursive", action="store_true", help="递归处理子目录")
    parser.add_argument("--advanced", action="store_true", help="高级清理：移除未使用的定义")
    parser.add_argument("--batch", action="store_true", help="批量处理模式：处理SVG_OUTPUT文件夹中的所有SVG文件")
    parser.add_argument("--Segmented", action="store_true", help="处理SVG_Segmented_fixed文件夹中的所有SVG文件")    
    args = parser.parse_args()
    
    cleaner = SVGCleaner()
    
    # 处理SVG_Segmented_fixed文件夹
    if args.Segmented:
        print("使用SVG_Segmented_fixed处理模式...")
        cleaner.batch_process_beforecombine_svg()
        return
    
    # 如果没有指定输入参数，默认使用批量处理模式
    if args.input is None or args.batch:
        print("使用批量处理模式...")
        cleaner.batch_process_svg_output()
        return
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"错误: 输入路径不存在: {input_path}")
        return
    
    if input_path.is_file():
        # 处理单个文件
        if args.advanced:
            cleaner.clean_svg_advanced(input_path, args.output)
        else:
            cleaner.clean_svg_file(input_path, args.output)
    
    elif input_path.is_dir():
        # 处理目录
        cleaner.clean_svg_directory(input_path, args.output)
    
    else:
        print(f"错误: 不支持的路径类型: {input_path}")

if __name__ == "__main__":
    main()