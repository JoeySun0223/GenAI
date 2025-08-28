#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级SVG到PNG转换器
支持批量转换、多种输出格式、自定义尺寸和质量设置
"""

import os
import sys
import argparse
from pathlib import Path
import cairosvg
from PIL import Image
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class AdvancedSVGConverter:
    """高级SVG转换器类"""
    
    def __init__(self, input_dir, output_dir, width=None, height=None, 
                 format='png', quality=95, max_workers=4):
        """
        初始化转换器
        
        Args:
            input_dir (str): 输入目录
            output_dir (str): 输出目录
            width (int, optional): 输出宽度
            height (int, optional): 输出高度
            format (str): 输出格式 ('png', 'jpg', 'jpeg', 'tiff', 'bmp')
            quality (int): 输出质量 (1-100)
            max_workers (int): 最大工作线程数
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.width = width
        self.height = height
        self.format = format.lower()
        self.quality = quality
        self.max_workers = max_workers
        
        # 支持的格式映射
        self.format_mapping = {
            'png': 'png',
            'jpg': 'jpeg',
            'jpeg': 'jpeg',
            'tiff': 'tiff',
            'bmp': 'bmp'
        }
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def convert_single_file(self, svg_file):
        """
        转换单个SVG文件
        
        Args:
            svg_file (Path): SVG文件路径
            
        Returns:
            tuple: (成功标志, 文件名, 错误信息)
        """
        try:
            # 生成输出文件名
            output_filename = svg_file.stem + "." + self.format
            output_path = self.output_dir / output_filename
            
            # 读取SVG文件内容并预处理
            with open(svg_file, 'r', encoding='utf-8') as f:
                svg_content = f.read()
            
            # 修复空的fill属性，将其设置为黑色
            # 匹配 fill="" 或 fill='' 的情况
            import re
            svg_content = re.sub(r'fill=["\']\s*["\']', 'fill="black"', svg_content)
            
            # 使用cairosvg转换
            png_data = cairosvg.svg2png(
                bytestring=svg_content.encode('utf-8'),
                output_width=self.width,
                output_height=self.height
            )
            
            # 如果需要其他格式，先转换为PIL Image
            if self.format != 'png':
                # 从PNG数据创建PIL Image
                img = Image.open(io.BytesIO(png_data))
                
                # 保存为指定格式
                if self.format in ['jpeg', 'jpg']:
                    # JPEG不支持透明，转换为RGB
                    if img.mode in ('RGBA', 'LA', 'P'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                        img = background
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(output_path, format='JPEG', quality=self.quality)
                else:
                    img.save(output_path, format=self.format_mapping[self.format])
            else:
                # 直接保存PNG
                with open(output_path, 'wb') as f:
                    f.write(png_data)
            
            return True, svg_file.name, None
            
        except Exception as e:
            return False, svg_file.name, str(e)
    
    def convert_batch(self):
        """
        批量转换SVG文件
        
        Returns:
            dict: 转换结果统计
        """
        # 查找所有SVG文件
        svg_files = list(self.input_dir.glob("*.svg"))
        
        if not svg_files:
            print(f"在 {self.input_dir} 中没有找到SVG文件")
            return {'success': 0, 'failed': 0, 'total': 0}
        
        print(f"找到 {len(svg_files)} 个SVG文件")
        print(f"输出格式: {self.format.upper()}")
        print(f"输出尺寸: {self.width or '原始尺寸'} x {self.height or '原始尺寸'}")
        print(f"使用 {self.max_workers} 个线程进行转换...")
        print()
        
        start_time = time.time()
        success_count = 0
        failed_count = 0
        
        # 使用线程池进行并行转换
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(self.convert_single_file, svg_file): svg_file 
                for svg_file in svg_files
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_file):
                success, filename, error = future.result()
                
                if success:
                    print(f"✓ 成功转换: {filename}")
                    success_count += 1
                else:
                    print(f"✗ 转换失败: {filename} - {error}")
                    failed_count += 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n转换完成!")
        print(f"总耗时: {duration:.2f} 秒")
        print(f"成功: {success_count} 个文件")
        print(f"失败: {failed_count} 个文件")
        print(f"平均速度: {len(svg_files)/duration:.2f} 文件/秒")
        
        return {
            'success': success_count,
            'failed': failed_count,
            'total': len(svg_files),
            'duration': duration
        }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='高级SVG到图片转换器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python advanced_svg_converter.py                    # 使用默认设置
  python advanced_svg_converter.py -i svg_files -o png_files  # 自定义目录
  python advanced_svg_converter.py -w 1024 -h 768 -f jpg      # 自定义尺寸和格式
  python advanced_svg_converter.py -w 800 -f png -q 90 -j 8   # 高质量PNG，8线程
        """
    )
    
    parser.add_argument('-i', '--input', default='example_svg',
                       help='输入SVG文件夹路径 (默认: example_svg)')
    parser.add_argument('-o', '--output', default='example_image',
                       help='输出图片文件夹路径 (默认: example_image)')
    parser.add_argument('-w', '--width', type=int, default=None,
                       help='输出图片宽度 (像素)')
    parser.add_argument('--height', type=int, default=None,
                       help='输出图片高度 (像素)')
    parser.add_argument('-f', '--format', default='png',
                       choices=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
                       help='输出格式 (默认: png)')
    parser.add_argument('-q', '--quality', type=int, default=95,
                       help='输出质量 (1-100, 默认: 95)')
    parser.add_argument('-j', '--jobs', type=int, default=4,
                       help='并行工作线程数 (默认: 4)')
    
    args = parser.parse_args()
    
    print("高级SVG到图片转换器")
    print("=" * 60)
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"输出格式: {args.format.upper()}")
    print(f"输出尺寸: {args.width or '原始尺寸'} x {args.height or '原始尺寸'}")
    print(f"输出质量: {args.quality}")
    print(f"工作线程: {args.jobs}")
    print()
    
    # 创建转换器并执行转换
    converter = AdvancedSVGConverter(
        input_dir=args.input,
        output_dir=args.output,
        width=args.width,
        height=args.height,
        format=args.format,
        quality=args.quality,
        max_workers=args.jobs
    )
    
    results = converter.convert_batch()
    
    # 返回适当的退出码
    sys.exit(0 if results['failed'] == 0 else 1)

if __name__ == "__main__":
    main() 