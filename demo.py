#!/usr/bin/env python3
"""
Pipeline脚本：按顺序运行多个处理脚本
运行顺序：segment.py -> seg_filter.py -> png2svg.py -> svg_filter.py -> merge_svg.py -> svg_color_corr.py
"""

import subprocess
import time
import sys
import os
from datetime import datetime

def run_script(script_name, description):
    """
    运行指定的Python脚本
    
    Args:
        script_name (str): 脚本文件名
        description (str): 脚本描述
    
    Returns:
        bool: 运行是否成功
    """
    print(f"\n{'='*60}")
    print(f"正在运行: {description}")
    print(f"脚本: {script_name}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # 检查脚本文件是否存在
        if not os.path.exists(script_name):
            print(f"错误: 脚本文件 {script_name} 不存在!")
            return False
        
        # 运行脚本
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, 
                              text=True, 
                              cwd=os.getcwd())
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} 运行成功!")
            print(f"运行时间: {duration:.2f} 秒")
            
            # 如果有输出，显示最后几行
            if result.stdout.strip():
                print("\n脚本输出:")
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines[-5:]:  # 显示最后5行
                    print(f"  {line}")
            
            return True
        else:
            print(f"❌ {description} 运行失败!")
            print(f"错误代码: {result.returncode}")
            print(f"运行时间: {duration:.2f} 秒")
            
            if result.stderr.strip():
                print("\n错误信息:")
                print(result.stderr)
            
            return False
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ 运行 {description} 时发生异常: {str(e)}")
        print(f"运行时间: {duration:.2f} 秒")
        return False

def main():
    """主函数"""
    print("🚀 开始运行处理管道...")
    print(f"工作目录: {os.getcwd()}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 记录总开始时间
    total_start_time = time.time()
    
    # 定义要运行的脚本列表
    scripts = [
        ("MVrender.py", "多视图渲染(Trellis)"),
        ("segment.py", "图像分割处理"),
        ("seg_filter.py", "分割结果过滤"),
        ("png2svg.py", "PNG转SVG"),
        ("svg_filter.py", "SVG过滤处理"),
        ("merge_svg.py", "SVG合并"),
        ("svg_color_corr.py", "SVG颜色校正")
    ]
    
    # 运行统计
    successful_runs = 0
    failed_runs = 0
    script_times = []
    
    # 按顺序运行每个脚本
    for i, (script_name, description) in enumerate(scripts, 1):
        print(f"\n📋 步骤 {i}/{len(scripts)}")
        
        script_start_time = time.time()
        success = run_script(script_name, description)
        script_end_time = time.time()
        script_duration = script_end_time - script_start_time
        
        script_times.append((script_name, description, script_duration, success))
        
        if success:
            successful_runs += 1
        else:
            failed_runs += 1
            print(f"\n⚠️  脚本 {script_name} 运行失败，是否继续运行后续脚本?")
            print("输入 'y' 继续，输入 'n' 停止:")
            
            try:
                user_input = input().strip().lower()
                if user_input != 'y':
                    print("用户选择停止运行管道")
                    break
            except KeyboardInterrupt:
                print("\n用户中断运行")
                break
    
    # 计算总运行时间
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # 显示运行总结
    print(f"\n{'='*80}")
    print("📊 运行总结")
    print(f"{'='*80}")
    print(f"总运行时间: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)")
    print(f"成功运行: {successful_runs} 个脚本")
    print(f"失败运行: {failed_runs} 个脚本")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n📋 详细运行时间:")
    for script_name, description, duration, success in script_times:
        status = "✅" if success else "❌"
        print(f"  {status} {script_name}: {duration:.2f} 秒 - {description}")
    
    if failed_runs == 0:
        print(f"\n🎉 所有脚本运行成功!")
    else:
        print(f"\n⚠️  有 {failed_runs} 个脚本运行失败，请检查错误信息")
    
    return failed_runs == 0

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n\n⚠️  用户中断运行")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 发生未预期的错误: {str(e)}")
        sys.exit(1)
