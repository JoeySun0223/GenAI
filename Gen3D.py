import os
#os.environ['ATTN_BACKEND'] = 'xformers'   # 使用 xformers 替代默认的 flash-attn
#os.environ['ATTN_BACKEND'] = 'flash-attn'   # 使用flash-attn可能提供更好的性能
os.environ['SPCONV_ALGO'] = 'native'          # 改回'auto'可能在某些情况下提供更好的精度
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

# 添加内存优化设置
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 更好的错误追踪

import sys
import imageio
from PIL import Image
import os.path as osp
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import torch
import numpy as np
from scipy.ndimage import binary_dilation
import random
import json

# torch设置（在导入后设置）
torch.backends.cudnn.benchmark = True     # 优化卷积性能
torch.backends.cudnn.deterministic = False # 允许非确定性算法以提高性能

def print_gpu_memory_info():
    """打印详细的GPU内存信息"""
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"总内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"已分配: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        print(f"已缓存: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")
        print(f"可用内存: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3:.1f} GB")
        print("-" * 50)

def force_clean_gpu_memory():
    """强制清理GPU内存"""
    if torch.cuda.is_available():
        # 清理PyTorch缓存
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        # 重置内存分配器
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        
        print("GPU内存已强制清理")
        print_gpu_memory_info()

# 强制清理GPU内存
force_clean_gpu_memory()

# 创建输出目录
output_dir = "MultiView"
os.makedirs(output_dir, exist_ok=True)

# 清空输出目录中的现有图片文件
import glob
existing_files = glob.glob(os.path.join(output_dir, "*.png"))
for file in existing_files:
    try:
        os.remove(file)
        print(f"已删除: {file}")
    except Exception as e:
        print(f"删除文件失败 {file}: {e}")

print(f"输出文件将保存至: {output_dir}/")

# 处理命令行参数
if len(sys.argv) > 1:
    # 获取命令行参数中的文件名
    file_name = sys.argv[1]
    # 检查文件名是否带有扩展名
    if not file_name.endswith('.png') and not file_name.endswith('.jpg'):
        file_name = f"{file_name}.png"  # 默认添加.png扩展名
    
    # 构建完整路径
    image_path = f"example_image/{file_name}"
else:
    # 默认图像路径
    image_path = "example_image/bluecar.png"

print(f"处理图像: {image_path}")

# 尝试使用相对路径
pipeline = TrellisImageTo3DPipeline.from_pretrained("TRELLIS-image-large")
pipeline.cuda()

print("Pipeline已加载到GPU")
print_gpu_memory_info()

# 提取不带扩展名的文件名
base_filename = osp.splitext(osp.basename(image_path))[0]

# 检查文件是否存在
if not osp.exists(image_path):
    print(f"错误: 找不到图像文件 {image_path}")
    sys.exit(1)

# Load an image
image = Image.open(image_path)

# 检查图像尺寸，如果太大则调整
max_size = 1024
if max(image.size) > max_size:
    ratio = max_size / max(image.size)
    new_size = tuple(int(dim * ratio) for dim in image.size)
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    print(f"图像已调整大小从 {image.size} 到 {new_size}")

print(f"处理图像尺寸: {image.size}")
print_gpu_memory_info()

# Run the pipeline with error handling
try:
    print("开始运行pipeline...")
    outputs = pipeline.run(
        image,
        seed=1,
        #seed=random.randint(0, 2**32-1),
        # 以下参数取消注释并调整
        #sparse_structure_sampler_params={
        #    "steps": 24,  # 从12增加到24提高采样精度
        #    "cfg_strength": 7.5,  # 可以尝试增加到9.0增强细节
        #},
        #slat_sampler_params={
        #    "steps": 24,  # 同样增加步数
        #    "cfg_strength": 4.0,  # 可以适当增加以提高细节表现
        #},
    )
    print("Pipeline运行完成")
    print_gpu_memory_info()
except RuntimeError as e:
    if "out of memory" in str(e):
        print("CUDA内存不足！尝试以下解决方案：")
        print("1. 关闭其他占用GPU的程序")
        print("2. 重启程序")
        print("3. 如果问题持续，可以尝试降低图像分辨率")
        print("4. 检查是否有其他Python进程占用GPU内存")
        print(f"错误详情: {e}")
        
        # 尝试清理内存后重新运行
        print("尝试清理内存后重新运行...")
        force_clean_gpu_memory()
        
        # 如果还是失败，尝试使用更小的图像
        if max(image.size) > 512:
            ratio = 512 / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            print(f"尝试使用更小的图像尺寸: {new_size}")
            
            try:
                outputs = pipeline.run(image, seed=1)
                print("使用较小图像成功运行")
            except RuntimeError as e2:
                print(f"即使使用较小图像仍然失败: {e2}")
                sys.exit(1)
        else:
            sys.exit(1)
    else:
        print(f"运行时错误: {e}")
        sys.exit(1)
except Exception as e:
    print(f"未知错误: {e}")
    sys.exit(1)
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes

# 修改渲染部分，每隔30度输出一张图片，从0度到330度（共12张）
num_images = 12
# 直接指定12个角度点，确保精确的30度间隔
angles_degrees = [i * 30 for i in range(12)]  # [0, 30, 60, ..., 330]
# 转换为弧度
yaws = [angle * np.pi / 180 for angle in angles_degrees]
pitch = [0.0] * num_images  # 保持水平视角

# 修正参数名称：使用rs而不是r，使用fovs而不是fov
extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, rs=2, fovs=40)

# 渲染图片 - 使用对比方法生成准确的alpha通道
print("正在渲染黑色背景...")
render_results_black = render_utils.render_frames(
    outputs['gaussian'][0], 
    extrinsics, 
    intrinsics,
    {'resolution': 512, 'bg_color': [0, 0, 0]}  # 黑色背景
)

print("正在渲染白色背景...")
render_results_white = render_utils.render_frames(
    outputs['gaussian'][0], 
    extrinsics, 
    intrinsics,
    {'resolution': 512, 'bg_color': [1, 1, 1]}  # 白色背景
)

def generate_alpha_from_contrast(black_img, white_img, threshold=0.1):
    """
    通过对比黑色背景和白色背景的渲染结果生成alpha通道
    """
    # 转换为float进行计算
    black_float = black_img.astype(np.float32) / 255.0
    white_float = white_img.astype(np.float32) / 255.0
    
    # 计算差异
    diff = np.abs(white_float - black_float)
    
    # 如果差异很大，说明是模型区域（应该不透明）
    # 如果差异很小，说明是背景区域（应该透明）
    # 但是我们需要反转逻辑：差异大的区域设为不透明
    alpha = np.mean(diff, axis=2) > threshold
    
    # 返回alpha通道：模型区域为255（不透明），背景区域为0（透明）
    return alpha.astype(np.uint8) * 255

# 保存每张图片
for i in range(len(render_results_black['color'])):
    # 使用预先计算的角度值，确保精确
    angle = angles_degrees[i]
    img_path = osp.join(output_dir, f"{base_filename}_{angle:03d}deg.png")  # 使用3位数格式化
    
    # 获取黑色和白色背景的渲染结果
    black_img = render_results_black['color'][i]
    white_img = render_results_white['color'][i]
    
    # 使用黑色背景的图像作为最终颜色
    final_color = black_img
    
    # 通过对比生成alpha通道
    alpha_channel = generate_alpha_from_contrast(black_img, white_img)
    
    # 如果alpha通道逻辑反了，直接反转
    alpha_channel = 255 - alpha_channel
    
    # 创建RGBA图像
    rgba = np.dstack([final_color.astype(np.uint8), alpha_channel])
    Image.fromarray(rgba, 'RGBA').save(img_path)
    
    print(f"已保存图片: {img_path}")

# 视频生成代码
#video = render_utils.render_video(outputs['gaussian'][0])['color']
#imageio.mimsave(osp.join(output_dir, f"{base_filename}_gs.mp4"), video, fps=30)

# 在pipeline运行完成后，添加3D数据保存代码
print("正在保存3D模型数据...")

# 保存3D Gaussian模型数据
gaussian_data = {
    'xyz': outputs['gaussian'][0].get_xyz.detach().cpu().numpy(),
    'opacity': outputs['gaussian'][0].get_opacity.detach().cpu().numpy(),
    'scaling': outputs['gaussian'][0].get_scaling.detach().cpu().numpy(),
    'rotation': outputs['gaussian'][0].get_rotation.detach().cpu().numpy(),
    'features': outputs['gaussian'][0].get_features.detach().cpu().numpy()
}

# 保存到numpy文件
np.savez_compressed(
    os.path.join(output_dir, f"{base_filename}_3d_model.npz"),
    **gaussian_data
)

# 保存相机参数 - 修复类型问题
camera_params = {
    'resolution': 512,
    'fov': 40,
    'camera_distance': 2.0,
    'angles_degrees': angles_degrees,
    'yaws': yaws,
    'pitch': pitch,
    'extrinsics': [ext.detach().cpu().numpy() if hasattr(ext, 'detach') else np.array(ext) for ext in extrinsics],
    'intrinsics': [intr.detach().cpu().numpy() if hasattr(intr, 'detach') else np.array(intr) for intr in intrinsics]
}

with open(os.path.join(output_dir, f"{base_filename}_camera_params.json"), 'w') as f:
    json.dump(camera_params, f, indent=2, default=str)

# 保存3D-2D映射关系
print("正在计算3D-2D映射关系...")
mapping_data = {}

for i, angle in enumerate(angles_degrees):
    print(f"计算角度 {angle}° 的3D-2D映射...")
    
    # 获取当前角度的渲染结果
    black_img = render_results_black['color'][i]
    white_img = render_results_white['color'][i]
    
    # 计算alpha通道
    alpha_channel = generate_alpha_from_contrast(black_img, white_img)
    alpha_channel = 255 - alpha_channel  # 反转逻辑
    
    # 获取3D点在这个角度下的2D投影
    # 使用相机参数计算3D到2D的投影
    current_extrinsics = extrinsics[i]
    current_intrinsics = intrinsics[i]
    
    # 确保是张量格式
    if not isinstance(current_extrinsics, torch.Tensor):
        current_extrinsics = torch.tensor(current_extrinsics, dtype=torch.float32)
    if not isinstance(current_intrinsics, torch.Tensor):
        current_intrinsics = torch.tensor(current_intrinsics, dtype=torch.float32)
    
    # 获取3D点坐标
    points_3d = outputs['gaussian'][0].get_xyz.detach()
    
    # 转换到相机坐标系
    points_cam = torch.matmul(current_extrinsics[:3, :3], points_3d.T) + current_extrinsics[:3, 3:4]
    
    # 投影到2D
    points_2d = torch.matmul(current_intrinsics, points_cam)
    points_2d = points_2d[:2] / points_2d[2:3]
    
    # 转换到像素坐标
    points_pixel = points_2d.T.detach().cpu().numpy()
    
    # 过滤在图像范围内的点
    valid_mask = (points_pixel[:, 0] >= 0) & (points_pixel[:, 0] < 512) & \
                 (points_pixel[:, 1] >= 0) & (points_pixel[:, 1] < 512)
    
    valid_points_3d = points_3d[valid_mask].detach().cpu().numpy()
    valid_points_2d = points_pixel[valid_mask]
    
    # 保存映射关系
    mapping_data[f'angle_{angle:03d}'] = {
        'points_3d': valid_points_3d,
        'points_2d': valid_points_2d,
        'alpha_mask': alpha_channel,
        'extrinsics': current_extrinsics.detach().cpu().numpy(),
        'intrinsics': current_intrinsics.detach().cpu().numpy()
    }

# 保存映射数据
np.savez_compressed(
    os.path.join(output_dir, f"{base_filename}_3d_2d_mapping.npz"),
    **mapping_data
)

# 保存PLY文件
print("正在保存PLY文件...")
output_ply_path = osp.join(output_dir, f"{base_filename}.ply")
outputs['gaussian'][0].save_ply(output_ply_path)
print(f"已保存PLY文件: {output_ply_path}")

# 保存GLB文件
print("正在保存GLB文件...")
glb = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    simplify=0.8,
    texture_size=2048,
)

glb_filename = f"{base_filename}.glb"
output_glb_path = osp.join(output_dir, glb_filename)
glb.export(output_glb_path)
print(f"已保存GLB文件: {output_glb_path}")

print(f"3D模型数据已保存到: {output_dir}")
print(f"包含以下文件:")
print(f"  - {base_filename}_3d_model.npz (3D模型数据)")
print(f"  - {base_filename}_camera_params.json (相机参数)")
print(f"  - {base_filename}_3d_2d_mapping.npz (3D-2D映射)")
print(f"  - {base_filename}.ply (PLY格式)")
print(f"  - {base_filename}.glb (GLB格式)")

print(f"已完成渲染，共生成 {num_images} 张图片")

# 程序结束时清理内存
if torch.cuda.is_available():
    # 清理pipeline
    del pipeline
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("GPU内存已清理")