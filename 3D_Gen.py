import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
#os.environ['ATTN_BACKEND'] = 'flash-attn'   # 使用flash-attn可能提供更好的性能
os.environ['SPCONV_ALGO'] = 'native'          # 改回'auto'可能在某些情况下提供更好的精度
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import sys
import imageio
from PIL import Image
import os.path as osp
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import torch
import numpy as np

# 创建输出目录
output_dir = "OUTPUTS"
os.makedirs(output_dir, exist_ok=True)
print(f"输出文件将保存至: {output_dir}/")

# 处理命令行参数
if len(sys.argv) > 1:
    # 获取命令行参数中的文件名
    file_name = sys.argv[1]
    # 检查文件名是否带有扩展名
    if not file_name.endswith('.png') and not file_name.endswith('.jpg'):
        file_name = f"{file_name}.png"  # 默认添加.png扩展名
    
    # 构建完整路径
    image_path = f"assets/example_image/{file_name}"
else:
    # 默认图像路径
    image_path = "assets/example_image/redcar.png"

print(f"处理图像: {image_path}")

# 尝试使用相对路径
pipeline = TrellisImageTo3DPipeline.from_pretrained("TRELLIS-image-large")
pipeline.cuda()

# 提取不带扩展名的文件名
base_filename = osp.splitext(osp.basename(image_path))[0]

# 检查文件是否存在
if not osp.exists(image_path):
    print(f"错误: 找不到图像文件 {image_path}")
    sys.exit(1)

# Load an image
image = Image.open(image_path)

# Run the pipeline
outputs = pipeline.run(
    image,
    seed=1,
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

# 渲染图片
render_results = render_utils.render_frames(
    outputs['gaussian'][0], 
    extrinsics, 
    intrinsics, 
    {'resolution': 512, 'bg_color': None}  # 设置为None以启用透明背景
)

# 保存每张图片
for i, img_data in enumerate(render_results['color']):
    # 使用预先计算的角度值，确保精确
    angle = angles_degrees[i]
    img_path = osp.join(output_dir, f"{base_filename}_{angle:03d}deg.png")  # 使用3位数格式化
    
    # 如果有alpha通道，创建RGBA图像
    if 'alpha' in render_results:
        alpha_data = render_results['alpha'][i]
        rgba_img = np.zeros((img_data.shape[0], img_data.shape[1], 4), dtype=np.uint8)
        rgba_img[:,:,0:3] = img_data
        rgba_img[:,:,3] = alpha_data.squeeze()
        Image.fromarray(rgba_img).save(img_path)
    else:
        # 否则保存为RGB图像
        Image.fromarray(img_data).save(img_path)
    
    print(f"已保存图片: {img_path}")

# 注释掉视频生成代码
#video = render_utils.render_video(outputs['gaussian'][0])['color']
#imageio.mimsave(osp.join(output_dir, f"{base_filename}_gs.mp4"), video, fps=30)

# 注释掉GLB文件生成和保存的代码
# GLB files can be extracted from the outputs
#glb = postprocessing_utils.to_glb(
#    outputs['gaussian'][0],
#    outputs['mesh'][0],
#    # 调整这些参数以提高精度
#    simplify=0.8,          # 从0.95减少到0.8，保留更多三角形（更少简化）
#    texture_size=2048,     # 从1024增加到2048，提高纹理分辨率
#)

# 保存glb文件到输出目录
#glb_filename = f"{base_filename}.glb"
#output_glb_path = osp.join(output_dir, glb_filename)
#glb.export(output_glb_path)
#print(f"已保存GLB文件: {output_glb_path}")

# 注释掉PLY文件保存的代码
# Save Gaussians as PLY files
#output_ply_path = osp.join(output_dir, f"{base_filename}.ply")
#outputs['gaussian'][0].save_ply(output_ply_path)
#print(f"已保存PLY文件: {output_ply_path}")

print(f"已完成渲染，共生成 {num_images} 张图片")