import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import sys
import imageio
from PIL import Image
import os.path as osp
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

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
    image_path = "assets/example_image/car.png"

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
    # Optional parameters
    # sparse_structure_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 7.5,
    # },
    # slat_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 3,
    # },
)
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes

# Render the outputs
#video = render_utils.render_video(outputs['gaussian'][0])['color']
#imageio.mimsave(osp.join(output_dir, f"{base_filename}_gs.mp4"), video, fps=30)
#video = render_utils.render_video(outputs['radiance_field'][0])['color']
#imageio.mimsave(osp.join(output_dir, f"{base_filename}_rf.mp4"), video, fps=30)
#video = render_utils.render_video(outputs['mesh'][0])['normal']
#imageio.mimsave(osp.join(output_dir, f"{base_filename}_mesh.mp4"), video, fps=30)

# GLB files can be extracted from the outputs
glb = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    # Optional parameters
    simplify=0.95,          # Ratio of triangles to remove in the simplification process
    texture_size=1024,      # Size of the texture used for the GLB
)

# 保存glb文件到输出目录
glb_filename = f"{base_filename}.glb"
output_glb_path = osp.join(output_dir, glb_filename)
glb.export(output_glb_path)
print(f"已保存GLB文件: {output_glb_path}")

# Save Gaussians as PLY files
output_ply_path = osp.join(output_dir, f"{base_filename}.ply")
outputs['gaussian'][0].save_ply(output_ply_path)
print(f"已保存PLY文件: {output_ply_path}")