import os
import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

def get_main_color(region_pixels):
    if len(region_pixels) == 0:
        return (0, 0, 0)
    kmeans = KMeans(n_clusters=1, n_init=1)
    kmeans.fit(region_pixels)
    return tuple(map(int, kmeans.cluster_centers_[0]))

def mask_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0

# 配置
input_dir = "MultiView"  # 存放所有待分割图片
output_dir = "output_parts"
os.makedirs(output_dir, exist_ok=True)

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
model = build_sam2(model_cfg, checkpoint)
mask_generator = SAM2AutomaticMaskGenerator(
    model,
    points_per_side=64,
    points_per_batch=128,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.92,
    stability_score_offset=0.7,
    crop_n_layers=1,
    box_nms_thresh=0.7,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=25,
    use_m2m=True
)

for img_name in os.listdir(input_dir):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    orig_img = Image.open(os.path.join(input_dir, img_name))
    image = np.array(orig_img.convert("RGB"))

    # 自动生成所有掩码
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks = mask_generator.generate(image)

    # 获取原图 alpha 通道（透明度）
    if orig_img.mode == "RGBA":
        orig_alpha = np.array(orig_img.split()[-1])
        non_transparent_mask = orig_alpha > 0
    else:
        non_transparent_mask = np.ones(image.shape[:2], dtype=bool)

    # 去除高度重叠的掩码（IoU>0.8）
    filtered_masks = []
    for mask_dict in masks:
        mask = mask_dict["segmentation"]
        if all(mask_iou(mask, m["segmentation"]) < 0.8 for m in filtered_masks):
            filtered_masks.append(mask_dict)

    # 输出每个分割部件
    base_name = os.path.splitext(img_name)[0]
    part_dir = os.path.join(output_dir, base_name)
    os.makedirs(part_dir, exist_ok=True)
    saved_count = 0
    for i, mask_dict in enumerate(filtered_masks):
        mask = mask_dict["segmentation"]
        effective_mask = np.logical_and(mask, non_transparent_mask)
        if np.sum(effective_mask) == 0:
            continue
        # 保持原图色彩
        rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgba[..., :3][effective_mask] = image[effective_mask]
        rgba[..., 3][effective_mask] = 255
        part_img = Image.fromarray(rgba, mode="RGBA")
        part_img.save(os.path.join(part_dir, f"part_{saved_count+1}.png"))
        saved_count += 1
    print(f"{img_name} 共保存 {saved_count} 个分割部件到 {part_dir}")

print("全部图片分割完成！")
