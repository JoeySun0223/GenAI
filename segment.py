#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAM 自动/框提示分割（使用 PNG 透明背景的 alpha 做打孔）
- 批量处理 ~/segment-anything/input/*.{jpg,png,...}
- 输出到 ~/segment-anything/output
  - *_vis.png         全图可视化
  - *_masks.npz       所有掩膜(0/1)与meta
  - *_summary.json    参数与统计
  - <basename>/       每个不重叠区域一张 RGBA PNG（保留原图颜色，背景透明）
"""

import os, sys, glob, json
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch

from segment_anything import (
    sam_model_registry,
    SamAutomaticMaskGenerator,
    SamPredictor,
)

# =================== 路径与模型配置（按需修改） ===================
HOME        = Path.home()
ROOT_DIR    = HOME / "GenAI"
INPUT_DIR   = ROOT_DIR / "MultiView_fixed"
OUTPUT_DIR  = ROOT_DIR / "Seg_output"
CKPT_PATH   = ROOT_DIR / "checkpoints" / "sam_vit_h_4b8939.pth"   # 改成你要用的权重
MODEL_TYPE  = "vit_h"  # 可选：vit_h / vit_l / vit_b
# ===============================================================

# —— 更接近官方 Demo 的自动分割参数（可按需微调）——
AUTO_KW = dict(
    points_per_side=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=25
)

# —— 导出控制（可通过命令行覆盖）——
EXPORT_PRIORITY = "area"  # "area" 或 "iou"
DEFAULT_MIN_AREA = 25    # 导出单块“唯一像素”最小阈值；如块偏少可降到 200~500

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def hsv_color(i: int, n: int):
    n = max(n, 1)
    hue = int(179 * (i % n) / n)  # OpenCV HSV: H∈[0,179]
    bgr = cv2.cvtColor(np.uint8([[[hue, 200, 240]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

def overlay_nicer(image_bgr: np.ndarray, masks: list) -> np.ndarray:
    out = image_bgr.copy()
    if not masks:
        return out
    ordered = sorted(masks, key=lambda m: -int(m.get("area", 0)))
    n = len(ordered)
    for i, m in enumerate(ordered):
        seg = m["segmentation"].astype(bool)
        color = np.array(hsv_color(i, n), dtype=np.uint8)
        out[seg] = (0.6 * out[seg] + 0.4 * color).astype(np.uint8)
    for m in ordered:
        seg = m["segmentation"].astype(np.uint8)
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 0, 0), 1)
    return out

def save_masks_npz(npz_path: Path, masks: list, h: int, w: int):
    stack = (
        np.stack([m["segmentation"].astype(np.uint8) for m in masks], axis=0)
        if masks else np.zeros((0, h, w), np.uint8)
    )
    meta = [
        {
            "area": int(m.get("area", 0)),
            "bbox": [int(x) for x in m.get("bbox", [0, 0, 0, 0])],
            "predicted_iou": float(m.get("predicted_iou", 0.0)),
            "stability_score": float(m.get("stability_score", 0.0)),
            "point_coords": m.get("point_coords", None),
            "crop_box": m.get("crop_box", None),
        }
        for m in masks
    ]
    # 注意：np.object_ 兼容旧 numpy；新版本可改 dtype=object
    np.savez_compressed(str(npz_path), masks=stack, meta=np.array(meta, dtype=np.object_))

# ========= 仅修改：读图时保留 alpha，并在预处理后同步缩放 alpha =========
def load_and_preprocess_with_alpha(path: str):
    """
    返回：bgr(3通道，原尺寸原像素)、bg_mask(bool, True=背景/alpha==0)
    仅用 alpha 做背景打孔，不做任何放大/锐化。
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # 可能是 RGBA
    if img is None:
        return None, None

    if img.ndim == 3 and img.shape[2] == 4:
        bgr   = img[:, :, :3].copy()
        alpha = img[:, :, 3]
        bg_mask = (alpha == 0)
    elif img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        bg_mask = None
    else:
        bgr = img
        bg_mask = None

    return bgr, bg_mask
# ===================================================================

def build_sam_and_device():
    if not CKPT_PATH.exists():
        print(f"[ERR] 找不到 checkpoint: {CKPT_PATH}", file=sys.stderr)
        sys.exit(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=str(CKPT_PATH))
    
    # 加载微调后的权重，允许额外的Adapter层
    checkpoint = torch.load(str(CKPT_PATH), map_location=device)
    sam.load_state_dict(checkpoint, strict=False)
    print(f"[INFO] 成功加载微调模型，包含额外Adapter层")
    
    sam.to(device)
    return sam, device

def run_auto(image_bgr: np.ndarray, sam) -> list:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask_generator = SamAutomaticMaskGenerator(sam, **AUTO_KW)
    masks = mask_generator.generate(rgb)
    return masks

# ========= 仅修改：导出时用 alpha 背景打孔，其他逻辑不变 =========
def save_non_overlapping_cutouts(
    mask_list: list,
    bgr: np.ndarray,
    save_dir: Path,
    priority: str = "area",
    min_area: int = DEFAULT_MIN_AREA,
    export_background: bool = False,
    bg_mask: np.ndarray | None = None
):
    """
    - 小块优先（面积升序），避免被大块吃掉
    - 导出 RGBA（保留原图颜色），非重叠
    - 用 alpha 背景打孔：seg &= ~bg_mask
    """
    ensure_dir(save_dir)
    H, W = bgr.shape[:2]
    taken = np.zeros((H, W), dtype=bool)

    # 排序：小块在前（次级按 predicted_iou 降序）
    ordered = sorted(
        mask_list,
        key=lambda m: (int(m.get("area", 0)), -float(m.get("predicted_iou", 0.0)))
    )

    count = 0
    for m in ordered:
        seg = m["segmentation"].astype(bool)
        if bg_mask is not None:
            seg = np.logical_and(seg, ~bg_mask)  # 打孔

        seg_unique = np.logical_and(seg, ~taken)
        uniq = int(seg_unique.sum())
        if uniq < min_area:
            continue

        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[..., :3] = bgr
        rgba[..., 3] = np.where(seg_unique, 255, 0).astype(np.uint8)

        out_path = save_dir / f"part_{count:03d}.png"
        cv2.imwrite(str(out_path), rgba)

        taken |= seg_unique
        count += 1

    if export_background:
        bg_left = ~taken
        if bg_left.sum() >= min_area:
            rgba = np.zeros((H, W, 4), dtype=np.uint8)
            rgba[..., :3] = bgr
            rgba[..., 3] = np.where(bg_left, 255, 0).astype(np.uint8)
            cv2.imwrite(str(save_dir / f"part_{count:03d}_background.png"), rgba)
# ===================================================================

def main():
    import argparse
    import shutil
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["auto"], default="auto",
                    help="auto=全图自动分割；box=整图内缩大框聚焦主体（单一mask）")
    ap.add_argument("--margin", type=int, default=8, help="box 模式下的内缩像素")
    ap.add_argument("--priority", choices=["area", "iou"], default=EXPORT_PRIORITY,
                    help="导出非重叠部分的优先级（仅影响次序，小块优先恒定）")
    ap.add_argument("--min-area", type=int, default=DEFAULT_MIN_AREA,
                    help="导出单块的最小唯一像素阈值，块偏少就调低（如 300~500）")
    ap.add_argument("--export-background", action="store_true",
                    help="把剩余像素另存为背景 PNG")
    args = ap.parse_args()

    # 每次运行都清空输出目录
    if OUTPUT_DIR.exists():
        print(f"[INFO] 清空输出目录: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    
    ensure_dir(OUTPUT_DIR)

    sam, device = build_sam_and_device()
    print(f"[INFO] model={MODEL_TYPE}  device={device}")
    print(f"[INFO] ckpt ={CKPT_PATH}")
    print(f"[INFO] mode ={args.mode}")
    print(f"[INFO] in   ={INPUT_DIR}")
    print(f"[INFO] out  ={OUTPUT_DIR}")

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.tif", "*.tiff")
    imgs = []
    for e in exts:
        imgs.extend(glob.glob(str(INPUT_DIR / e)))
    imgs.sort()
    if not imgs:
        print(f"[WARN] {INPUT_DIR} 下未发现图片。")
        return

    for p in imgs:
        name = Path(p).stem
        print(f"[PROC] {p}")
        bgr, bg_mask = load_and_preprocess_with_alpha(p)
        if bgr is None:
            print(f"[SKIP] 读图失败：{p}")
            continue

        vis_path  = OUTPUT_DIR / f"{name}_vis.png"
        npz_path  = OUTPUT_DIR / f"{name}_masks.npz"
        json_path = OUTPUT_DIR / f"{name}_summary.json"
        part_dir  = OUTPUT_DIR / name  # 存每个非重叠 cutout 的文件夹

        if args.mode == "auto":
            masks = run_auto(bgr, sam)
            #vis = overlay_nicer(bgr, masks)
            #cv2.imwrite(str(vis_path), vis)
            #save_masks_npz(npz_path, masks, bgr.shape[0], bgr.shape[1])

            # —— 用 alpha 背景打孔，导出非重叠 RGBA —— #
            save_non_overlapping_cutouts(
                masks, bgr, part_dir,
                priority=args.priority,
                min_area=args.min_area,
                export_background=args.export_background,
                bg_mask=bg_mask
            )

            #summary = dict(
            #    image=p,
            #    output_vis=str(vis_path),
            #    output_npz=str(npz_path),
            #    parts_dir=str(part_dir),
            #    num_masks=len(masks),
            #    export_priority=args.priority,
            #    mode="auto",
            #    model_type=MODEL_TYPE,
            #    device=device,
            #    params=AUTO_KW,
            #    min_area=args.min_area,
            #    used_alpha_bg=(bg_mask is not None),
            #    time=datetime.now().isoformat(timespec="seconds"),
            #)
            #with open(json_path, "w", encoding="utf-8") as f:
            #    json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"[DONE] {name}: masks={len(masks)} → {vis_path}, {part_dir}")

if __name__ == "__main__":
    main()