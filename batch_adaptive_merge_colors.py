#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量颜色归并：
- 默认：自适应主色发现与合并（面积阈值 + 感知色差阈值）
- 可选：--snap-majority 把每张图所有非透明像素都吸附为“该图出现次数最多的 RGB”
输入根目录: output_parts/
输出根目录: final_output/ （镜像目录 + 每个角度 summary.json）
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
import colorsys

# ---------- 可选依赖 ----------
def _maybe_import_cv2():
    try:
        import cv2
        return cv2
    except Exception:
        return None

def _maybe_import_ciede2000():
    try:
        from skimage.color import deltaE_ciede2000
        return deltaE_ciede2000
    except Exception:
        return None

# ---------- 工具函数 ----------
def rgb01_to_hsv_deg(rgb01: np.ndarray) -> np.ndarray:
    hsv = []
    for r,g,b in rgb01:
        h,s,v = colorsys.rgb_to_hsv(float(r), float(g), float(b))
        hsv.append((h*360.0, s, v))
    return np.array(hsv, dtype=np.float32)

def circ_hue_diff_deg(h1, h2):
    d = abs(h1 - h2) % 360.0
    return min(d, 360.0 - d)

def rgb_to_lab_opencv(rgb_arr: np.ndarray) -> np.ndarray:
    cv2 = _maybe_import_cv2()
    if cv2 is None:
        raise RuntimeError("需要安装 opencv-python 才能使用 --lab")
    arr = rgb_arr.reshape(1, -1, 3).astype(np.uint8)[:, :, ::-1]  # RGB->BGR
    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3)
    return lab

# ---------- 自适应主色（非 snap-majority 模式时使用） ----------
def choose_adaptive_dominants(unique_rgb: np.ndarray,
                              counts: np.ndarray,
                              total_fg: int,
                              use_lab: bool,
                              merge_threshold: float,
                              min_percent: float|None,
                              min_pixels: int|None,
                              use_ciede2000: bool,
                              prototype: str,
                              hue_guard: float,
                              sat_thr: float):
    order = np.argsort(-counts)  # 频次降序
    colors = unique_rgb[order]
    freqs  = counts[order]

    if min_percent is None: min_percent = 0.0
    thr_by_percent = int(np.ceil(total_fg * (min_percent / 100.0)))
    hard_thr = max(thr_by_percent, int(min_pixels or 0))

    # 预转换空间/HSV（护栏）
    colors_hsv = rgb01_to_hsv_deg(colors.astype(np.float32)/255.0)
    if use_lab:
        colors_space = rgb_to_lab_opencv(colors.astype(np.uint8))
    else:
        colors_space = colors.astype(np.float32)

    dom_rgb, dom_space, dom_hsv, dom_count, trace = [], [], [], [], []

    for i, (rgb, sp, c) in enumerate(zip(colors, colors_space, freqs)):
        if c < hard_thr:
            trace.append({"type": "skip_small", "rgb": rgb.tolist(), "count": int(c)})
            continue

        if not dom_rgb:
            dom_rgb.append(rgb.astype(np.float64))
            dom_space.append(sp.astype(np.float64))
            dom_hsv.append(colors_hsv[i].astype(np.float64))
            dom_count.append(float(c))
            trace.append({"type": "new", "rgb": rgb.tolist(), "count": int(c)})
            continue

        dom_space_arr = np.vstack(dom_space)
        if use_lab:
            # 欧氏距离（足够快）；如需 CIEDE2000 可改这里
            dists = np.sqrt(((dom_space_arr - sp) ** 2).sum(axis=1))
        else:
            dists = np.sqrt(((dom_space_arr - sp) ** 2).sum(axis=1))

        j = int(np.argmin(dists))
        d = float(dists[j])

        # 色相护栏：彩色-彩色且色相差过大 -> 不合并
        cand_h, cand_s, _ = colors_hsv[i]
        base_h, base_s, _ = dom_hsv[j]
        hue_far = (cand_s >= sat_thr and base_s >= sat_thr and
                   circ_hue_diff_deg(cand_h, base_h) > hue_guard)

        if hue_far:
            dom_rgb.append(rgb.astype(np.float64))
            dom_space.append(sp.astype(np.float64))
            dom_hsv.append(colors_hsv[i].astype(np.float64))
            dom_count.append(float(c))
            trace.append({"type":"new_hue_guard","rgb":rgb.tolist(),"count":int(c),
                          "hue_diff": circ_hue_diff_deg(cand_h, base_h)})
            continue

        if d <= merge_threshold:
            w_old = dom_count[j]; w_new = w_old + c
            if prototype == "mean":
                dom_rgb[j]   = (dom_rgb[j]   * w_old + rgb * c) / w_new
                dom_space[j] = (dom_space[j] * w_old + sp  * c) / w_new
            # prototype=="first": 锚点不更新
            dom_count[j] = w_new
            trace.append({"type":"merge","rgb":rgb.tolist(),"count":int(c),
                          "merged_into": j, "distance": d})
        else:
            dom_rgb.append(rgb.astype(np.float64))
            dom_space.append(sp.astype(np.float64))
            dom_hsv.append(colors_hsv[i].astype(np.float64))
            dom_count.append(float(c))
            trace.append({"type":"new","rgb":rgb.tolist(),"count":int(c),"distance_to_nearest": d})

    dom_rgb   = np.clip(np.rint(np.vstack(dom_rgb)), 0, 255).astype(np.uint8) if dom_rgb else np.zeros((0,3), np.uint8)
    dom_count = np.array(dom_count, dtype=np.int64) if dom_count else np.zeros((0,), np.int64)
    return dom_rgb, dom_count, trace

# ---------- 单图处理 ----------
def process_one_image(img_path: Path,
                      out_path: Path,
                      params: dict) -> dict:
    img = Image.open(img_path).convert("RGBA")
    arr = np.array(img)
    H, W = arr.shape[:2]
    alpha = arr[..., 3]
    fg_mask = (alpha > 0)
    total_fg = int(fg_mask.sum())
    if total_fg == 0:
        raise RuntimeError("no nontransparent pixels")

    rgb_fg = arr[..., :3][fg_mask]

    # 统计唯一 RGB & 频次
    uniq_rgb, inverse, counts = np.unique(
        rgb_fg.reshape(-1,3), axis=0, return_inverse=True, return_counts=True
    )

    # ---------- 模式 A：snap-majority（你现在想要的） ----------
    if params["snap_majority"]:
        idx_max = int(np.argmax(counts))
        majority_rgb = uniq_rgb[idx_max].astype(np.uint8)
        mapped_rgb = np.tile(majority_rgb, (rgb_fg.shape[0], 1))
        out_arr = arr.copy()
        out_arr[..., :3][fg_mask] = mapped_rgb
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out_arr, mode="RGBA").save(out_path)

        dom_list = [{
            "rank": 1,
            "rgb": majority_rgb.tolist(),
            "mapped_pixels": int(total_fg),
            "mapped_percent": 100.0,
            "accumulated_pixels_during_selection": int(counts[idx_max])
        }]
        mse = float(((rgb_fg.astype(np.float32) - mapped_rgb.astype(np.float32))**2).mean())

        return {
            "file": str(img_path.name),
            "size": [H, W],
            "nontransparent_pixels": total_fg,
            "original_unique_colors": int(uniq_rgb.shape[0]),
            "dominant_colors": dom_list,
            "mse_rgb": mse,
            "K": 1
        }, dom_list

    # ---------- 模式 B：自适应主色合并（默认） ----------
    dom_rgb, dom_count, trace = choose_adaptive_dominants(
        uniq_rgb, counts, total_fg,
        use_lab=params["lab"],
        merge_threshold=params["merge_threshold"],
        min_percent=params["min_percent"],
        min_pixels=params["min_pixels"],
        use_ciede2000=False,
        prototype=params["prototype"],
        hue_guard=params["hue_guard"],
        sat_thr=params["sat_thr"],
    )
    K = dom_rgb.shape[0]

    if params["lab"]:
        src_space = rgb_to_lab_opencv(rgb_fg.astype(np.uint8))
        dst_space = rgb_to_lab_opencv(dom_rgb.astype(np.uint8)) if K>0 else np.zeros((0,3), np.float32)
    else:
        src_space = rgb_fg.astype(np.float32)
        dst_space = dom_rgb.astype(np.float32)

    if K > 0:
        D = ((src_space[:, None, :] - dst_space[None, :, :]) ** 2).sum(axis=2)
        nearest = np.argmin(D, axis=1)
        mapped_rgb = dom_rgb[nearest].astype(np.uint8)
    else:
        nearest = np.zeros((rgb_fg.shape[0],), dtype=int)
        mapped_rgb = rgb_fg

    out_arr = arr.copy()
    out_arr[..., :3][fg_mask] = mapped_rgb
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out_arr, mode="RGBA").save(out_path)

    mapped_counts = np.bincount(nearest, minlength=K).astype(int) if K>0 else np.array([], dtype=int)
    mse = float(((rgb_fg.astype(np.float32) - mapped_rgb.astype(np.float32))**2).mean())

    order = np.argsort(-mapped_counts) if K>0 else np.array([], int)
    dom_list = []
    for rank, j in enumerate(order, 1):
        dom_list.append({
            "rank": rank,
            "rgb": dom_rgb[j].tolist(),
            "mapped_pixels": int(mapped_counts[j]),
            "mapped_percent": float(mapped_counts[j] / total_fg * 100.0),
            "accumulated_pixels_during_selection": int(dom_count[j]),
        })

    return {
        "file": str(img_path.name),
        "size": [H, W],
        "nontransparent_pixels": total_fg,
        "original_unique_colors": int(uniq_rgb.shape[0]),
        "dominant_colors": dom_list,
        "mse_rgb": mse,
        "K": int(K),
    }, dom_list

# ---------- 批量 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root",  default="output_parts", help="输入根目录（角度子文件夹）")
    ap.add_argument("--output-root", default="final_output",  help="输出根目录")
    ap.add_argument("--exts", nargs="+", default=[".png"], help="处理的文件扩展名")
    # snap-majority 模式
    ap.add_argument("--snap-majority", action="store_true",
                    help="将每张图所有非透明像素都改成该图中出现次数最多的 RGB（单色吸附）")
    # 自适应模式参数（snap-majority 关闭时生效）
    ap.add_argument("--min-percent", type=float, default=0.5, help="候选主色最小占比(%)")
    ap.add_argument("--min-pixels",  type=int,   default=0,   help="候选主色最小像素数")
    ap.add_argument("--lab", action="store_true", help="在 CIE Lab 空间做距离")
    ap.add_argument("--merge-threshold", type=float, default=8.0, help="主色合并阈值（Lab 6~12）")
    ap.add_argument("--prototype", choices=["first","mean"], default="first",
                    help="主色锚点：first=不更新(稳)，mean=加权平均")
    ap.add_argument("--hue-guard", type=float, default=18.0, help="色相护栏阈值（度）")
    ap.add_argument("--sat-thr",   type=float, default=0.12, help="彩色判定最小饱和度(0~1)")
    args = ap.parse_args()

    params = {
        "snap_majority": args.snap_majority,
        "min_percent": args.min_percent,
        "min_pixels":  args.min_pixels,
        "lab": args.lab,
        "merge_threshold": args.merge_threshold,
        "prototype": args.prototype,
        "hue_guard": args.hue_guard,
        "sat_thr": args.sat_thr,
    }

    in_root  = Path(args.input_root)
    out_root = Path(args.output_root)
    if not in_root.exists():
        print(f"[ERR] 输入根目录不存在：{in_root}", file=sys.stderr)
        sys.exit(1)

    angle_dirs = [p for p in sorted(in_root.iterdir()) if p.is_dir()]
    if not angle_dirs:
        print(f"[WARN] 未发现角度子文件夹：{in_root}")
        sys.exit(0)

    for angle_dir in angle_dirs:
        out_dir = out_root / angle_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        files = [p for p in sorted(angle_dir.iterdir())
                 if p.is_file() and p.suffix.lower() in {e.lower() for e in args.exts}]
        if not files:
            print(f"[INFO] 跳过（无图片）：{angle_dir}")
            continue

        print(f"[PROC] 角度文件夹：{angle_dir} -> {out_dir}")

        per_image_reports = []
        agg_counts = {}      # 聚合调色板：RGB -> 像素计数
        total_fg_pixels = 0

        for img_path in files:
            out_path = out_dir / img_path.name
            try:
                report_img, dom_list = process_one_image(img_path, out_path, params)
            except Exception as e:
                print(f"[SKIP] {img_path.name}: {e}")
                continue

            per_image_reports.append(report_img)
            total_fg_pixels += report_img["nontransparent_pixels"]
            for d in dom_list:
                key = tuple(d["rgb"])
                agg_counts[key] = agg_counts.get(key, 0) + int(d["mapped_pixels"])

        agg_sorted = sorted(agg_counts.items(), key=lambda kv: -kv[1])
        agg_palette = [
            {
                "rank": i+1,
                "rgb": list(rgb),
                "mapped_pixels": int(cnt),
                "mapped_percent": float(cnt / total_fg_pixels * 100.0) if total_fg_pixels>0 else 0.0
            }
            for i, (rgb, cnt) in enumerate(agg_sorted)
        ]

        summary = {
            "angle_dir": str(angle_dir),
            "output_dir": str(out_dir),
            "num_images": len(per_image_reports),
            "mode": "snap-majority" if params["snap_majority"] else "adaptive-merge",
            "params": params,
            "aggregate": {
                "total_nontransparent_pixels": int(total_fg_pixels),
                "dominant_palette": agg_palette
            },
            "images": per_image_reports
        }
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"[DONE] 写入 {out_dir/'summary.json'}，图片数：{len(per_image_reports)}")

if __name__ == "__main__":
    main()
