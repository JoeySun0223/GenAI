# filter_parts_connected_recursive.py
# 递归批量筛选"部件PNG"（透明背景）——连通度优先
# - 遍历 in_dir 下所有子文件夹的图片
# - 在 out_dir 镜像输出目录结构
# - 连通度判据：lcc>=阈值 或 n<=阈值（两者其一即通过）
# - 图片等比例缩放：最大边长限制为512像素
# 依赖: opencv-python, numpy
# 用法:
#   python filter_parts_connected_recursive.py --in-dir before --out-dir after
# 常用参数:
#   --lcc-min 0.92   最大连通域占比阈值
#   --max-comp 10    允许的有效连通域数上限
#   --tiny-ratio 0.001  忽略更小的碎屑（相对前景）
#   --alpha-thr 0.5  alpha>thr 视为前景
#   --max-size 512   图片最大边长（等比例缩放）
#   --exts ".png,.jpg,.jpeg,.webp" 自定义后缀

import os, json, argparse
from pathlib import Path
import numpy as np
import cv2

# ---------- 基础：读取与Alpha转掩膜 ----------

def resize_image_proportional(img, max_size=512):
    """等比例缩放图片，最大边长不超过max_size"""
    height, width = img.shape[:2]
    
    # 如果图片已经小于等于目标尺寸，直接返回
    if max(height, width) <= max_size:
        return img
    
    # 计算缩放比例
    scale = max_size / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # 使用INTER_AREA进行缩小（质量更好）
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized

def analyze_mask(img_rgba, alpha_thr=0.5):
    H, W = img_rgba.shape[:2]
    if img_rgba.ndim == 2:
        img_rgba = cv2.cvtColor(img_rgba, cv2.COLOR_GRAY2BGRA)
    if img_rgba.shape[2] == 3:
        alpha = np.full((H, W), 1.0, np.float32)
        img_rgba = np.concatenate([img_rgba, (alpha*255).astype(np.uint8)[...,None]], axis=2)
    else:
        alpha = img_rgba[:, :, 3].astype(np.float32) / 255.0
    mask = (alpha > alpha_thr).astype(np.uint8)
    return mask, int(mask.sum()), alpha, img_rgba

# ---------- 连通度指标 ----------

def connectedness_stats(mask, tiny_ratio=0.001):
    total = int(mask.sum())
    if total == 0:
        return 0.0, 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return 1.0, 0
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    tiny_th = max(1, int(total * tiny_ratio))
    valid = areas >= tiny_th
    areas_valid = areas[valid]
    if areas_valid.size == 0:
        return 0.0, 0
    lcc = int(areas_valid.max())
    lcc_ratio = lcc / float(total)
    n_comp = int(valid.sum())
    return lcc_ratio, n_comp

def reconnect_then_check(mask, tiny_ratio=0.001):
    H, W = mask.shape
    k = max(3, int(round(min(H, W) * 0.005)))  # ~0.5%
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return connectedness_stats(closed, tiny_ratio=tiny_ratio)

# ---------- 判据：连通度（独立条件） ----------

def should_keep_by_connectedness(img_rgba,
                                 alpha_thr=0.5,
                                 lcc_ratio_min=0.5,
                                 max_components=8,
                                 tiny_ratio=0.001):
    mask, total, _, _ = analyze_mask(img_rgba, alpha_thr=alpha_thr)
    if total == 0:
        return False, ["empty_alpha"]
    lcc_ratio, n_comp = connectedness_stats(mask, tiny_ratio=tiny_ratio)
    if (lcc_ratio >= lcc_ratio_min) and (n_comp <= max_components):
        return True, [f"conn_ok:lcc={lcc_ratio:.3f},n={n_comp}"]
    lcc2, n2 = reconnect_then_check(mask, tiny_ratio=tiny_ratio)
    if (lcc2 >= lcc_ratio_min) and (n2 <= max_components):
        return True, [f"conn_ok_after_close:lcc={lcc2:.3f},n={n2}"]
    return False, [f"low_connectedness:lcc={lcc_ratio:.3f},n={n_comp}"]

# ---------- 单目录处理 ----------

def process_dir(in_dir: Path, out_dir: Path, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    exts = tuple([e.strip().lower() for e in args.exts.split(",") if e.strip()])
    files = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort()

    report = {"kept": [], "dropped": []}
    for fp in files:
        name = fp.name
        img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
        if img is None:
            report["dropped"].append({"file": name, "reasons": ["cannot_read"]})
            continue
        
        # 统一 RGBA
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            alpha = np.full((img.shape[0], img.shape[1], 1), 255, np.uint8)
            img = np.concatenate([img, alpha], axis=2)

        keep, reasons = should_keep_by_connectedness(
            img_rgba=img,
            alpha_thr=args.alpha_thr,
            lcc_ratio_min=args.lcc_min,
            max_components=args.max_comp,
            tiny_ratio=args.tiny_ratio
        )
        
        if keep:
            # 先记录原始尺寸
            original_size = img.shape[:2]
            
            # 在输出前最后一步进行等比例缩放
            img_resized = resize_image_proportional(img, args.max_size)
            new_size = img_resized.shape[:2]
            
            # 添加缩放信息到原因中
            if original_size != new_size:
                reasons.append(f"resized:{original_size[1]}x{original_size[0]}->{new_size[1]}x{new_size[0]}")
            
            # 保存缩放后的图片
            cv2.imwrite(str(out_dir / name), img_resized)
            report["kept"].append({"name": name, "reasons": reasons})
        else:
            report["dropped"].append({"file": name, "reasons": reasons})

    # 写本目录报告（若目录内有文件）
    if files:
        with open(out_dir / "report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return report, len(files)

# ---------- 递归主流程 ----------

def run_recursive(in_root: Path, out_root: Path, args):
    master = {
        "settings": {
            "alpha_thr": args.alpha_thr,
            "lcc_min": args.lcc_min,
            "max_comp": args.max_comp,
            "tiny_ratio": args.tiny_ratio,
            "max_size": args.max_size,
            "exts": args.exts
        },
        "summary": {"dirs": 0, "files": 0, "kept": 0, "dropped": 0},
        "details": []  # 每个子目录的统计
    }
    for root, dirs, files in os.walk(in_root):
        in_dir = Path(root)
        rel = in_dir.relative_to(in_root)
        out_dir = out_root / rel
        report, num_files = process_dir(in_dir, out_dir, args)
        if num_files > 0:
            master["summary"]["dirs"] += 1
            master["summary"]["files"] += num_files
            master["summary"]["kept"] += len(report["kept"])
            master["summary"]["dropped"] += len(report["dropped"])
            master["details"].append({
                "dir": str(rel) if str(rel) != "." else ".",
                "files": num_files,
                "kept": len(report["kept"]),
                "dropped": len(report["dropped"])
            })

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "report_master.json", "w", encoding="utf-8") as f:
        json.dump(master, f, ensure_ascii=False, indent=2)

    print(f"[完成] 目录数={master['summary']['dirs']}, 文件数={master['summary']['files']}, "
          f"保留={master['summary']['kept']}, 丢弃={master['summary']['dropped']}.")
    print(f"总报表: {out_root/'report_master.json'}")

# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Recursively filter part PNGs by connectedness with proportional resizing.")
    p.add_argument("--in-dir", default="memory_output_parts", help="输入根目录")
    p.add_argument("--out-dir", default="output_parts", help="输出根目录（将镜像结构）")
    p.add_argument("--alpha-thr", type=float, default=0.5, help="alpha>thr 视为前景")
    p.add_argument("--lcc-min", type=float, default=0.5, help="最大连通域占比阈值（≥通过）")
    p.add_argument("--max-comp", type=int, default=8, help="允许的有效连通域数量上限（独立判据）")
    p.add_argument("--tiny-ratio", type=float, default=0.001, help="忽略小碎屑的占比阈值")
    p.add_argument("--max-size", type=int, default=512, help="图片最大边长（等比例缩放）")
    p.add_argument("--exts", type=str, default=".png,.jpg,.jpeg,.webp", help="逗号分隔的文件后缀")
    return p.parse_args()

if __name__ == "__main__":
    import shutil
    args = parse_args()
    in_root = Path(args.in_dir)
    out_root = Path(args.out_dir)
    
    # 每次运行都清空输出目录
    if out_root.exists():
        print(f"[INFO] 清空输出目录: {out_root}")
        shutil.rmtree(out_root)
    
    run_recursive(in_root, out_root, args)
