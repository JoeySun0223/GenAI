#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section 6.3 Evaluation (Palette ΔE + IoU + SSIM) + MV-PNG Palette vs SVG
------------------------------------------------------------------------
- Color (ours): position-independent palette ΔE between canonical SVG palette
  and corrected SVG palette (应为 0，验证颜色矫正成功).
- New: MV-PNG (raster) palette vs canonical SVG palette, also position-independent.
  Before comparing, MV-PNG colors are clustered in Lab space with a ΔE threshold,
  and tiny-area colors are dropped. This shows how multi-view PNG had color drift
  and how correction brought it back.

- Shape: IoU (alpha masks) + SSIM (grayscale) between vectorized render and the
  MultiView raster for that view.

- Two-row CSV per object:
    1) "corresponding view to original (same viewpoint)"  ← default angle=270
    2) "other views (min–max)"

Usage:
  python self_eval.py \
    --svg_dir example_svg \
    --multiview_dir MultiView \
    --corrected_dir SVG_OUTPUT_CORRECTED \
    --out_root EVAL_6_3_RESULTS \
    --width 512 --height 512 --montage_height 256 \
    --corresponding_angle 270 \
    --png_palette_cluster_de 3.0 \
    --png_palette_min_area 0.001

Dependencies:
  pip install pillow numpy lxml cairosvg scikit-image
"""

import os, re, math, json, shutil, subprocess, argparse, csv
from glob import glob
from collections import defaultdict
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import lxml.etree as ET

# ---------------- Rendering ---------------- #
def render_svg_to_png(svg_path, png_path, width, height):
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    try:
        import cairosvg
        cairosvg.svg2png(url=svg_path, write_to=png_path,
                         output_width=width, output_height=height)
    except Exception as e:
        inkscape = shutil.which("inkscape")
        if not inkscape:
            raise RuntimeError("Need cairosvg or Inkscape for rendering") from e
        subprocess.run([inkscape, svg_path, "--export-type=png",
                        f"--export-width={width}", f"--export-height={height}",
                        f"--export-filename={png_path}"], check=True)

def load_png(path, size=None):
    im = Image.open(path).convert("RGBA")
    if size:
        im = im.resize(size, Image.NEAREST)
    return im

def pil_to_np(im: Image.Image) -> np.ndarray:
    return np.array(im, dtype=np.uint8)

def concat_row(images, out_h):
    if not images:
        return None
    rs=[]
    for im in images:
        w,h=im.size
        if h!=out_h:
            im = im.resize((int(round(w*(out_h/h))), out_h), Image.BICUBIC)
        rs.append(im)
    total_w = sum(im.size[0] for im in rs)
    out = Image.new("RGBA",(total_w,out_h),(0,0,0,0))
    x=0
    for im in rs:
        out.paste(im,(x,0))
        x += im.size[0]
    return out

# ---------------- ΔE2000 over colors (palette, position-independent) ---------------- #
HEX_OR_RGB = re.compile(r'#[0-9A-Fa-f]{3,8}|rgba?\([^)]*\)|hsla?\([^)]*\)')
URL_GRAD = re.compile(r'url\(#([^)]+)\)')

def _parse_style(style_str):
    out={}
    for part in style_str.split(';'):
        if ':' in part:
            k,v=part.split(':',1)
            out[k.strip().lower()]=v.strip()
    return out

def _parse_color_token(token):
    if not token:
        return None
    t = token.strip().lower()
    if t in ('none','transparent','currentcolor','inherit'):
        return None
    m = HEX_OR_RGB.search(token)
    if not m:
        return None
    val = m.group(0)
    if val.startswith('#'):
        h = val[1:]
        if len(h)==3: h=''.join([c*2 for c in h])
        try:
            return (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))
        except:
            return None
    if val.lower().startswith('rgb'):
        nums = re.findall(r'[\d.]+', val)
        if len(nums)>=3:
            r,g,b = [max(0,min(255,int(round(float(x))))) for x in nums[:3]]
            return (r,g,b)
    if val.lower().startswith('hsl'):
        nums = re.findall(r'[\d.]+', val)
        if len(nums)>=3:
            h,s,l = float(nums[0])%360, float(nums[1])/100.0, float(nums[2])/100.0
            c=(1-abs(2*l-1))*s; x=c*(1-abs((h/60)%2-1)); m=l-c/2
            if   0<=h<60:  rp,gp,bp=c,x,0
            elif 60<=h<120:rp,gp,bp=x,c,0
            elif 120<=h<180:rp,gp,bp=0,c,x
            elif 180<=h<240:rp,gp,bp=0,x,c
            elif 240<=h<300:rp,gp,bp=x,0,c
            else:           rp,gp,bp=c,0,x
            return (int(round((rp+m)*255)), int(round((gp+m)*255)), int(round((bp+m)*255)))
    return None

def _collect_gradient_colors(root):
    grad = {}
    for tag in ('linearGradient','radialGradient'):
        for g in root.findall(f'.//{{*}}{tag}'):
            gid = g.attrib.get('id')
            if not gid: continue
            cols=[]
            for stop in g.findall('.//{*}stop'):
                sc = stop.attrib.get('stop-color')
                if not sc and 'style' in stop.attrib:
                    sc = _parse_style(stop.attrib['style']).get('stop-color')
                col = _parse_color_token(sc)
                if col: cols.append(col)
            if cols: grad[gid]=cols
    return grad

def _resolve_elem_colors(elem, grad_colors):
    cols=[]
    for k in ('fill','stroke','stop-color','color'):
        v = elem.attrib.get(k)
        if v:
            m = URL_GRAD.search(v)
            if m and m.group(1) in grad_colors: cols += grad_colors[m.group(1)]
            else:
                c = _parse_color_token(v)
                if c: cols.append(c)
    if 'style' in elem.attrib:
        st = _parse_style(elem.attrib['style'])
        for k in ('fill','stroke','stop-color','color'):
            if k in st:
                v = st[k]
                m = URL_GRAD.search(v)
                if m and m.group(1) in grad_colors: cols += grad_colors[m.group(1)]
                else:
                    c = _parse_color_token(v)
                    if c: cols.append(c)
    if not cols and elem.getparent() is not None:
        cols += _resolve_elem_colors(elem.getparent(), grad_colors)
    return cols

def parse_svg_palette(svg_path):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    grad_colors = _collect_gradient_colors(root)
    palette=[]
    for el in root.iter():
        palette += _resolve_elem_colors(el, grad_colors)
    if not palette:
        return np.empty((0,3), dtype=np.uint8)
    uniq = sorted(set(palette))
    return np.array(uniq, dtype=np.uint8)

# sRGB→Lab for color triplets
def srgb_to_lab_colors(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)/255.0
    a=0.055
    rgb=np.where(rgb<=0.04045, rgb/12.92, ((rgb+a)/(1+a))**2.4)
    M=np.array([[0.4124564,0.3575761,0.1804375],
                [0.2126729,0.7151522,0.0721750],
                [0.0193339,0.1191920,0.9503041]])
    XYZ=rgb@M.T
    Xn,Yn,Zn=0.95047,1.0,1.08883
    X,Y,Z=XYZ[:,0]/Xn, XYZ[:,1]/Yn, XYZ[:,2]/Zn
    d=6/29
    f=lambda t: np.where(t>d**3,np.cbrt(t), t/(3*d**2)+4/29)
    fX,fY,fZ=f(X),f(Y),f(Z)
    L=116*fY-16; a=500*(fX-fY); b=200*(fY-fZ)
    return np.stack([L,a,b],axis=1)

# ΔE2000 for two Lab arrays
def deltaE2000(lab1, lab2):
    L1,a1,b1=lab1[...,0],lab1[...,1],lab1[...,2]
    L2,a2,b2=lab2[...,0],lab2[...,1],lab2[...,2]
    C1=np.hypot(a1,b1); C2=np.hypot(a2,b2)
    Cm=(C1+C2)/2
    G=0.5*(1-np.sqrt((Cm**7)/(Cm**7+25**7)))
    a1p=(1+G)*a1; a2p=(1+G)*a2
    C1p=np.hypot(a1p,b1); C2p=np.hypot(a2p,b2)
    h1p=np.degrees(np.arctan2(b1,a1p))%360
    h2p=np.degrees(np.arctan2(b2,a2p))%360
    dLp=L2-L1; dCp=C2p-C1p
    dh=h2p-h1p
    dh=np.where(dh>180,dh-360,dh)
    dh=np.where(dh<-180,dh+360,dh)
    dHp=2*np.sqrt(C1p*C2p)*np.sin(np.radians(dh/2))
    Lpm=(L1+L2)/2; Cpm=(C1p+C2p)/2
    hsum=h1p+h2p; hdiff=np.abs(h1p-h2p)
    hpm=np.where((C1p*C2p)==0,hsum,
                 np.where(hdiff<=180,0.5*hsum,0.5*(hsum+360)))%360
    T=(1-0.17*np.cos(np.radians(hpm-30))+0.24*np.cos(np.radians(2*hpm))
       +0.32*np.cos(np.radians(3*hpm+6))-0.20*np.cos(np.radians(4*hpm-63)))
    dRo=30*np.exp(-(((hpm-275)/25)**2))
    RC=2*np.sqrt((Cpm**7)/(Cpm**7+25**7))
    SL=1+(0.015*((Lpm-50)**2))/np.sqrt(20+(Lpm-50)**2)
    SC=1+0.045*Cpm; SH=1+0.015*Cpm*T
    RT=-np.sin(np.radians(2*dRo))*RC
    dE=np.sqrt((dLp/SL)**2+(dCp/SC)**2+(dHp/SH)**2+RT*(dCp/SC)*(dHp/SH))
    return dE

def palette_distance(labA, labB):
    """For each color in B, find nearest in A; report mean/min/max."""
    if labA.size == 0 or labB.size == 0:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
    best=[]
    for i in range(labB.shape[0]):
        dE = deltaE2000(labA, np.broadcast_to(labB[i], labA.shape))
        best.append(np.min(dE))
    best=np.array(best)
    return {"mean": float(np.mean(best)),
            "min": float(np.min(best)),
            "max": float(np.max(best))}

# ---------------- MV-PNG palette extraction & clustering ---------------- #
def png_palette_clustered_lab(rgba_u8: np.ndarray, de_thresh: float = 3.0, min_area_ratio: float = 0.001):
    """
    Extract visible colors from RGBA PNG, cluster them in Lab space with a ΔE threshold,
    and return cluster centers (Lab) + stats.

    - de_thresh: ΔE2000 threshold for merging a color into an existing cluster.
    - min_area_ratio: drop colors whose pixel count < ratio * visible_pixels.
    """
    assert rgba_u8.dtype == np.uint8
    alpha = rgba_u8[..., 3] > 0
    if not np.any(alpha):
        return np.empty((0,3), dtype=np.float64), dict(num_clusters=0, extra_ratio=float("nan"))

    rgb = rgba_u8[..., :3][alpha]          # (N,3)
    # count unique RGBs
    uniq, counts = np.unique(rgb.reshape(-1,3), axis=0, return_counts=True)
    visible = counts.sum()
    # drop tiny colors
    keep = counts >= max(1, int(round(min_area_ratio * visible)))
    uniq = uniq[keep]
    counts = counts[keep]
    if uniq.size == 0:
        return np.empty((0,3), dtype=np.float64), dict(num_clusters=0, extra_ratio=float("nan"))

    # convert to Lab
    uniq_lab = srgb_to_lab_colors(uniq)  # (K,3)
    # greedy clustering by popularity (counts desc)
    order = np.argsort(-counts)
    centers = []          # list of Lab centers
    sizes = []            # counts per cluster
    for idx in order:
        c_lab = uniq_lab[idx]
        if not centers:
            centers.append(c_lab.copy()); sizes.append(int(counts[idx]))
            continue
        # find nearest cluster by ΔE
        dists = [np.min(deltaE2000(np.array([cen]), np.array([c_lab]))) for cen in centers]
        j = int(np.argmin(dists))
        if dists[j] <= de_thresh:
            # merge: weighted average in Lab space
            w_old = sizes[j]; w_new = counts[idx]
            centers[j] = (centers[j]*w_old + c_lab*w_new) / (w_old + w_new)
            sizes[j] += int(w_new)
        else:
            centers.append(c_lab.copy()); sizes.append(int(counts[idx]))

    centers = np.vstack(centers) if centers else np.empty((0,3))
    stats = dict(num_clusters=int(len(sizes)))
    return centers, stats  # centers in Lab

def mvpng_palette_vs_svg(svg_lab: np.ndarray, mv_lab: np.ndarray):
    """
    Compare clustered MV-PNG palette (Lab) vs canonical SVG palette (Lab).
    - Returns weighted mean/min/max nearest ΔE (weights uniform per cluster by default),
      and an 'extra_ratio' proxy: fraction of clusters whose nearest δE > 5 (perceptually off).
      (If you prefer pixel-weighted, adapt clustering to carry cluster weights.)
    """
    if svg_lab.size == 0 or mv_lab.size == 0:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan"), "extra_ratio": float("nan")}
    best=[]
    for i in range(mv_lab.shape[0]):
        dE = deltaE2000(svg_lab, np.broadcast_to(mv_lab[i], svg_lab.shape))
        best.append(np.min(dE))
    best = np.array(best)
    extra_ratio = float(np.mean(best > 5.0))  # fraction of clusters far from any SVG color
    return {"mean": float(np.mean(best)),
            "min": float(np.min(best)),
            "max": float(np.max(best)),
            "extra_ratio": extra_ratio}

# ---------------- Shape metrics (IoU + SSIM) ---------------- #
def compute_iou(mask1, mask2):
    inter = np.logical_and(mask1,mask2).sum()
    union = np.logical_or(mask1,mask2).sum()
    return inter/union if union>0 else float("nan")

def compute_ssim(gray1_u8, gray2_u8):
    g1 = gray1_u8.astype(np.float32)/255.0
    g2 = gray2_u8.astype(np.float32)/255.0
    score, diff = ssim(g1, g2, data_range=1.0, full=True)
    return float(score), diff

# ---------------- SVG path counting ---------------- #
def count_paths(svg_path):
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        return len(root.findall('.//{*}path'))
    except Exception:
        return None

# ---------------- Discovery helpers (DRIVE FROM MULTIVIEW) ---------------- #
MV_PAT = re.compile(r'^(?P<name>.+)_(?P<ang>\d{1,3})deg\.(?:png|PNG)$')

def collect_objects_from_multiview(multiview_dir):
    """Return dict: name -> list of (ang_int, ang_token, mv_png)"""
    groups = defaultdict(list)
    files = glob(os.path.join(multiview_dir, '*_*deg.png')) + \
            glob(os.path.join(multiview_dir, '*_*deg.PNG'))
    for p in files:
        m = MV_PAT.match(os.path.basename(p))
        if not m: continue
        name = m.group('name')
        ang_token = m.group('ang')
        ang_int = int(ang_token)
        groups[name].append((ang_int, ang_token, p))
    for k in groups:
        groups[k].sort(key=lambda x: x[0])
    return groups

def find_corrected_svg(corrected_dir, name, ang_int, ang_token):
    """Try multiple filename variants for corrected SVG."""
    cands = [
        os.path.join(corrected_dir, f"{name}_{ang_token}deg_merged.svg"),
        os.path.join(corrected_dir, f"{name}_{ang_int}deg_merged.svg"),
        os.path.join(corrected_dir, f"{name}_{ang_int:02d}deg_merged.svg"),
        os.path.join(corrected_dir, f"{name}_{ang_int:03d}deg_merged.svg"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    return None

# ---------------- Main ---------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--svg_dir', default='example_svg')
    ap.add_argument('--multiview_dir', default='MultiView')
    ap.add_argument('--corrected_dir', default='SVG_OUTPUT_CORRECTED')
    ap.add_argument('--out_root', default='EVAL_6_3_RESULTS')
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--height', type=int, default=512)
    ap.add_argument('--montage_height', type=int, default=256)
    ap.add_argument('--corresponding_angle', type=int, default=270,
                    help='Treat this angle as the view corresponding to the original viewpoint.')
    ap.add_argument('--png_palette_cluster_de', type=float, default=3.0,
                    help='ΔE2000 threshold for merging MV-PNG colors in palette clustering.')
    ap.add_argument('--png_palette_min_area', type=float, default=0.001,
                    help='Minimum visible-area ratio to keep a PNG color (e.g., 0.001=0.1%).')
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    objects = collect_objects_from_multiview(args.multiview_dir)
    if not objects:
        print(f"[WARN] No *_*deg.png found in {args.multiview_dir}")
        return

    for name, items in objects.items():
        canonical_svg = os.path.join(args.svg_dir, f"{name}.svg")
        if not os.path.exists(canonical_svg):
            print(f"[WARN] Missing canonical SVG: {canonical_svg}; skip {name}.")
            continue

        out_dir = os.path.join(args.out_root, name)
        os.makedirs(out_dir, exist_ok=True)
        vec_dir = os.path.join(out_dir, "vectorized_pngs")
        os.makedirs(vec_dir, exist_ok=True)

        # Render canonical (visual ref)
        ref_png = os.path.join(out_dir, "reference_render.png")
        render_svg_to_png(canonical_svg, ref_png, args.width, args.height)

        # Canonical palette + original paths
        pal_orig = parse_svg_palette(canonical_svg)
        pal_orig_lab = srgb_to_lab_colors(pal_orig)
        paths_orig = count_paths(canonical_svg)

        per_view=[]
        mv_row_imgs=[]
        vec_row_imgs=[]

        print(f"[INFO] {name}: {len(items)} views")
        for ang_int, ang_token, mv_png in items:
            corr_svg = find_corrected_svg(args.corrected_dir, name, ang_int, ang_token)
            if not corr_svg:
                print(f"[WARN] Missing corrected SVG for {name} {ang_token}deg; skip this view.")
                continue

            vec_png = os.path.join(vec_dir, f"{name}_{ang_token}deg_vec.png")
            render_svg_to_png(corr_svg, vec_png, args.width, args.height)

            mv_im = load_png(mv_png, size=(args.width,args.height))
            vec_im = load_png(vec_png)
            mv_np = pil_to_np(mv_im)
            vec_np = pil_to_np(vec_im)

            # Palette ΔE (corrected SVG vs canonical SVG) — should be ~0
            pal_corr = parse_svg_palette(corr_svg)
            pal_corr_lab = srgb_to_lab_colors(pal_corr)
            pal_diff_corr = palette_distance(pal_orig_lab, pal_corr_lab)

            # NEW: MV-PNG (raster) palette vs canonical SVG palette (position-independent)
            mv_centers_lab, mv_stats = png_palette_clustered_lab(
                mv_np, de_thresh=args.png_palette_cluster_de, min_area_ratio=args.png_palette_min_area
            )
            mv_pal_metrics = mvpng_palette_vs_svg(pal_orig_lab, mv_centers_lab)
            mv_pal_metrics.update(mv_stats)  # add num_clusters, etc.

            # Shape metrics (position-dependent): IoU & SSIM
            mask_mv = mv_np[...,3] > 0
            mask_vec = vec_np[...,3] > 0
            iou = compute_iou(mask_mv, mask_vec)
            g1 = np.array(Image.fromarray(mv_np).convert("L"))
            g2 = np.array(Image.fromarray(vec_np).convert("L"))
            ssim_val, diff = compute_ssim(g1, g2)

            if ang_int == args.corresponding_angle:
                Image.fromarray((diff*255).astype(np.uint8)).save(os.path.join(out_dir, "diff_ssim_corresponding.png"))

            paths_corr = count_paths(corr_svg)

            per_view.append({
                "angle": ang_int,
                "angle_token": ang_token,
                "files": {
                    "mv_png": os.path.relpath(mv_png, out_dir),
                    "vec_png": os.path.relpath(vec_png, out_dir),
                    "corr_svg": os.path.relpath(corr_svg, out_dir)
                },
                "palette_diff_correctedSVG": pal_diff_corr,
                "mv_palette_vs_svg": mv_pal_metrics,  # <-- NEW
                "shape_metrics": {"iou": iou, "ssim": ssim_val},
                "paths_corr": paths_corr
            })

            mv_row_imgs.append(mv_im)
            vec_row_imgs.append(vec_im)

        if not per_view:
            print(f"[WARN] No valid views computed for {name}.")
            continue

        per_view.sort(key=lambda x: x["angle"])
        focus = next((v for v in per_view if v["angle"] == args.corresponding_angle), None)
        others = [v for v in per_view if v is not focus]

        def minmax(vals):
            vals = [x for x in vals if x is not None and not (isinstance(x,float) and math.isnan(x))]
            if not vals: return (float("nan"), float("nan"))
            return (min(vals), max(vals))

        # Ranges for others
        pal_corr_rng = minmax([v["palette_diff_correctedSVG"]["mean"] for v in others])
        mv_pal_rng   = minmax([v["mv_palette_vs_svg"]["mean"] for v in others])
        iou_rng      = minmax([v["shape_metrics"]["iou"] for v in others])
        ssim_rng     = minmax([v["shape_metrics"]["ssim"] for v in others])
        paths_corr_rng = minmax([v["paths_corr"] for v in others])

        summary = {
            "object": name,
            "num_views": len(per_view),
            "angles": [v["angle"] for v in per_view],
            "corresponding_angle": args.corresponding_angle,
            "paths_orig": paths_orig,
            "focus": focus,
            "others_range": {
                "palette_correctedSVG_mean_minmax": pal_corr_rng,
                "mv_png_palette_mean_minmax": mv_pal_rng,
                "iou_minmax": iou_rng,
                "ssim_minmax": ssim_rng,
                "paths_corr_minmax": paths_corr_rng
            },
            "params": {
                "png_palette_cluster_de": args.png_palette_cluster_de,
                "png_palette_min_area": args.png_palette_min_area
            }
        }

        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({
                "summary": summary,
                "per_view": per_view,
                "reference_render": os.path.relpath(ref_png, out_dir)
            }, f, ensure_ascii=False, indent=2)

        # Montages for paper
        try:
            concat_row(mv_row_imgs, args.montage_height).save(os.path.join(out_dir, "montage_input_row.png"))
        except Exception as e:
            print(f"[WARN] montage_input_row failed: {e}")
        try:
            concat_row(vec_row_imgs, args.montage_height).save(os.path.join(out_dir, "montage_vectorized_row.png"))
        except Exception as e:
            print(f"[WARN] montage_vectorized_row failed: {e}")

        # Two-row CSV per object (新增 MVPNG_PaletteΔE_mean 列)
        csv_path = os.path.join(out_dir, "summary.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "Object","Views","Group",
                "PaletteΔE_mean(correctedSVG↔SVG)",
                "MVPNG_PaletteΔE_mean(PNG↔SVG)",
                "IoU","SSIM","Paths","Paths(orig)"
            ])

            def fmt_rng(t):
                a,b=t
                if any(math.isnan(x) for x in (a,b)):
                    return "nan–nan"
                return f"{a:.3f}–{b:.3f}"
            def fmt_rng_int(t):
                a,b=t
                if a!=a or b!=b:
                    return "n/a"
                return f"{int(a)}–{int(b)}"

            # Row 1: corresponding view
            if focus is not None:
                w.writerow([
                    name, len(per_view),
                    "corresponding view to original (same viewpoint)",
                    f"{focus['palette_diff_correctedSVG']['mean']:.3f}" if not math.isnan(focus['palette_diff_correctedSVG']['mean']) else "nan",
                    f"{focus['mv_palette_vs_svg']['mean']:.3f}" if not math.isnan(focus['mv_palette_vs_svg']['mean']) else "nan",
                    f"{focus['shape_metrics']['iou']:.3f}" if not math.isnan(focus['shape_metrics']['iou']) else "nan",
                    f"{focus['shape_metrics']['ssim']:.3f}" if not math.isnan(focus['shape_metrics']['ssim']) else "nan",
                    (str(focus['paths_corr']) if focus['paths_corr'] is not None else "n/a"),
                    (str(paths_orig) if paths_orig is not None else "n/a")
                ])
            else:
                w.writerow([name, len(per_view),
                            "corresponding view to original (same viewpoint)",
                            "nan","nan","nan","nan","n/a",
                            (str(paths_orig) if paths_orig is not None else "n/a")])

            # Row 2: other views (min–max)
            w.writerow([
                name, len(per_view),
                "other views (min–max)",
                fmt_rng(pal_corr_rng),
                fmt_rng(mv_pal_rng),
                fmt_rng(iou_rng),
                fmt_rng(ssim_rng),
                fmt_rng_int(paths_corr_rng) if paths_corr_rng[0]==paths_corr_rng[0] else "n/a",
                (str(paths_orig) if paths_orig is not None else "n/a")
            ])

        print(f"[OK] {name}: results saved → {out_dir}")

if __name__ == "__main__":
    main()
