#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVG 批处理器（单文件、通用版 + 极短描边路径清理）

用法：
  把本文件保存为 svg_color_batch.py，与 beforecombine_svg/ 同级。
  然后执行：  python svg_color_batch.py

功能：
- 清空并重建输出目录 deal_svg/
- 遍历 beforecombine_svg/ 的一级子目录（若没有子目录，则直接处理根目录）
- 对每个 SVG 执行：
  1) 颜色面积统计 + “三步法 + ΔE”判定（删除或保留颜色）
  2) 清理“极短描边路径”（通常渲染成小黑点）

输出：
- 处理后的 SVG 到 deal_svg/<同名子目录>/
- 每个子目录一份 report.json（包含颜色判定与清理统计）

阈值（可在命令行指定）：
- 三步法：--tiny 0.1  --small 5  --deltaE 8
- 噪点清理：
  --cleanup 1              # 开/关：1 开启（默认），0 关闭
  --minStrokeLen 6.0       # 线长阈值（≤ 则判为噪点）
  --maxStrokeBBoxW 4.0     # 包围盒最大宽度阈值（≤ 则判为噪点）
  --maxStrokeBBoxH 3.0     # 包围盒最大高度阈值（≤ 则判为噪点）
  --bottomY  None/0.0      # 仅靠近底边才清理（给一个 y 值则仅清理 bbox_min_y≤该值 的小段；默认不限位置）
  --onlyStroked 1          # 仅对有描边且无填充的路径生效（默认 1）
  --setLineCapButt 0       # 若设为 1，则把所有路径 stroke-linecap 统一改为 butt（可减少端帽点）
"""

import os
import re
import math
import json
import shutil
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# ===================== 可调参数（全局默认） =====================
# 三步法阈值（单位：百分比）
TINY_PCT = 0.1       # < 0.1% 直接删除
SMALL_PCT = 8.0      # [0.1%, 8%) 进入 ΔE 相似性判断
DELTAE_SMALL_SIMILAR = 30.0  # ≤ 30 视为相近色 -> 删除；>30 -> 保留

# 噪点清理默认阈值
CLEANUP_ENABLED = True
CLEAN_MIN_STROKE_LEN = 6.0     # 线长 ≤ 6.0 视为“极短”
CLEAN_MAX_BBOX_W = 4.0         # 包围盒宽 ≤ 4.0
CLEAN_MAX_BBOX_H = 3.0         # 包围盒高 ≤ 3.0
CLEAN_ONLY_STROKED = True      # 只清理“无填充 + 有描边”的路径
CLEAN_BOTTOM_Y = None          # 仅清理靠近底边（如 0.0）；None 表示不限位置
FORCE_LINECAP_BUTT = False     # 把所有路径 stroke-linecap 统一为 butt（可减少端帽点）

INPUT_ROOT = "beforecombine_svg"
OUTPUT_ROOT = "deal_svg"

# ===================== 几何工具 =====================
def polygon_area(points: List[Tuple[float, float]]) -> float:
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

def lerp(a, b, t):
    return (a[0] * (1 - t) + b[0] * t, a[1] * (1 - t) + b[1] * t)

def cubic_points(p0, p1, p2, p3, tol=0.25, depth=0, max_depth=10):
    """自适应细分三次贝塞尔为折线。"""
    def max_dist_line(a, d, b, c):
        ax, ay = a; dx, dy = d
        vx, vy = dx - ax, dy - ay
        lv = (vx * vx + vy * vy) ** 0.5 or 1e-9
        def dist(p):
            px, py = p
            return abs(vx * (ay - py) - vy * (ax - px)) / lv
        return max(dist(b), dist(c))
    if depth > max_depth or max_dist_line(p0, p3, p1, p2) < tol:
        return [p0, p3]
    p01 = lerp(p0, p1, 0.5)
    p12 = lerp(p1, p2, 0.5)
    p23 = lerp(p2, p3, 0.5)
    p012 = lerp(p01, p12, 0.5)
    p123 = lerp(p12, p23, 0.5)
    p0123 = lerp(p012, p123, 0.5)
    L = cubic_points(p0, p01, p012, p0123, tol, depth + 1, max_depth)
    R = cubic_points(p0123, p123, p23, p3, tol, depth + 1, max_depth)
    return L[:-1] + R

def parse_points_attr(s: str) -> List[Tuple[float, float]]:
    nums = re.findall(r"-?\d*\.?\d+(?:e[-+]?\d+)?", s or "")
    pts = []
    for i in range(0, len(nums), 2):
        try:
            pts.append((float(nums[i]), float(nums[i + 1])))
        except Exception:
            break
    return pts

def parse_path_to_polygons(d: str) -> List[List[Tuple[float, float]]]:
    """解析 path 的 M/L/H/V/C/Q/Z 指令，曲线细分为折线，返回闭合多边形集合。"""
    tokens = re.findall(r"[MLHVCSQTAZmlhvcsqtaz]|-?\d*\.?\d+(?:e[-+]?\d+)?", d or "")
    polys, poly = [], []
    i, cmd = 0, None
    cur = (0.0, 0.0)

    def add(pt):
        poly.append(pt)
    def flush(close=True):
        nonlocal poly, polys
        if not poly:
            return
        if close and poly[0] != poly[-1]:
            poly.append(poly[0])
        if len(poly) >= 3:
            polys.append(poly.copy())
        poly.clear()

    while i < len(tokens):
        t = tokens[i]
        if t.isalpha():
            cmd = t
            i += 1
            continue
        if cmd in ("M", "m"):
            x = float(t); y = float(tokens[i + 1]); i += 2
            cur = (cur[0] + x, cur[1] + y) if cmd == "m" else (x, y)
            flush(close=False)
            add(cur)
            cmd = "L" if cmd == "M" else "l"
        elif cmd in ("L", "l"):
            x = float(t); y = float(tokens[i + 1]); i += 2
            cur = (cur[0] + x, cur[1] + y) if cmd == "l" else (x, y)
            add(cur)
        elif cmd in ("H", "h"):
            x = float(t); i += 1
            cur = (cur[0] + x, cur[1]) if cmd == "h" else (x, cur[1])
            add(cur)
        elif cmd in ("V", "v"):
            y = float(t); i += 1
            cur = (cur[0], cur[1] + y) if cmd == "v" else (cur[0], y)
            add(cur)
        elif cmd in ("C", "c"):
            x1 = float(t); y1 = float(tokens[i + 1])
            x2 = float(tokens[i + 2]); y2 = float(tokens[i + 3])
            x3 = float(tokens[i + 4]); y3 = float(tokens[i + 5]); i += 6
            if cmd == "c":
                p0 = cur; p1 = (cur[0] + x1, cur[1] + y1)
                p2 = (cur[0] + x2, cur[1] + y2); p3 = (cur[0] + x3, cur[1] + y3)
            else:
                p0 = cur; p1 = (x1, y1); p2 = (x2, y2); p3 = (x3, y3)
            pts = cubic_points(p0, p1, p2, p3)
            for pt in pts[1:]:
                add(pt)
            cur = p3
        elif cmd in ("Z", "z"):
            flush(close=True)
            i += 1
        else:
            # 未实现的 A/S/T 等：跳过一个数字避免死循环
            i += 1
    flush(close=True)
    return polys

# ------- 为“极短描边路径清理”额外提供的采样/度量 -------
def parse_path_points_for_length(d: str) -> List[Tuple[float, float]]:
    """生成适合度量长度/包围盒的折线点（M/L/H/V/C；Q 可按需补充）。"""
    tokens = re.findall(r"[MLHVCSQTAZmlhvcsqtaz]|-?\d*\.?\d+(?:e[-+]?\d+)?", d or "")
    pts = []
    i, cmd = 0, None
    cur = (0.0, 0.0)

    def add(p):
        pts.append(p)

    while i < len(tokens):
        t = tokens[i]
        if t.isalpha():
            cmd = t; i += 1; continue
        if cmd in ("M", "m"):
            x = float(t); y = float(tokens[i + 1]); i += 2
            cur = (cur[0] + x, cur[1] + y) if cmd == "m" else (x, y)
            add(cur)
            cmd = "L" if cmd == "M" else "l"
        elif cmd in ("L", "l"):
            x = float(t); y = float(tokens[i + 1]); i += 2
            cur = (cur[0] + x, cur[1] + y) if cmd == "l" else (x, y)
            add(cur)
        elif cmd in ("H", "h"):
            x = float(t); i += 1
            cur = (cur[0] + x, cur[1]) if cmd == "h" else (x, cur[1])
            add(cur)
        elif cmd in ("V", "v"):
            y = float(t); i += 1
            cur = (cur[0], cur[1] + y) if cmd == "v" else (cur[0], y)
            add(cur)
        elif cmd in ("C", "c"):
            x1 = float(t); y1 = float(tokens[i + 1])
            x2 = float(tokens[i + 2]); y2 = float(tokens[i + 3])
            x3 = float(tokens[i + 4]); y3 = float(tokens[i + 5]); i += 6
            if cmd == "c":
                p0 = cur; p1 = (cur[0] + x1, cur[1] + y1)
                p2 = (cur[0] + x2, cur[1] + y2); p3 = (cur[0] + x3, cur[1] + y3)
            else:
                p0 = cur; p1 = (x1, y1); p2 = (x2, y2); p3 = (x3, y3)
            # 采样 t=0.25,0.5,0.75,1.0
            def bez(t, p0, p1, p2, p3):
                u = 1 - t
                x = u**3*p0[0] + 3*u**2*t*p1[0] + 3*u*t**2*p2[0] + t**3*p3[0]
                y = u**3*p0[1] + 3*u**2*t*p1[1] + 3*u*t**2*p2[1] + t**3*p3[1]
                return (x, y)
            for tt in (0.25, 0.5, 0.75, 1.0):
                cur = bez(tt, p0, p1, p2, p3)
                add(cur)
        elif cmd in ("Z", "z"):
            i += 1
        else:
            i += 1
    return pts

def path_length(pts: List[Tuple[float, float]]) -> float:
    L = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        L += (dx*dx + dy*dy) ** 0.5
    return L

def bbox_of_pts(pts: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))

# ===================== 颜色工具（规范化 + Lab） =====================
def normalize_color_token(color: str) -> Optional[str]:
    if not color:
        return None
    s = color.strip()
    if s.startswith("#"):
        s = s[1:]
        if len(s) == 3:
            r = int(s[0] * 2, 16); g = int(s[1] * 2, 16); b = int(s[2] * 2, 16)
        elif len(s) == 6:
            r = int(s[0:2], 16); g = int(s[2:4], 16); b = int(s[4:6], 16)
        else:
            return None
        return f"#{r:02X}{g:02X}{b:02X}"
    m = re.match(r"rgb\(\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*\)", s, flags=re.I)
    if m:
        r = max(0, min(255, int(m.group(1))))
        g = max(0, min(255, int(m.group(2))))
        b = max(0, min(255, int(m.group(3))))
        return f"#{r:02X}{g:02X}{b:02X}"
    return s

def parse_rgb01(hex_or_rgb: str) -> Optional[Tuple[float, float, float]]:
    s = hex_or_rgb.strip()
    if s.startswith("#"):
        h = s[1:]
        if len(h) != 6:
            return None
        return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)
    m = re.match(r"rgb\(\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*\)", s, flags=re.I)
    if m:
        return (int(m.group(1)) / 255.0, int(m.group(2)) / 255.0, int(m.group(3)) / 255.0)
    return None

def rgb_to_lab(r, g, b):
    def inv_gamma(u):
        return ((u + 0.055) / 1.055) ** 2.4 if u > 0.04045 else (u / 12.92)
    R = inv_gamma(r); G = inv_gamma(g); B = inv_gamma(b)
    X = R * 0.4124 + G * 0.3576 + B * 0.1805
    Y = R * 0.2126 + G * 0.7152 + B * 0.0722
    Z = R * 0.0193 + G * 0.1192 + B * 0.9505
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x, y, z = X / Xn, Y / Yn, Z / Zn
    def f(t): return t ** (1 / 3) if t > 0.008856 else (7.787 * t + 16 / 116)
    L = 116 * f(y) - 16
    a = 500 * (f(x) - f(y))
    b = 200 * (f(y) - f(z))
    return (L, a, b)

def deltaE76(l1, l2):
    return ((l1[0] - l2[0]) ** 2 + (l1[1] - l2[1]) ** 2 + (l1[2] - l2[2]) ** 2) ** 0.5

# ===================== 样式解析（style/class/父组继承） =====================
def collect_css_class_fills(root: ET.Element) -> Dict[str, str]:
    css_text = []
    for style in root.iter():
        if style.tag.split("}")[-1] == "style" and style.text:
            css_text.append(style.text)
    css = "\n".join(css_text)
    class_fill = {}
    for m in re.finditer(r"\.([A-Za-z0-9_\-]+)\s*\{([^}]*)\}", css, flags=re.DOTALL):
        cls = m.group(1)
        body = m.group(2)
        fm = re.search(r"fill\s*:\s*([^;]+);", body, flags=re.IGNORECASE)
        if fm:
            class_fill[cls] = fm.group(1).strip()
    return class_fill

def get_effective_fill(elem: ET.Element, parent_map: Dict[ET.Element, ET.Element], class_fill: Dict[str, str]) -> Optional[str]:
    style = elem.attrib.get("style", "")
    fill = elem.attrib.get("fill", "").strip() if elem.attrib.get("fill") else ""
    if "fill:" in style:
        for part in style.split(";"):
            s = part.strip()
            if s.lower().startswith("fill:"):
                fill = s.split(":", 1)[1].strip()
    if (not fill or fill == "none"):
        cls = elem.attrib.get("class", "")
        if cls:
            for c in re.split(r"\s+", cls.strip()):
                if c in class_fill:
                    fill = class_fill[c]
                    break
    if (not fill or fill == "none"):
        p = parent_map.get(elem)
        while p is not None:
            p_style = p.attrib.get("style", "")
            p_fill = p.attrib.get("fill", "").strip() if p.attrib.get("fill") else ""
            if "fill:" in p_style:
                for part in p_style.split(";"):
                    s = part.strip()
                    if s.lower().startswith("fill:"):
                        p_fill = s.split(":", 1)[1].strip()
            if p_fill and p_fill != "none":
                fill = p_fill
                break
            p = parent_map.get(p)
    if not fill or fill == "none":
        return None
    return normalize_color_token(fill)

def get_effective_stroke(elem: ET.Element, parent_map: Dict[ET.Element, ET.Element]) -> Tuple[Optional[str], Optional[str]]:
    """返回 (stroke_color, stroke_linecap)，均已去空格，若无则为 None。"""
    # stroke
    stroke = elem.attrib.get("stroke")
    style = elem.attrib.get("style", "")
    if (not stroke or stroke == "none") and "stroke:" in style:
        for part in style.split(";"):
            s = part.strip()
            if s.lower().startswith("stroke:"):
                stroke = s.split(":", 1)[1].strip()
                break
    # 继承
    if (not stroke or stroke == "none"):
        p = parent_map.get(elem)
        while p is not None:
            s = p.attrib.get("stroke")
            sty = p.attrib.get("style", "")
            if (not s or s == "none") and "stroke:" in sty:
                for part in sty.split(";"):
                    ss = part.strip()
                    if ss.lower().startswith("stroke:"):
                        s = ss.split(":", 1)[1].strip()
                        break
            if s and s != "none":
                stroke = s
                break
            p = parent_map.get(p)
    if stroke:
        stroke = stroke.strip()

    # linecap
    linecap = elem.attrib.get("stroke-linecap")
    if (not linecap) and "stroke-linecap:" in style:
        for part in style.split(";"):
            s = part.strip()
            if s.lower().startswith("stroke-linecap:"):
                linecap = s.split(":", 1)[1].strip()
                break
    if not linecap:
        p = parent_map.get(elem)
        while p is not None:
            lc = p.attrib.get("stroke-linecap")
            sty = p.attrib.get("style", "")
            if (not lc) and "stroke-linecap:" in sty:
                for part in sty.split(";"):
                    ss = part.strip()
                    if ss.lower().startswith("stroke-linecap:"):
                        lc = ss.split(":", 1)[1].strip()
                        break
            if lc:
                linecap = lc
                break
            p = parent_map.get(p)

    return (stroke if stroke else None, linecap if linecap else None)

# ===================== 元素面积 =====================
def element_filled_area(elem: ET.Element) -> float:
    tag = elem.tag.split("}")[-1]
    if tag == "path":
        d = elem.attrib.get("d", "")
        polys = parse_path_to_polygons(d)
        return sum(polygon_area(poly) for poly in polys)
    elif tag == "rect":
        w = float(elem.attrib.get("width", "0") or 0.0)
        h = float(elem.attrib.get("height", "0") or 0.0)
        return abs(w * h)
    elif tag == "circle":
        r = float(elem.attrib.get("r", "0") or 0.0)
        return math.pi * r * r
    elif tag == "ellipse":
        rx = float(elem.attrib.get("rx", "0") or 0.0)
        ry = float(elem.attrib.get("ry", "0") or 0.0)
        return math.pi * rx * ry
    elif tag == "polygon":
        pts = parse_points_attr(elem.attrib.get("points", ""))
        return polygon_area(pts)
    elif tag == "polyline":
        pts = parse_points_attr(elem.attrib.get("points", ""))
        if pts and pts[0] != pts[-1]:
            pts = pts + [pts[0]]
        return polygon_area(pts)
    return 0.0

# ===================== 颜色排名 & 三步法决策 =====================
def rank_colors_by_area(root: ET.Element) -> Tuple[List[Tuple[str, float]], Dict[ET.Element, str]]:
    parent_map = {c: p for p in root.iter() for c in p}
    class_fill = collect_css_class_fills(root)
    color_area: Dict[str, float] = {}
    elem_color: Dict[ET.Element, str] = {}
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag not in ("path", "rect", "circle", "ellipse", "polygon", "polyline"):
            continue
        fill = get_effective_fill(elem, parent_map, class_fill)
        if not fill:
            continue
        area = element_filled_area(elem)
        if area <= 0:
            continue
        color_area[fill] = color_area.get(fill, 0.0) + area
        elem_color[elem] = fill
    ranked = sorted(color_area.items(), key=lambda x: x[1], reverse=True)
    return ranked, elem_color

def tiered_policy_keep_colors(
    ranked: List[Tuple[str, float]],
    tiny_pct: float = TINY_PCT,
    small_pct: float = SMALL_PCT,
    deltaE_thr: float = DELTAE_SMALL_SIMILAR
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """三步法：返回保留颜色列表 + 决策明细。"""
    total = sum(a for _, a in ranked) or 1.0
    perc = {c: a / total * 100.0 for c, a in ranked}
    # 预计算 Lab
    labs: Dict[str, Optional[Tuple[float, float, float]]] = {}
    for c, _ in ranked:
        rgb = parse_rgb01(c)
        labs[c] = rgb_to_lab(*rgb) if rgb else None

    keep, drop = set(), set()
    details: List[Dict[str, Any]] = []

    for i, (c, a) in enumerate(ranked):
        p = perc[c]
        reason = ""
        nearest = None
        dEmin = None
        if p < tiny_pct:
            drop.add(c)
            reason = f"percent {p:.4f}% < tiny {tiny_pct}% -> DROP"
        elif p >= small_pct:
            keep.add(c)
            reason = f"percent {p:.4f}% ≥ small {small_pct}% -> KEEP"
        else:
            # 与所有更大面积的颜色比较 ΔE，取最小
            lab_c = labs.get(c)
            if lab_c is not None:
                for j in range(i):
                    cj, _ = ranked[j]
                    lab_j = labs.get(cj)
                    if lab_j is None:
                        continue
                    dE = deltaE76(lab_c, lab_j)
                    if dEmin is None or dE < dEmin:
                        dEmin = dE
                        nearest = cj
            if dEmin is not None and dEmin <= deltaE_thr:
                drop.add(c)
                reason = (f"{tiny_pct}% ≤ percent {p:.4f}% < {small_pct}%, "
                          f"ΔE({c} vs {nearest})={dEmin:.2f} ≤ {deltaE_thr} -> DROP")
            else:
                keep.add(c)
                extra = f" (nearest={nearest}, ΔE={dEmin:.2f})" if dEmin is not None else ""
                reason = (f"{tiny_pct}% ≤ percent {p:.4f}% < {small_pct}%"
                          f"{extra} -> KEEP")

        details.append({
            "color": c,
            "area": a,
            "percent": p,
            "decision": "KEEP" if c in keep else "DROP",
            "reason": reason,
            "nearest_larger_color": nearest,
            "deltaE_to_nearest": (None if dEmin is None else round(dEmin, 4))
        })

    return list(keep), details

# ===================== 清理极短描边路径 =====================
def cleanup_micro_strokes(root: ET.Element,
                          min_len: float,
                          max_bbox_w: float,
                          max_bbox_h: float,
                          only_stroked: bool,
                          bottom_y: Optional[float],
                          force_linecap_butt: bool) -> Dict[str, Any]:
    """删除疑似“端帽黑点”的极短路径；可选把所有路径端帽改为 butt。"""
    parent_map = {c: p for p in root.iter() for c in p}
    class_fill = collect_css_class_fills(root)

    removed = 0
    scanned = 0
    reasons = []

    for el in list(root.iter()):
        if el.tag.split("}")[-1] != "path":
            continue

        # 端帽修正（可选）
        if force_linecap_butt:
            # 如果没有显式设置，直接加上；若有设置且不是 butt，也改为 butt
            if el.attrib.get("stroke-linecap") != "butt":
                el.set("stroke-linecap", "butt")

        # 是否参与清理
        fill = get_effective_fill(el, parent_map, class_fill)
        stroke, linecap = get_effective_stroke(el, parent_map)

        if only_stroked:
            # 仅在 “没有填充 + 有描边” 时考虑清理
            if (fill is not None) or (stroke is None or stroke == "none"):
                continue

        d = el.attrib.get("d", "")
        pts = parse_path_points_for_length(d)
        if len(pts) < 2:
            # 基本无长度
            par = parent_map.get(el)
            if par is not None:
                par.remove(el); removed += 1
                reasons.append({"index": id(el), "why": "len<2 points"})
            continue

        scanned += 1
        L = path_length(pts)
        x0, y0, x1, y1 = bbox_of_pts(pts)
        w = x1 - x0
        h = y1 - y0

        if bottom_y is not None and y0 > bottom_y:
            # 若限定仅底边附近清理（例如 bottom_y=0.0），而该路径不在底部，则跳过
            continue

        if L <= min_len or (w <= max_bbox_w and h <= max_bbox_h):
            par = parent_map.get(el)
            if par is not None:
                par.remove(el); removed += 1
                reasons.append({
                    "index": id(el),
                    "length": round(L, 3),
                    "bbox_w": round(w, 3),
                    "bbox_h": round(h, 3),
                    "y_min": round(y0, 3),
                    "stroke": stroke,
                    "linecap": linecap,
                    "why": f"length<={min_len} or (bbox_w<={max_bbox_w} & bbox_h<={max_bbox_h})"
                })

    return {"scanned_paths": scanned, "removed_micro_strokes": removed, "details": reasons}

# ===================== 过滤写回 =====================
def filter_svg_by_keep_colors(svg_in: Path, svg_out: Path, keep_colors: List[str]) -> int:
    tree = ET.parse(svg_in)
    root = tree.getroot()
    ranked, elem_color = rank_colors_by_area(root)
    keep = set(keep_colors)

    parent_map = {c: p for p in root.iter() for c in p}
    removed = 0
    for elem, col in list(elem_color.items()):
        if col not in keep:
            parent = parent_map.get(elem)
            if parent is not None:
                try:
                    parent.remove(elem)
                    removed += 1
                except Exception:
                    pass

    svg_out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(svg_out, encoding="utf-8", xml_declaration=True)
    return removed

# ===================== 批处理主流程 =====================
def process_dir(input_dir: Path, output_dir: Path,
                tiny_pct=TINY_PCT, small_pct=SMALL_PCT, deltaE_thr=DELTAE_SMALL_SIMILAR,
                cleanup=True, min_len=CLEAN_MIN_STROKE_LEN, max_w=CLEAN_MAX_BBOX_W, max_h=CLEAN_MAX_BBOX_H,
                only_stroked=CLEAN_ONLY_STROKED, bottom_y=CLEAN_BOTTOM_Y, force_linecap_butt=FORCE_LINECAP_BUTT
                ) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for svg_path in sorted(input_dir.glob("*.svg")):
        file_result = {"file": svg_path.name}
        try:
            # 1) 颜色判定（先读取一次以拿到排名等信息）
            tree = ET.parse(svg_path)
            root = tree.getroot()
            ranked, _ = rank_colors_by_area(root)
            keep_colors, details = tiered_policy_keep_colors(ranked, tiny_pct, small_pct, deltaE_thr)
            file_result["colors"] = details
            file_result["kept_colors"] = keep_colors

            # 2) 删除非保留颜色元素，写临时树
            removed_color_elems = filter_svg_by_keep_colors(svg_path, output_dir / svg_path.name, keep_colors)
            file_result["removed_elements_by_color"] = removed_color_elems

            # 3) 对输出文件做“极短描边路径清理”
            if cleanup or force_linecap_butt:
                tree2 = ET.parse(output_dir / svg_path.name)
                root2 = tree2.getroot()
                cln = cleanup_micro_strokes(root2, min_len, max_w, max_h, only_stroked, bottom_y, force_linecap_butt)
                file_result["cleanup"] = cln
                tree2.write(output_dir / svg_path.name, encoding="utf-8", xml_declaration=True)

            results.append(file_result)

        except Exception as e:
            file_result["error"] = f"Failed: {e}"
            results.append(file_result)

    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return {"dir": str(input_dir), "count": len(results)}

def main():
    parser = argparse.ArgumentParser()
    # 颜色“三步法”参数
    parser.add_argument("--input", type=str, default=INPUT_ROOT, help="输入根目录（包含多个子文件夹）")
    parser.add_argument("--output", type=str, default=OUTPUT_ROOT, help="输出根目录（每次运行会清空）")
    parser.add_argument("--tiny", type=float, default=TINY_PCT, help="tiny 阈值，默认 0.1（%）")
    parser.add_argument("--small", type=float, default=SMALL_PCT, help="small 阈值，默认 5（%）")
    parser.add_argument("--deltaE", type=float, default=DELTAE_SMALL_SIMILAR, help="ΔE 相似阈值，默认 30.0")
    # 噪点清理参数
    parser.add_argument("--cleanup", type=int, default=1, help="是否启用极短描边路径清理：1 启用（默认），0 禁用")
    parser.add_argument("--minStrokeLen", type=float, default=CLEAN_MIN_STROKE_LEN, help="极短路径线长阈值（≤即删），默认 3.0")
    parser.add_argument("--maxStrokeBBoxW", type=float, default=CLEAN_MAX_BBOX_W, help="极小包围盒宽阈值（≤即删），默认 2.0")
    parser.add_argument("--maxStrokeBBoxH", type=float, default=CLEAN_MAX_BBOX_H, help="极小包围盒高阈值（≤即删），默认 2.0")
    parser.add_argument("--onlyStroked", type=int, default=1, help="仅清理“无填充+有描边”的路径：1 是（默认），0 否")
    parser.add_argument("--bottomY", type=float, default=None, help="仅清理 bbox_min_y≤该值 的极短路径；默认 None（不限位置）")
    parser.add_argument("--setLineCapButt", type=int, default=0, help="把所有路径的端帽统一改为 butt：1 是，0 否（默认）")

    args = parser.parse_args()

    inroot = Path(args.input)
    outroot = Path(args.output)

    if not inroot.exists():
        print(f"[ERROR] 输入目录不存在：{inroot}")
        return

    # 每次运行清空输出
    if outroot.exists():
        shutil.rmtree(outroot)
    outroot.mkdir(parents=True, exist_ok=True)

    subdirs = [p for p in inroot.iterdir() if p.is_dir()]
    if not subdirs:
        print(f"[INFO] 未发现子目录，直接处理 {inroot}")
        process_dir(
            inroot, outroot,
            tiny_pct=args.tiny, small_pct=args.small, deltaE_thr=args.deltaE,
            cleanup=bool(args.cleanup),
            min_len=args.minStrokeLen, max_w=args.maxStrokeBBoxW, max_h=args.maxStrokeBBoxH,
            only_stroked=bool(args.onlyStroked),
            bottom_y=args.bottomY if not (args.bottomY != args.bottomY) else None, # 兼容 NaN
            force_linecap_butt=bool(args.setLineCapButt)
        )
    else:
        for sub in sorted(subdirs):
            print(f"[RUN] 处理子目录：{sub.name}")
            process_dir(
                sub, outroot / sub.name,
                tiny_pct=args.tiny, small_pct=args.small, deltaE_thr=args.deltaE,
                cleanup=bool(args.cleanup),
                min_len=args.minStrokeLen, max_w=args.maxStrokeBBoxW, max_h=args.maxStrokeBBoxH,
                only_stroked=bool(args.onlyStroked),
                bottom_y=args.bottomY if not (args.bottomY != args.bottomY) else None,
                force_linecap_butt=bool(args.setLineCapButt)
            )

    print(f"[OK] 全部完成，输出在：{outroot}")

if __name__ == "__main__":
    main()
