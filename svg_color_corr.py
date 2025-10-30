#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量颜色矫正（SVG → 参考色卡最近色），并输出 JSON 报告。
- 参考：example_svg/ 中每个 *.svg（仅当在 SVG_OUTPUT/ 中真的被“同名前缀”使用到，才处理并出报告）
- 目标：SVG_OUTPUT/ 中以参考名前缀开头的 *.svg（如 bluecar*.svg）
- 输出：SVG_OUTPUT_CORRECTED/；以及 report_<ref>.json（仅对“用到的参考”生成）

更新点：
- 用 ΔE2000 代替 Lab 欧氏距离，修正高饱和红附近的感知偏差（让 eb424f/ef4351/ed4352/e64954/e64853 更稳定去 f74f73）。
- patch_style：颜色键遇到空值（如 "fill:"）直接删除该条目，避免挡住继承/默认值。
- 默认黑注入：不再受无关 style 改动影响，只依据“是否显式声明 fill + 是否封闭形状”。
- 参考色卡提取：对 url(...) 统一用 startswith 判断。
"""

import os, re, math, json, glob
from xml.etree import ElementTree as ET
from PIL import ImageColor
from math import sqrt, atan2, cos, sin, radians, degrees, exp

EXAMPLE_DIR = "example_svg"
INPUT_DIR   = "SVG_OUTPUT"
OUTPUT_DIR  = "SVG_OUTPUT_CORRECTED"

COLOR_KEYS = ("fill","stroke","stop-color","color")

# ==== 阈值（更强的“近黑→黑”偏置，同时避免有色深色被吸黑） ====
L_BLACK_T     = 8.0    # 亮度判定“极暗”
C_GRAY_T      = 4.0    # 色度判定“近灰”
D_BLACK_ABS_T = 6.0    # 绝对接近黑：ΔE<=6 直接判黑（几乎纯黑）
BLACK_MARGIN  = 6.2    # 有色深色：黑要比最近非黑好 ≥6.2 ΔE 才能抢走
NEARBLACK_MARGIN = 2.0 # 近黑区：允许黑比最近非黑“多差”≤2 ΔE 仍可判黑

# ========== 颜色解析/格式化 ==========
def parse_color(s):
    """Return (kind, (r,g,b), alpha, fmt) where kind in {'rgb','none','ref'}; fmt 标记原始格式"""
    if not s: return None
    t = s.strip()
    tl = t.lower()
    if tl in ("none","transparent"): return ("none", None, None, None)
    if t.startswith("url("): return ("ref", None, None, None)

    m = re.fullmatch(r"#([0-9a-fA-F]{3,4}|[0-9a-fA-F]{6,8})", t)
    if m:
        h = m.group(1)
        if len(h) == 3:  # RGB
            r,g,b = int(h[0]*2,16), int(h[1]*2,16), int(h[2]*2,16); a=None; fmt="hex3"
        elif len(h) == 4:  # RGBA (nibble alpha)
            r,g,b = int(h[0]*2,16), int(h[1]*2,16), int(h[2]*2,16); a = int(h[3]*2,16)/255.0; fmt="hex4"
        elif len(h) == 6:  # RRGGBB
            r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16); a=None; fmt="hex6"
        else:              # RRGGBBAA
            r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16); a = int(h[6:8],16)/255.0; fmt="hex8"
        return ("rgb", (r,g,b), a, fmt)

    m = re.fullmatch(r"rgba?\((.+)\)", t, re.IGNORECASE)
    if m:
        ps = [p.strip() for p in m.group(1).split(",")]
        if len(ps) >= 3:
            def ch(x): return int(round(float(x[:-1])*2.55)) if x.endswith("%") else int(float(x))
            r,g,b = ch(ps[0]), ch(ps[1]), ch(ps[2])
            a = None
            if len(ps) >= 4:
                try: a = float(ps[3]); a = max(0.0, min(1.0, a))
                except: a = None
            return ("rgb", (max(0,min(255,r)), max(0,min(255,g)), max(0,min(255,b))), a, "rgba" if a is not None else "rgb")

    m = re.fullmatch(r"hsla?\((.+)\)", t, re.IGNORECASE)
    if m:
        ps = [p.strip() for p in m.group(1).split(",")]
        if len(ps) >= 3:
            h = float(ps[0].rstrip("deg")); s = float(ps[1].rstrip("%"))/100.0; l = float(ps[2].rstrip("%"))/100.0
            C = (1-abs(2*l-1))*s; x = (h/60)%2-1; X = C*(1-abs(x))
            if   0<=h%360<60:  r1,g1,b1=C,X,0
            elif 60<=h%360<120:r1,g1,b1=X,C,0
            elif 120<=h%360<180:r1,g1,b1=0,C,X
            elif 180<=h%360<240:r1,g1,b1=0,X,C
            elif 240<=h%360<300:r1,g1,b1=X,0,C
            else:               r1,g1,b1=C,0,X
            m_ = l - C/2
            r,g,b = int(round((r1+m_)*255)), int(round((g1+m_)*255)), int(round((b1+m_)*255))
            a = None
            if len(ps) >= 4:
                try: a = float(ps[3]); a = max(0.0, min(1.0, a))
                except: a = None
            return ("rgb", (max(0,min(255,r)), max(0,min(255,g)), max(0,min(255,b))), a, "rgba" if a is not None else "rgb")

    try:
        r,g,b = ImageColor.getrgb(t)
        return ("rgb", (r,g,b), None, None)
    except Exception:
        return None

def rgb_hex(rgb): return "#{:02X}{:02X}{:02X}".format(*rgb)

def fmt_with_alpha(new_rgb, alpha, prefer_fmt=None):
    """优先保留原格式；否则：不透明→#RRGGBB；半透明→rgba()."""
    if prefer_fmt in ("hex4","hex8") and alpha is not None:
        if prefer_fmt == "hex4":
            nib = max(0, min(15, int(round(alpha * 15))))
            return "#{:02X}{:02X}{:02X}{:X}".format(new_rgb[0], new_rgb[1], new_rgb[2], nib)
        if prefer_fmt == "hex8":
            aa = max(0, min(255, int(round(alpha * 255))))
            return "#{:02X}{:02X}{:02X}{:02X}".format(new_rgb[0], new_rgb[1], new_rgb[2], aa)
    if alpha is None or abs(alpha-1.0) < 1e-6:
        return rgb_hex(new_rgb)
    return "rgba({},{},{},{:.4g})".format(new_rgb[0], new_rgb[1], new_rgb[2], alpha)

# ========== sRGB→Lab ==========
def srgb_to_xyz(rgb):
    r,g,b = [c/255.0 for c in rgb]
    def inv(u): return ((u+0.055)/1.055)**2.4 if u>0.04045 else u/12.92
    r,g,b = inv(r),inv(g),inv(b)
    X = r*0.4124564 + g*0.3575761 + b*0.1804375
    Y = r*0.2126729 + g*0.7151522 + b*0.0721750
    Z = r*0.0193339 + g*0.1191920 + b*0.9503041
    return (X,Y,Z)
def xyz_to_lab(xyz):
    X,Y,Z = xyz; Xr,Yr,Zr = 0.95047,1.0,1.08883
    x,y,z = X/Xr, Y/Yr, Z/Zr
    def f(t): return t**(1/3) if t>(6/29)**3 else (1/3)*(29/6)**2*t + 4/29
    fx,fy,fz = f(x),f(y),f(z)
    L = 116*fy-16; a = 500*(fx-fy); b = 200*(fy-fz)
    return (L,a,b)
def rgb_to_lab(rgb): return xyz_to_lab(srgb_to_xyz(rgb))

# ---- CIEDE2000 ----
def deltaE2000(lab1, lab2):
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    avg_L = (L1 + L2) / 2.0
    C1 = sqrt(a1*a1 + b1*b1)
    C2 = sqrt(a2*a2 + b2*b2)
    avg_C = (C1 + C2) / 2.0
    if avg_C == 0:
        G = 0.0
    else:
        G = 0.5 * (1 - sqrt((avg_C**7) / (avg_C**7 + 25**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = sqrt(a1p*a1p + b1*b1)
    C2p = sqrt(a2p*a2p + b2*b2)
    avg_Cp = (C1p + C2p) / 2.0
    h1p = (degrees(atan2(b1, a1p)) + 360.0) % 360.0
    h2p = (degrees(atan2(b2, a2p)) + 360.0) % 360.0
    if abs(h1p - h2p) > 180:
        avg_hp = (h1p + h2p + 360.0) / 2.0 if (h1p + h2p) < 360.0 else (h1p + h2p - 360.0) / 2.0
    else:
        avg_hp = (h1p + h2p) / 2.0
    T = (1
         - 0.17 * cos(radians(avg_hp - 30))
         + 0.24 * cos(radians(2 * avg_hp))
         + 0.32 * cos(radians(3 * avg_hp + 6))
         - 0.20 * cos(radians(4 * avg_hp - 63)))
    d_hp = h2p - h1p
    if abs(d_hp) > 180:
        d_hp += 360 if h2p <= h1p else -360
    dLp = L2 - L1
    dCp = C2p - C1p
    dHp = 2 * sqrt(C1p * C2p) * sin(radians(d_hp / 2.0))
    S_L = 1 + (0.015 * (avg_L - 50) ** 2) / (200 + (avg_L - 50) ** 2)
    S_C = 1 + 0.045 * avg_Cp
    S_H = 1 + 0.015 * avg_Cp * T
    delta_ro = 30 * exp(-(((avg_hp - 275) / 25) ** 2))
    R_C = 2 * (avg_Cp ** 7 / (avg_Cp ** 7 + 25 ** 7))
    R_T = -R_C * sin(radians(2 * delta_ro))
    k_L = k_C = k_H = 1.0
    return sqrt((dLp / (k_L * S_L)) ** 2
              + (dCp / (k_C * S_C)) ** 2
              + (dHp / (k_H * S_H)) ** 2
              + R_T * (dCp / (k_C * S_C)) * (dHp / (k_H * S_H)))

# 用 ΔE2000 替换原欧氏 ΔE
def deltaE(l1, l2): 
    return deltaE2000(l1, l2)

# ========== 工具：style / 继承 / 闭合 ==========
def parse_inline_style(style_str):
    """只保留非空值（如 'fill:' 空值不计），避免空属性/空样式挡住解析。"""
    m = {}
    if not isinstance(style_str, str):
        return m
    for part in style_str.split(";"):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip().lower()
        v = v.strip()
        if v != "":
            m[k] = v
    return m

def get_attr_or_style(el, key):
    """优先属性；若属性值为空字符串则视为未声明，再查 style。返回 (value, 'attr'/'style') 或 (None, None)。"""
    if key in el.attrib:
        v = (el.attrib.get(key) or "").strip()
        if v != "":
            return v, "attr"
    st = parse_inline_style(el.attrib.get("style"))
    if key in st:
        return st[key], "style"
    return None, None

def build_parent_map(root):
    m = {}
    for p in root.iter():
        for c in p: m[c] = p
    return m

def tag_of(el): return el.tag.split('}')[-1].lower()

def is_closed(el):
    t = tag_of(el)
    if t in ("rect","circle","ellipse","polygon"): return True
    if t == "path":
        d = el.attrib.get("d","")
        return bool(re.search(r"[zZ]", d))
    return False

def effective_prop(el, key, parent_map, css_map, allow_inherit=False):
    """inline attr -> inline style -> css(id/class/tag) -> inherit(color)"""
    v, src = get_attr_or_style(el, key)
    if v is not None: return v, "inline"
    # css by id
    elid = el.attrib.get("id")
    if elid and f"#{elid}" in css_map and key in css_map[f"#{elid}"]:
        return css_map[f"#{elid}"][key], "css_id"
    # css by class
    for c in el.attrib.get("class","").split():
        sel = f".{c}"
        if sel in css_map and key in css_map[sel]:
            return css_map[sel][key], "css_class"
    # css by tag
    sel = tag_of(el)
    if sel in css_map and key in css_map[sel]:
        return css_map[sel][key], "css_tag"
    if allow_inherit:
        cur = parent_map.get(el)
        while cur is not None:
            vv, ss = effective_prop(cur, key, parent_map, css_map, allow_inherit=False)
            if vv is not None: return vv, "inherit:"+ss
            cur = parent_map.get(cur)
        return "black", "initial_default"  # 'color' 的初始值按黑处理
    return None, None

def parse_css_simple(style_text):
    """极简 CSS 解析，仅 .class/#id/tag 三类选择器"""
    css = {}
    for sel, body in re.findall(r"([^{]+)\{([^}]+)\}", style_text or "", flags=re.S):
        props = {}
        for part in body.split(";"):
            if ":" not in part: continue
            k,v = part.split(":",1)
            props[k.strip().lower()] = (v.strip())
        for s in sel.split(","):
            key = s.strip()
            if key: css.setdefault(key, {}).update(props)
    return css

def text_of_style_tags(root):
    return "\n".join([el.text or "" for el in root.iter() if tag_of(el)=="style"])

# ========== 参考色卡 ==========
def extract_reference_palette(svg_path):
    tree = ET.parse(svg_path); root = tree.getroot()
    parent_map = build_parent_map(root)
    css_map = parse_css_simple(text_of_style_tags(root))
    colors = []

    for el in root.iter():
        t = tag_of(el)
        if t in ("path","rect","circle","ellipse","polygon","polyline","g","use","text"):
            fval, _ = effective_prop(el, "fill", parent_map, css_map, allow_inherit=False)
            sval, _ = effective_prop(el, "stroke", parent_map, css_map, allow_inherit=False)

            # “未声明 fill 的封闭图形” → 默认黑
            declared_any_fill = (get_attr_or_style(el,"fill")[0] is not None)
            if not declared_any_fill:
                elid = el.attrib.get("id")
                if elid and f"#{elid}" in css_map and "fill" in css_map[f"#{elid}"]: declared_any_fill = True
                if not declared_any_fill:
                    for c in el.attrib.get("class","").split():
                        if f".{c}" in css_map and "fill" in css_map[f".{c}"]: declared_any_fill = True; break
                if not declared_any_fill and t in css_map and "fill" in css_map[t]: declared_any_fill = True
            if not declared_any_fill and is_closed(el):
                colors.append("#000000")
            else:
                if fval and (not fval.lower().startswith("url(")) and fval.lower() not in ("none","transparent"):
                    p = parse_color(fval)
                    if p and p[0]=="rgb": colors.append(rgb_hex(p[1]))
                if sval and sval.lower() not in ("none","transparent"):
                    p = parse_color(sval)
                    if p and p[0]=="rgb": colors.append(rgb_hex(p[1]))

        elif t == "stop":
            sc, _ = get_attr_or_style(el, "stop-color")
            if sc:
                p = parse_color(sc)
                if p and p[0]=="rgb": colors.append(rgb_hex(p[1]))

    # 去重保序
    seen=set(); uniq=[]
    for h in colors:
        if h not in seen:
            seen.add(h); uniq.append(h)
    return uniq

# ========== 最近色映射（含“近黑优先黑 / 有色优先非黑”） ==========
class NearestPalette:
    def __init__(self, hex_list):
        self.hex = list(hex_list)
        self.lab = [rgb_to_lab(tuple(int(h[i:i+2],16) for i in (1,3,5))) for h in self.hex]
        self.black_idx = next((i for i,h in enumerate(self.hex) if h.upper()=="#000000"), None)

    def decide(self, rgb):
        lab = rgb_to_lab(rgb)
        L, a, b = lab
        C = math.sqrt(a*a + b*b)

        # 计算：全局最近 / 最近非黑 / 到黑的距离（用 ΔE2000）
        best_i, best_d = None, 1e9
        best_nonblack_i, best_nonblack_d = None, 1e9
        d_black = 1e9
        for i, lab2 in enumerate(self.lab):
            d = deltaE(lab, lab2)
            if d < best_d:
                best_d, best_i = d, i
            if self.black_idx is not None and i == self.black_idx:
                d_black = d
            else:
                if d < best_nonblack_d:
                    best_nonblack_d, best_nonblack_i = d, i

        # 无黑色参考：直接最近邻
        if self.black_idx is None:
            chosen = best_i
            rule = "nearest_neighbor"
        else:
            # A) 近黑区（极暗且近灰）：只有当黑不比最近非黑差太多，才判黑
            if L <= L_BLACK_T and C <= C_GRAY_T:
                if best_nonblack_i is None or d_black <= best_nonblack_d + NEARBLACK_MARGIN:
                    chosen = self.black_idx; rule = "near_black_margin_to_black"
                else:
                    chosen = best_nonblack_i; rule = "prefer_nonblack_inside_near_black"
            # B) 绝对接近黑：无视色度，直接判黑
            elif d_black <= D_BLACK_ABS_T:
                chosen = self.black_idx; rule = "absolute_black_distance"
            else:
                # C) 一般情况：有色（C大）优先非黑；黑要显著更近才能抢走
                if C > C_GRAY_T and best_nonblack_i is not None:
                    if d_black + BLACK_MARGIN < best_nonblack_d:
                        chosen = self.black_idx; rule = "black_margin_beats_nonblack"
                    else:
                        chosen = best_nonblack_i; rule = "prefer_nonblack_for_colored"
                else:
                    chosen = best_i; rule = "nearest_neighbor_grayish"

        target_hex = self.hex[chosen]
        decision = {
            "rule": rule,
            "source_rgb_hex": rgb_hex(rgb),
            "source_L": round(L, 3),
            "source_C": round(C, 3),
            "d_black": None if self.black_idx is None else round(d_black, 3),
            "best_nonblack_hex": None if best_nonblack_i is None else self.hex[best_nonblack_i],
            "best_nonblack_d": None if best_nonblack_i is None else round(best_nonblack_d, 3),
            "best_overall_hex": self.hex[best_i],
            "best_overall_d": round(best_d, 3),
            "chosen_hex": target_hex
        }
        new_rgb = tuple(int(target_hex[i:i+2],16) for i in (1,3,5))
        return new_rgb, decision

# ========== style 处理（健壮，支持 currentColor & 空值剔除） ==========
def patch_style(style_str, mapper, resolver=None, logger=None):
    if not isinstance(style_str, str) or not style_str.strip():
        return style_str
    parts = style_str.split(";")
    out = []
    for part in parts:
        s = (part or "").strip()
        if not s:
            continue
        if ":" not in s:
            out.append(s)
            continue
        try:
            k, v = s.split(":", 1)
            kk = k.strip().lower()
            vv = v.strip()
            if kk in COLOR_KEYS:
                if vv == "":
                    # 空值当未声明：直接跳过
                    continue
                orig_v = vv
                if vv.lower() == "currentcolor" and resolver:
                    rv, _ = resolver("color")
                    vv = rv or "#000000"
                new_v = remap_value(vv, mapper, logger, orig_literal=orig_v)
                out.append(f"{k}:{new_v}")
            else:
                out.append(f"{k}:{vv}")
        except Exception:
            out.append(s)
    return ";".join(out)

def remap_value(v, mapper: NearestPalette, logger=None, orig_literal=None):
    p = parse_color(v)
    if not p: return v
    kind, rgb, alpha, fmt = p
    if kind in ("ref","none"): return v
    new_rgb, decision = mapper.decide(rgb)
    out = fmt_with_alpha(new_rgb, alpha, prefer_fmt=fmt)
    if logger:
        try:
            key = (orig_literal if orig_literal is not None else v.strip())
            logger(key, rgb_hex(rgb), alpha, out, rgb_hex(new_rgb), decision)
        except Exception:
            pass
    return out

# ========== 单文件矫正 ==========
def correct_file(in_path, out_path, mapper: NearestPalette, file_logger=None):
    tree = ET.parse(in_path); root = tree.getroot()
    parent_map = build_parent_map(root)
    css_map = parse_css_simple(text_of_style_tags(root))

    # 渐变 stop
    for el in root.iter():
        if tag_of(el) == "stop":
            if "stop-color" in el.attrib:
                val = (el.attrib["stop-color"] or "").strip()
                if val != "":
                    el.attrib["stop-color"] = remap_value(val, mapper, logger=file_logger)
                else:
                    del el.attrib["stop-color"]
            if "style" in el.attrib:
                el.attrib["style"] = patch_style(el.attrib["style"], mapper,
                    resolver=lambda key: effective_prop(el, key, parent_map, css_map, allow_inherit=True),
                    logger=file_logger)

    # 普通元素
    for el in root.iter():
        t = tag_of(el)
        if t not in ("path","rect","circle","ellipse","polygon","polyline","g","use","text"):
            if "style" in el.attrib:
                try:
                    el.attrib["style"] = patch_style(el.attrib["style"], mapper,
                        resolver=lambda key: effective_prop(el, key, parent_map, css_map, allow_inherit=True),
                        logger=file_logger)
                except Exception: pass
            continue

        # 属性键：空值→删除；currentColor 解析；正常 remap
        for key in COLOR_KEYS:
            if key in el.attrib:
                val = (el.attrib[key] or "").strip()
                if val == "":
                    # 空值等价未声明
                    del el.attrib[key]
                else:
                    if val.lower() == "currentcolor":
                        base, _ = effective_prop(el, "color", parent_map, css_map, allow_inherit=True)
                        base = base or "#000000"
                        el.attrib[key] = remap_value(base, mapper, logger=file_logger, orig_literal=val)
                    else:
                        el.attrib[key] = remap_value(val, mapper, logger=file_logger)

        # style 键
        if "style" in el.attrib:
            try:
                el.attrib["style"] = patch_style(el.attrib["style"], mapper,
                    resolver=lambda k: effective_prop(el, k, parent_map, css_map, allow_inherit=True),
                    logger=file_logger)
            except Exception:
                pass

        # 未声明 fill 且为封闭形状 → 默认黑映射后显式写入（不再依赖 changed）
        if is_closed(el):
            # 判断是否 anywhere 声明过 fill（属性/行内style/css）
            declared_any_fill = (get_attr_or_style(el,"fill")[0] is not None)
            if not declared_any_fill:
                elid = el.attrib.get("id")
                if elid and f"#{elid}" in css_map and "fill" in css_map[f"#{elid}"]: declared_any_fill = True
                if not declared_any_fill:
                    for c in el.attrib.get("class","").split():
                        if f".{c}" in css_map and "fill" in css_map[f".{c}"]:
                            declared_any_fill = True; break
                if not declared_any_fill and t in css_map and "fill" in css_map[t]:
                    declared_any_fill = True

            if not declared_any_fill:
                el.attrib["fill"] = remap_value("#000000", mapper, logger=file_logger, orig_literal="(default-black)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if '}' in root.tag:
        ET.register_namespace('', root.tag.split('}')[0][1:])
    tree.write(out_path, encoding="utf-8", xml_declaration=True)

# ========== 清空输出目录 ==========
def clear_output_directory(output_dir):
    """清空输出目录中的所有文件"""
    if not os.path.isdir(output_dir):
        return
    
    # 删除所有文件
    for file_path in glob.glob(os.path.join(output_dir, "*")):
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"已删除: {file_path}")
        except Exception as e:
            print(f"删除失败 {file_path}: {e}")

# ========== 主流程：只对"用到的参考"处理并出报告 ==========
def main():
    # 清空输出目录
    print(f"[清理] 清空输出目录: {OUTPUT_DIR}")
    clear_output_directory(OUTPUT_DIR)
    
    # 目录存在性检查
    if not os.path.isdir(INPUT_DIR):
        print(f"[错误] 输入目录不存在：{INPUT_DIR}")
        return
    if not os.path.isdir(EXAMPLE_DIR):
        print(f"[错误] 参考目录不存在：{EXAMPLE_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_svgs = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".svg")]
    refs = [f for f in os.listdir(EXAMPLE_DIR) if f.lower().endswith(".svg")]

    total_out = 0
    for ref in refs:
        base = os.path.splitext(ref)[0].lower()
        matched = [n for n in input_svgs if n.lower().startswith(base)]
        if not matched:
            continue  # 这个参考未被使用，不处理、不出报告

        ref_path = os.path.join(EXAMPLE_DIR, ref)
        palette = extract_reference_palette(ref_path)
        if not palette:
            print(f"[跳过] 参考未提取到颜色：{ref_path}")
            continue

        mapper = NearestPalette(palette)

        report = {
            "reference_svg": ref_path,
            "reference_name": base,
            "palette_hex": palette,
            "thresholds": {
                "L_BLACK_T": L_BLACK_T,
                "C_GRAY_T": C_GRAY_T,
                "D_BLACK_ABS_T": D_BLACK_ABS_T,
                "BLACK_MARGIN": BLACK_MARGIN,
                "NEARBLACK_MARGIN": NEARBLACK_MARGIN
            },
            "notes": {
                "css_styles_parsed": True,
                "default_black_from_unfilled_closed_shapes": True,
                "empty_attr_treated_as_undeclared": True,
                "distance_metric": "CIEDE2000"
            },
            "files": []
        }

        for name in matched:
            in_path  = os.path.join(INPUT_DIR, name)
            out_path = os.path.join(OUTPUT_DIR, name)

            file_log = {}
            def file_logger(orig_literal, src_hex, alpha, mapped_literal, mapped_hex, decision):
                if orig_literal not in file_log:
                    file_log[orig_literal] = {
                        "source_literal": orig_literal,
                        "source_rgb_hex": src_hex,
                        "alpha": None if alpha is None else float(alpha),
                        "target_literal": mapped_literal,
                        "target_rgb_hex": mapped_hex,
                        "decision": decision
                    }

            try:
                correct_file(in_path, out_path, mapper, file_logger=file_logger)
                report["files"].append({
                    "input_svg": in_path,
                    "output_svg": out_path,
                    "mappings": list(file_log.values())
                })
                total_out += 1
                print(f"[OK] {name}  参考: {ref}")
            except Exception as e:
                print(f"[失败] {name}: {e}")
                report["files"].append({
                    "input_svg": in_path,
                    "output_svg": out_path,
                    "error": str(e)
                })

        if report["files"]:
            report_path = os.path.join(OUTPUT_DIR, f"report_{base}.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"[报告] {report_path}")

    print(f"[完成] 实际输出文件数：{total_out}")

if __name__ == "__main__":
    main()
