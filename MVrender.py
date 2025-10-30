#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多视图透明渲染（Trellis）
- 输入：example_image/bluecar.png（如需更换，将 DEFAULT_IMAGE_PATH 改为目标路径；若仅给文件名，默认从 example_image/ 下读取）
- 处理：TrellisImageTo3DPipeline → 按 0..345° 每 15° 渲染 24 张 → 黑/白背景对比生成透明通道
- 输出：MultiView 目录，<basename>_000deg.png … _345deg.png（RGBA，背景透明）；MultiView_additional 目录，<basename>_xxxyaw_+xxxpitch.png,（RGBA，背景透明）
"""

from __future__ import annotations
import os
import sys
import glob
import json

import numpy as np
from PIL import Image

import torch
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils  # 保留以支持下方可选功能

# =====================
# 固定配置（可在此处修改）
# =====================
DEFAULT_IMAGE_PATH = "example_image/bluecar.png" 
OUTPUT_DIR = "MultiView"
OUTPUT_DIR_FIXED = "MultiView_fixed"  # 固定输出目录
ADDITIONAL_OUTPUT_DIR = "MultiView_additional"  # 新增角度输出目录（已注释）
ZERO_PITCH_OUTPUT_DIR = "MultiView_horizontal"  # 俯仰为0度的水平角度输出目录（已注释）
PITCH_MINUS_45_OUTPUT_DIR = "MultiView_pitch_minus_45"  # 俯仰-45度角度输出目录
PITCH_PLUS_45_OUTPUT_DIR = "MultiView_pitch_plus_45"  # 俯仰+45度角度输出目录
RESOLUTION = 512
FOV_DEG = 40.0
CAMERA_RADIUS = 2.0
SEED = 1
ALPHA_THRESHOLD = 0.1
CLEAR_OUTPUT_BEFORE_RENDER = True
USE_FP16 = False    # True: autocast(FP16)；False: 关闭
USE_BF16 = False    # True: autocast(BF16)；False: 关闭（两者仅启用其一）

# 环境（根据需要取消注释）
# os.environ['ATTN_BACKEND'] = 'xformers'
# os.environ['ATTN_BACKEND'] = 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 便于报错定位；如追求速度可注释

# cuDNN 设置（性能为主）
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


# ===============
# 工具函数
# ===============

def _ensure_input_path(path_or_name: str) -> str:
    """返回可用的输入路径。"""
    p = path_or_name
    if not os.path.exists(p):
        base = path_or_name
        if not (base.endswith('.png') or base.endswith('.jpg') or base.endswith('.jpeg')):
            base += '.png'
        cand = os.path.join('example_image', base)
        if os.path.exists(cand):
            p = cand
    if not os.path.exists(p):
        raise FileNotFoundError(f"找不到输入图像: {path_or_name}")
    return p


def _load_image_to_res(path: str, target_res: int) -> Image.Image:
    """读取图像并将长边缩放到 target_res。"""
    im = Image.open(path).convert('RGBA')
    w, h = im.size
    if max(w, h) != target_res:
        scale = target_res / max(w, h)
        im = im.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.LANCZOS)
    return im


# 原来的30度间隔函数（已注释，可随时恢复）
# def _angles_degrees_12() -> list[int]:
#     """0..330，每 30° 共 12 个视角。"""
#     return [i * 30 for i in range(12)]

def _angles_degrees_12() -> list[int]:
    """0..330，每 30° 共 12 个视角。"""
    return [i * 30 for i in range(12)]

# 新增俯仰-45度角度组（每30度12个视角）
def _angles_pitch_minus_45() -> list[tuple[int, int]]:
    """俯仰-45度角度组：每30度12个视角，俯仰角固定为-45度"""
    return [(i * 30, -45) for i in range(12)]

# 新增俯仰+45度角度组（每30度12个视角）
def _angles_pitch_plus_45() -> list[tuple[int, int]]:
    """俯仰+45度角度组：每30度12个视角，俯仰角固定为+45度"""
    return [(i * 30, 45) for i in range(12)]

#实验查看不同角度使用此函数（已注释）
# def _angles_additional() -> list[tuple[int, int]]:
#     """新增的特定角度：(yaw, pitch) - (30,+30),(90,+30),(150,+30),(210,-20),(270,-20),(330,-20),(330,+45),(240,+45)"""
#     return [(120, 30), (0, 30), (240, 30), (300, -20), (180, -20), (60, -20), (330, 45), (240, 45)]

# def _angles_zero_pitch() -> list[tuple[int, int]]:
#     """俯仰为0度的水平角度：(yaw, pitch) - 0,45,90,135,180,225,270,315度"""
#     return [(0, 0), (45, 0), (90, 0), (135, 0), (180, 0), (225, 0), (270, 0), (315, 0)]


def _alpha_from_contrast(black_img: np.ndarray, white_img: np.ndarray, thr: float) -> np.ndarray:
    """黑白对比生成 alpha，返回 H×W uint8。"""
    diff = np.abs(white_img.astype(np.int16) - black_img.astype(np.int16))
    diff_mean = diff.mean(axis=2).astype(np.float32) / 255.0
    alpha = (diff_mean > thr).astype(np.uint8) * 255
    return 255 - alpha  # 与渲染逻辑配合


def _render_colors(gaussian, angles_deg: list[int], r: float, fov_deg: float, res: int):
    """分别以黑/白背景渲染，返回 (colors_black, colors_white)。"""
    yaws = [np.deg2rad(a) for a in angles_deg]
    pitch = [0.0] * len(angles_deg)
    extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws, pitch, rs=r, fovs=fov_deg
    )
    rb = render_utils.render_frames(gaussian, extrinsics, intrinsics,
                                    {'resolution': res, 'bg_color': [0, 0, 0]})
    rw = render_utils.render_frames(gaussian, extrinsics, intrinsics,
                                    {'resolution': res, 'bg_color': [1, 1, 1]})
    return rb['color'], rw['color']

#实验查看不同角度使用此函数
def _render_colors_3d(gaussian, angles_3d: list[tuple[int, int]], r: float, fov_deg: float, res: int):
    """分别以黑/白背景渲染3D角度，返回 (colors_black, colors_white)。"""
    yaws = [np.deg2rad(yaw) for yaw, pitch in angles_3d]
    pitches = [np.deg2rad(pitch) for yaw, pitch in angles_3d]
    extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws, pitches, rs=r, fovs=fov_deg
    )
    rb = render_utils.render_frames(gaussian, extrinsics, intrinsics,
                                    {'resolution': res, 'bg_color': [0, 0, 0]})
    rw = render_utils.render_frames(gaussian, extrinsics, intrinsics,
                                    {'resolution': res, 'bg_color': [1, 1, 1]})
    return rb['color'], rw['color']

def _render_colors_3d_white_bg(gaussian, angles_3d: list[tuple[int, int]], r: float, fov_deg: float, res: int):
    """以白色背景渲染3D角度，返回白色背景的RGB图片。"""
    yaws = [np.deg2rad(yaw) for yaw, pitch in angles_3d]
    pitches = [np.deg2rad(pitch) for yaw, pitch in angles_3d]
    extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws, pitches, rs=r, fovs=fov_deg
    )
    rw = render_utils.render_frames(gaussian, extrinsics, intrinsics,
                                    {'resolution': res, 'bg_color': [1, 1, 1]})
    return rw['color']

def _render_colors_pitch_minus_45(gaussian, angles_pitch_minus_45: list[tuple[int, int]], r: float, fov_deg: float, res: int):
    """分别以黑/白背景渲染俯仰-45度角度，返回 (colors_black, colors_white)。"""
    yaws = [np.deg2rad(yaw) for yaw, pitch in angles_pitch_minus_45]
    pitches = [np.deg2rad(pitch) for yaw, pitch in angles_pitch_minus_45]
    extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws, pitches, rs=r, fovs=fov_deg
    )
    rb = render_utils.render_frames(gaussian, extrinsics, intrinsics,
                                    {'resolution': res, 'bg_color': [0, 0, 0]})
    rw = render_utils.render_frames(gaussian, extrinsics, intrinsics,
                                    {'resolution': res, 'bg_color': [1, 1, 1]})
    return rb['color'], rw['color']

def _render_colors_pitch_plus_45(gaussian, angles_pitch_plus_45: list[tuple[int, int]], r: float, fov_deg: float, res: int):
    """分别以黑/白背景渲染俯仰+45度角度，返回 (colors_black, colors_white)。"""
    yaws = [np.deg2rad(yaw) for yaw, pitch in angles_pitch_plus_45]
    pitches = [np.deg2rad(pitch) for yaw, pitch in angles_pitch_plus_45]
    extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws, pitches, rs=r, fovs=fov_deg
    )
    rb = render_utils.render_frames(gaussian, extrinsics, intrinsics,
                                    {'resolution': res, 'bg_color': [0, 0, 0]})
    rw = render_utils.render_frames(gaussian, extrinsics, intrinsics,
                                    {'resolution': res, 'bg_color': [1, 1, 1]})
    return rb['color'], rw['color']

def _render_colors_zero_pitch(gaussian, angles_zero_pitch: list[tuple[int, int]], r: float, fov_deg: float, res: int):
    """分别以黑/白背景渲染俯仰为0度的水平角度，返回 (colors_black, colors_white)。"""
    yaws = [np.deg2rad(yaw) for yaw, pitch in angles_zero_pitch]
    pitches = [np.deg2rad(pitch) for yaw, pitch in angles_zero_pitch]
    extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws, pitches, rs=r, fovs=fov_deg
    )
    rb = render_utils.render_frames(gaussian, extrinsics, intrinsics,
                                    {'resolution': res, 'bg_color': [0, 0, 0]})
    rw = render_utils.render_frames(gaussian, extrinsics, intrinsics,
                                    {'resolution': res, 'bg_color': [1, 1, 1]})
    return rb['color'], rw['color']


def _save_rgba_series(colors_black, colors_white, angles_deg, out_dir: str, basename: str, thr: float) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(OUTPUT_DIR_FIXED, exist_ok=True)
    for i, angle in enumerate(angles_deg):
        black = colors_black[i]
        white = colors_white[i]
        alpha = _alpha_from_contrast(black, white, thr)
        rgba = np.dstack((black.astype(np.uint8), alpha))
        
        # 保存到原始目录
        out_path = os.path.join(out_dir, f"{basename}_{angle:03d}deg.png")
        Image.fromarray(rgba, mode='RGBA').save(out_path, optimize=True)
        print(f"已保存: {out_path}")
        
        # 同时保存到固定目录
        out_path_fixed = os.path.join(OUTPUT_DIR_FIXED, f"{basename}_{angle:03d}deg.png")
        Image.fromarray(rgba, mode='RGBA').save(out_path_fixed, optimize=True)
        print(f"已保存到固定目录: {out_path_fixed}")

#实验查看不同角度使用此函数
def _save_rgba_series_3d(colors_black, colors_white, angles_3d, out_dir: str, basename: str, thr: float) -> None:
    """保存3D角度的RGBA系列"""
    os.makedirs(out_dir, exist_ok=True)
    for i, (yaw, pitch) in enumerate(angles_3d):
        black = colors_black[i]
        white = colors_white[i]
        alpha = _alpha_from_contrast(black, white, thr)
        rgba = np.dstack((black.astype(np.uint8), alpha))
        out_path = os.path.join(out_dir, f"{basename}_{yaw:03d}yaw_{pitch:+03d}pitch.png")
        Image.fromarray(rgba, mode='RGBA').save(out_path, optimize=True)
        print(f"已保存3D角度: {out_path}")

def _save_rgb_series_3d_white_bg(colors_white, angles_3d, out_dir: str, basename: str) -> None:
    """保存3D角度的RGB系列（白色背景）"""
    os.makedirs(out_dir, exist_ok=True)
    for i, (yaw, pitch) in enumerate(angles_3d):
        white_bg = colors_white[i]
        rgb = white_bg.astype(np.uint8)
        out_path = os.path.join(out_dir, f"{basename}_{yaw:03d}yaw_{pitch:+03d}pitch.png")
        Image.fromarray(rgb, mode='RGB').save(out_path, optimize=True)
        print(f"已保存3D角度（白色背景）: {out_path}")

def _save_rgba_series_pitch_minus_45(colors_black, colors_white, angles_pitch_minus_45, out_dir: str, basename: str, thr: float) -> None:
    """保存俯仰-45度角度的RGBA系列"""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(OUTPUT_DIR_FIXED, exist_ok=True)
    for i, (yaw, pitch) in enumerate(angles_pitch_minus_45):
        black = colors_black[i]
        white = colors_white[i]
        alpha = _alpha_from_contrast(black, white, thr)
        rgba = np.dstack((black.astype(np.uint8), alpha))
        
        # 保存到原始目录
        out_path = os.path.join(out_dir, f"{basename}_{yaw:03d}yaw_{pitch:+03d}pitch.png")
        Image.fromarray(rgba, mode='RGBA').save(out_path, optimize=True)
        print(f"已保存俯仰-45度角度: {out_path}")
        
        # 同时保存到固定目录
        out_path_fixed = os.path.join(OUTPUT_DIR_FIXED, f"{basename}_{yaw:03d}yaw_{pitch:+03d}pitch.png")
        Image.fromarray(rgba, mode='RGBA').save(out_path_fixed, optimize=True)
        print(f"已保存俯仰-45度角度到固定目录: {out_path_fixed}")

def _save_rgba_series_pitch_plus_45(colors_black, colors_white, angles_pitch_plus_45, out_dir: str, basename: str, thr: float) -> None:
    """保存俯仰+45度角度的RGBA系列"""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(OUTPUT_DIR_FIXED, exist_ok=True)
    for i, (yaw, pitch) in enumerate(angles_pitch_plus_45):
        black = colors_black[i]
        white = colors_white[i]
        alpha = _alpha_from_contrast(black, white, thr)
        rgba = np.dstack((black.astype(np.uint8), alpha))
        
        # 保存到原始目录
        out_path = os.path.join(out_dir, f"{basename}_{yaw:03d}yaw_{pitch:+03d}pitch.png")
        Image.fromarray(rgba, mode='RGBA').save(out_path, optimize=True)
        print(f"已保存俯仰+45度角度: {out_path}")
        
        # 同时保存到固定目录
        out_path_fixed = os.path.join(OUTPUT_DIR_FIXED, f"{basename}_{yaw:03d}yaw_{pitch:+03d}pitch.png")
        Image.fromarray(rgba, mode='RGBA').save(out_path_fixed, optimize=True)
        print(f"已保存俯仰+45度角度到固定目录: {out_path_fixed}")

def _save_rgba_series_zero_pitch(colors_black, colors_white, angles_zero_pitch, out_dir: str, basename: str, thr: float) -> None:
    """保存俯仰为0度水平角度的RGBA系列"""
    os.makedirs(out_dir, exist_ok=True)
    for i, (yaw, pitch) in enumerate(angles_zero_pitch):
        black = colors_black[i]
        white = colors_white[i]
        alpha = _alpha_from_contrast(black, white, thr)
        rgba = np.dstack((black.astype(np.uint8), alpha))
        out_path = os.path.join(out_dir, f"{basename}_{yaw:03d}yaw_{pitch:+03d}pitch.png")
        Image.fromarray(rgba, mode='RGBA').save(out_path, optimize=True)
        print(f"已保存水平角度: {out_path}")


def _maybe_clear_output_pngs(out_dir: str):
    if not os.path.isdir(out_dir):
        return
    for p in glob.glob(os.path.join(out_dir, "*.png")):
        try:
            os.remove(p)
            print(f"已删除: {p}")
        except Exception as e:
            print(f"删除失败 {p}: {e}")


# ===============
# 主流程
# ===============

def main() -> int:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    image_path = _ensure_input_path(DEFAULT_IMAGE_PATH)
    print(f"处理图像: {image_path}")

    if CLEAR_OUTPUT_BEFORE_RENDER:
        _maybe_clear_output_pngs(OUTPUT_DIR)
        _maybe_clear_output_pngs(OUTPUT_DIR_FIXED)
        # _maybe_clear_output_pngs(ADDITIONAL_OUTPUT_DIR)  # 已注释
        # _maybe_clear_output_pngs(ZERO_PITCH_OUTPUT_DIR)  # 已注释
        _maybe_clear_output_pngs(PITCH_MINUS_45_OUTPUT_DIR)
        _maybe_clear_output_pngs(PITCH_PLUS_45_OUTPUT_DIR)
    print(f"输出目录: {OUTPUT_DIR}/")
    print(f"固定输出目录: {OUTPUT_DIR_FIXED}/")
    # print(f"新增角度输出目录: {ADDITIONAL_OUTPUT_DIR}/")  # 已注释
    # print(f"水平角度输出目录: {ZERO_PITCH_OUTPUT_DIR}/")  # 已注释
    print(f"俯仰-45度角度输出目录: {PITCH_MINUS_45_OUTPUT_DIR}/")
    print(f"俯仰+45度角度输出目录: {PITCH_PLUS_45_OUTPUT_DIR}/")

    image = _load_image_to_res(image_path, RESOLUTION)
    print(f"处理图像尺寸: {image.size}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")

    try:
        pipeline = TrellisImageTo3DPipeline.from_pretrained("TRELLIS-image-large")
        if device == 'cuda':
            pipeline.cuda()
        print("Pipeline 已就绪")
    except Exception as e:
        print(f"加载 Pipeline 失败: {e}")
        return 1

    try:
        autocast_dtype = None
        if USE_FP16 and torch.cuda.is_available():
            autocast_dtype = torch.float16
        if USE_BF16 and torch.cuda.is_available():
            autocast_dtype = torch.bfloat16

        print("开始运行 pipeline…")
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=autocast_dtype is not None, dtype=autocast_dtype):
            outputs = pipeline.run(image, seed=SEED)
        print("Pipeline 运行完成")
    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower() and torch.cuda.is_available():
            print("CUDA OOM：可降低 RESOLUTION，或关闭 FP16/BF16，或清理其他进程。")
        print(f"运行失败: {e}")
        return 1
    except KeyboardInterrupt:
        print("中断。")
        return 130

    try:
        # 原有的12角度渲染（每30度）
        # 如需恢复15度间隔，请取消注释下面的代码并注释掉当前代码
        # angles_deg = _angles_degrees_24()  # 15度间隔，24张图片
        angles_deg = _angles_degrees_12()  # 30度间隔，12张图片
        colors_black, colors_white = _render_colors(outputs['gaussian'][0], angles_deg, CAMERA_RADIUS, FOV_DEG, RESOLUTION)
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        _save_rgba_series(colors_black, colors_white, angles_deg, OUTPUT_DIR, base_filename, ALPHA_THRESHOLD)
        print(f"标准多视图渲染已完成，共生成 {len(angles_deg)} 张图片")
        print(f"生成的文件: {base_filename}_000deg.png … {base_filename}_330deg.png")
        
        # 新增3D角度渲染（白色背景 + 透明背景）（已注释）
        # additional_angles_3d = _angles_additional()
        # print(f"\n开始渲染新增3D角度: {additional_angles_3d}")
        # 
        # # 渲染透明背景版本
        # colors_black_add, colors_white_add = _render_colors_3d(outputs['gaussian'][0], additional_angles_3d, CAMERA_RADIUS, FOV_DEG, RESOLUTION)
        # _save_rgba_series_3d(colors_black_add, colors_white_add, additional_angles_3d, ADDITIONAL_OUTPUT_DIR, base_filename, ALPHA_THRESHOLD)
        # print(f"新增3D角度渲染已完成，共生成 {len(additional_angles_3d)} 张图片（透明背景）")
        # 
        # # 渲染白色背景版本
        # colors_white_add_only = _render_colors_3d_white_bg(outputs['gaussian'][0], additional_angles_3d, CAMERA_RADIUS, FOV_DEG, RESOLUTION)
        # _save_rgb_series_3d_white_bg(colors_white_add_only, additional_angles_3d, ADDITIONAL_OUTPUT_DIR, f"{base_filename}_white_bg")
        # print(f"新增3D角度渲染已完成，共生成 {len(additional_angles_3d)} 张图片（白色背景）")
        # 
        # print(f"生成的文件: {base_filename}_120yaw_+030pitch.png, {base_filename}_000yaw_+030pitch.png, {base_filename}_240yaw_+030pitch.png, {base_filename}_300yaw_-020pitch.png, {base_filename}_180yaw_-020pitch.png, {base_filename}_060yaw_-020pitch.png, {base_filename}_330yaw_+045pitch.png, {base_filename}_240yaw_+045pitch.png")
        # print(f"白色背景文件: {base_filename}_white_bg_120yaw_+030pitch.png, {base_filename}_white_bg_000yaw_+030pitch.png, {base_filename}_white_bg_240yaw_+030pitch.png, {base_filename}_white_bg_300yaw_-020pitch.png, {base_filename}_white_bg_180yaw_-020pitch.png, {base_filename}_white_bg_060yaw_-020pitch.png, {base_filename}_white_bg_330yaw_+045pitch.png, {base_filename}_white_bg_240yaw_+045pitch.png")
        
        # 水平角度渲染（俯仰为0度）（已注释）
        # horizontal_angles = _angles_zero_pitch()
        # print(f"\n开始渲染水平角度: {horizontal_angles}")
        # colors_black_horz, colors_white_horz = _render_colors_zero_pitch(outputs['gaussian'][0], horizontal_angles, CAMERA_RADIUS, FOV_DEG, RESOLUTION)
        # _save_rgba_series_zero_pitch(colors_black_horz, colors_white_horz, horizontal_angles, ZERO_PITCH_OUTPUT_DIR, base_filename, ALPHA_THRESHOLD)
        # print(f"水平角度渲染已完成，共生成 {len(horizontal_angles)} 张图片")
        # print(f"生成的文件: {base_filename}_000yaw_+000pitch.png, {base_filename}_045yaw_+000pitch.png, {base_filename}_090yaw_+000pitch.png, {base_filename}_135yaw_+000pitch.png, {base_filename}_180yaw_+000pitch.png, {base_filename}_225yaw_+000pitch.png, {base_filename}_270yaw_+000pitch.png, {base_filename}_315yaw_+000pitch.png")
        
        # 俯仰-45度角度渲染（每30度12个视角）
        pitch_minus_45_angles = _angles_pitch_minus_45()
        print(f"\n开始渲染俯仰-45度角度: {len(pitch_minus_45_angles)} 个角度")
        colors_black_minus45, colors_white_minus45 = _render_colors_pitch_minus_45(outputs['gaussian'][0], pitch_minus_45_angles, CAMERA_RADIUS, FOV_DEG, RESOLUTION)
        _save_rgba_series_pitch_minus_45(colors_black_minus45, colors_white_minus45, pitch_minus_45_angles, PITCH_MINUS_45_OUTPUT_DIR, base_filename, ALPHA_THRESHOLD)
        print(f"俯仰-45度角度渲染已完成，共生成 {len(pitch_minus_45_angles)} 张图片")
        print(f"生成的文件: {base_filename}_000yaw_-045pitch.png … {base_filename}_330yaw_-045pitch.png")
        
        # 俯仰+45度角度渲染（每30度12个视角）
        pitch_plus_45_angles = _angles_pitch_plus_45()
        print(f"\n开始渲染俯仰+45度角度: {len(pitch_plus_45_angles)} 个角度")
        colors_black_plus45, colors_white_plus45 = _render_colors_pitch_plus_45(outputs['gaussian'][0], pitch_plus_45_angles, CAMERA_RADIUS, FOV_DEG, RESOLUTION)
        _save_rgba_series_pitch_plus_45(colors_black_plus45, colors_white_plus45, pitch_plus_45_angles, PITCH_PLUS_45_OUTPUT_DIR, base_filename, ALPHA_THRESHOLD)
        print(f"俯仰+45度角度渲染已完成，共生成 {len(pitch_plus_45_angles)} 张图片")
        print(f"生成的文件: {base_filename}_000yaw_+045pitch.png … {base_filename}_330yaw_+045pitch.png")
        
        # 导出 GLB 格式（已注释）
        # print(f"\n开始导出 GLB 格式...")
        # if 'mesh' in outputs and outputs['mesh']:
        #     glb_output_path = os.path.join(OUTPUT_DIR, f"{base_filename}.glb")
        #     glb = postprocessing_utils.to_glb(outputs['gaussian'][0], outputs['mesh'][0], simplify=0.8, texture_size=2048)
        #     glb.export(glb_output_path)
        #     print(f"已保存 GLB 格式: {glb_output_path}")
        #     print(f"GLB 导出完成")
        # else:
        #     print("警告: 未检测到 mesh 输出，跳过 GLB 导出")
        
    except Exception as e:
        print(f"渲染失败: {e}")
        return 1
    finally:
        if torch.cuda.is_available():
            del pipeline
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("GPU 内存已清理")

    return 0


if __name__ == "__main__":
    sys.exit(main())

# ===================== 可选功能（按需取消注释） =====================
# 保存 3D 模型数据（Gaussian 参数）
# gaussian_data = {
#     'xyz': outputs['gaussian'][0].get_xyz.detach().cpu().numpy(),
#     'opacity': outputs['gaussian'][0].get_opacity.detach().cpu().numpy(),
#     'scaling': outputs['gaussian'][0].get_scaling.detach().cpu().numpy(),
#     'rotation': outputs['gaussian'][0].get_rotation.detach().cpu().numpy(),
#     'features': outputs['gaussian'][0].get_features.detach().cpu().numpy(),
# }
# np.savez_compressed(os.path.join(OUTPUT_DIR, f"{base_filename}_3d_model.npz"), **gaussian_data)

# 保存相机参数
# camera_params = {
#     'resolution': RESOLUTION,
#     'fov': FOV_DEG,
#     'camera_distance': CAMERA_RADIUS,
#     'angles_degrees': angles_deg,
#     'extrinsics': [ext.detach().cpu().numpy() if hasattr(ext, 'detach') else np.array(ext) for ext in extrinsics],
#     'intrinsics': [intr.detach().cpu().numpy() if hasattr(intr, 'detach') else np.array(intr) for intr in intrinsics],
# }
# with open(os.path.join(OUTPUT_DIR, f"{base_filename}_camera_params.json"), 'w') as f:
#     json.dump(camera_params, f, indent=2, default=str)

# 计算 3D-2D 投影映射（按角度保存）
# mapping_data = {}
# for i, angle in enumerate(angles_deg):
#     alpha = _alpha_from_contrast(colors_black[i], colors_white[i], ALPHA_THRESHOLD)
#     current_extrinsics = extrinsics[i]
#     current_intrinsics = intrinsics[i]
#     if not isinstance(current_extrinsics, torch.Tensor):
#         current_extrinsics = torch.tensor(current_extrinsics, dtype=torch.float32)
#     if not isinstance(current_intrinsics, torch.Tensor):
#         current_intrinsics = torch.tensor(current_intrinsics, dtype=torch.float32)
#     points_3d = outputs['gaussian'][0].get_xyz.detach()
#     points_cam = torch.matmul(current_extrinsics[:3, :3], points_3d.T) + current_extrinsics[:3, 3:4]
#     points_2d = torch.matmul(current_intrinsics, points_cam)
#     points_2d = points_2d[:2] / points_2d[2:3]
#     points_pixel = points_2d.T.detach().cpu().numpy()
#     mapping_data[f'angle_{angle:03d}'] = {
#         'alpha_mask': alpha,
#         'points_2d': points_pixel,
#     }
# np.savez_compressed(os.path.join(OUTPUT_DIR, f"{base_filename}_3d_2d_mapping.npz"), **mapping_data)

# GLB 导出功能已集成到主函数中（仅导出GLB格式）

# 保存渲染视频（高开销，不默认启用）
# video = render_utils.render_video(outputs['gaussian'][0])['color']
# imageio.mimsave(os.path.join(OUTPUT_DIR, f"{base_filename}_gs.mp4"), video, fps=30)
