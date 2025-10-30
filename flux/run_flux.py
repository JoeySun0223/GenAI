#!/usr/bin/env python3
"""
Flux LoRA处理脚本 - 生成flux_repair_开头的文件
基于参考图片和校正样本的颜色偏差来指导所有图片的处理
支持动态参数调整和批量处理（不启用颜色矫正）
"""

# 在导入任何可能使用OpenBLAS的库之前设置线程数限制
import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import logging
import time
import uuid

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FluxRepairProcessor:
    """集成动态颜色校正的Flux处理器（不启用颜色矫正）"""
    
    def __init__(self, config_path="configs/style_reference.yaml"):
        """初始化处理器"""
        self.config = self.load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        
        # 批次信息
        self.batch_id = self._generate_batch_id()
        
        self.setup_pipeline()
        
    def _generate_batch_id(self) -> str:
        """生成批次ID"""
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        return f"batch_{timestamp}_{unique_id}"
    
    def load_config(self, config_path):
        """加载配置文件"""
        config_file = Path(__file__).parent / config_path
        if not config_file.exists():
            logger.error(f"配置文件不存在: {config_file}")
            return None
            
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def setup_pipeline(self):
        """设置Flux + LoRA pipeline"""
        if not self.config:
            logger.error("配置加载失败")
            return
            
        logger.info("正在加载Flux LoRA...")
        
        try:
            from diffusers import FluxImg2ImgPipeline
            
            # 加载基础模型
            base_model = self.config['model']['base_model']
            logger.info(f"加载基础模型: {base_model}")
            
            self.pipeline = FluxImg2ImgPipeline.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                use_safetensors=True
            )
            
            # 加载LoRA
            lora_path = Path(__file__).parent / self.config['model']['lora_path']
            if lora_path.exists():
                logger.info(f"加载LoRA: {lora_path}")
                self.pipeline.load_lora_weights(str(lora_path))
                logger.info("✅ LoRA加载成功")
            else:
                logger.warning(f"LoRA文件不存在: {lora_path}")
            
            # 移动到设备
            self.pipeline = self.pipeline.to(self.device)
            
            # 启用内存优化
            if self.config.get('advanced', {}).get('enable_memory_efficient', True):
                self.pipeline.enable_model_cpu_offload()
            if self.config.get('advanced', {}).get('enable_attention_slicing', True):
                self.pipeline.enable_attention_slicing()
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("✅ Pipeline设置完成")
            
        except Exception as e:
            logger.error(f"Pipeline设置失败: {e}")
            raise
    
    def process_image(self, input_path, output_path):
        """处理单张图片"""
        try:
            # 加载输入图片
            input_image = Image.open(input_path).convert("RGB")
            
            # 调整图片大小
            output_size = self.config['processing']['output_size']
            input_image = input_image.resize((output_size, output_size), Image.Resampling.LANCZOS)
            
            logger.info(f"🎨 开始处理图片: {input_path.name}")
            
            # 1. 跳过颜色校正（不启用颜色矫正）
            color_correction_params = None
            
            # 2. 构建增强的prompt（缩短以避免token超限）
            base_prompt = "v3ct0r style, simple flat vector art, isolated on white bg, high quality, crisp edges, clean design, professional vector illustration, smooth curves, perfect geometry, minimalist style, modern design, sharp details, clean lines, geometric precision, vector graphics, flat design, clean composition, professional artwork"
            
            # 增强的负面prompt
            negative_prompt = "no extra objects, no layout changes, no new patterns, no gradients, no heavy re-stylization, no color shifts, no over-smoothing, no artifacts, no blur, no noise, no pixelation, no jagged edges, no rough textures, no complex backgrounds, no shadows, no 3d effects, no realistic rendering, no photographic elements, no hand-drawn elements, no sketchy lines, no messy details, no watermarks, no text, no logos, no signatures"
            
            # 3. 获取处理参数
            processing_config = self.config['processing']
            
            # 根据颜色校正参数动态调整处理参数（跳过，因为没有颜色校正）
            strength = processing_config['strength']
            guidance_scale = processing_config['guidance_scale']
            
            logger.info(f"🎯 最终处理参数: strength={strength:.2f}, guidance_scale={guidance_scale:.1f}")
            
            # 4. 生成图片
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(self.device.type == "cuda")):
                result = self.pipeline(
                    prompt=base_prompt,
                    negative_prompt=negative_prompt,
                    image=input_image,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=processing_config['num_inference_steps'],
                    height=output_size,
                    width=output_size
                )
            
            # 保存结果
            output_image = result.images[0]
            
            # 调整输出图片大小为512x512
            final_output_size = 512
            output_image = output_image.resize((final_output_size, final_output_size), Image.Resampling.LANCZOS)
            
            output_image.save(output_path)
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 5. 记录处理信息
            self._save_processing_info(input_path, output_path, color_correction_params, 
                                     strength, guidance_scale, base_prompt)
            
            logger.info(f"✅ 处理完成: {input_path.name} -> {output_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 处理失败 {input_path.name}: {e}")
            return False
    
    def _save_processing_info(self, input_path, output_path, color_correction_params, 
                            strength, guidance_scale, prompt):
        """保存处理信息"""
        try:
            info_dir = Path(output_path).parent / "processing_info"
            info_dir.mkdir(exist_ok=True)
            
            info_file = info_dir / f"{Path(input_path).stem}_info.json"
            
            info = {
                'batch_id': self.batch_id,
                'input_file': str(input_path),
                'output_file': str(output_path),
                'processing_parameters': {
                    'strength': strength,
                    'guidance_scale': guidance_scale,
                    'num_inference_steps': self.config['processing']['num_inference_steps'],
                    'processing_size': self.config['processing']['output_size'],
                    'final_output_size': 512,
                    'prompt': prompt,
                    'negative_prompt': "no extra objects, no layout changes, no new patterns, no gradients, no heavy re-stylization, no color shifts, no over-smoothing, no artifacts, no blur, no noise, no pixelation, no jagged edges, no rough textures, no complex backgrounds, no shadows, no 3d effects, no realistic rendering, no photographic elements, no hand-drawn elements, no sketchy lines, no messy details, no watermarks, no text, no logos, no signatures"
                },
                'color_correction': {
                    'applied': color_correction_params is not None,
                    'parameters': color_correction_params if color_correction_params else {},
                    'correction_strength': color_correction_params.get('rgb_correction', {}).get('intensity', 0) if color_correction_params else 0
                },
                'timestamp': time.time()
            }
            
            import json
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.warning(f"⚠️ 处理信息保存失败: {e}")
    
    def process_batch(self, input_dir, output_dir):
        """批量处理图片"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # 创建输出目录
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取输入图片
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        input_images = []
        
        for ext in image_extensions:
            input_images.extend(input_path.glob(f"*{ext}"))
        
        # 排除ori_svg目录
        input_images = [img for img in input_images if img.parent == input_path]
        
        if not input_images:
            logger.error(f"在 {input_dir} 中没有找到图片文件")
            return
        
        logger.info(f"找到 {len(input_images)} 张图片需要处理")
        logger.info(f"📊 批次ID: {self.batch_id}")
        
        # 批量处理
        success_count = 0
        for input_image in tqdm(input_images, desc="处理图片"):
            output_filename = f"flux_repair_{input_image.stem}.png"
            output_file = output_path / output_filename
            
            if self.process_image(input_image, output_file):
                success_count += 1
        
        # 总结
        logger.info("=" * 60)
        logger.info(f"🎉 Flux修复处理完成!")
        logger.info(f"📊 统计信息:")
        logger.info(f"   - 批次ID: {self.batch_id}")
        logger.info(f"   - 总图片数: {len(input_images)}")
        logger.info(f"   - 成功处理: {success_count}")
        logger.info(f"   - 失败数量: {len(input_images) - success_count}")
        logger.info(f"   - 成功率: {success_count/len(input_images)*100:.1f}%")
        logger.info(f"📁 输出目录: {output_path}")
        logger.info(f"📁 处理信息: {output_path}/processing_info/")
        logger.info("=" * 60)

def main():
    """主函数"""
    logger.info("🚀 开始使用Flux LoRA处理bluecar图片...")
    
    # 路径配置
    base_dir = Path(__file__).parent
    input_dir = base_dir / "input" / "your_images" / "bluecar_"
    output_dir = base_dir / "output" / "bluecar_flux_repair"
    config_path = "configs/style_reference.yaml"
    
    # 检查输入目录
    if not input_dir.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        return
    
    # 初始化处理器
    try:
        processor = FluxRepairProcessor(config_path)
        logger.info("✅ Flux修复处理器初始化成功")
    except Exception as e:
        logger.error(f"❌ Flux修复处理器初始化失败: {e}")
        return
    
    # 批量处理
    processor.process_batch(input_dir, output_dir)

if __name__ == "__main__":
    main()
