#!/usr/bin/env python3
"""
Flux LoRA+ Style Main Processing Pipeline
Integrated style reference image enhancement processing workflow
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import logging

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from style_reference_processor import StyleReferenceProcessor
from utils.image_utils import load_images, save_image, create_output_dirs

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FluxLoRAPipeline:
    """Flux LoRA+ Style main processing pipeline"""
    
    def __init__(self, config_path="configs/style_reference.yaml"):
        """Initialize pipeline"""
        self.config = self.load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = StyleReferenceProcessor(self.config, self.device)
        
        # Create output directories
        self.output_dirs = create_output_dirs()
        
    def load_config(self, config_path):
        """Load configuration file"""
        config_file = Path(__file__).parent.parent / config_path
        if not config_file.exists():
            logger.warning(f"Configuration file not found: {config_file}, using default config")
            return self.get_default_config()
            
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get_default_config(self):
        """Get default configuration"""
        return {
            'model': {
                'base_model': 'black-forest-labs/FLUX.1-dev',
                'lora_path': 'path/to/your/lora.safetensors'
            },
            'processing': {
                'strength': 0.2,
                'guidance_scale': 4.0,
                'num_inference_steps': 4,
                'style_weight': 0.5,
                'output_size': 1024
            },
            'paths': {
                'input_dir': 'input/your_images',
                'style_dir': 'input/your_images/ori_svg',
                'output_dir': 'output'
            }
        }
    
    def process_images(self):
        """Main image processing workflow"""
        logger.info("Starting Flux LoRA+ Style processing workflow")
        
        # 1. Load input images
        input_dir = Path(__file__).parent.parent / self.config['paths']['input_dir']
        images = load_images(input_dir, exclude_dirs=['ori_svg'])
        
        if not images:
            error_msg = f"❌ ERROR: No image files found in {input_dir}"
            logger.error(error_msg)
            print(f"\n{error_msg}")
            print("📁 Please place your images in the following directory:")
            print(f"   {input_dir}")
            print("📋 Supported formats: PNG, JPG, JPEG, BMP, TIFF")
            return
        
        logger.info(f"Found {len(images)} images to process")
        print(f"✅ Found {len(images)} images to process")
        
        # 2. Load style reference
        style_dir = Path(__file__).parent.parent / self.config['paths']['style_dir']
        style_images = load_images(style_dir)
        
        if not style_images:
            warning_msg = f"⚠️  WARNING: No style reference images found in {style_dir}"
            logger.warning(warning_msg)
            print(f"\n{warning_msg}")
            print("📁 Please place your original SVG reference images in:")
            print(f"   {style_dir}")
            print("📋 This is required for optimal style learning")
            print("🔄 Processing will continue without style reference...")
            style_reference = None
        else:
            style_reference = style_images[0]  # Use first style reference
            logger.info(f"Using style reference: {style_reference}")
            print(f"✅ Using style reference: {style_reference.name}")
        
        # 3. Process each image
        output_dir = Path(__file__).parent.parent / self.config['paths']['output_dir']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🚀 Starting image processing...")
        successful_count = 0
        
        for i, image_path in enumerate(tqdm(images, desc="Processing images")):
            try:
                # Load image
                image = Image.open(image_path).convert('RGB')
                
                # Process image
                processed_image = self.processor.process_image(
                    image, 
                    style_reference=style_reference
                )
                
                # Save result
                output_path = output_dir / f"processed_{image_path.stem}.png"
                save_image(processed_image, output_path)
                
                logger.info(f"Processing completed: {image_path.name} -> {output_path.name}")
                successful_count += 1
                
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
                print(f"❌ Failed to process: {image_path.name}")
                continue
        
        # 4. Print summary
        print(f"\n📊 Processing Summary:")
        print(f"   Total images: {len(images)}")
        print(f"   Successfully processed: {successful_count}")
        print(f"   Success rate: {successful_count/len(images)*100:.1f}%")
        print(f"   Output directory: {output_dir}")
        
        if successful_count > 0:
            print(f"\n✅ Processing completed successfully!")
        else:
            print(f"\n❌ No images were processed successfully")
        
        logger.info("All images processed successfully!")
        
        # 5. Generate processing report
        self.generate_report(images, output_dir)
    
    def process_image(self, image_path, style_reference_path=None):
        """Process single image"""
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Load style reference
        style_reference = None
        if style_reference_path:
            style_reference = Image.open(style_reference_path).convert('RGB')
        
        # Process image
        processed_image = self.processor.process_image(image, style_reference)
        
        return processed_image
    
    def generate_report(self, input_images, output_dir):
        """Generate processing report"""
        logger.info("Generating processing report...")
        
        # Statistics
        total_images = len(input_images)
        processed_images = len(list(output_dir.glob("processed_*.png")))
        
        report = {
            "Processing Statistics": {
                "Total Images": total_images,
                "Successfully Processed": processed_images,
                "Success Rate": f"{processed_images/total_images*100:.1f}%"
            },
            "Configuration Parameters": self.config['processing'],
            "Output Directory": str(output_dir)
        }
        
        # Save report
        report_path = Path(__file__).parent.parent / "processing_report.yaml"
        with open(report_path, 'w', encoding='utf-8') as f:
            yaml.dump(report, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Processing report saved: {report_path}")

def main():
    """Main function"""
    print("🎨 Flux LoRA+ Style Image Enhancement Tool")
    print("=" * 50)
    
    try:
        # Create pipeline
        print("🔧 Initializing pipeline...")
        pipeline = FluxLoRAPipeline()
        
        # Process images
        pipeline.process_images()
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        print(f"\n❌ Fatal error: {e}")
        print("💡 Please check your configuration and try again")
        sys.exit(1)

if __name__ == "__main__":
    main()
