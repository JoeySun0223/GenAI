#!/usr/bin/env python3
"""
Style Reference Processing Core Module
Implements Flux LoRA+ Style image enhancement functionality
"""

import torch
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPImageProcessor
import logging

logger = logging.getLogger(__name__)

class StyleReferenceProcessor:
    """Style reference processor"""
    
    def __init__(self, config, device):
        """Initialize processor"""
        self.config = config
        self.device = device
        self.pipeline = None
        self.clip_processor = None
        
        # Initialize models
        self._load_models()
    
    def _load_models(self):
        """Load models"""
        try:
            logger.info("Loading Flux model...")
            
            # Load base model
            model_id = self.config['model']['base_model']
            self.pipeline = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            
            # Set scheduler
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            # Load LoRA weights (if exists)
            lora_path = self.config['model'].get('lora_path')
            if lora_path and lora_path != 'path/to/your/lora.safetensors':
                logger.info(f"Loading LoRA weights: {lora_path}")
                self.pipeline.load_lora_weights(lora_path)
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Load CLIP processor for style extraction
            self.clip_processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            
            logger.info("Model loading completed")
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    def extract_style_features(self, style_image):
        """Extract style features"""
        if style_image is None:
            return None
        
        try:
            # Preprocess style image
            style_inputs = self.clip_processor(
                images=style_image, 
                return_tensors="pt"
            )
            
            # Extract features
            with torch.no_grad():
                style_features = self.pipeline.encode_prompt(
                    style_inputs.pixel_values.to(self.device)
                )
            
            return style_features
            
        except Exception as e:
            logger.error(f"Style feature extraction failed: {e}")
            return None
    
    def process_image(self, image, style_reference=None):
        """Process image"""
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Extract style features
            style_features = None
            if style_reference is not None:
                style_features = self.extract_style_features(style_reference)
            
            # Generate processing parameters
            generation_params = self._get_generation_params(style_features)
            
            # Execute image processing
            result = self.pipeline(
                image=processed_image,
                **generation_params
            )
            
            # Post-process result
            processed_result = self._postprocess_result(result)
            
            return processed_result
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise
    
    def _preprocess_image(self, image):
        """Preprocess input image"""
        # Resize image
        target_size = self.config['processing']['output_size']
        if image.size != (target_size, target_size):
            image = image.resize((target_size, target_size), Image.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Normalize to [0, 1]
        if image_array.dtype == np.uint8:
            image_array = image_array.astype(np.float32) / 255.0
        
        return image_array
    
    def _get_generation_params(self, style_features=None):
        """Get generation parameters"""
        params = {
            'strength': self.config['processing']['strength'],
            'guidance_scale': self.config['processing']['guidance_scale'],
            'num_inference_steps': self.config['processing']['num_inference_steps'],
            'output_type': 'pil'
        }
        
        # Add style weight if style features exist
        if style_features is not None:
            style_weight = self.config['processing']['style_weight']
            params['style_features'] = style_features
            params['style_weight'] = style_weight
        
        return params
    
    def _postprocess_result(self, result):
        """Post-process result"""
        # Get processed image
        if hasattr(result, 'images') and len(result.images) > 0:
            processed_image = result.images[0]
        else:
            processed_image = result
        
        # Ensure correct output format
        if isinstance(processed_image, np.ndarray):
            # Convert to PIL image
            if processed_image.dtype == np.float32:
                processed_image = (processed_image * 255).astype(np.uint8)
            processed_image = Image.fromarray(processed_image)
        
        return processed_image
    
    def batch_process(self, images, style_reference=None):
        """Batch process images"""
        results = []
        
        for i, image in enumerate(images):
            try:
                logger.info(f"Processing image {i+1}/{len(images)}")
                result = self.process_image(image, style_reference)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process image {i+1}: {e}")
                results.append(None)
        
        return results
    
    def get_processing_info(self):
        """Get processing information"""
        return {
            'model': self.config['model']['base_model'],
            'device': str(self.device),
            'parameters': self.config['processing'],
            'lora_loaded': self.config['model'].get('lora_path') != 'path/to/your/lora.safetensors'
        }
