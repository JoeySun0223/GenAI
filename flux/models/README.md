# Models Directory

This directory contains the LoRA weights and model files for the Flux LoRA+ Style tool.

## 📁 Directory Structure

```
models/
├── README.md                    # This file
├── lora_weights.safetensors     # ✅ LoRA fine-tuned weights (INCLUDED)
└── config.json                  # LoRA configuration (if needed)
```

## 🔧 Model Files

### LoRA Weights (`lora_weights.safetensors`) ✅ INCLUDED
- **Purpose**: Fine-tuned LoRA weights for style enhancement
- **Size**: ~4.1MB
- **Format**: SafeTensors format for security
- **Status**: ✅ Available in the repository
- **Source**: Trained on Flux.1 model with custom dataset

### Configuration (`config.json`)
- **Purpose**: LoRA model configuration parameters
- **Content**: Model architecture and training parameters
- **Required**: For proper model loading

## 📥 Download Instructions

The LoRA weights are **already included** in this repository:

```bash
# Clone the repository
git clone https://github.com/MengnanJiangNan/GenAI_Zhaolin.git
cd GenAI_Zhaolin/flux_lora_style_tool

# The LoRA weights are already in models/lora_weights.safetensors
ls -la models/
```

## ⚠️ Important Notes

- **Security**: Models are in SafeTensors format for enhanced security
- **Version**: Model version matches tool version v1.0.0
- **Path**: Configured in `configs/style_reference.yaml`
- **Size**: ~4.1MB (relatively small and efficient)

## 🔄 Model Updates

When new model versions are released:
1. Download the new model file
2. Replace the old file in `models/`
3. Update the configuration if needed
4. Test with a small batch first

## 🎯 Usage

The LoRA weights are automatically loaded by the tool:

```python
# The tool will automatically load:
# models/lora_weights.safetensors
# when you run:
python scripts/run_pipeline_with_style.py
```
