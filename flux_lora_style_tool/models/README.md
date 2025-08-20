# Models Directory

This directory contains the LoRA weights and model files for the Flux LoRA+ Style tool.

## 📁 Directory Structure

```
models/
├── README.md                    # This file
├── lora_weights.safetensors     # LoRA fine-tuned weights (to be uploaded)
└── config.json                  # LoRA configuration (if needed)
```

## 🔧 Model Files

### LoRA Weights (`lora_weights.safetensors`)
- **Purpose**: Fine-tuned LoRA weights for style enhancement
- **Size**: ~100-500MB (depending on model size)
- **Format**: SafeTensors format for security
- **Download**: Available in GitHub releases

### Configuration (`config.json`)
- **Purpose**: LoRA model configuration parameters
- **Content**: Model architecture and training parameters
- **Required**: For proper model loading

## 📥 Download Instructions

1. **From GitHub Releases**:
   ```bash
   # Download the latest release
   wget https://github.com/JoeySun0223/GenAI/releases/latest/download/lora_weights.safetensors
   mv lora_weights.safetensors models/
   ```

2. **Manual Download**:
   - Go to [Releases page](https://github.com/JoeySun0223/GenAI/releases)
   - Download `lora_weights.safetensors`
   - Place in `models/` directory

## ⚠️ Important Notes

- **Security**: Models are in SafeTensors format for enhanced security
- **Version**: Ensure model version matches tool version
- **Path**: Update `configs/style_reference.yaml` with correct model path
- **Size**: Models may be large, ensure sufficient storage space

## 🔄 Model Updates

When new model versions are released:
1. Download the new model file
2. Replace the old file in `models/`
3. Update the configuration if needed
4. Test with a small batch first
