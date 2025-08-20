# 🎨 Flux LoRA+ Style Image Enhancement Tool

An intelligent image enhancement tool based on Flux.1 + LoRA + style reference, capable of automatically correcting geometric errors and applying specific styles.

## 📦 Version & Download

**Current Version**: v1.0.0  
**Last Updated**: December 2024  
**Flux Model Version**: black-forest-labs/FLUX.1-dev

### 🚀 Quick Download

```bash
# Clone the repository
git clone https://github.com/JoeySun0223/GenAI.git
cd GenAI/flux_lora_style_tool

# Or download directly
wget https://github.com/JoeySun0223/GenAI/archive/refs/heads/main.zip
unzip main.zip
cd GenAI-main/flux_lora_style_tool
```

### 📋 Prerequisites

- **Python**: 3.8+
- **CUDA**: 11.8+ (for GPU acceleration)
- **RAM**: 8GB+ (16GB recommended)
- **Storage**: 10GB+ for models

### 🔧 Model Downloads

**LoRA Weights**: Available in `models/` directory
- Download from: [LoRA Weights Release](https://github.com/JoeySun0223/GenAI/releases)
- Place in: `flux_lora_style_tool/models/lora_weights.safetensors`

**Flux Model**: Automatically downloaded on first run
- Model: `black-forest-labs/FLUX.1-dev`
- Size: ~8GB
- Location: `~/.cache/huggingface/hub/`

## ✨ Key Features

### 🎯 Core Capabilities
- **Geometric Correction**: Automatically corrects circles, lines, angles, and other geometric shapes
- **Style Learning**: Automatically learns and applies aesthetic styles from reference images
- **Quality Enhancement**: Enhances image details, clarity, and overall quality
- **Batch Processing**: Supports automated processing of large batches of images

### 🔧 Technical Features
- **Automated Style Detection**: Automatically finds style references in the `ori_svg` folder
- **Intelligent Parameter Optimization**: Best configurations from systematic experiments
- **Fast Processing**: 4-step processing, ~1.2 seconds per image
- **High-Quality Output**: 1024x1024 high-resolution output

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data Structure
```
input/
└── your_images/
    ├── image1.png          # Images to process
    ├── image2.png
    └── ori_svg/            # ⚠️ REQUIRED: Style reference directory
        └── style_reference.png  # Style reference image
```

### 3. Important: Style Reference Setup
**⚠️ CRITICAL**: You must provide style reference images in the `ori_svg` folder:

- **Path**: `input/your_images/ori_svg/`
- **Format**: PNG, JPG, JPEG, BMP, TIFF
- **Content**: Original SVG reference images that define the target style
- **Quantity**: At least one reference image (first one will be used)

The tool will automatically detect and use these reference images for style learning. If no reference images are found, the tool will show a warning and continue without style reference.

### 4. Run Processing
```bash
# Process with style reference
python scripts/run_pipeline_with_style.py

# View results
python scripts/view_comparison.py
```

## 📁 Project Structure

```
flux_lora_style_tool/
├── input/                          # Input image directory
│   └── your_images/
│       ├── *.png                   # Images to process
│       └── ori_svg/                # ⚠️ REQUIRED: Style reference images
│           └── style_reference.png # Original SVG reference for style
├── output/                         # Processing results output
├── configs/                        # Configuration files
├── parameter_exp/                  # Experiment data and records
├── scripts/                        # Core scripts
├── requirements.txt                # Dependencies list
└── README.md                       # Project documentation
```

## 🎛️ Optimal Configuration

### Recommended Parameter Settings
| Processing Method | Strength | Guidance | Steps | Style Weight | Use Case |
|-------------------|----------|----------|-------|--------------|----------|
| **Style Reference Processing** | 0.2 | 4.0 | 4 | 0.5 | Product display, formal projects |
| **No Style Reference Processing** | 0.4 | 3.0 | 4 | - | Batch processing, quick preview |

## 📊 Processing Effects

### Input → Output Comparison
- **Original Image**: May have geometric imperfections, unclear lines, etc.
- **Processed**: Regular geometric shapes, clear lines, unified style, enhanced quality

### Supported Processing Types
- ✅ Geometric shape correction (circles, lines, angles)
- ✅ Edge sharpening
- ✅ Detail enhancement
- ✅ Style unification
- ✅ Color optimization

## 🔧 Core Scripts

### Main Scripts
- **`run_pipeline_with_style.py`**: Main processing pipeline with style reference integration
- **`style_reference_processor.py`**: Core style reference processing
- **`create_style_comparison.py`**: Generate comparison analysis images
- **`view_comparison.py`**: View processing results

### Configuration Files
- **`configs/style_reference.yaml`**: Style reference processing configuration
- **`configs/enhanced_lora.yaml`**: LoRA processing configuration

## 📈 Performance Metrics

- **Processing Speed**: ~1.2 seconds per image (4-step processing)
- **Success Rate**: 100%
- **Output Quality**: 1024x1024 high resolution
- **Supported Formats**: PNG, JPG, JPEG, BMP, TIFF

## 🎨 Use Cases

### Applicable Fields
- **Product Design**: Unify product image style and quality
- **UI/UX Design**: Optimize interface element visual effects
- **Illustration Creation**: Batch process illustration images
- **Design Assets**: Improve design asset quality and consistency

### Typical Applications
1. **Automotive Design**: Correct geometric shapes, unify design style
2. **Product Display**: Enhance image quality, maintain brand consistency
3. **Illustration Works**: Batch optimize illustration quality and style
4. **UI Components**: Unify interface element visual effects

## 💡 Usage Tips

### Style Reference Preparation
- **Required**: Place original SVG reference images in `input/your_images/ori_svg/`
- Choose clear, representative style images
- Ensure style reference is similar to target image style
- Use the same style reference for the same batch
- **Warning**: If no reference images found, processing will continue without style learning

### Parameter Tuning
- **Style too strong**: Reduce Style Weight (0.5 → 0.4)
- **Style too weak**: Increase Style Weight (0.5 → 0.6)
- **Too much modification**: Reduce Strength (0.2 → 0.15)
- **Too little modification**: Increase Strength (0.2 → 0.25)

## ⚠️ Important Notes

### Input Requirements
1. **Images to Process**: Place target images in `input/your_images/`
2. **Style References**: **MUST** place original SVG reference images in `input/your_images/ori_svg/`
3. **File Formats**: Supported formats are PNG, JPG, JPEG, BMP, TIFF

### Error Handling
- If no images found in input directory: Tool will show error and exit
- If no style references found: Tool will show warning and continue without style
- If processing fails: Individual image errors are logged, processing continues

## 🔍 Technical Architecture

### Core Technology Stack
- **Base Model**: black-forest-labs/FLUX.1-dev
- **LoRA Fine-tuning**: Custom fine-tuned models
- **Style Learning**: CLIP image embedding + style weighting
- **Image Processing**: PIL, NumPy, Matplotlib

### Processing Pipeline
1. **Image Loading**: Automatically detect and load input images
2. **Style Extraction**: Extract style features from `ori_svg` folder
3. **LoRA Processing**: Apply LoRA fine-tuned models for image enhancement
4. **Style Application**: Apply learned styles to target images
5. **Result Output**: Generate high-quality processing results

## 🤝 Contributing

Welcome to submit Issues and Pull Requests to improve this tool!

## 📄 License

This project is licensed under the MIT License.

---

**Last Updated**: December 2024  
**Version**: v1.0  
**Optimal Configuration**: Strength=0.2, Guidance=4.0, Steps=4, Style Weight=0.5
