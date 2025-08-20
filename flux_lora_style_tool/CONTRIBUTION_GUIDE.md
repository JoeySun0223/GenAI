# 🚀 Contribution Guide for Flux LoRA+ Style Tool

This guide will help you contribute the Flux LoRA+ Style tool to the JoeySun0223/GenAI repository.

## 📋 Prerequisites

- GitHub account: [@MengnanJiangNan](https://github.com/MengnanJiangNan)
- Git installed on your system
- Access to the target repository: [JoeySun0223/GenAI](https://github.com/JoeySun0223/GenAI)

## 🔄 Step-by-Step Process

### Step 1: Fork the Repository

1. **Visit the target repository**:
   - Go to: https://github.com/JoeySun0223/GenAI
   - Click the **"Fork"** button in the top-right corner
   - Select your account (MengnanJiangNan)
   - Wait for the fork to complete

2. **Verify the fork**:
   - You should now have: https://github.com/MengnanJiangNan/GenAI
   - This is your copy of the repository

### Step 2: Clone Your Fork

```bash
# Clone your forked repository
git clone https://github.com/MengnanJiangNan/GenAI.git
cd GenAI

# Add the original repository as upstream
git remote add upstream https://github.com/JoeySun0223/GenAI.git
```

### Step 3: Create a Feature Branch

```bash
# Create and switch to a new branch
git checkout -b add-flux-lora-tool

# Verify you're on the new branch
git branch
```

### Step 4: Add Your Code

```bash
# Copy the flux_lora_style_tool directory
cp -r /path/to/flux_lora_style_tool ./

# Add all files
git add flux_lora_style_tool/

# Commit with a descriptive message
git commit -m "Add Flux LoRA+ Style Image Enhancement Tool v1.0.0

Features:
- Intelligent image enhancement with Flux.1 + LoRA
- Style reference processing capabilities
- Comprehensive error handling and user feedback
- Complete documentation and usage examples
- Optimized for batch processing

Technical details:
- Based on black-forest-labs/FLUX.1-dev
- Supports multiple image formats
- Configurable processing parameters
- User-friendly CLI interface"
```

### Step 5: Push to Your Fork

```bash
# Push the branch to your fork
git push origin add-flux-lora-tool
```

### Step 6: Create Pull Request

1. **Go to your fork**: https://github.com/MengnanJiangNan/GenAI
2. **Click "Compare & pull request"** (should appear automatically)
3. **Fill in the PR details**:

**Title**: `Add Flux LoRA+ Style Image Enhancement Tool v1.0.0`

**Description**:
```markdown
## 🎨 Flux LoRA+ Style Image Enhancement Tool

This PR adds a comprehensive image enhancement tool based on Flux.1 + LoRA + style reference processing.

### ✨ Key Features

- **Geometric Correction**: Automatically corrects circles, lines, angles, and other geometric shapes
- **Style Learning**: Automatically learns and applies aesthetic styles from reference images
- **Quality Enhancement**: Enhances image details, clarity, and overall quality
- **Batch Processing**: Supports automated processing of large batches of images
- **User-Friendly**: Comprehensive error handling and progress feedback

### 🔧 Technical Details

- **Base Model**: black-forest-labs/FLUX.1-dev
- **Processing Speed**: ~1.2 seconds per image (4-step processing)
- **Output Quality**: 1024x1024 high resolution
- **Supported Formats**: PNG, JPG, JPEG, BMP, TIFF

### 📁 Project Structure

```
flux_lora_style_tool/
├── README.md                    # Comprehensive documentation
├── requirements.txt             # Dependencies
├── configs/                     # Configuration files
├── scripts/                     # Core processing scripts
├── models/                      # LoRA weights directory
├── input/                       # Input directory structure
├── output/                      # Output directory
└── parameter_exp/               # Experiment documentation
```

### 🚀 Quick Start

```bash
cd flux_lora_style_tool
pip install -r requirements.txt
python scripts/run_pipeline_with_style.py
```

### 📋 Requirements

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ RAM (16GB recommended)
- 10GB+ storage for models

### 🔗 Related

- **LoRA Weights**: Will be available in GitHub releases
- **Documentation**: Complete usage guide included
- **Examples**: Sample configurations and test cases

### ✅ Testing

- [x] Code review completed
- [x] Documentation updated
- [x] Error handling implemented
- [x] User feedback improved
- [x] Performance optimized
```

4. **Click "Create pull request"**

## 🔧 Alternative: Web Interface Upload

If you prefer using the web interface:

1. **Go to your fork**: https://github.com/MengnanJiangNan/GenAI
2. **Click "Add file"** → **"Upload files"**
3. **Drag and drop** the entire `flux_lora_style_tool` folder
4. **Add commit message**: "Add Flux LoRA+ Style Tool v1.0.0"
5. **Click "Commit changes"**
6. **Create Pull Request** as described above

## 📦 Creating a Release

After the PR is merged, create a release for the LoRA weights:

1. **Go to Releases**: https://github.com/JoeySun0223/GenAI/releases
2. **Click "Create a new release"**
3. **Tag**: `v1.0.0`
4. **Title**: `Flux LoRA+ Style Tool v1.0.0`
5. **Upload Assets**: Add `lora_weights.safetensors`
6. **Publish release**

## 🎯 Success Criteria

- [ ] Fork created successfully
- [ ] Code uploaded to your fork
- [ ] Pull request created
- [ ] PR description is comprehensive
- [ ] Code review completed
- [ ] PR merged to main repository
- [ ] Release created for LoRA weights

## 🐛 Troubleshooting

### Permission Issues
- Ensure you have write access to your fork
- Check GitHub authentication
- Use personal access token if needed

### Merge Conflicts
- Sync your fork with upstream: `git pull upstream main`
- Resolve conflicts locally
- Push updated branch

### Large Files
- LoRA weights should be uploaded via releases
- Use `.gitignore` to exclude large files
- Document download instructions
