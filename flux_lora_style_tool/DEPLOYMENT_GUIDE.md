# 🚀 Deployment Guide

This guide explains how to deploy the Flux LoRA+ Style tool to GitHub.

## 📋 Prerequisites

1. **GitHub Account**: Ensure you have access to [JoeySun0223/GenAI](https://github.com/JoeySun0223/GenAI)
2. **Git**: Install Git on your system
3. **GitHub CLI** (optional): For easier GitHub operations

## 🔧 Manual Upload Steps

### Method 1: Using GitHub Web Interface

1. **Navigate to Repository**:
   - Go to https://github.com/JoeySun0223/GenAI
   - Click "Add file" → "Upload files"

2. **Upload Directory**:
   - Drag and drop the entire `flux_lora_style_tool` folder
   - Add commit message: "Add Flux LoRA+ Style Tool v1.0.0"
   - Click "Commit changes"

### Method 2: Using Git Commands

```bash
# Clone the existing repository
git clone https://github.com/JoeySun0223/GenAI.git
cd GenAI

# Copy the flux_lora_style_tool directory
cp -r /path/to/flux_lora_style_tool ./

# Add and commit
git add flux_lora_style_tool/
git commit -m "Add Flux LoRA+ Style Tool v1.0.0"

# Push to GitHub
git push origin main
```

### Method 3: Using GitHub CLI

```bash
# Install GitHub CLI if not installed
# macOS: brew install gh
# Ubuntu: sudo apt install gh

# Login to GitHub
gh auth login

# Clone and add files
git clone https://github.com/JoeySun0223/GenAI.git
cd GenAI
cp -r /path/to/flux_lora_style_tool ./
git add flux_lora_style_tool/
git commit -m "Add Flux LoRA+ Style Tool v1.0.0"
git push origin main
```

## 📦 Creating a Release

After uploading the code, create a release for the LoRA weights:

1. **Go to Releases**:
   - Navigate to https://github.com/JoeySun0223/GenAI/releases
   - Click "Create a new release"

2. **Release Information**:
   - **Tag**: `v1.0.0`
   - **Title**: `Flux LoRA+ Style Tool v1.0.0`
   - **Description**: Include installation and usage instructions

3. **Upload Assets**:
   - Upload `lora_weights.safetensors` as a release asset
   - Users can download it directly from the release page

## 🔗 Update README Links

After deployment, update the main repository README to include:

```markdown
## 🎨 Flux LoRA+ Style Tool

An intelligent image enhancement tool based on Flux.1 + LoRA + style reference.

**Quick Start**:
```bash
cd flux_lora_style_tool
pip install -r requirements.txt
python scripts/run_pipeline_with_style.py
```

**Download LoRA Weights**: [Release v1.0.0](https://github.com/JoeySun0223/GenAI/releases/tag/v1.0.0)
```

## ✅ Verification

After deployment, verify:

1. **Code Upload**: Check that all files are present in the repository
2. **Release**: Ensure LoRA weights are available for download
3. **Documentation**: README links work correctly
4. **Installation**: Test the installation process

## 🐛 Troubleshooting

### Permission Denied (403 Error)
- Ensure you have write access to the repository
- Check your GitHub authentication
- Use personal access token if needed

### Large File Upload
- LoRA weights may be large (>100MB)
- Use Git LFS if needed: `git lfs track "*.safetensors"`
- Or upload via GitHub releases

### Branch Protection
- If main branch is protected, create a pull request
- Request review from repository maintainers
