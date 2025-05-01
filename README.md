# 3D Model Renderer with SVG Converter

这个项目包含两个主要功能：
1. 3D模型渲染器 (3D_Gen.py)
   - 从单张图片生成3D模型
   - 支持360度渲染，每30度输出一张图片
   - 支持透明背景输出

2. PNG到SVG转换器 (png_to_svg_converter.py)
   - 将PNG图片转换为SVG格式
   - 支持批量转换
   - 保持图层顺序
   - 支持透明背景

## 使用方法

### 3D模型渲染

```bash
python 3D_Gen.py [图片名称]
```

### PNG到SVG转换

```bash
# 批量转换
python png_to_svg_converter.py

# 单文件转换
python png_to_svg_converter.py 图片路径
```

## 依赖项

```bash
pip install -r requirements.txt
```

5. 创建 requirements.txt 文件：

```text:requirements.txt
torch
numpy
Pillow
opencv-python
scikit-learn
imageio
```

6. 然后提交这些新文件：

```bash
git add .gitignore README.md requirements.txt
git commit -m "Add project documentation and configuration files"
git push
```

这样您的项目就会被完整地推送到GitHub上，包含了所有必要的文档和配置文件。其他人可以轻松地克隆和使用您的项目。

如果您之后修改了代码，可以使用以下命令推送更新：

```bash
git add .
git commit -m "描述您的更改"
git push 