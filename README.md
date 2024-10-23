# 图像压缩与解压缩系统

本代码仓库包含了一个用于压缩、解压缩和查看大型卫星图像的系统，该系统使用自定义神经网络模型。

## 安装

### Linux 环境
对于 Linux 环境，您可以直接使用 pip 安装所需依赖：

```bash
pip install compressai
```


### Windows 环境
对于 Windows 环境，安装过程较为复杂：

1. 从源码编译 compressai。
2. 安装 PyTorch < 2.3.0 的 GPU 版本，以覆盖原有的 CPU 版本。

## 使用方法

文件 `Untitled-1.sh` 包含了运行系统的示例命令。以下是使用流程的简要概述：

1. 下载 AerialImageDataset。
2. 将图像分割成 19x19 个 256x256 像素的图像块，剩余部分裁剪。
3. 使用分割后的图像训练模型。
4. 将图像压缩成 BIN 文件。
5. 对压缩文件进行分类和打包。
6. 使用 `decompress_frontend.py` 浏览压缩后的图像。

## 主要组件

- `train.py`: 训练压缩模型。
- `compress_images.py`: 将图像压缩成 BIN 文件。
- `decompress_images.py`: 将 BIN 文件解压缩回图像。
- `decompress_frontend.py`: 提供用于浏览压缩图像的图形用户界面。

## 模型架构

压缩模型基于 `models/tcm.py` 中定义的自定义架构。