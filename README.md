# VQ-VAE 潜在空间分析工具

这个项目提供了一套全面的分析工具，用于评估VQ-VAE模型潜在空间的质量，特别是针对下游扩散模型的应用场景。

## 项目背景

VQ-VAE (Vector Quantized Variational AutoEncoder) 是一种强大的生成模型，能够将图像压缩到离散的潜在空间，并实现高质量的重建。然而，即使重建质量很好，潜在空间的结构和特性也可能不适合下游任务，特别是扩散模型。

常见问题包括：
- 潜在空间分布不均匀或不连续
- 码本利用率低或分布不均
- 潜在空间中的插值不平滑
- 量化导致的信息丢失

本项目旨在提供客观、可视化的评估手段，帮助识别和解决这些问题。

## 主要功能

该工具包提供以下分析功能：

1. **潜在空间分布分析**
   - 分析潜在向量的统计特性
   - 与正态分布的距离测量
   - 维度间相关性分析

2. **码本结构分析**
   - 码本向量之间的距离分析
   - 码本利用率评估
   - 码本向量的可视化

3. **潜在空间连续性分析**
   - 插值平滑度评估
   - 量化前后一致性测量
   - 插值结果可视化

4. **重建误差分析**
   - 像素级和特征级误差分布
   - 误差统计和可视化

5. **综合扩散适应性评分**
   - 基于多维指标的综合评分
   - 针对性改进建议
   - 雷达图可视化结果

## 安装要求

- Python 3.7+
- PyTorch 1.8+
- scikit-learn
- matplotlib
- seaborn
- scipy
- tqdm

## 快速开始

### 本地安装

```bash
# 克隆仓库
git clone https://github.com/你的用户名/VQ-VAE.git
cd VQ-VAE

# 安装依赖
pip install -r requirements.txt
```

### 使用方法

```python
# 基本用法
from vqvae.latent_space_analyzer import LatentSpaceAnalyzer
from vqvae.custom_vqgan import CustomVQGAN

# 加载模型
model = CustomVQGAN(**model_config)
model.load_state_dict(torch.load('path/to/model.pt')['vqgan_state_dict'])

# 创建分析器
analyzer = LatentSpaceAnalyzer(model, device='cuda')

# 运行综合分析
report = analyzer.analyze_for_diffusion(dataloader)
```

### 命令行工具

```bash
python -m vqvae.diffusion_readiness_test \
    --model_path path/to/model.pt \
    --data_dir path/to/dataset \
    --batch_size 32 \
    --max_samples 1000
```

## 在Kaggle上使用

1. 上传预训练模型到Kaggle数据集
2. 在Kaggle笔记本中运行以下代码：

```python
# 克隆项目
!git clone https://github.com/你的用户名/VQ-VAE.git
!cp -r /kaggle/input/vq-gan/*.pt /kaggle/working/VQ-VAE/model/

# 添加项目路径
import sys
sys.path.append('/kaggle/working/VQ-VAE')

# 运行分析
!python -m vqvae.diffusion_readiness_test \
    --model_path /kaggle/input/vq-gan/vqgan_model_best_best.pt \
    --data_dir /kaggle/input/your-dataset/images \
    --batch_size 16 \
    --max_samples 500
```

## 项目结构

```
VQ-VAE/
├── vqvae/
│   ├── latent_space_analyzer.py  # 潜在空间分析工具
│   ├── diffusion_readiness_test.py  # 命令行工具
│   ├── custom_vqgan.py  # VQ-GAN模型定义
│   ├── ema_vector_quantizer.py  # EMA向量量化器
│   ├── models/  # 模型组件
│   └── README_latent_analysis.md  # 详细文档
├── model/  # 预训练模型存放目录
└── README.md  # 项目说明
```

## 许可证

MIT

## 致谢

感谢所有为VQ-VAE和扩散模型研究做出贡献的研究者。