# Kaggle上使用VQ-VAE潜在空间分析工具

本文档提供在Kaggle平台上使用VQ-VAE潜在空间分析工具的详细指南。

## 准备工作

### 1. 上传预训练模型到Kaggle数据集

1. 在Kaggle上创建一个新的数据集
   - 点击"Create" > "Dataset"
   - 数据集名称设为"vq-gan"
   - 上传您的预训练模型文件（vqgan_model_best_best.pt）
   - 点击"Create"按钮完成创建

2. 确保您有一个测试数据集
   - 可以使用您自己的数据集
   - 或者使用Kaggle上的公开数据集

## 在Kaggle笔记本中运行分析

### 1. 创建新的Kaggle笔记本

1. 点击"Create" > "Notebook"
2. 在设置中添加数据集：
   - 您创建的"vq-gan"数据集
   - 您要用于测试的图像数据集

### 2. 设置环境

将以下代码复制到笔记本的第一个代码单元：

```python
# 安装必要的依赖
!pip install scikit-learn matplotlib seaborn tqdm

# 克隆项目代码
!git clone https://github.com/你的用户名/VQ-VAE.git
!mkdir -p /kaggle/working/VQ-VAE/model
!cp -r /kaggle/input/vq-gan/*.pt /kaggle/working/VQ-VAE/model/

# 添加项目路径
import sys
sys.path.append('/kaggle/working/VQ-VAE')

# 检查文件是否存在
!ls -la /kaggle/working/VQ-VAE/vqvae/
```

### 3. 检查数据集和模型

```python
# 检查数据集
import os

# 替换为您的数据集路径
data_dir = '/kaggle/input/your-dataset/images'

if os.path.exists(data_dir):
    print(f"数据集路径存在: {data_dir}")
    print(f"文件数量: {len(os.listdir(data_dir))}")
else:
    print(f"数据集路径不存在: {data_dir}")
    print("请更新为正确的数据集路径")

# 检查预训练模型
model_path = '/kaggle/input/vq-gan/vqgan_model_best_best.pt'

if os.path.exists(model_path):
    print(f"模型文件存在: {model_path}")
    print(f"文件大小: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
else:
    print(f"模型文件不存在: {model_path}")
    print("请确保已上传模型文件")
```

### 4. 运行分析

```python
# 运行分析脚本
!python -m vqvae.diffusion_readiness_test \
    --model_path {model_path} \
    --data_dir {data_dir} \
    --batch_size 16 \
    --max_samples 500 \
    --n_embed 8192 \
    --embed_dim 256 \
    --output_dir vqvae_analysis
```

### 5. 查看分析结果

```python
# 显示雷达图
from IPython.display import Image
Image('/kaggle/working/vqvae_analysis/diffusion_readiness_radar.png')

# 查看分析报告
with open('/kaggle/working/vqvae_analysis/diffusion_readiness_report.txt', 'r') as f:
    report = f.read()
    
print(report)
```

## 参数说明

分析脚本支持以下参数：

- `--model_path`: 预训练模型路径（默认为'/kaggle/input/vq-gan/vqgan_model_best_best.pt'）
- `--data_dir`: 数据集目录，必须指定
- `--batch_size`: 批次大小，默认为32
- `--max_samples`: 分析的最大样本数，默认为1000
- `--n_embed`: 码本大小，默认为8192
- `--embed_dim`: 嵌入维度，默认为256
- `--output_dir`: 输出目录，默认为'analysis_results'

## 注意事项

1. **模型参数匹配**：确保`--n_embed`和`--embed_dim`参数与训练模型时使用的值一致，否则模型加载可能失败。

2. **内存限制**：如果遇到内存不足问题，可以减小`--batch_size`和`--max_samples`的值。

3. **运行时间**：完整分析可能需要几分钟到几十分钟，取决于数据集大小和GPU性能。

4. **输出文件**：分析结果将保存在`/kaggle/working/vqvae_analysis/`目录下，包括雷达图和详细报告。 