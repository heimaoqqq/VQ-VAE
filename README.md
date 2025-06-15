# 微多普勒时频图数据增广系统

基于VQ-VAE+LDM (潜在扩散模型) 的微多普勒时频图数据增广方案，使用Hugging Face的diffusers库实现。

## 项目结构

```
.
├── dataset/                    # 数据集目录
├── vqvae/                      # VQ-VAE模块化实现
│   ├── models/                 # 模型定义
│   │   ├── vqmodel.py          # VQ模型创建
│   │   └── losses.py           # 损失函数(包含位置感知损失)
│   ├── trainers/               # 训练器
│   │   └── vqvae_trainer.py    # VQ-VAE完整训练器
│   └── utils/                  # 工具函数
│       ├── training.py         # 训练相关函数
│       ├── normalization.py    # 分段自适应标准化实现
│       ├── config.py           # 配置参数解析
│       └── visualization.py    # 可视化函数
├── ldm/                        # LDM模块化实现
│   ├── models/                 # 模型定义
│   │   └── unet.py             # UNet模型定义
│   ├── trainers/               # 训练器
│   │   └── ldm_trainer.py      # LDM模型训练器
│   └── utils/                  # 工具函数
│       ├── training.py         # 训练相关函数
│       ├── config.py           # 配置参数解析
│       └── visualization.py    # 可视化函数
├── dataset.py                  # 数据集加载和处理
├── train_vqvae.py              # VQ-VAE模型训练入口脚本
├── train_ldm.py                # LDM扩散模型训练入口脚本  
├── generate.py                 # 图像生成脚本
├── test_normalization.py       # 分段自适应标准化测试脚本
└── requirements.txt            # 项目依赖
```

## 环境配置

```bash
# 创建并激活conda环境（推荐）
conda create -n vq-ldm python=3.10
conda activate vq-ldm

# 安装依赖
pip install -r requirements.txt

# 如果使用GPU，确保安装了对应版本的PyTorch
```

## 训练流程

整体训练流程分为两个阶段：
1. 训练VQ-VAE模型，将微多普勒图像压缩到离散的潜在空间
2. 训练LDM模型，学习在VQ-VAE的潜在空间中生成新样本

### 1. 训练VQ-VAE模型

```bash
python train_vqvae.py \
  --data_dir dataset \
  --output_dir vqvae_model \
  --batch_size 12 \
  --image_size 256 \
  --epochs 100 \
  --lr 5e-5 \
  --use_perceptual \
  --lambda_perceptual 0.1 \
  --use_positional \
  --lambda_positional 0.5 \
  --lambda_vertical 1.0 \
  --lambda_horizontal 0.5 \
  --use_adaptive_norm \
  --split_ratio 0.5 \
  --lower_quantile 0.01 \
  --upper_quantile 0.99 \
  --fp16 \
  --save_images \
  --logging_steps 3600 \
  --save_epochs 5
```

主要参数说明：
- `--data_dir`: 数据集目录
- `--output_dir`: 模型保存目录
- `--batch_size`: 批次大小
- `--image_size`: 图像尺寸
- `--epochs`: 训练轮数
- `--lr`: 学习率
- `--latent_channels`: 潜变量通道数（默认为4）
- `--vq_embed_dim`: VQ嵌入维度（默认为4）
- `--vq_num_embed`: VQ嵌入数量（默认为128）
- `--n_layers`: 下采样层数（默认为3，生成32x32潜在空间）
- `--save_epochs`: 每多少轮保存一次模型（默认为5）
- `--save_images`: 是否保存重建图像对比图
- `--logging_steps`: 每多少步保存一次重建图像（默认为100）
- `--use_wandb`: 是否使用wandb记录训练过程
- `--use_perceptual`: 是否使用感知损失
- `--lambda_perceptual`: 感知损失权重（默认为0.1）
- `--use_positional`: 是否使用位置感知损失（特别适合微多普勒图像）
- `--lambda_positional`: 位置感知损失权重（默认为0.5）
- `--lambda_vertical`: 垂直方向分布一致性权重（默认为1.0）
- `--lambda_horizontal`: 水平方向分布一致性权重（默认为0.5）
- `--use_adaptive_norm`: 是否使用分段自适应标准化（专为微多普勒数据优化）
- `--split_ratio`: 图像垂直分割比例（默认为0.5）
- `--lower_quantile`: 下半部分的下分位数（默认为0.01）
- `--upper_quantile`: 下半部分的上分位数（默认为0.99）
- `--fp16`: 是否使用半精度训练

### 2. 训练LDM模型

```bash
python train_ldm.py \
  --data_dir dataset \
  --vqvae_model_path vqvae_model \
  --output_dir ldm_model \
  --batch_size 16 \
  --image_size 256 \
  --epochs 100 \
  --lr 1e-4 \
  --save_images \
  --eval_steps 1000 \
  --num_inference_steps 50 \
  --beta_schedule cosine \
  --prediction_type epsilon \
  --scheduler_type ddim \
  --unet_channels 192,384,576,768
```

主要参数说明：
- `--vqvae_model_path`: 预训练的VQ-VAE模型路径
- `--latent_channels`: 潜变量通道数（默认为4）
- `--save_epochs`: 每多少轮保存一次模型（默认为5）
- `--save_images`: 是否保存生成图像
- `--eval_steps`: 每多少步评估并生成样本图像（默认为1000）
- `--logging_steps`: 日志记录间隔步数（默认为100）
- `--mixed_precision`: 混合精度训练，可选["no", "fp16", "bf16"]
- `--num_inference_steps`: 推理步数（默认为50）
- `--beta_schedule`: beta噪声调度方式（默认为cosine）
- `--prediction_type`: 预测类型（默认为epsilon）
- `--scheduler_type`: 采样器类型，可选[ddpm, ddim, pndm]（默认为ddim）
- `--unet_channels`: UNet基础通道数（默认为192,384,576,768）

## 生成新图像

训练完成后，可以使用训练好的模型生成新的微多普勒时频图：

```bash
python generate.py \
  --vqvae_path vqvae_model \
  --ldm_path ldm_model \
  --output_dir generated_images \
  --num_samples 16 \
  --batch_size 4 \
  --num_inference_steps 50 \
  --grid
```

主要参数说明：
- `--vqvae_path`: VQ-VAE模型路径
- `--ldm_path`: LDM模型路径
- `--output_dir`: 生成图像保存目录
- `--num_samples`: 生成样本数量
- `--num_inference_steps`: 推理步数
- `--grid`: 是否生成网格图像
- `--seed`: 随机种子，用于生成可重复的结果
- `--scheduler_type`: 采样器类型，可选[ddpm, ddim, pndm]（默认为ddim）
- `--use_adaptive_norm`: 是否使用分段自适应标准化
- `--split_ratio`: 图像分割比例，用于分段标准化
- `--lower_quantile`: 下半部分的下分位数
- `--upper_quantile`: 下半部分的上分位数

## 自定义Pipeline

运行生成脚本时，可以选择创建一个自定义Pipeline，将VQ-VAE和LDM模型组合在一起：

```bash
python generate.py --vqvae_path vqvae_model --ldm_path ldm_model
```

自定义Pipeline将被保存到指定的输出目录中，可以直接使用diffusers的接口加载。

## 模块化架构

项目采用高度模块化的设计，使代码更加清晰和可维护：

### VQ-VAE模块
- **models**: 包含VQ-VAE模型定义和各种损失函数
- **trainers**: 包含完整的训练器，处理训练循环、验证、早停和检查点保存
- **utils**: 包含工具函数，如标准化、可视化和配置解析

### LDM模块
- **models**: 包含UNet模型定义
- **trainers**: 包含LDM训练器，处理扩散模型的训练
- **utils**: 包含工具函数，如配置解析和可视化

### 主要脚本
- **train_vqvae.py**: 仅包含VQ-VAE训练的入口点，实际训练逻辑位于vqvae/trainers/vqvae_trainer.py
- **train_ldm.py**: 仅包含LDM训练的入口点，实际训练逻辑位于ldm/trainers/ldm_trainer.py
- **generate.py**: 提供了图像生成功能，支持多种采样器和自定义Pipeline创建

这种模块化设计使得代码更易于理解和扩展，同时减少了主脚本的复杂性。

## 核心特性

### 1. 位置感知损失 (Positional Loss)

位置感知损失专为微多普勒时频图特性设计，它通过对比原始图像和重建图像的垂直/水平分布特性，确保模型能够准确重建微多普勒时频图的关键特征：

- **垂直分布一致性**：微多普勒图像在垂直方向上有显著的强度分布特征，该损失函数确保重建图像保持这些分布特征
- **水平分布一致性**：同时关注水平方向的频率分布，但权重较低
- **频率域一致性**：通过傅里叶变换比较图像在频率域的特征分布

位置感知损失的实现位于`vqvae/models/losses.py`中的`PositionalLoss`类，可以通过以下参数控制：
- `--use_positional`: 是否启用位置感知损失
- `--lambda_positional`: 位置感知损失的总体权重
- `--lambda_vertical`: 垂直方向一致性的权重
- `--lambda_horizontal`: 水平方向一致性的权重

### 2. 分段自适应标准化 (Segmented Adaptive Normalization)

针对微多普勒时频图上下部分信号强度差异显著的特点，设计了专用的分段自适应标准化方法：

- **区域分割**：将图像按比例分为上下两部分，分别处理
- **下部强信号区域**：使用分位数标准化，保留关键目标信息
- **上部弱信号区域**：使用增强对比度的均值-方差标准化，凸显微弱特征
- **自适应参数**：标准化参数根据每张图像的特性自动调整

分段自适应标准化的实现位于`vqvae/utils/normalization.py`中的`MicroDopplerNormalizer`类，可以通过以下参数控制：
- `--use_adaptive_norm`: 是否启用分段自适应标准化
- `--split_ratio`: 图像垂直分割比例（默认0.5）
- `--lower_quantile`: 下半部分的下分位数（默认0.01）
- `--upper_quantile`: 下半部分的上分位数（默认0.99）

### 3. 模型架构优化

项目针对微多普勒时频图的特点对模型架构进行了优化：

- **VQ-VAE**: 
  - 使用较小的嵌入维度（4）和适中的码本大小（128）
  - 默认3层下采样，生成32×32的潜在空间
  - 支持感知损失和位置感知损失的组合使用

- **LDM**:
  - 使用DDIM采样器提高采样效率
  - 支持多种噪声调度方式（cosine, linear等）
  - 优化的UNet通道配置（192,384,576,768）

## GPU显存适配配置

项目已针对不同显存大小的GPU进行了优化配置：

### 6GB显存GPU配置 (例如GTX 1060)

```bash
# 6GB显存优化配置
python train_vqvae.py \
  --batch_size 8 \
  --lr 1e-4 \
  --latent_channels 4 \
  --vq_embed_dim 4 \
  --vq_num_embed 128 \
  --n_layers 2 \
  --image_size 128 \
  --fp16  # 开启混合精度训练
```

主要优化点：
- 降低批次大小到8或更低
- 减小模型容量(嵌入维度和码本大小)
- 减少下采样层数
- 使用较小的输入尺寸
- 启用FP16混合精度训练，提高显存利用率

### 16GB+ 显存GPU配置

```bash
# 16GB显存优化配置
python train_vqvae.py \
  --batch_size 32 \
  --lr 1e-4 \
  --latent_channels 4 \
  --vq_embed_dim 4 \
  --vq_num_embed 256 \
  --n_layers 4 \
  --fp16  # 开启混合精度训练
  --use_perceptual  # 使用感知损失
  --lambda_perceptual 0.1  # 感知损失权重
  --use_positional  # 使用位置感知损失
  --lambda_positional 0.5  # 位置损失权重
  --use_adaptive_norm  # 使用分段自适应标准化
```

主要优化点：
- 批次大小增加到32
- 增大模型容量（码本大小）
- 使用所有4层下采样结构
- 启用感知损失提升重建质量
- 启用位置感知损失捕获微多普勒图像的特殊分布
- 启用分段自适应标准化，增强对微多普勒图像上下区域不同信号特性的处理

## 重要提示

1. 潜在通道数(latent_channels)和VQ嵌入维度(vq_embed_dim)应该保持一致
2. 在模型中，实际下采样倍数与层数并不完全对应，请参考控制台输出信息
3. 对于大型数据集，建议使用更大的码本(vq_num_embed)以增强表示能力
4. 如使用感知损失，请确保有足够的显存
5. 位置感知损失和分段自适应标准化专为微多普勒数据优化，可显著提升重建质量和生成质量
6. 生成图像时，如果模型训练时使用了分段自适应标准化，生成时也应启用该功能

## Kaggle环境使用指南

项目已针对Kaggle环境进行了兼容性优化。如在Kaggle上使用，请遵循以下步骤：

### Kaggle训练命令

```python
# 训练VQ-VAE（完整指令）
!python train_vqvae.py \
  --data_dir /kaggle/input/dataset \
  --output_dir vqvae_model \
  --batch_size 16 \
  --image_size 256 \
  --epochs 100 \
  --lr 1e-4 \
  --use_perceptual \
  --use_positional \
  --use_adaptive_norm \
  --fp16 \
  --save_images \
  --logging_steps 383  # 每1轮保存一次重建图像（假设每轮383批次）

# 如果想每10轮保存一次重建图像：
# --logging_steps 3830  # 383批次/轮 × 10轮

# 训练LDM
!python train_ldm.py \
  --data_dir /kaggle/input/dataset \
  --vqvae_model_path vqvae_model \
  --output_dir ldm_model \
  --batch_size 16 \
  --image_size 256 \
  --epochs 100 \
  --lr 1e-4 \
  --mixed_precision fp16 \
  --save_images \
  --eval_steps 1000  # 每1000步生成一次样本图像
  --scheduler_type ddim \
  --num_inference_steps 50
```

## 模型配置说明

最新版本针对微多普勒时频图的特点，对模型大小进行了优化：

### 默认模型配置
- 下采样层数: 3（生成32x32潜在空间）
- 起始通道数: 32（原为64）
- 最大通道数: 256（原为512）
- 潜变量通道数: 4（优化为更大容量）
- 码本大小: 128（优化为更大容量，适应复杂特征）
- UNet通道配置: [192, 384, 576, 768]（提升LDM生成能力）
- LDM采样器: DDIM（默认50步，提供更高质量和更快速度）

这些配置基于微多普勒时频图的特征复杂度进行了优化，在保持训练效率的同时提供更好的表示能力。

## 数据集要求

1. 数据集应组织为以下结构：
   ```
   dataset/
   ├── class1/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ├── class2/
   │   ├── image1.jpg
   │   └── ...
   └── ...
   ```

2. 图像应为RGB格式，推荐尺寸为256×256
3. 支持自动调整图像尺寸，但为了最佳效果，建议预处理为统一尺寸
4. 数据集会自动按8:2比例分为训练集和验证集

## 注意事项

1. 确保数据集中的图像均为RGB彩色图像
2. VQ-VAE训练通常比LDM训练收敛更快
3. 如果码本利用率较低(低于30%)，可以考虑进一步减小码本大小
4. 在大规模数据集上可能需要调整模型架构和训练策略
5. 默认已配置模型保存策略，包括：
   - 每5轮保存一次模型（可通过--save_epochs调整）
   - 保存验证损失最佳的模型
   - 验证损失连续10轮无改善自动早停（已从3轮增加到10轮）
   - 自动删除旧模型，只保留最新和最佳模型
6. 重建图像保存路径：
   - VQ-VAE重建图像：保存在{output_dir}/images/目录下
   - LDM生成图像：保存在{output_dir}/generated_images/目录下
7. 生成的图像质量受以下因素影响：
   - VQ-VAE的重建质量
   - LDM的训练轮数
   - 采样器类型和推理步数
   - 分段自适应标准化的参数设置 