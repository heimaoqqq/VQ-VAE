# 微多普勒时频图数据增广系统

基于VQ-VAE+LDM (潜在扩散模型) 的微多普勒时频图数据增广方案，使用Hugging Face的diffusers库实现。

## 项目结构

```
.
├── dataset/                    # 数据集目录 
├── dataset.py                  # 数据集加载和处理
├── train_vqvae.py              # VQ-VAE模型训练脚本
├── train_ldm.py                # LDM扩散模型训练脚本  
├── generate.py                 # 图像生成脚本
├── check_vqmodel_api.py        # VQModel API检查工具
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
  --batch_size 16 \
  --image_size 256 \
  --epochs 100 \
  --lr 1e-4
```

主要参数说明：
- `--data_dir`: 数据集目录
- `--output_dir`: 模型保存目录
- `--batch_size`: 批次大小
- `--image_size`: 图像尺寸
- `--epochs`: 训练轮数
- `--lr`: 学习率
- `--vq_embed_dim`: VQ嵌入维度
- `--vq_num_embed`: VQ嵌入数量
- `--use_wandb`: 是否使用wandb记录训练过程
- `--use_perceptual`: 是否使用感知损失
- `--lambda_perceptual`: 感知损失权重

### 2. 训练LDM模型

```bash
python train_ldm.py \
  --data_dir dataset \
  --vqvae_model_path vqvae_model \
  --output_dir ldm_model \
  --batch_size 16 \
  --image_size 256 \
  --epochs 100 \
  --lr 1e-4
```

主要参数说明：
- `--vqvae_model_path`: 预训练的VQ-VAE模型路径
- `--mixed_precision`: 混合精度训练，可选["no", "fp16", "bf16"]
- `--save_epochs`: 每多少个epoch保存一次模型
- `--save_images`: 是否保存生成图像

## 生成新图像

训练完成后，可以使用训练好的模型生成新的微多普勒时频图：

```bash
python generate.py \
  --vqvae_path vqvae_model \
  --ldm_path ldm_model \
  --output_dir generated_images \
  --num_samples 16 \
  --batch_size 4 \
  --num_inference_steps 1000 \
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

## 自定义Pipeline

运行生成脚本时，可以选择创建一个自定义Pipeline，将VQ-VAE和LDM模型组合在一起：

```bash
python generate.py --vqvae_path vqvae_model --ldm_path ldm_model
# 当询问是否创建自定义Pipeline时，输入 y
```

自定义Pipeline将被保存到指定的输出目录中，可以直接使用diffusers的接口加载。

## 16GB GPU显存优化配置

项目已针对16GB显存的GPU进行了优化配置：

### VQ-VAE模型配置

```bash
# 16GB显存优化配置
python train_vqvae.py \
  --batch_size 32 \
  --lr 1e-4 \
  --latent_channels 4 \
  --vq_embed_dim 128 \
  --vq_num_embed 256 \
  --n_layers 4 \
  --fp16  # 开启混合精度训练
  --use_perceptual  # 使用感知损失
  --lambda_perceptual 0.1  # 感知损失权重
```

主要优化点：
- 批次大小增加到32
- 增大模型容量（嵌入维度和通道数）
- 使用所有4层下采样结构
- 启用FP16混合精度训练，提高显存利用率
- 结合感知损失提升重建质量

### LDM模型配置

```bash
# 16GB显存优化配置
python train_ldm.py \
  --batch_size 32 \
  --lr 1e-4 \
  --latent_channels 4 \
  --mixed_precision fp16  # 默认开启FP16
  --save_images  # 定期保存生成结果
```

### 训练提示

1. 先训练VQ-VAE模型直至收敛（重建损失稳定）
2. 使用训练好的VQ-VAE模型训练LDM
3. 注意监控码本利用率，理想情况下应接近100%
4. 如显存不足，可降低批次大小或模型参数

## Kaggle环境使用指南

项目已针对Kaggle环境进行了兼容性优化。如在Kaggle上使用，请遵循以下步骤：

### 环境设置

在notebook开头运行以下代码下载并配置项目：

```python
# 克隆项目
!git clone https://github.com/heimaoqqq/VQ-VAE.git
%cd VQ-VAE

# 设置环境
!python kaggle_setup.py
```

### 数据处理

根据您的数据集位置进行相应配置：

```python
# 如果您的数据集在Kaggle dataset中：
!ln -s /kaggle/input/your-dataset-name dataset

# 或者复制数据集
!cp -r /kaggle/input/your-dataset-name dataset
```

### Kaggle训练命令

```python
# 训练VQ-VAE
!python train_vqvae.py \
  --data_dir dataset \
  --output_dir vqvae_model \
  --batch_size 16 \
  --kaggle \
  --fp16 \
  --epochs 20 \
  --use_perceptual \
  --lambda_perceptual 0.1

# 训练LDM
!python train_ldm.py \
  --data_dir dataset \
  --vqvae_model_path vqvae_model \
  --output_dir ldm_model \
  --batch_size 16 \
  --kaggle \
  --mixed_precision fp16 \
  --epochs 20
```

### 常见问题解决

如遇到依赖问题，执行：

```python
# 安装兼容版本的依赖
!pip install huggingface_hub>=0.20.2
!pip install diffusers>=0.26.3 --no-deps
!pip install torch torchvision
```

## 注意事项

1. 确保数据集中的图像均为256×256彩色图像
2. VQ-VAE训练通常比LDM训练收敛更快
3. 生成效果与训练数据质量、模型架构和训练参数密切相关
4. 在大规模数据集上可能需要调整模型架构和训练策略
5. 默认已配置模型保存策略，包括：
   - 每5轮保存一次模型（可通过--save_epochs调整）
   - 保存验证损失最佳的模型
   - 验证损失连续3轮无改善自动早停
   - 自动删除旧模型，只保留最新和最佳模型 