# 微多普勒时频图数据增广系统

基于VQ-GAN+LDM (潜在扩散模型) 的微多普勒时频图数据增广方案，使用Hugging Face的diffusers库实现。

## 项目结构

```
.
├── dataset/                    # 数据集目录 
├── dataset.py                  # 数据集加载和处理
├── train_vqvae.py              # VQ-VAE模型训练脚本
├── train_ldm.py                # LDM扩散模型训练脚本  
├── generate.py                 # 图像生成脚本
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

## 注意事项

1. 确保数据集中的图像均为256×256彩色图像
2. VQ-VAE训练通常比LDM训练收敛更快
3. 生成效果与训练数据质量、模型架构和训练参数密切相关
4. 在大规模数据集上可能需要调整模型架构和训练策略 