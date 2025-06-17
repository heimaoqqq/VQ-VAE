# 微多普勒时频图数据增广系统 (VQ-GAN + LDM)

本项目旨在构建一个两阶段的深度学习流程，用于生成高质量的微多普勒时频图，以实现有效的数据增广。系统首先训练一个**向量量化生成对抗网络（VQ-GAN）**，将高分辨率的时频图压缩到一个高质量的、离散的潜在空间中。随后，在这一潜在空间上训练一个**潜在扩散模型（LDM）**，以学习并生成全新的、多样化的时频图数据。

## 开发历程与技术选型

本项目在开发过程中经历了一系列关键的技术迭代，最终方案是解决核心问题的最佳实践总结：

1.  **初始问题：LDM生成模糊**
    项目最初采用标准的VQ-VAE + LDM架构。尽管VQ-VAE在视觉上重建效果良好，但以此为基础训练的LDM生成的图像却非常模糊，缺乏关键的高频细节。

2.  **初步尝试：优化LDM架构**
    我们首先怀疑是LDM的网络结构问题，并对其U-Net中的注意力机制进行了优化，但生成质量改善有限，表明问题根源不在于第二阶段的LDM。

3.  **核心突破：引入VQ-GAN**
    通过分析相关领域的先进实践，我们断定，单纯依赖重建损失（如L1）和感知损失（LPIPS）的VQ-VAE，本质上难以完美还原真实数据中所有锐利的高频细节，这是导致LDM模糊的根本原因。解决方案是引入一个**对抗性判别器**，将VQ-VAE升级为**VQ-GAN**。判别器的存在会"强迫"生成器（即VQ-VAE的解码器）去学习和重建那些最精细、最锐利的细节，从而产生一个信息保真度极高的潜在空间。

4.  **最终方案：构建自定义VQ-GAN模型**
    在实现VQ-GAN的过程中，我们遇到了大量由Hugging Face `diffusers`库版本不兼容导致的问题（如模块初始化参数错误、内部数据流不匹配等）。为了彻底解决这些"黑盒"问题并获得对模型行为的完全控制，我们最终放弃了使用不稳定的高层`VQModel`封装，转而采用其经过充分测试的底层模块（`Encoder`, `Decoder`, `VectorQuantizer`），手动构建了一个我们自己的`CustomVQGAN`模型。这一最终方案稳定、可靠，且逻辑清晰。

## 项目结构

```
.
├── dataset/                      # 数据集目录
├── vqvae/                        # VQ-GAN 模块
│   ├── custom_vqgan.py           # ✅ 自定义的VQ-GAN模型实现 (核心)
│   ├── discriminator.py          # ✅ PatchGAN判别器实现
│   ├── vqgan_trainer.py          # ✅ VQ-GAN训练器 (含WGAN-GP损失)
│   └── utils.py                  # 工具函数
├── ldm/                          # LDM 模块 (后续阶段)
│   └── ...
├── dataset.py                    # 数据集加载和处理
├── train_vqvae.py                # ✅ VQ-GAN 训练脚本
├── train_ldm.py                  # LDM 扩散模型训练脚本 (后续阶段)
└── requirements.txt              # 项目依赖
```

## 环境配置

```bash
# 创建并激活conda环境
conda create -n vqgan-ldm python=3.11
conda activate vqgan-ldm

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 训练流程

### 1. 训练VQ-GAN模型

此阶段的目标是获得一个高质量的自编码器。

```bash
python train_vqvae.py \
  --data_dir "path/to/your/dataset" \
  --output_dir "vqgan_model_output" \
  --batch_size 8 \
  --epochs 300 \
  --lr 1e-4 \
  --disc_lr 1e-4 \
  --use_wandb
```

**主要参数说明 (`train_vqvae.py`):**

- **路径参数**
  - `--data_dir`: 数据集目录。
  - `--output_dir`: 模型和日志的输出目录。
- **数据参数**
  - `--batch_size`: 批次大小 (推荐为8)。
  - `--image_size`: 图像尺寸 (默认为256)。
- **训练参数**
  - `--epochs`: 训练轮数 (GAN通常需要较多轮数, e.g., 300)。
  - `--lr`: 生成器（VQ-GAN）的学习率。
  - `--disc_lr`: 判别器的学习率。
- **模型结构参数**
  - `--in_channels`, `--out_channels`: 输入/输出通道数 (彩色图为3)。
  - `--latent_channels`: 潜在空间通道数 (默认为4)。
  - `--num_vq_embeddings`: 码本中编码的数量 (默认为8192)。
  - `--block_out_channels`: VQ-GAN每个下采样块的输出通道 (e.g., `64 128 256`)。
  - `--layers_per_block`: 每个U-Net块中的ResNet层数 (默认为2)。
- **损失权重参数**
  - `--reconstruction_loss_weight`: 重建损失（L1）的权重。
  - `--perceptual_loss_weight`: 感知损失（LPIPS）的权重。
  - `--g_loss_adv_weight`: 生成器对抗性损失的权重。
  - `--vq_embed_loss_weight`: VQ承诺损失的权重。
  - `--gradient_penalty_weight`: WGAN-GP梯度惩罚的权重。
- **日志与保存**
  - `--save_epochs`: 每N轮保存一次模型。
  - `--logging_steps`: 每N步在wandb上记录一次日志。
  - `--use_wandb`: 是否使用Weights & Biases进行可视化。
  - `--wandb_project`: wandb项目名。
  - `--wandb_name`: 本次运行的wandb名称。


### 2. 训练LDM模型 (后续工作)

在获得高质量的VQ-GAN后，可进行第二阶段的LDM训练。

```bash
# 示例命令 (根据train_ldm.py的最终实现可能需要调整)
python train_ldm.py \
  --data_dir "path/to/your/dataset" \
  --vqgan_path "vqgan_model_output/vqgan_epoch_300.pt" \
  --output_dir "ldm_model_output" \
  --batch_size 4 \
  --epochs 500 \
  --lr 1e-4
```

---

**感谢您在整个调试过程中的耐心与协作，我们共同构建了一个稳健且先进的生成模型框架。** 