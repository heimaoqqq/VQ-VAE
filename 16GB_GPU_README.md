# 16GB GPU显存配置说明

本项目已针对16GB显存的GPU进行了优化配置，主要包括：

## 1. VQ-VAE模型配置

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
```

主要优化点：
- 批次大小增加到32
- 增大模型容量（嵌入维度和通道数）
- 使用所有4层下采样结构
- 启用FP16混合精度训练，提高显存利用率

## 2. LDM模型配置

```bash
# 16GB显存优化配置
python train_ldm.py \
  --batch_size 32 \
  --lr 1e-4 \
  --latent_channels 4 \
  --mixed_precision fp16  # 默认开启FP16
  --save_images  # 定期保存生成结果
```

主要优化点：
- 匹配VQ-VAE配置的潜在通道数
- 使用更大的UNet模型（更多通道和层）
- 默认开启混合精度训练
- 批次大小增加到32

## 3. 推送到GitHub

代码已提交，但推送到GitHub可能需要认证。请运行以下命令或使用批处理文件：

```bash
# 检查远程仓库配置
git remote -v

# 推送代码
git push -u origin master
```

如果遇到SSH认证问题，可以切换到HTTPS：

```bash
git remote set-url origin https://github.com/heimaoqqq/VQ-VAE.git
```

## 4. 训练提示

1. 先训练VQ-VAE模型直至收敛（重建损失稳定）
2. 使用训练好的VQ-VAE模型训练LDM
3. 注意监控码本利用率，理想情况下应接近100%
4. 如显存不足，可降低批次大小或模型参数 