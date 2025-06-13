# Kaggle环境使用指南

本项目已针对Kaggle环境进行了兼容性优化。如在Kaggle上使用，请遵循以下步骤：

## 1. 环境设置

在notebook开头运行以下代码下载并配置项目：

```python
# 克隆项目
!git clone https://github.com/heimaoqqq/VQ-VAE.git
%cd VQ-VAE

# 设置环境
!python kaggle_setup.py
```

## 2. 数据处理

根据您的数据集位置进行相应配置：

```python
# 如果您的数据集在Kaggle dataset中：
!ln -s /kaggle/input/your-dataset-name dataset

# 或者复制数据集
!cp -r /kaggle/input/your-dataset-name dataset
```

## 3. 训练VQ-VAE模型

```python
!python train_vqvae.py \
  --data_dir dataset \
  --output_dir vqvae_model \
  --batch_size 16 \
  --kaggle \
  --fp16 \
  --epochs 20
```

## 4. 训练LDM模型

```python
!python train_ldm.py \
  --data_dir dataset \
  --vqvae_model_path vqvae_model \
  --output_dir ldm_model \
  --batch_size 16 \
  --kaggle \
  --epochs 20
```

## 5. 生成样本

```python
!python generate.py \
  --vqvae_path vqvae_model \
  --ldm_path ldm_model \
  --output_dir generated_images \
  --num_samples 16 \
  --grid
```

## 常见问题解决

如遇到依赖问题，执行：

```python
# 安装兼容版本的依赖
!pip install huggingface_hub>=0.20.2
!pip install diffusers>=0.26.3 --no-deps
!pip install torch torchvision
```

由于Kaggle笔记本环境会话结束时状态会丢失，请注意定期保存模型文件和生成结果。另外，如果需要使用GPU加速，可以在Kaggle设置中启用GPU。 