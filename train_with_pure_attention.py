"""
使用纯自注意力的LDM训练脚本 (优化版本)
"""
import os
import torch
import argparse
from dataset import get_dataloaders
from diffusers import VQModel
from ldm.models.unet import create_unet_model
from ldm.trainers.ldm_trainer import LDMTrainer
from ldm.utils.config import parse_args

def main():
    # 解析参数
    args = parse_args()
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.cuda.empty_cache()
        
        # 显示GPU信息
        print(f"使用设备: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"总GPU显存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"当前可用显存: {torch.cuda.memory_reserved(0) / (1024**3):.2f} GB")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载预训练的VQ-VAE模型
    vq_model = VQModel.from_pretrained(args.vqvae_path)
    vq_model.eval().requires_grad_(False)  # 冻结VQ-VAE参数
    
    # 计算潜在空间大小
    latent_size = args.image_size // (2 ** (len(vq_model.config.down_block_types)))
    
    print(f"\n{'='*50}")
    print(f"纯自注意力LDM训练 (优化版本)")
    print(f"{'='*50}")
    print(f"图像大小: {args.image_size}x{args.image_size}")
    print(f"潜在空间大小: {latent_size}x{latent_size}")
    print(f"批次大小: {args.batch_size} (梯度累积步数: {args.gradient_accumulation_steps})")
    print(f"有效批次大小: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.epochs}")
    print(f"混合精度: {args.mixed_precision}")
    print(f"Beta调度: {args.beta_schedule}")
    print(f"采样器: {args.scheduler_type}")
    print(f"{'='*50}\n")
    
    # 创建数据加载器
    train_dataloader, val_dataloader, _ = get_dataloaders(
        args.data_dir, args.batch_size, args.image_size
    )
    
    total_samples = len(train_dataloader.dataset) + len(val_dataloader.dataset)
    print(f"数据集总样本数: {total_samples}")
    print(f"训练集样本数: {len(train_dataloader.dataset)} ({len(train_dataloader.dataset)/total_samples*100:.1f}%)")
    print(f"验证集样本数: {len(val_dataloader.dataset)} ({len(val_dataloader.dataset)/total_samples*100:.1f}%)")
    print(f"训练批次数: {len(train_dataloader)}")
    print(f"每轮训练步数: {len(train_dataloader)}")
    print(f"{'='*50}\n")
    
    # 创建纯自注意力UNet模型
    model = create_unet_model(
        latent_size=latent_size, 
        latent_channels=args.latent_channels
    )
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params/1e6:.2f}M")
    print(f"可训练参数: {trainable_params/1e6:.2f}M")
    
    # 监控初始显存占用
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print(f"初始GPU内存占用: {torch.cuda.memory_allocated() / (1024**3):.4f} GB")
        print(f"初始GPU最大内存占用: {torch.cuda.max_memory_allocated() / (1024**3):.4f} GB")
    
    # 创建LDM训练器
    trainer = LDMTrainer(
        unet=model,
        vq_model=vq_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        args=args,
        beta_schedule=args.beta_schedule
    )
    
    # 监控模型和优化器分配后的显存
    if torch.cuda.is_available():
        print(f"模型加载后GPU内存占用: {torch.cuda.memory_allocated() / (1024**3):.4f} GB")
        print(f"模型加载后GPU峰值内存占用: {torch.cuda.max_memory_allocated() / (1024**3):.4f} GB")
    
    # 如果指定了检查点路径，则从检查点恢复训练
    if args.resume_from_checkpoint:
        print(f"正在从检查点恢复训练: {args.resume_from_checkpoint}")
        trainer.load_checkpoint(args.resume_from_checkpoint)
    
    # 开始训练
    trainer.train()
    
    # 训练结束后显示最终显存使用情况
    if torch.cuda.is_available():
        print(f"\n训练结束信息:")
        print(f"{'='*50}")
        print(f"最终GPU内存占用: {torch.cuda.memory_allocated() / (1024**3):.4f} GB")
        print(f"训练过程中峰值GPU内存占用: {torch.cuda.max_memory_allocated() / (1024**3):.4f} GB")
        print(f"{'='*50}")

if __name__ == "__main__":
    main() 