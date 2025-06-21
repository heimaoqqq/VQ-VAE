"""
改进版VQ-VAE训练脚本，专门针对解决码本坍塌问题
"""

import os
import torch
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random

from vqvae.custom_vqgan import CustomVQGAN
from vqvae.discriminator import Discriminator
from vqvae.vqgan_trainer import VQGANTrainer
from vqvae.improved_vq_config import CODEBOOK_CONFIG, TRAINER_CONFIG, OPTIMIZER_CONFIG, TRAINING_STRATEGY

def set_seed(seed):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device)
    
    # 创建数据变换
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 归一化到[-1, 1]
    ])
    
    # 加载数据集
    train_dataset = ImageFolder(root=args.train_dir, transform=transform)
    val_dataset = ImageFolder(root=args.val_dir, transform=transform)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_STRATEGY["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_STRATEGY["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"训练数据集大小: {len(train_dataset)}")
    print(f"验证数据集大小: {len(val_dataset)}")
    
    # 创建模型目录
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    
    # 创建VQGAN模型
    vqgan = CustomVQGAN(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        n_embed=CODEBOOK_CONFIG["n_embed"],
        embed_dim=CODEBOOK_CONFIG["embed_dim"],
        ema_decay=CODEBOOK_CONFIG["ema_decay"],
        commitment_loss_beta=CODEBOOK_CONFIG["commitment_loss_beta"]
    ).to(device)
    
    # 创建判别器
    discriminator = Discriminator(
        in_channels=args.out_channels,
        n_layers=3,
        use_actnorm=False
    ).to(device)
    
    # 创建优化器
    g_optimizer = torch.optim.Adam(
        vqgan.parameters(),
        lr=OPTIMIZER_CONFIG["learning_rate"],
        betas=(OPTIMIZER_CONFIG["beta1"], OPTIMIZER_CONFIG["beta2"]),
        weight_decay=OPTIMIZER_CONFIG["weight_decay"]
    )
    
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=OPTIMIZER_CONFIG["learning_rate"],
        betas=(OPTIMIZER_CONFIG["beta1"], OPTIMIZER_CONFIG["beta2"]),
        weight_decay=OPTIMIZER_CONFIG["weight_decay"]
    )
    
    # 创建学习率调度器
    g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    
    # 创建训练器
    trainer = VQGANTrainer(
        vqgan=vqgan,
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        lr_scheduler_g=g_scheduler,
        lr_scheduler_d=d_scheduler,
        device=device,
        use_amp=args.use_amp,
        checkpoint_path=os.path.join(args.model_dir, f"{args.model_name}.pt"),
        sample_dir=args.sample_dir,
        lambda_gp=10.0,
        l1_weight=TRAINER_CONFIG["l1_weight"],
        perceptual_weight=TRAINER_CONFIG["perceptual_weight"],
        adversarial_weight=TRAINER_CONFIG["adversarial_weight"],
        entropy_weight=TRAINER_CONFIG["entropy_weight"],
        log_interval=args.log_interval,
        reset_low_usage_interval=TRAINER_CONFIG["reset_low_usage_interval"],
        reset_low_usage_percentage=TRAINER_CONFIG["reset_low_usage_percentage"],
        temperature=TRAINING_STRATEGY["temperature"],
        disable_codebook_expansion=TRAINING_STRATEGY["disable_codebook_expansion"]
    )
    
    # 训练模型
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        skip_optimizer=args.skip_optimizer,
        resume_training=args.resume_training
    )
    
    print("训练完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="改进版VQ-VAE训练脚本")
    
    # 数据相关参数
    parser.add_argument("--train_dir", type=str, required=True, help="训练数据集目录")
    parser.add_argument("--val_dir", type=str, required=True, help="验证数据集目录")
    parser.add_argument("--image_size", type=int, default=256, help="图像大小")
    parser.add_argument("--in_channels", type=int, default=3, help="输入通道数")
    parser.add_argument("--out_channels", type=int, default=3, help="输出通道数")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器工作线程数")
    
    # 模型相关参数
    parser.add_argument("--model_dir", type=str, default="./models", help="模型保存目录")
    parser.add_argument("--model_name", type=str, default="vqgan_model", help="模型名称")
    parser.add_argument("--sample_dir", type=str, default="./samples", help="样本保存目录")
    
    # 训练相关参数
    parser.add_argument("--epochs", type=int, default=100, help="训练轮次")
    parser.add_argument("--lr_step_size", type=int, default=30, help="学习率调度器步长")
    parser.add_argument("--lr_gamma", type=float, default=0.5, help="学习率衰减系数")
    parser.add_argument("--log_interval", type=int, default=50, help="日志打印间隔")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="早停耐心值")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--use_amp", action="store_true", help="是否使用混合精度训练")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--skip_optimizer", action="store_true", help="加载检查点时是否跳过优化器状态")
    parser.add_argument("--resume_training", action="store_true", help="是否从检查点恢复训练")
    
    args = parser.parse_args()
    main(args) 