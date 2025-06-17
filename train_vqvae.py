"""
VQ-GAN 训练脚本
使用带有对抗性判别器的向量量化自编码器（VQ-GAN）训练微多普勒时频图的生成模型
"""

import os
import argparse
import torch
import shutil
from torch.optim import AdamW
from tqdm import tqdm

# 导入数据集
from dataset import get_dataloaders

# 导入自定义模块
from vqvae.models import create_vq_model
from vqvae.discriminator import Discriminator, weights_init
from vqvae.vqgan_trainer import VQGANTrainer
from vqvae.utils import save_reconstructed_images, validate

# 导入其他库
try:
    import wandb
except ImportError:
    print("未找到wandb库，将不使用wandb进行可视化")
    wandb = None

def parse_args():
    parser = argparse.ArgumentParser(description="训练VQ-GAN模型")
    parser.add_argument("--data_dir", type=str, default="dataset", help="数据集目录")
    parser.add_argument("--output_dir", type=str, default="vqgan_model", help="模型保存目录")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小 (GAN训练可能需要更小批次)")
    parser.add_argument("--image_size", type=int, default=256, help="图像尺寸")
    parser.add_argument("--epochs", type=int, default=300, help="训练轮数 (GAN需要更多轮数收敛)")
    parser.add_argument("--lr", type=float, default=1e-4, help="生成器学习率")
    parser.add_argument("--save_epochs", type=int, default=20, help="每多少个epoch保存一次模型")
    parser.add_argument("--logging_steps", type=int, default=100, help="日志间隔步数")
    parser.add_argument("--latent_channels", type=int, default=4, help="潜变量通道数")
    parser.add_argument("--vq_embed_dim", type=int, default=4, help="VQ嵌入维度")
    parser.add_argument("--vq_num_embed", type=int, default=8192, help="VQ嵌入数量")
    parser.add_argument("--n_layers", type=int, default=3, help="下采样层数")
    parser.add_argument("--save_images", action="store_true", help="是否保存重建图像")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb记录训练")
    parser.add_argument("--wandb_project", type=str, default="vq-gan-microdoppler", help="wandb项目名")
    parser.add_argument("--wandb_name", type=str, default="vqgan-training", help="wandb运行名")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")
    
    # GAN 和 感知损失相关参数
    parser.add_argument("--no_perceptual", action="store_true", help="不使用感知损失")
    parser.add_argument("--lambda_perceptual", type=float, default=0.1, help="感知损失权重")
    parser.add_argument("--gan_loss_weight", type=float, default=0.8, help="对抗损失权重")
    parser.add_argument("--disc_lr", type=float, default=1e-4, help="判别器学习率")

    return parser.parse_args()

def train_vqgan(args):
    """训练VQ-GAN模型"""
    # 创建输出目录和可视化目录
    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # 初始化wandb
    if args.use_wandb and wandb is not None:
        wandb.init(project=args.wandb_project, name=args.wandb_name)
        wandb.config.update(args)

    # 创建数据加载器
    train_dataloader, val_dataloader, _ = get_dataloaders(
        args.data_dir, args.batch_size, args.image_size
    )
    
    # 打印数据集划分信息
    total_samples = len(train_dataloader.dataset) + len(val_dataloader.dataset)
    
    print(f"数据集统计信息:")
    print(f"总样本数: {total_samples}")
    print(f"训练集: {len(train_dataloader.dataset)}个样本 ({len(train_dataloader.dataset)/total_samples*100:.1f}%), {len(train_dataloader)}个批次")
    print(f"验证集: {len(val_dataloader.dataset)}个样本 ({len(val_dataloader.dataset)/total_samples*100:.1f}%), {len(val_dataloader)}个批次")
    print(f"批次大小: {args.batch_size}")
    
    # 创建 VQ-VAE (生成器)
    vq_model = create_vq_model(args).to(args.device)
    
    # 创建判别器
    discriminator = Discriminator(input_channels=3, n_layers=3).to(args.device)
    discriminator.apply(weights_init)

    # 创建优化器
    vq_optimizer = AdamW(vq_model.parameters(), lr=args.lr, betas=(0.5, 0.9))
    disc_optimizer = AdamW(discriminator.parameters(), lr=args.disc_lr, betas=(0.5, 0.9))
    
    # 创建训练器
    trainer = VQGANTrainer(
        vq_model=vq_model,
        discriminator=discriminator,
        vq_optimizer=vq_optimizer,
        disc_optimizer=disc_optimizer,
        device=args.device,
        lambda_perceptual=args.lambda_perceptual,
        gan_loss_weight=args.gan_loss_weight,
        use_perceptual=not args.no_perceptual
    )
    
    # 训练循环
    global_step = 0
    for epoch in range(args.epochs):
        vq_model.train()
        discriminator.train()
        
        progress_bar = tqdm(total=len(train_dataloader), 
                           desc=f"Epoch {epoch+1}/{args.epochs}", 
                           leave=True,
                           ncols=120)
        
        for step, batch in enumerate(train_dataloader):
            images = batch.to(args.device)
            
            # 训练步骤
            results, reconstructed = trainer.train_step(images)
            
            # 更新进度条
            status_dict = {
                "g_loss": f"{results['g_loss']:.3f}",
                "d_loss": f"{results['d_loss']:.3f}",
                "recon": f"{results['recon_loss']:.3f}",
                "perp": f"{results['perplexity']:.1f}"
            }
            progress_bar.set_postfix(status_dict)
            progress_bar.update(1)
            
            # wandb记录
            if args.use_wandb and wandb is not None and global_step % args.logging_steps == 0:
                wandb.log(results, step=global_step)
            
            # 保存重建图像
            if args.save_images and global_step % args.logging_steps == 0:
                img_path = save_reconstructed_images(
                    images.cpu(), reconstructed.cpu(), epoch, global_step, images_dir
                )
                if args.use_wandb and wandb is not None:
                    wandb.log({"reconstruction": wandb.Image(img_path)}, step=global_step)

            global_step += 1
        
        progress_bar.close()

        # 验证 (可选, GAN的验证通常是看图)
        # ...

        # 保存模型
        if (epoch + 1) % args.save_epochs == 0:
            vq_model_path = os.path.join(args.output_dir, f"vq_model_epoch_{epoch+1}.pt")
            disc_path = os.path.join(args.output_dir, f"discriminator_epoch_{epoch+1}.pt")
            torch.save(vq_model.state_dict(), vq_model_path)
            torch.save(discriminator.state_dict(), disc_path)
            print(f"Epoch {epoch+1} 模型已保存。")

if __name__ == "__main__":
    args = parse_args()
    # FP16对于GAN训练不稳定, 默认不使用
    train_vqgan(args) 