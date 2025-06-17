"""
VQ-GAN 训练脚本
使用带有对抗性判别器的向量量化自编码器（VQ-GAN）训练微多普勒时频图的生成模型
"""

import os
import argparse
import torch
from torch.optim import AdamW
from tqdm import tqdm
from accelerate import Accelerator
from diffusers import VQModel
from lpips import LPIPS

# 导入数据集
from dataset import get_dataloaders

# 导入自定义模块
from vqvae.discriminator import Discriminator, weights_init
from vqvae.vqgan_trainer import VQGANTrainer
from vqvae.utils import save_reconstructed_images

# 导入其他库
try:
    import wandb
except ImportError:
    print("未找到wandb库，将不使用wandb进行可视化")
    wandb = None

def parse_args():
    parser = argparse.ArgumentParser(description="Train a VQ-GAN model.")
    # Paths
    parser.add_argument("--data_dir", type=str, default="dataset", help="Directory for the dataset.")
    parser.add_argument("--output_dir", type=str, default="vqgan_model_output", help="Directory to save model and logs.")
    
    # Data
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--image_size", type=int, default=256, help="Image size for training.")

    # Training
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the generator (VQ-GAN).")
    parser.add_argument("--disc_lr", type=float, default=1e-4, help="Learning rate for the discriminator.")
    
    # Model Config
    parser.add_argument("--in_channels", type=int, default=3, help="Input channels for the VQ-GAN.")
    parser.add_argument("--out_channels", type=int, default=3, help="Output channels for the VQ-GAN.")
    parser.add_argument("--latent_channels", type=int, default=4, help="Number of channels in the latent space.")
    parser.add_argument("--num_vq_embeddings", type=int, default=8192, help="Number of embeddings in the codebook.")
    parser.add_argument("--block_out_channels", nargs='+', type=int, default=[64, 128, 256], help="Channel configurations for VQ-GAN blocks.")
    
    # Loss Weights
    parser.add_argument("--reconstruction_loss_weight", type=float, default=1.0, help="Weight for reconstruction loss.")
    parser.add_argument("--perceptual_loss_weight", type=float, default=1.0, help="Weight for perceptual loss.")
    parser.add_argument("--g_loss_adv_weight", type=float, default=0.8, help="Weight for adversarial generator loss.")
    parser.add_argument("--vq_embed_loss_weight", type=float, default=1.0, help="Weight for VQ embedding loss (commitment loss).")
    parser.add_argument("--gradient_penalty_weight", type=float, default=10.0, help="Weight for WGAN-GP gradient penalty.")

    # Logging & Saving
    parser.add_argument("--save_epochs", type=int, default=20, help="Save model every N epochs.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log metrics every N steps.")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging.")
    parser.add_argument("--wandb_project", type=str, default="vq-gan-microdoppler", help="Wandb project name.")
    parser.add_argument("--wandb_name", type=str, default="vqgan-training-run", help="Wandb run name.")

    return parser.parse_args()

def train_vqgan(args):
    accelerator = Accelerator(log_with="wandb" if args.use_wandb else None)
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        images_dir = os.path.join(args.output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        if args.use_wandb:
            accelerator.init_trackers(
                project_name=args.wandb_project,
                config=vars(args),
                init_kwargs={"wandb": {"name": args.wandb_name}}
            )

    train_dataloader, val_dataloader, _ = get_dataloaders(args.data_dir, args.batch_size, args.image_size)
    
    if accelerator.is_main_process:
        print(f"找到 {len(train_dataloader.dataset) + len(val_dataloader.dataset)} 张图像")
        total_samples = len(train_dataloader.dataset) + len(val_dataloader.dataset)
        train_samples = len(train_dataloader.dataset)
        val_samples = len(val_dataloader.dataset)
        print("数据集统计信息:")
        print(f"总样本数: {total_samples}")
        print(f"训练集: {train_samples}个样本 ({train_samples/total_samples*100:.1f}%), {len(train_dataloader)}个批次")
        print(f"验证集: {val_samples}个样本 ({val_samples/total_samples*100:.1f}%), {len(val_dataloader)}个批次")
        print(f"批次大小: {args.batch_size}")

    # Create VQ-GAN (Generator)
    vqgan = VQModel(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        latent_channels=args.latent_channels,
        num_vq_embeddings=args.num_vq_embeddings,
        vq_embed_dim=args.latent_channels,
        block_out_channels=args.block_out_channels,
    )

    # Create Discriminator
    discriminator = Discriminator(input_channels=args.in_channels, n_layers=len(args.block_out_channels))
    discriminator.apply(weights_init)

    # Optimizers
    g_optimizer = AdamW(vqgan.parameters(), lr=args.lr, betas=(0.5, 0.9))
    d_optimizer = AdamW(discriminator.parameters(), lr=args.disc_lr, betas=(0.5, 0.9))
    
    # Prepare with accelerator
    vqgan, discriminator, g_optimizer, d_optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        vqgan, discriminator, g_optimizer, d_optimizer, train_dataloader, val_dataloader
    )

    # Initialize the trainer
    trainer = VQGANTrainer(
        vqgan=vqgan,
        discriminator=discriminator,
        config=args,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        accelerator=accelerator
    )

    global_step = 0
    for epoch in range(args.epochs):
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.epochs}", disable=not accelerator.is_local_main_process)
        
        for step, batch in enumerate(train_dataloader):
            results, reconstructed = trainer.train_step(batch)
            
            progress_bar.set_postfix({
                "G_Loss": f"{results['g_loss']:.3f}",
                "D_Loss": f"{results['d_loss']:.3f}",
                "Recon": f"{results['reconstruction_loss']:.3f}",
                "Perceptual": f"{results['perceptual_loss']:.3f}",
                "Adv_G": f"{results['g_loss_adv']:.3f}",
                "VQ": f"{results['vq_embed_loss']:.3f}",
            })
            progress_bar.update(1)
            
            if accelerator.is_main_process:
                if global_step % args.logging_steps == 0:
                    accelerator.log(results, step=global_step)
                
                if global_step % args.logging_steps == 0:
                    save_reconstructed_images(batch, reconstructed, epoch, global_step, images_dir)

            global_step += 1
        
        progress_bar.close()

        if accelerator.is_main_process and (epoch + 1) % args.save_epochs == 0:
            unwrapped_vqgan = accelerator.unwrap_model(vqgan)
            unwrapped_discriminator = accelerator.unwrap_model(discriminator)
            
            vqgan_path = os.path.join(args.output_dir, f"vqgan_epoch_{epoch+1}.pt")
            discriminator_path = os.path.join(args.output_dir, f"discriminator_epoch_{epoch+1}.pt")

            torch.save(unwrapped_vqgan.state_dict(), vqgan_path)
            torch.save(unwrapped_discriminator.state_dict(), discriminator_path)
            accelerator.print(f"Epoch {epoch+1}: Models saved to {args.output_dir}")

if __name__ == "__main__":
    args = parse_args()
    train_vqgan(args) 