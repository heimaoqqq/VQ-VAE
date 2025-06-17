"""
VQ-GAN 训练脚本
使用带有对抗性判别器的向量量化自编码器（VQ-GAN）训练微多普勒时频图的生成模型
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from tqdm import tqdm
import collections

from diffusers import VQModel
from vqvae.models import Discriminator, PerceptualLoss
from vqvae.vqgan_trainer import VQGANTrainer
from dataset import create_dataset

def main(config):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset and dataloaders
    dataset = create_dataset(config.data_path, image_size=config.image_size)
    
    # Split dataset into training and validation
    # Use a fixed generator for reproducibility
    train_split = int(len(dataset) * (1.0 - config.val_split_ratio))
    val_split = len(dataset) - train_split
    train_dataset, val_dataset = random_split(dataset, [train_split, val_split], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # Create VQ-GAN model (Generator part)
    vq_model = VQModel(
        in_channels=1,
        out_channels=1,
        down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
        block_out_channels=[128, 256, 512],
        latent_channels=config.vq_embed_dim, # diffusers uses latent_channels for the VQ embedding dimension
        num_vq_embeddings=config.vq_num_embed,
        vq_embed_dim=config.vq_embed_dim,
    ).to(device)

    # Create Discriminator
    discriminator = Discriminator(
        in_channels=1,
        num_layers=3,
        initial_channels=config.disc_channels,
    ).to(device)

    # Create Perceptual Loss
    perceptual_loss = PerceptualLoss().to(device)

    # Optimizers
    optimizer_g = torch.optim.Adam(vq_model.parameters(), lr=config.lr, betas=(0.5, 0.9))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.lr, betas=(0.5, 0.9))

    # Create Trainer
    trainer = VQGANTrainer(
        vqgan=vq_model,
        discriminator=discriminator,
        perceptual_loss=perceptual_loss,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        adv_weight=config.adv_weight,
        commitment_weight=config.commitment_weight,
        gp_weight=config.gp_weight,
        device=device
    )

    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(config.output_dir, "checkpoints")
    samples_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    # Get a fixed batch from validation set for consistent visualization
    fixed_val_images, _ = next(iter(val_loader))
    fixed_val_images = fixed_val_images.to(device)

    best_val_recon_loss = float('inf')

    print("Starting training...")
    for epoch in range(config.num_epochs):
        # =======================
        #      Training
        # =======================
        vq_model.train()
        discriminator.train()
        
        train_metrics = collections.defaultdict(float)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]")
        for i, (images, _) in enumerate(progress_bar):
            images = images.to(device)
            metrics_dict = trainer.train_step(images)
            
            for k, v in metrics_dict.items():
                train_metrics[k] += v
            
            avg_metrics = {k: f"{v / (i + 1):.4f}" for k, v in train_metrics.items()}
            progress_bar.set_postfix(avg_metrics)

        # =======================
        #     Validation
        # =======================
        vq_model.eval()
        discriminator.eval()
        
        val_metrics = collections.defaultdict(float)
        
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Val]")
        with torch.no_grad():
            for i, (images, _) in enumerate(val_progress_bar):
                images = images.to(device)
                metrics_dict = trainer.validate_step(images)

                for k, v in metrics_dict.items():
                    val_metrics[k] += v
                
                avg_metrics = {k: f"{v / (i + 1):.4f}" for k, v in val_metrics.items()}
                val_progress_bar.set_postfix(avg_metrics)

        # Log metrics for the epoch
        print(f"\n--- Epoch {epoch+1} Summary ---")
        train_log_str = " | ".join([f"Train {k}: {v / len(train_loader):.4f}" for k, v in train_metrics.items()])
        print(train_log_str)
        val_log_str = " | ".join([f"Val {k}: {v / len(val_loader):.4f}" for k, v in val_metrics.items()])
        print(val_log_str)
        print("---------------------------\n")

        # Save a sample of reconstructed images from the fixed validation batch
        with torch.no_grad():
            recon_images = vq_model(fixed_val_images).sample
        
        comparison = torch.cat([fixed_val_images[:8], recon_images[:8]])
        save_image(comparison.cpu(), os.path.join(samples_dir, f"recon_epoch_{epoch+1}.png"), nrow=8)

        # Save model checkpoint if it's the best one on the validation set
        current_val_recon_loss = val_metrics["val_recon_loss"] / len(val_loader)
        if current_val_recon_loss < best_val_recon_loss:
            best_val_recon_loss = current_val_recon_loss
            checkpoint_path = os.path.join(checkpoints_dir, "best_model.pt")
            torch.save({
                'vq_model_state_dict': vq_model.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'epoch': epoch + 1,
                'config': config,
                'val_recon_loss': best_val_recon_loss
            }, checkpoint_path)
            print(f"Epoch {epoch+1}: New best model saved with Val Recon Loss: {best_val_recon_loss:.4f} to {checkpoint_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a VQ-GAN model with detailed logging.")
    parser.add_argument('--data_path', type=str, default='data/processed', help='Path to the processed data')
    parser.add_argument('--output_dir', type=str, default='checkpoints/vqgan_final', help='Directory to save checkpoints and samples')
    parser.add_argument('--image_size', type=int, default=256, help='Size to resize images to')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for generator and discriminator')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of training epochs')

    # VQ-GAN Model parameters
    parser.add_argument('--vq_embed_dim', type=int, default=256, help='Dimension of the codebook embeddings')
    parser.add_argument('--vq_num_embed', type=int, default=8192, help='Number of codebook embeddings')
    
    # Discriminator parameters
    parser.add_argument('--disc_channels', type=int, default=64, help='Initial channels for discriminator')

    # Loss weights
    parser.add_argument('--adv_weight', type=float, default=0.8, help='Weight for adversarial loss')
    parser.add_argument('--commitment_weight', type=float, default=0.25, help='Weight for VQ commitment loss')
    parser.add_argument('--gp_weight', type=float, default=10.0, help='Weight for gradient penalty')

    # Dataset split
    parser.add_argument('--val_split_ratio', type=float, default=0.05, help='Ratio of dataset to be used for validation')

    config = parser.parse_args()
    main(config) 