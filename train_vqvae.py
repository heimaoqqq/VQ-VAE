"""
VQ-GAN 训练脚本 (已重构)
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR

from diffusers import UNet2DModel, VQModel, AutoencoderKL
from diffusers.optimization import get_scheduler
from vqvae.discriminator import PatchGANDiscriminator
from vqvae.vqgan_trainer import VQGANTrainer
from dataset import MicroDopplerDataset

def create_vq_model(config):
    """
    Creates an Autoencoder with a VQ-quantizer.
    We use AutoencoderKL as a backbone and then manually attach quantizer parts
    from a dummy VQModel to bypass environment/import issues.
    """
    # 1. Create the AutoencoderKL backbone
    model = AutoencoderKL(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(128, 256, 512, config.latent_channels),
        layers_per_block=2,
        latent_channels=config.latent_channels,
        norm_num_groups=32,
    )

    # 2. Create a dummy VQModel instance to "steal" its quantizer components
    dummy_vq_model = VQModel(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        latent_channels=config.latent_channels,
        num_vq_embeddings=config.vq_num_embed,
        vq_embed_dim=config.latent_channels,
    )

    # 3. Steal the components and attach them to our main model
    model.quantize = dummy_vq_model.quantize
    model.quant_conv = dummy_vq_model.quant_conv
    model.post_quant_conv = dummy_vq_model.post_quant_conv

    return model

def main(config):
    # Setup device and AMP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == 'cuda'
    print(f"Using device: {device}, AMP enabled: {use_amp}")

    # Create dataset and dataloaders
    dataset = MicroDopplerDataset(config.data_path, image_size=config.image_size)
    
    train_size = int(len(dataset) * (1.0 - config.val_split_ratio))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")

    # Create VQ-GAN model (Generator part)
    print("Setting up models...")
    vq_model = create_vq_model(config).to(device)
    discriminator = PatchGANDiscriminator(input_channels=config.in_channels).to(device)

    # Optimizers for Generator (VQ-Model) and Discriminator
    optimizer_g = torch.optim.Adam(vq_model.parameters(), lr=config.lr, betas=(0.5, 0.9))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.lr, betas=(0.5, 0.9))

    # Learning Rate Schedulers
    scheduler_g = StepLR(optimizer_g, step_size=config.lr_decay_step, gamma=0.5)
    scheduler_d = StepLR(optimizer_d, step_size=config.lr_decay_step, gamma=0.5)

    # Create output directory for checkpoints
    checkpoints_dir = os.path.join(config.output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoints_dir, "checkpoint.pt")
    
    # Create Trainer
    trainer = VQGANTrainer(
        vqvae=vq_model,
        discriminator=discriminator,
        g_optimizer=optimizer_g,
        d_optimizer=optimizer_d,
        lr_scheduler_g=scheduler_g,
        lr_scheduler_d=scheduler_d,
        device=device,
        use_amp=use_amp,
        checkpoint_path=checkpoint_path,
        l1_weight=config.l1_weight,
        perceptual_weight=config.perceptual_weight,
        lambda_gp=config.gp_weight
    )

    # Start training
    print("Starting training...")
    trainer.train(train_loader, val_loader, config.num_epochs)
    print("Training finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a VQ-GAN model.")
    # Paths and dirs
    parser.add_argument('--data_path', type=str, default='data/processed', help='Path to the processed data')
    parser.add_argument('--output_dir', type=str, default='checkpoints/vqgan_final', help='Directory to save checkpoints and samples')
    
    # Training params
    parser.add_argument('--image_size', type=int, default=256, help='Size to resize images to')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for optimizers')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--lr_decay_step', type=int, default=100, help='Step size for LR decay')

    # Model params
    parser.add_argument('--in_channels', type=int, default=3, help='Input channels for the VQ-VAE model')
    parser.add_argument('--out_channels', type=int, default=3, help='Output channels for the VQ-VAE model')
    parser.add_argument('--latent_channels', type=int, default=512, help='Number of channels in the latent space')
    parser.add_argument('--vq_num_embed', type=int, default=8192, help='Number of codebook embeddings')
    parser.add_argument('--disc_channels', type=int, default=64, help='Initial channels for discriminator')

    # Loss weights
    parser.add_argument('--l1_weight', type=float, default=1.0, help='Weight for L1 reconstruction loss')
    parser.add_argument('--perceptual_weight', type=float, default=1.0, help='Weight for perceptual loss')
    parser.add_argument('--gp_weight', type=float, default=10.0, help='Weight for gradient penalty')

    # Dataset params
    parser.add_argument('--val_split_ratio', type=float, default=0.05, help='Ratio of dataset to be used for validation')

    config = parser.parse_args()
    main(config) 