"""
VQ-GAN 训练脚本 (已重构)
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
import itertools

from vqvae.custom_vqgan import CustomVQGAN
from vqvae.discriminator import Discriminator
from vqvae.vqgan_trainer import VQGANTrainer
from dataset import MicroDopplerDataset

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

    # Create VQ-GAN model using our custom implementation
    model = CustomVQGAN(
        in_channels=3, 
        out_channels=3,
        block_out_channels=[128, 256, 512],
        latent_channels=config.vq_embed_dim, # This is the channel depth of the encoder's output
        num_vq_embeddings=config.vq_num_embed,
        vq_embed_dim=config.vq_embed_dim # This is the dimension of each codebook vector
    ).to(device)
    
    # Create Discriminator
    discriminator = Discriminator(input_channels=3, n_layers=3, n_filters_start=config.disc_channels).to(device)

    # Optimizers for Generator (VQ-Model) and Discriminator
    # IMPORTANT: We exclude the codebook parameters from the generator's optimizer,
    # as it will be updated by EMA.
    encoder_decoder_params = itertools.chain(
        model.encoder.parameters(),
        model.decoder.parameters(),
        model.quant_conv.parameters(),
        model.post_quant_conv.parameters()
    )
    optimizer_g = torch.optim.Adam(encoder_decoder_params, lr=config.lr, betas=(config.beta1, config.beta2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.lr_d, betas=(config.beta1, config.beta2))
    
    # Learning Rate Schedulers
    scheduler_g = StepLR(optimizer_g, step_size=config.lr_decay_step, gamma=0.5)
    scheduler_d = StepLR(optimizer_d, step_size=config.lr_decay_step, gamma=0.5)

    # Create output directory for checkpoints
    os.makedirs(config.output_dir, exist_ok=True)
    # The trainer will save the best model to this path
    checkpoint_path = os.path.join(config.output_dir, "vqgan_model_best.pt")
    sample_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create Trainer
    trainer = VQGANTrainer(
        vqgan=model,
        discriminator=discriminator,
        g_optimizer=optimizer_g,
        d_optimizer=optimizer_d,
        lr_scheduler_g=scheduler_g,
        lr_scheduler_d=scheduler_d,
        device=device,
        use_amp=use_amp,
        checkpoint_path=checkpoint_path,
        sample_dir=sample_dir,
        lambda_gp=config.gp_weight,
        l1_weight=config.l1_weight,
        perceptual_weight=config.perceptual_weight,
        adversarial_weight=config.adversarial_weight,
        log_interval=config.log_interval
    )

    # Start training
    print("Starting training...")
    trainer.train(train_loader, val_loader, config.epochs, smoke_test=config.smoke_test)
    print("Training finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a VQ-GAN model.")
    # Paths and dirs
    parser.add_argument('--data_path', type=str, default='data/processed', help='Path to the processed data')
    parser.add_argument('--output_dir', type=str, default='checkpoints/vqgan_final', help='Directory to save checkpoints and samples')
    
    # Training params
    parser.add_argument('--image_size', type=int, default=256, help='Size to resize images to')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate for generator')
    parser.add_argument('--lr_d', type=float, default=2.5e-4, help='Learning rate for discriminator')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta2')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--lr_decay_step', type=int, default=100, help='Step size for LR decay')
    parser.add_argument('--log_interval', type=int, default=100, help='How many batches to wait before logging training status')
        
    # Model params
    parser.add_argument('--vq_embed_dim', type=int, default=256, help='Dimension of the codebook embeddings')
    parser.add_argument('--vq_num_embed', type=int, default=8192, help='Number of codebook embeddings')
    parser.add_argument('--disc_channels', type=int, default=64, help='Initial channels for discriminator')
    parser.add_argument('--vq_channels', type=int, default=256, help='Number of channels in VQ latent space')
        
    # Loss weights
    parser.add_argument('--l1_weight', type=float, default=1.0, help='Weight for L1 reconstruction loss')
    parser.add_argument('--perceptual_weight', type=float, default=1.0, help='Weight for perceptual loss')
    parser.add_argument('--gp_weight', type=float, default=10.0, help='Weight for gradient penalty')
    parser.add_argument('--adversarial_weight', type=float, default=1.5, help='Weight for adversarial loss')
            
    # Dataset params
    parser.add_argument('--val_split_ratio', type=float, default=0.05, help='Ratio of dataset to be used for validation')

    # Utility params
    parser.add_argument('--smoke-test', action='store_true', help='Run a quick smoke test with a few batches instead of a full training.')

    config = parser.parse_args()
    main(config) 