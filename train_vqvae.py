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
from vqvae.models.micro_doppler_encoder import MicroDopplerEncoder
from vqvae.models.micro_doppler_decoder import MicroDopplerDecoder
from diffusers.models.autoencoders.vae import Encoder, Decoder

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
        embed_dim=config.vq_embed_dim,
        n_embed=config.vq_num_embed,
        commitment_loss_beta=config.commitment_loss_beta,
        ema_decay=config.ema_decay
    ).to(device)
    
    # 根据命令行参数选择使用自定义编码器/解码器
    if config.use_micro_doppler_encoder:
        print("使用微多普勒专用编码器")
        model.encoder = MicroDopplerEncoder(
            in_channels=3,
            latent_dim=config.vq_embed_dim,
            base_channels=64
        ).to(device)
    
    if config.use_micro_doppler_decoder:
        print("使用微多普勒专用解码器")
        model.decoder = MicroDopplerDecoder(
            in_channels=config.vq_embed_dim,
            out_channels=3,
            base_channels=64
        ).to(device)
    
    # Create Discriminator
    discriminator = Discriminator(input_channels=3, n_layers=3, n_filters_start=config.disc_channels).to(device)
    
    # Optimizers for Generator (VQ-Model) and Discriminator
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
    checkpoint_path = os.path.join(config.output_dir, "vqgan_model_best.pt")
    
    # 如果指定了自定义检查点路径，使用它
    if config.checkpoint_path:
        checkpoint_path = config.checkpoint_path
        print(f"使用自定义检查点路径: {checkpoint_path}")
    
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
        entropy_weight=config.entropy_weight,
        log_interval=config.log_interval,
        reset_low_usage_interval=config.reset_low_usage_interval,
        reset_low_usage_percentage=config.reset_low_usage_percentage
    )

    # Start training
    print("Starting training...")
    trainer.train(
        train_loader, 
        val_loader, 
        config.epochs, 
        early_stopping_patience=config.early_stopping_patience,
        skip_optimizer=config.skip_optimizer,
        resume_training=not config.fresh_start
    )
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
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Patience for early stopping. Set to 0 to disable.')
        
    # Model params
    parser.add_argument('--vq_embed_dim', type=int, default=256, help='Dimension of the codebook embeddings')
    parser.add_argument('--vq_num_embed', type=int, default=512, help='Number of codebook embeddings (默认值从8192降低为512)')
    parser.add_argument('--disc_channels', type=int, default=64, help='Initial channels for discriminator')
    parser.add_argument('--commitment_loss_beta', type=float, default=3.0, help='Commitment loss beta factor (从2.0增加到3.0)')
    parser.add_argument('--ema_decay', type=float, default=0.95, help='EMA decay rate for codebook updates (从0.999降低为0.95)')
    parser.add_argument('--reset_low_usage_interval', type=int, default=5, help='每多少个epoch重置低使用率码元')
    parser.add_argument('--reset_low_usage_percentage', type=float, default=0.1, help='重置使用率最低的码元比例')
    parser.add_argument('--use_micro_doppler_encoder', action='store_true', help='使用微多普勒专用编码器')
    parser.add_argument('--use_micro_doppler_decoder', action='store_true', help='使用微多普勒专用解码器')
        
    # Loss weights
    parser.add_argument('--l1_weight', type=float, default=0.6, help='Weight for L1 reconstruction loss')
    parser.add_argument('--perceptual_weight', type=float, default=0.005, help='Weight for perceptual loss (从0.01降低为0.005)')
    parser.add_argument('--adversarial_weight', type=float, default=0.8, help='Weight for adversarial loss')
    parser.add_argument('--gp_weight', type=float, default=10.0, help='Weight for gradient penalty in WGAN-GP')
    parser.add_argument('--entropy_weight', type=float, default=0.6, help='Weight for codebook entropy regularization (从0.1增加到0.6)')
            
    # Dataset params
    parser.add_argument('--val_split_ratio', type=float, default=0.05, help='Ratio of dataset to be used for validation')

    # 恢复训练选项
    parser.add_argument('--checkpoint_path', type=str, default=None, help='自定义检查点文件路径')
    parser.add_argument('--skip_optimizer', action='store_true', help='跳过加载优化器状态')
    parser.add_argument('--fresh_start', action='store_true', help='从头开始训练，忽略现有检查点')

    config = parser.parse_args()
    main(config) 