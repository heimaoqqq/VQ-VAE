"""
VQ-GAN 训练器
"""
import torch
import torch.nn.functional as F
from lpips import LPIPS
from accelerate import Accelerator
from torch.amp import autocast

from .models import PerceptualLoss

class VQGANTrainer:
    def __init__(self, vqgan, discriminator, perceptual_loss, optimizer_g, optimizer_d, adv_weight=0.5, commitment_weight=0.25, gp_weight=10.0, device='cuda'):
        self.vqgan = vqgan
        self.discriminator = discriminator
        self.perceptual_loss = perceptual_loss
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.adv_weight = adv_weight
        self.commitment_weight = commitment_weight
        self.gp_weight = gp_weight
        self.device = device
        
        # Initialize LPIPS loss. It's important to use a pre-trained model for perceptual loss and keep it in eval mode.
        self.lpips_loss = LPIPS(net='vgg').to(self.device).eval()

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=self.device, requires_grad=False)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        # We need a tensor of ones with the same shape as the discriminator output
        fake = torch.ones_like(d_interpolates, device=self.device, requires_grad=False)
        
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_step(self, images, scaler_g, scaler_d):
        # ====================================================
        # 1. Train the Discriminator
        # ====================================================
        self.discriminator.train()
        self.optimizer_d.zero_grad()

        with autocast(device_type=self.device.type):
            # Generate fake images first to be used in both D and G training
            h = self.vqgan.encoder(images)
            h = self.vqgan.quant_conv(h)
            quant_states, commitment_loss, vq_info = self.vqgan.quantize(h)
            recon_images = self.vqgan.decode(quant_states).sample
            
            # Judge fake images
            d_loss_fake = self.discriminator(recon_images.detach()).mean()
            # Judge real images
            d_loss_real = self.discriminator(images.detach()).mean()
            # Calculate Gradient Penalty
            gradient_penalty = self.compute_gradient_penalty(images.data, recon_images.data)
            # Total Discriminator Loss
            d_loss = d_loss_fake - d_loss_real + self.gp_weight * gradient_penalty
        
        # Scale and backpropagate for Discriminator
        scaler_d.scale(d_loss).backward()
        scaler_d.step(self.optimizer_d)
        scaler_d.update()

        # ====================================================
        # 2. Train the Generator (VQ-GAN)
        # ====================================================
        self.vqgan.train()
        self.optimizer_g.zero_grad()

        # We need perplexity for logging, but don't need its gradient
        with torch.no_grad():
            perplexity = vq_info[1]
            perplexity_item = perplexity.item() if perplexity is not None else -1.0
        
        with autocast(device_type=self.device.type):
            # Adversarial Loss (Generator's goal is to fool the discriminator)
            g_loss_adv = -self.discriminator(recon_images).mean()
            
            # Reconstruction and Perceptual Loss
            recon_loss_l1 = F.l1_loss(recon_images, images)
            recon_loss_perceptual = self.perceptual_loss(recon_images, images)
            recon_loss = recon_loss_l1 + recon_loss_perceptual

            # Total Generator Loss
            g_loss = recon_loss + self.adv_weight * g_loss_adv + self.commitment_weight * commitment_loss
        
        # Scale and backpropagate for Generator
        scaler_g.scale(g_loss).backward()
        scaler_g.step(self.optimizer_g)
        scaler_g.update()
        
        # ====================================================
        # 3. Compile and return metrics (as float32)
        # ====================================================
        return {
            "g_loss": g_loss.item(),
            "d_loss": d_loss.item(),
            "recon_loss": recon_loss.item(),
            "adv_g_loss": g_loss_adv.item(),
            "commitment_loss": commitment_loss.item(),
            "perplexity": perplexity_item,
            "d_loss_real": d_loss_real.item(),
            "d_loss_fake": d_loss_fake.item(),
            "gradient_penalty": gradient_penalty.item(),
        }

    @torch.no_grad()
    def validate_step(self, images):
        self.vqgan.eval()
        
        h = self.vqgan.encoder(images)
        h = self.vqgan.quant_conv(h)
        quant_states, commitment_loss, vq_info = self.vqgan.quantize(h)
        recon_images = self.vqgan.decode(quant_states).sample

        perplexity = vq_info[1]
        perplexity_item = perplexity.item() if perplexity is not None else -1.0

        recon_loss_l1 = F.l1_loss(recon_images, images)
        recon_loss_perceptual = self.perceptual_loss(recon_images, images)
        recon_loss = recon_loss_l1 + recon_loss_perceptual
        
        return {
            "val_recon_loss": recon_loss.item(),
            "val_commitment_loss": commitment_loss.item(),
            "val_perplexity": perplexity_item
        } 