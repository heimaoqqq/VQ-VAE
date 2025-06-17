"""
VQ-GAN 训练器
"""
import torch
import torch.nn.functional as F
from lpips import LPIPS
from accelerate import Accelerator

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

    def train_step(self, images):
        # ====================================================
        # Manual VQ-GAN forward pass to extract perplexity
        # ====================================================
        h = self.vqgan.encoder(images)
        h = self.vqgan.quant_conv(h)
        # The 'quantize' method in diffusers' VQModel returns:
        # (quant_states, vq_loss_dict, info) where vq_loss_dict is not used here but vq_loss is inside.
        # The vq_loss that VQModel's forward returns is the commitment loss.
        # We need the full output from quantize.
        quant_states, vq_loss_dict, vq_info = self.vqgan.quantize(h)
        recon_images = self.vqgan.decode(quant_states).sample

        commitment_loss = vq_loss_dict.get("commitment_loss", torch.tensor(0.0, device=self.device))
        perplexity = vq_info[2]
        
        # ====================================================
        # 1. Train the Discriminator
        # ====================================================
        self.discriminator.train()
        self.optimizer_d.zero_grad()

        # Judge fake images
        d_loss_fake = self.discriminator(recon_images.detach()).mean()

        # Judge real images
        d_loss_real = self.discriminator(images.detach()).mean()
        
        # Calculate Gradient Penalty
        gradient_penalty = self.compute_gradient_penalty(images.data, recon_images.data)
        
        # Total Discriminator Loss with Gradient Penalty
        d_loss = d_loss_fake - d_loss_real + self.gp_weight * gradient_penalty
        
        d_loss.backward()
        self.optimizer_d.step()

        # ====================================================
        # 2. Train the Generator (VQ-GAN)
        # ====================================================
        self.vqgan.train()
        self.optimizer_g.zero_grad()

        # Reconstruction and Perceptual Loss
        recon_loss_l1 = F.l1_loss(recon_images, images)
        recon_loss_perceptual = self.perceptual_loss(recon_images, images)
        recon_loss = recon_loss_l1 + recon_loss_perceptual

        # Adversarial Loss (Generator's goal is to fool the discriminator)
        g_loss_adv = -self.discriminator(recon_images).mean()

        # Total Generator Loss
        g_loss = recon_loss + self.adv_weight * g_loss_adv + self.commitment_weight * commitment_loss

        g_loss.backward()
        self.optimizer_g.step()
        
        # ====================================================
        # 3. Compile and return metrics
        # ====================================================
        return {
            "g_loss": g_loss.item(),
            "d_loss": d_loss.item(),
            "recon_loss": recon_loss.item(),
            "adv_g_loss": g_loss_adv.item(),
            "commitment_loss": commitment_loss.item(),
            "perplexity": perplexity.item(),
            "d_loss_real": d_loss_real.item(),
            "d_loss_fake": d_loss_fake.item(),
            "gradient_penalty": gradient_penalty.item(),
        }

    @torch.no_grad()
    def validate_step(self, images):
        self.vqgan.eval()
        
        h = self.vqgan.encoder(images)
        h = self.vqgan.quant_conv(h)
        quant_states, vq_loss_dict, vq_info = self.vqgan.quantize(h)
        recon_images = self.vqgan.decode(quant_states).sample

        commitment_loss = vq_loss_dict.get("commitment_loss", torch.tensor(0.0, device=self.device))
        perplexity = vq_info[2]

        recon_loss_l1 = F.l1_loss(recon_images, images)
        recon_loss_perceptual = self.perceptual_loss(recon_images, images)
        recon_loss = recon_loss_l1 + recon_loss_perceptual
        
        return {
            "val_recon_loss": recon_loss.item(),
            "val_commitment_loss": commitment_loss.item(),
            "val_perplexity": perplexity.item()
        }

    def train_step_fp16(self, images, vq_scaler, disc_scaler):
        # FP16的实现相对复杂，此处暂时留空
        # 推荐在验证GAN训练稳定后再实现
        raise NotImplementedError("FP16 training for VQ-GAN is not implemented yet.") 