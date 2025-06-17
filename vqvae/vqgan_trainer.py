"""
VQ-GAN 训练器
"""
import torch
import torch.nn.functional as F
from lpips import LPIPS
from accelerate import Accelerator

from .models import PerceptualLoss

class VQGANTrainer:
    def __init__(self, vqgan, discriminator, config, g_optimizer, d_optimizer, accelerator: Accelerator):
        self.vqgan = vqgan
        self.discriminator = discriminator
        self.config = config
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.accelerator = accelerator
        
        # Initialize LPIPS loss. It's important to use a pre-trained model for perceptual loss and keep it in eval mode.
        self.lpips_loss = LPIPS(net='vgg').to(self.accelerator.device).eval()

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=self.accelerator.device, requires_grad=False)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        # We need a tensor of ones with the same shape as the discriminator output
        fake = torch.ones_like(d_interpolates, device=self.accelerator.device, requires_grad=False)
        
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

    def train_step(self, x):
        # Set models to train mode
        self.vqgan.train()
        self.discriminator.train()

        # -------------------
        #  Train Generator
        # -------------------
        self.g_optimizer.zero_grad()

        with self.accelerator.autocast():
            # Perform a full forward pass through the VQGAN to get all necessary outputs
            model_output = self.vqgan(x, return_dict=True)
            reconstructed_x = model_output["sample"]
            
            # Extract losses and metrics from the model output.
            reconstruction_loss = model_output["reconstruction_loss"]
            vq_embed_loss = model_output["vq_loss"]
            perplexity = model_output["perplexity"]

            # Perceptual loss
            # Make sure input and target are in the expected range for LPIPS ([-1, 1])
            perceptual_loss = self.lpips_loss(reconstructed_x, x)

            # Discriminator forward pass for generator loss
            # We want the discriminator to think the fake images are real
            disc_fake_pred = self.discriminator(reconstructed_x)
            g_loss_adv = -torch.mean(disc_fake_pred)
            
            # Combine losses for the generator
            g_loss = (
                self.config.reconstruction_loss_weight * reconstruction_loss +
                self.config.perceptual_loss_weight * perceptual_loss +
                self.config.g_loss_adv_weight * g_loss_adv +
                self.config.vq_embed_loss_weight * vq_embed_loss
            )

        self.accelerator.backward(g_loss)
        self.g_optimizer.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        self.d_optimizer.zero_grad()

        with self.accelerator.autocast():
            # Detach reconstructed_x to avoid backpropagating through the generator
            disc_fake_pred = self.discriminator(reconstructed_x.detach())
            disc_real_pred = self.discriminator(x)

            # WGAN-GP gradient penalty
            gradient_penalty = self.compute_gradient_penalty(x, reconstructed_x.detach())

            # Discriminator loss based on WGAN-GP
            d_loss = torch.mean(disc_fake_pred) - torch.mean(disc_real_pred) + self.config.gradient_penalty_weight * gradient_penalty
        
        self.accelerator.backward(d_loss)
        self.d_optimizer.step()

        return {
            "g_loss": g_loss.item(),
            "d_loss": d_loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "perceptual_loss": perceptual_loss.item(),
            "g_loss_adv": g_loss_adv.item(),
            "vq_embed_loss": vq_embed_loss.item(),
            "gradient_penalty": gradient_penalty.item(),
            "perplexity": perplexity.item()
        }, reconstructed_x

    def train_step_fp16(self, images, vq_scaler, disc_scaler):
        # FP16的实现相对复杂，此处暂时留空
        # 推荐在验证GAN训练稳定后再实现
        raise NotImplementedError("FP16 training for VQ-GAN is not implemented yet.") 