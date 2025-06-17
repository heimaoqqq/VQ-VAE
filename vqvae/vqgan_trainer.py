"""
VQ-GAN 训练器
"""
import torch
import torch.nn.functional as F

from .models import PerceptualLoss

class VQGANTrainer:
    def __init__(self, vq_model, discriminator, vq_optimizer, disc_optimizer, device,
                 lambda_perceptual=0.1, gan_loss_weight=0.8, use_perceptual=True):
        self.vq_model = vq_model
        self.discriminator = discriminator
        self.vq_optimizer = vq_optimizer
        self.disc_optimizer = disc_optimizer
        self.device = device
        self.gan_loss_weight = gan_loss_weight
        
        # 损失函数
        self.perceptual_loss = PerceptualLoss(self.device) if use_perceptual else None
        self.lambda_perceptual = lambda_perceptual

    def train_step(self, images):
        self.vq_model.train()
        self.discriminator.train()

        # ----- 训练生成器 (VQ-VAE) -----
        self.vq_optimizer.zero_grad()
        
        # 重建图像
        reconstructed, vq_loss_dict = self.vq_model(images)
        
        # 计算重建损失
        recon_loss = F.l1_loss(reconstructed, images)
        
        # 计算感知损失
        perceptual_loss = self.perceptual_loss(reconstructed, images) if self.perceptual_loss else torch.tensor(0.0)
        
        # 计算对抗损失 (生成器部分)
        fake_logits = self.discriminator(reconstructed)
        # 目标是让判别器认为生成的图像是"真"的 (即输出全1)
        g_loss = -torch.mean(fake_logits)
        
        # VQ-VAE的总损失
        vq_loss = recon_loss + \
                  vq_loss_dict['loss'] + \
                  self.lambda_perceptual * perceptual_loss + \
                  self.gan_loss_weight * g_loss
                  
        vq_loss.backward()
        self.vq_optimizer.step()

        # ----- 训练判别器 -----
        self.disc_optimizer.zero_grad()
        
        # 对真实图像的判别
        real_logits = self.discriminator(images)
        
        # 对生成图像的判别
        # 使用 .detach() 来避免梯度流回生成器
        fake_logits = self.discriminator(reconstructed.detach())
        
        # WGAN-GP的判别器损失 (一种更稳定的GAN损失)
        # 目标: real_logits -> +inf, fake_logits -> -inf
        disc_loss = torch.mean(fake_logits) - torch.mean(real_logits)
        
        # WGAN-GP的梯度惩罚
        alpha = torch.rand(images.size(0), 1, 1, 1, device=self.device)
        interpolated = (alpha * images.data + (1 - alpha) * reconstructed.data).requires_grad_(True)
        interpolated_logits = self.discriminator(interpolated)
        
        gradients = torch.autograd.grad(
            outputs=interpolated_logits,
            inputs=interpolated,
            grad_outputs=torch.ones(interpolated_logits.size(), device=self.device),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        disc_loss += 10 * gradient_penalty # 梯度惩罚的权重通常为10
        
        disc_loss.backward()
        self.disc_optimizer.step()

        # 返回损失信息和重建图像
        results = {
            'total_vq_loss': vq_loss.item(),
            'recon_loss': recon_loss.item(),
            'vq_embed_loss': vq_loss_dict['loss'].item(),
            'g_loss': g_loss.item(),
            'd_loss': disc_loss.item(),
            'perplexity': vq_loss_dict['perplexity'].item()
        }
        if self.perceptual_loss:
            results['perceptual_loss'] = perceptual_loss.item()
            
        return results, reconstructed.detach()

    def train_step_fp16(self, images, vq_scaler, disc_scaler):
        # FP16的实现相对复杂，此处暂时留空
        # 推荐在验证GAN训练稳定后再实现
        raise NotImplementedError("FP16 training for VQ-GAN is not implemented yet.") 