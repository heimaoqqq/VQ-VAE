"""
VQ-GAN 训练器 (最终修复版)
"""
import torch
import torch.nn.functional as F
import os
import shutil
from tqdm import tqdm
from lpips import LPIPS

class VQGANTrainer:
    def __init__(self, vqvae, discriminator, g_optimizer, d_optimizer, lr_scheduler_g, lr_scheduler_d, device, use_amp, checkpoint_path, l1_weight=1.0, perceptual_weight=1.0, lambda_gp=10.0):
        self.vqvae = vqvae
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.lr_scheduler_g = lr_scheduler_g
        self.lr_scheduler_d = lr_scheduler_d
        self.device = device
        self.use_amp = use_amp
        
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.lambda_gp = lambda_gp

        # Checkpoint paths
        self.checkpoint_dir = os.path.dirname(checkpoint_path)
        self.checkpoint_path = checkpoint_path # This is now the 'latest' checkpoint
        self.best_model_path = os.path.join(self.checkpoint_dir, "best_model.pt")

        self.best_val_loss = float('inf')
        self.start_epoch = 1
        
        self.perceptual_loss = LPIPS(net='vgg').to(self.device).eval()
        for param in self.perceptual_loss.parameters():
            param.requires_grad = False

        self.g_scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.d_scaler = torch.amp.GradScaler(enabled=self.use_amp)

    def _get_vq_output(self, real_imgs):
        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            h = self.vqvae.encoder(real_imgs)
            h = self.vqvae.quant_conv(h)
            _, commitment_loss, vq_info = self.vqvae.quantize(h)
            perplexity = vq_info['perplexity']
            quantized = vq_info['quantized']
            decoded_imgs = self.vqvae.decoder(self.vqvae.post_quant_conv(quantized))
        return decoded_imgs, commitment_loss, perplexity

    def _train_batch(self, batch):
        real_imgs = batch
        decoded_imgs, commitment_loss, perplexity = self._get_vq_output(real_imgs)

        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            l1_loss = F.l1_loss(decoded_imgs, real_imgs)
            perceptual_loss = self.perceptual_loss(decoded_imgs, real_imgs).mean()

        # Train Discriminator
        self.d_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            real_output = self.discriminator(real_imgs)
            fake_output = self.discriminator(decoded_imgs.detach())
        gradient_penalty = self.compute_gradient_penalty(real_imgs, decoded_imgs.detach())
        d_loss = fake_output.mean() - real_output.mean() + self.lambda_gp * gradient_penalty
        self.d_scaler.scale(d_loss).backward()
        self.d_scaler.step(self.d_optimizer)
        self.d_scaler.update()

        # Train Generator
        self.g_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            g_output = self.discriminator(decoded_imgs)
            g_loss_adv = -g_output.mean()
            g_loss = self.l1_weight * l1_loss + self.perceptual_weight * perceptual_loss + g_loss_adv + commitment_loss
        self.g_scaler.scale(g_loss).backward()
        self.g_scaler.step(self.g_optimizer)
        self.g_scaler.update()

        return {'L1': l1_loss.item(), 'Perceptual': perceptual_loss.item(), 'Adv': g_loss_adv.item(),
                'Commit': commitment_loss.item(), 'D': d_loss.item(), 'GP': gradient_penalty.item(),
                'Perplexity': perplexity.item()}

    def _validate_batch(self, batch):
        real_imgs = batch
        decoded_imgs, commitment_loss, perplexity = self._get_vq_output(real_imgs)
        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            l1_loss = F.l1_loss(decoded_imgs, real_imgs)
            perceptual_loss = self.perceptual_loss(decoded_imgs, real_imgs).mean()
        return {'L1': l1_loss.item(), 'Perceptual': perceptual_loss.item(),
                'Commit': commitment_loss.item(), 'Perplexity': perplexity.item()}

    def _run_epoch(self, dataloader, is_train, epoch, num_epochs):
        phase = "Train" if is_train else "Validation"
        self.vqvae.train(is_train)
        self.discriminator.train(is_train)
        
        epoch_metrics = {}
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs} [{phase}]", leave=True)

        batch_fn = self._train_batch if is_train else self._validate_batch
        context = torch.enable_grad() if is_train else torch.no_grad()

        with context:
            for images, _ in progress_bar:
                images = images.to(self.device)
                batch_metrics = batch_fn(images)
                if not epoch_metrics:
                    epoch_metrics = {key: 0.0 for key in batch_metrics}
                for key, val in batch_metrics.items():
                    epoch_metrics[key] += val
                progress_bar.set_postfix(self._format_metrics(batch_metrics, short=True))
        
        return {key: val / len(dataloader) for key, val in epoch_metrics.items()}

    def train(self, train_loader, val_loader, epochs):
        self.load_checkpoint()
        
        for epoch in range(self.start_epoch, epochs + 1):
            train_metrics = self._run_epoch(train_loader, True, epoch, epochs)
            print(f"Epoch {epoch}/{epochs} [Train] Summary: {self._format_metrics(train_metrics)}")

            val_metrics = self._run_epoch(val_loader, False, epoch, epochs)
            print(f"Epoch {epoch}/{epochs} [Validation] Summary: {self._format_metrics(val_metrics)}")
            
            if self.lr_scheduler_g: self.lr_scheduler_g.step()
            if self.lr_scheduler_d: self.lr_scheduler_d.step()
                
            self.save_checkpoint(epoch)
            current_val_loss = val_metrics.get('L1', float('inf'))
            if current_val_loss < self.best_val_loss:
                print(f"Val loss improved ({self.best_val_loss:.4f} --> {current_val_loss:.4f}). Saving best model...")
                self.best_val_loss = current_val_loss
                shutil.copyfile(self.checkpoint_path, self.best_model_path)

    def save_checkpoint(self, epoch):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint = {'epoch': epoch, 'best_val_loss': self.best_val_loss,
                      'vqvae_state_dict': self.vqvae.state_dict(),
                      'discriminator_state_dict': self.discriminator.state_dict(),
                      'g_optimizer_state_dict': self.g_optimizer.state_dict(),
                      'd_optimizer_state_dict': self.d_optimizer.state_dict(),
                      'lr_scheduler_g_state_dict': self.lr_scheduler_g.state_dict() if self.lr_scheduler_g else None,
                      'lr_scheduler_d_state_dict': self.lr_scheduler_d.state_dict() if self.lr_scheduler_d else None,
                      'g_scaler_state_dict': self.g_scaler.state_dict(),
                      'd_scaler_state_dict': self.d_scaler.state_dict()}
        torch.save(checkpoint, self.checkpoint_path)
        
    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            print("No checkpoint found. Starting from scratch.")
            return

        print(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.vqvae.load_state_dict(checkpoint['vqvae_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        if self.lr_scheduler_g and checkpoint.get('lr_scheduler_g_state_dict'):
            self.lr_scheduler_g.load_state_dict(checkpoint['lr_scheduler_g_state_dict'])
        if self.lr_scheduler_d and checkpoint.get('lr_scheduler_d_state_dict'):
            self.lr_scheduler_d.load_state_dict(checkpoint['lr_scheduler_d_state_dict'])
            
        self.g_scaler.load_state_dict(checkpoint['g_scaler_state_dict'])
        self.d_scaler.load_state_dict(checkpoint['d_scaler_state_dict'])
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resuming training from epoch {self.start_epoch}")

    def _format_metrics(self, metrics, short=False):
        log_str = []
        if not metrics: return ""
        for key, val in metrics.items():
            if short:
                key_map = {'L1': 'L1', 'Perceptual': 'Perc', 'Adv': 'Adv', 'Commit': 'Com', 'D': 'D', 'GP': 'GP', 'Perplexity': 'Perp'}
                name = key_map.get(key, key)
                log_str.append(f"{name}={val:.2f}" if key != 'Perplexity' else f"{name}={val:.0f}")
            else:
                log_str.append(f"{key}={val:.4f}")
        return ", ".join(log_str)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        fake = torch.ones_like(d_interpolates, device=self.device)
        gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean() 