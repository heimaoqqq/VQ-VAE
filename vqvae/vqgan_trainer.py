"""
全新的、重构后的 VQ-GAN 训练器
"""
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from lpips import LPIPS

class VQGANTrainer:
    def __init__(self, vqvae, discriminator, g_optimizer, d_optimizer, lr_scheduler_g, lr_scheduler_d, device, use_amp, checkpoint_path, lambda_gp=10.0, l1_weight=1.0, perceptual_weight=1.0):
        self.vqvae = vqvae
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.lr_scheduler_g = lr_scheduler_g
        self.lr_scheduler_d = lr_scheduler_d
        self.device = device
        self.use_amp = use_amp
        
        self.lambda_gp = lambda_gp
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight

        self.checkpoint_path = checkpoint_path
        self.best_val_loss = float('inf')
        self.start_epoch = 1
        
        # Initialize LPIPS loss
        self.perceptual_loss = LPIPS(net='vgg').to(self.device).eval()
        for param in self.perceptual_loss.parameters():
            param.requires_grad = False

        # AMP Scalers
        self.g_scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.d_scaler = torch.amp.GradScaler(enabled=self.use_amp)

    def _train_batch(self, batch):
        real_imgs = batch
        
        # VQVAE forward pass to get reconstructed images and losses
        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            # Manually perform the forward pass to access the commitment loss from the quantizer
            h = self.vqvae.encoder(real_imgs)
            h = self.vqvae.quant_conv(h)
            quant, commitment_loss, _ = self.vqvae.quantize(h)
            decoded_imgs = self.vqvae.decoder(self.vqvae.post_quant_conv(quant))

            l1_loss = F.l1_loss(decoded_imgs, real_imgs)
            perceptual_loss = self.perceptual_loss(decoded_imgs, real_imgs).mean()

        # ====================================================
        # 1. Train the Discriminator
        # ====================================================
        self.d_optimizer.zero_grad()

        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            real_output = self.discriminator(real_imgs)
            d_loss_real = real_output.mean()
            
            fake_output = self.discriminator(decoded_imgs.detach())
            d_loss_fake = fake_output.mean()
        
        # Gradient penalty calculation should be in float32 for stability
        gradient_penalty = self.compute_gradient_penalty(real_imgs, decoded_imgs.detach())
        
        # Final discriminator loss, combining components.
        # The addition happens in float32.
        d_loss = d_loss_fake - d_loss_real + self.lambda_gp * gradient_penalty

        self.d_scaler.scale(d_loss).backward()
        self.d_scaler.step(self.d_optimizer)
        self.d_scaler.update()

        # ====================================================
        # 2. Train the Generator (VQ-GAN)
        # ====================================================
        self.g_optimizer.zero_grad()

        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            # Adversarial Loss from discriminator's perspective on the new reconstructions
            g_output = self.discriminator(decoded_imgs)
            g_loss_adv = -g_output.mean()
            
            # Total Generator Loss
            g_loss = self.l1_weight * l1_loss + self.perceptual_weight * perceptual_loss + g_loss_adv + commitment_loss

        self.g_scaler.scale(g_loss).backward()
        self.g_scaler.step(self.g_optimizer)
        self.g_scaler.update()

        return {
            'L1': l1_loss.item(),
            'Perceptual': perceptual_loss.item(),
            'Adv': g_loss_adv.item(),
            'Commit': commitment_loss.item(),
            'D': d_loss.item(),
            'D_real': d_loss_real.item(),
            'D_fake': d_loss_fake.item(),
            'GP': gradient_penalty.item(),
        }

    def _validate_batch(self, batch):
        real_imgs = batch
        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            # Manually perform the forward pass to access the commitment loss from the quantizer
            h = self.vqvae.encoder(real_imgs)
            h = self.vqvae.quant_conv(h)
            quant, commitment_loss, _ = self.vqvae.quantize(h)
            decoded_imgs = self.vqvae.decoder(self.vqvae.post_quant_conv(quant))

            l1_loss = F.l1_loss(decoded_imgs, real_imgs)
            perceptual_loss = self.perceptual_loss(decoded_imgs, real_imgs).mean()

        return {
            'L1': l1_loss.item(),
            'Perceptual': perceptual_loss.item(),
            'Commit': commitment_loss.item(),
            'Adv': 0, 'D': 0, 'D_real': 0, 'D_fake': 0, 'GP': 0, # GAN metrics are not relevant for validation
        }

    def _get_short_metric_names(self):
        return {
            'L1': 'L1', 'Perceptual': 'Perc', 'Adv': 'G_adv', 'Commit': 'Commit',
            'D': 'D_loss', 'D_real': 'D_r', 'D_fake': 'D_f', 'GP': 'GP'
        }

    def _format_metrics(self, metrics, short_names=False):
        name_map = self._get_short_metric_names() if short_names else {}
        log_str = []
        for key, val in metrics.items():
            if val == 0 and key in ['Adv', 'D', 'D_real', 'D_fake', 'GP']: # Don't show GAN metrics in validation summary
                continue
            name = name_map.get(key, key)
            log_str.append(f"{name}={val:.4f}")
        return ', '.join(log_str)

    def _run_epoch(self, dataloader, is_train, epoch, num_epochs):
        phase = "Train" if is_train else "Validation"
        self.vqvae.train(is_train)
        self.discriminator.train(is_train)
        
        epoch_metrics = {}
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs} [{phase}]", leave=False)

        for images, _ in progress_bar:
            images = images.to(self.device)
            
            if is_train:
                batch_metrics = self._train_batch(images)
            else:
                with torch.no_grad():
                    batch_metrics = self._validate_batch(images)
            
            for key, val in batch_metrics.items():
                epoch_metrics[key] = epoch_metrics.get(key, 0) + val

            progress_bar.set_postfix_str(self._format_metrics(batch_metrics, short_names=True))
        
        avg_metrics = {key: val / len(dataloader) for key, val in epoch_metrics.items()}
        return avg_metrics

    def _log_epoch_summary(self, epoch, num_epochs, avg_metrics, phase):
        print(f"Epoch {epoch}/{num_epochs} [{phase}] Summary: {self._format_metrics(avg_metrics)}")

    def train(self, train_loader, val_loader, epochs):
        self.load_checkpoint()
        
        for epoch in range(self.start_epoch, epochs + 1):
            # Training phase
            train_metrics = self._run_epoch(train_loader, is_train=True, epoch=epoch, num_epochs=epochs)
            self._log_epoch_summary(epoch, epochs, train_metrics, "Train")

            # Validation phase
            val_metrics = self._run_epoch(val_loader, is_train=False, epoch=epoch, num_epochs=epochs)
            self._log_epoch_summary(epoch, epochs, val_metrics, "Validation")
            
            # Step LR schedulers
            if self.lr_scheduler_g: self.lr_scheduler_g.step()
            if self.lr_scheduler_d: self.lr_scheduler_d.step()
                
            # Save model checkpoint based on perceptual loss
            current_val_loss = val_metrics.get('Perceptual', float('inf'))
            if current_val_loss < self.best_val_loss:
                print(f"Validation Perceptual loss improved from {self.best_val_loss:.4f} to {current_val_loss:.4f}. Saving best model...")
                self.best_val_loss = current_val_loss
                self.save_checkpoint(epoch, is_best=True)

    def save_checkpoint(self, epoch, is_best=False):
        # Always save the latest checkpoint
        latest_checkpoint_path = os.path.join(os.path.dirname(self.checkpoint_path), 'vqgan_checkpoint_latest.pth')
        
        checkpoint = {
            'epoch': epoch,
            'best_val_loss': self.best_val_loss,
            'vqvae_state_dict': self.vqvae.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'lr_scheduler_g_state_dict': self.lr_scheduler_g.state_dict() if self.lr_scheduler_g else None,
            'lr_scheduler_d_state_dict': self.lr_scheduler_d.state_dict() if self.lr_scheduler_d else None,
            'g_scaler_state_dict': self.g_scaler.state_dict(),
            'd_scaler_state_dict': self.d_scaler.state_dict()
        }
        
        torch.save(checkpoint, latest_checkpoint_path)
        print(f"Saved latest checkpoint to {latest_checkpoint_path}")

        # If it's the best model, save it to the main checkpoint path
        if is_best:
            torch.save(checkpoint, self.checkpoint_path)
            print(f"Saved best model checkpoint to {self.checkpoint_path}")

    def load_checkpoint(self):
        # Try to load the latest checkpoint first
        latest_checkpoint_path = os.path.join(os.path.dirname(self.checkpoint_path), 'vqgan_checkpoint_latest.pth')

        load_path = None
        if os.path.isfile(latest_checkpoint_path):
            load_path = latest_checkpoint_path
        elif os.path.isfile(self.checkpoint_path):
            # Fallback to the best model checkpoint if latest doesn't exist
            load_path = self.checkpoint_path

        if load_path and os.path.isfile(load_path):
            print(f"Loading checkpoint from {load_path}")
            checkpoint = torch.load(load_path, map_location=self.device)
            
            # Load states
            self.vqvae.load_state_dict(checkpoint['vqvae_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
            if self.lr_scheduler_g and checkpoint.get('lr_scheduler_g_state_dict'): self.lr_scheduler_g.load_state_dict(checkpoint['lr_scheduler_g_state_dict'])
            if self.lr_scheduler_d and checkpoint.get('lr_scheduler_d_state_dict'): self.lr_scheduler_d.load_state_dict(checkpoint['lr_scheduler_d_state_dict'])
            self.g_scaler.load_state_dict(checkpoint['g_scaler_state_dict'])
            self.d_scaler.load_state_dict(checkpoint['d_scaler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint['best_val_loss']
            print(f"Resuming training from epoch {self.start_epoch}")
        else:
            print("No checkpoint found. Starting from scratch.")

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP. Always run in float32 for stability."""
        alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        
        d_interpolates = self.discriminator(interpolates)
        fake = torch.ones(d_interpolates.size(), device=self.device, requires_grad=False)
        
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