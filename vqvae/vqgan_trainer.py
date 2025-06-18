"""
全新的、重构后的 VQ-GAN 训练器
"""
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from lpips import LPIPS
from torchvision.utils import save_image

class VQGANTrainer:
    def __init__(self, vqgan, discriminator, g_optimizer, d_optimizer, lr_scheduler_g, lr_scheduler_d, device, use_amp, checkpoint_path, sample_dir, lambda_gp=10.0, l1_weight=1.0, perceptual_weight=1.0, adversarial_weight=0.8, log_interval=50):
        self.vqgan = vqgan
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
        self.adversarial_weight = adversarial_weight

        self.checkpoint_path = checkpoint_path
        self.sample_dir = sample_dir
        os.makedirs(self.sample_dir, exist_ok=True)

        self.best_val_loss = float('inf')
        self.start_epoch = 1
        self.log_interval = log_interval
        
        # Early stopping attributes
        self.early_stopping_counter = 0
        
        # Initialize LPIPS loss
        self.perceptual_loss = LPIPS(net='vgg').to(self.device).eval()
        for param in self.perceptual_loss.parameters():
            param.requires_grad = False

        # AMP Scalers (using the new torch.amp API)
        self.g_scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)
        self.d_scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)

    def _train_batch(self, batch):
        real_imgs, _ = batch
        real_imgs = real_imgs.to(self.device)
        
        # --- Train Discriminator ---
        self.d_optimizer.zero_grad()
        with torch.amp.autocast(self.device.type, enabled=self.use_amp):
            # We only need the decoded images for the discriminator, so we detach them
            decoded_imgs_detached = self.vqgan(real_imgs, return_dict=True)["decoded_imgs"].detach()
            
            real_output = self.discriminator(real_imgs)
            fake_output = self.discriminator(decoded_imgs_detached)
            
            gradient_penalty = self.compute_gradient_penalty(real_imgs, decoded_imgs_detached)
            d_loss = fake_output.mean() - real_output.mean() + self.lambda_gp * gradient_penalty
            
        self.d_scaler.scale(d_loss).backward()
        self.d_scaler.step(self.d_optimizer)
        self.d_scaler.update()

        # --- Train Generator ---
        self.g_optimizer.zero_grad()
        with torch.amp.autocast(self.device.type, enabled=self.use_amp):
            # Forward pass for generator
            vqgan_output = self.vqgan(real_imgs, return_dict=True)
            decoded_imgs = vqgan_output["decoded_imgs"]
            commitment_loss = vqgan_output["commitment_loss"]
            perplexity = vqgan_output["perplexity"]
            
            # Reconstruction losses
            l1_loss = F.l1_loss(decoded_imgs, real_imgs)
            perceptual_loss = self.perceptual_loss(decoded_imgs, real_imgs).mean()
            
            # Adversarial loss
            g_output = self.discriminator(decoded_imgs)
            g_loss_adv = -g_output.mean()
            
            # Combine all losses for the generator
            # The commitment loss is already scaled by its beta inside the VQGAN model
            g_loss = (self.l1_weight * l1_loss + 
                      self.perceptual_weight * perceptual_loss + 
                      self.adversarial_weight * g_loss_adv + 
                      commitment_loss) # Directly add the commitment loss

        self.g_scaler.scale(g_loss).backward()
        self.g_scaler.step(self.g_optimizer)
        self.g_scaler.update()

        return {
            'L1': l1_loss.item(), 'Perceptual': perceptual_loss.item(), 'Adv': g_loss_adv.item(),
            'Commit': commitment_loss.item(), 'Perplexity': perplexity.item(), 'D': d_loss.item()
        }

    def _validate_batch(self, batch):
        real_imgs, _ = batch
        real_imgs = real_imgs.to(self.device)
        with torch.no_grad():
            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                vqgan_output = self.vqgan(real_imgs, return_dict=True)
                decoded_imgs = vqgan_output["decoded_imgs"]
                commitment_loss = vqgan_output["commitment_loss"]
                perplexity = vqgan_output["perplexity"]
                
                l1_loss = F.l1_loss(decoded_imgs, real_imgs)
                perceptual_loss = self.perceptual_loss(decoded_imgs, real_imgs).mean()
                
        return {
            'L1': l1_loss.item(), 
            'Perceptual': perceptual_loss.item(),
            'Commit': commitment_loss.item(), 
            'Perplexity': perplexity.item(),
            'originals': real_imgs, 
            'reconstructions': decoded_imgs.clamp(0, 1) # Clamp for valid image range
        }

    def _run_epoch(self, dataloader, is_train, epoch, num_epochs):
        phase = "Train" if is_train else "Validation"
        self.vqgan.train(is_train)
        self.discriminator.train(is_train)
        
        epoch_metrics = {}
        all_samples = {'originals': [], 'reconstructions': []}

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs} [{phase}]")
        for i, batch in enumerate(pbar):
            batch_metrics = self._train_batch(batch) if is_train else self._validate_batch(batch)
            
            if not is_train:
                # Always pop image tensors, but only store for the first batch
                originals = batch_metrics.pop('originals')
                reconstructions = batch_metrics.pop('reconstructions')
                if i == 0: 
                    all_samples['originals'].append(originals.cpu())
                    all_samples['reconstructions'].append(reconstructions.cpu())

            # Now, batch_metrics only contains scalar metrics
            for key, val in batch_metrics.items():
                epoch_metrics[key] = epoch_metrics.get(key, 0) + val
            
            # Update the progress bar with the latest metrics
            pbar.set_postfix({k: f"{v:.3f}" for k, v in batch_metrics.items()})

            # Print detailed log at the specified interval during training
            if is_train and i > 0 and (i + 1) % self.log_interval == 0:
                log_str = f"Epoch {epoch} Step [{i+1}/{len(dataloader)}]: "
                log_str += ", ".join([f"{k}: {v:.4f}" for k, v in batch_metrics.items()])
                # Print to the same line as tqdm to avoid breaking the bar
                pbar.write(log_str)

        avg_metrics = {key: val / len(dataloader) for key, val in epoch_metrics.items()}
        print(f"Epoch {epoch} [{phase}] Avg Metrics: " + ", ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()]))
        
        return avg_metrics, all_samples

    def train(self, train_loader, val_loader, epochs, early_stopping_patience=0, smoke_test=False):
        self.load_checkpoint()
        
        if smoke_test:
            epochs = 1
            print("--- Smoke test mode enabled: training will run for only 1 epoch. ---")

        for epoch in range(self.start_epoch, epochs + 1):
            train_metrics, _ = self._run_epoch(train_loader, is_train=True, epoch=epoch, num_epochs=epochs)
            
            val_metrics, val_samples = self._run_epoch(val_loader, is_train=False, epoch=epoch, num_epochs=epochs)
            current_val_loss = val_metrics.get('Perceptual', float('inf'))

            if self.lr_scheduler_g: self.lr_scheduler_g.step()
            if self.lr_scheduler_d: self.lr_scheduler_d.step()

            # --- Checkpoint and Early Stopping ---
            if current_val_loss < self.best_val_loss:
                print(f"Validation loss improved from {self.best_val_loss:.4f} to {current_val_loss:.4f}. Saving model...")
                self.best_val_loss = current_val_loss
                self.save_checkpoint(epoch, is_best=True)
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                print(f"Validation loss did not improve. Counter: {self.early_stopping_counter}/{early_stopping_patience}")
            
            if val_samples['originals']:
                self._save_sample_images(epoch, torch.cat(val_samples['originals']), torch.cat(val_samples['reconstructions']))

            if early_stopping_patience > 0 and self.early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
        
        print("Training finished.")

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'best_val_loss': self.best_val_loss,
            'vqgan_state_dict': self.vqgan.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_scheduler_state_dict': self.lr_scheduler_g.state_dict() if self.lr_scheduler_g else None,
            'd_scheduler_state_dict': self.lr_scheduler_d.state_dict() if self.lr_scheduler_d else None,
            'g_scaler_state_dict': self.g_scaler.state_dict(),
            'd_scaler_state_dict': self.d_scaler.state_dict(),
        }
        
        if is_best:
            torch.save(checkpoint, self.checkpoint_path)
            print(f"Saved best model checkpoint to {self.checkpoint_path}")

    def _save_sample_images(self, epoch, originals, reconstructions):
        num_samples = min(originals.size(0), 8)
        comparison = torch.cat([originals[:num_samples], reconstructions[:num_samples]])
        save_path = os.path.join(self.sample_dir, f'reconstruction_epoch_{epoch:04d}.png')
        save_image(comparison, save_path, nrow=num_samples, normalize=True)

    def load_checkpoint(self):
        if not os.path.isfile(self.checkpoint_path):
            print("No checkpoint found, starting from scratch.")
            return

        print(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        self.vqgan.load_state_dict(checkpoint['vqgan_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        if self.lr_scheduler_g and 'g_scheduler_state_dict' in checkpoint:
            self.lr_scheduler_g.load_state_dict(checkpoint['g_scheduler_state_dict'])
        if self.lr_scheduler_d and 'd_scheduler_state_dict' in checkpoint:
            self.lr_scheduler_d.load_state_dict(checkpoint['d_scheduler_state_dict'])
        
        self.g_scaler.load_state_dict(checkpoint['g_scaler_state_dict'])
        self.d_scaler.load_state_dict(checkpoint['d_scaler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Resuming training from epoch {self.start_epoch}")

    def compute_gradient_penalty(self, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty 