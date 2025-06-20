"""
全新的、重构后的 VQ-GAN 训练器
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from .models.losses import PerceptualLoss

class VQGANTrainer:
    def __init__(self, vqgan, discriminator, g_optimizer, d_optimizer, lr_scheduler_g, lr_scheduler_d, device, use_amp, checkpoint_path, sample_dir, lambda_gp=10.0, l1_weight=1.0, perceptual_weight=0.005, adversarial_weight=0.8, entropy_weight=0.3, log_interval=50, reset_low_usage_interval=5, reset_low_usage_percentage=0.1, temperature=1.0, disable_codebook_expansion=False):
        self.vqgan = vqgan
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.sample_dir = sample_dir
        self.lr_scheduler_g = lr_scheduler_g
        self.lr_scheduler_d = lr_scheduler_d
        self.use_amp = use_amp
        self.g_scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.d_scaler = torch.amp.GradScaler(enabled=self.use_amp)
        
        # 损失权重
        self.lambda_gp = lambda_gp
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
        self.entropy_weight = entropy_weight
        
        # 温度参数
        self.temperature = temperature
        
        # 日志和检查点
        self.log_interval = log_interval
        self.best_val_loss = float('inf')
        self.start_epoch = 1
        
        # 早停相关
        self.early_stopping_counter = 0
        
        # 确保样本目录存在
        os.makedirs(self.sample_dir, exist_ok=True)
        
        # 初始化步数计数器
        self.steps = 0
        
        # 低使用率码元重置参数
        self.reset_low_usage_interval = reset_low_usage_interval
        self.reset_low_usage_percentage = reset_low_usage_percentage
        
        # 初始化感知损失
        self.perceptual_loss = PerceptualLoss(device=device).to(device)

        # 码本监控相关
        self.codebook_stats = []
        self.max_codebook_size = 512  # 最大码本大小设置为512
        self.expansion_threshold = 0.8  # 码本利用率超过此值时扩展
        
        # 是否禁用码本自动扩展
        self.disable_codebook_expansion = disable_codebook_expansion

    def _train_batch(self, batch):
        real_imgs, _ = batch
        real_imgs = real_imgs.to(self.device)
        
        # --- Train Discriminator ---
        # 在训练初期减少判别器更新频率，让生成器先学习
        should_train_disc = True
        if self.steps < 500:  # 前500步
            should_train_disc = (self.steps % 5 == 0)  # 每5步更新一次判别器
        elif self.steps < 1000:  # 500-1000步
            should_train_disc = (self.steps % 3 == 0)  # 每3步更新一次判别器
            
        if should_train_disc:
            self.d_optimizer.zero_grad()
            
            # Get reconstructed images
            with torch.no_grad():
                decoded_imgs_detached = self.vqgan(real_imgs, return_dict=True, temperature=self.temperature)["decoded_imgs"].detach()
            
            # Real images loss
            real_pred = self.discriminator(real_imgs)
            d_real_loss = -torch.mean(real_pred)
            
            # Fake images loss
            fake_pred = self.discriminator(decoded_imgs_detached)
            d_fake_loss = torch.mean(fake_pred)
            
            # Gradient penalty
            gp_loss = self._gradient_penalty(real_imgs, decoded_imgs_detached)
            
            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss + self.lambda_gp * gp_loss
            
            # 判别器损失检查 - 如果损失过大，跳过此次更新
            # 在训练初期（前100步）使用更宽松的阈值，允许更大的损失值
            loss_threshold = 10000.0 if self.steps < 100 else 1000.0
            
            if not torch.isnan(d_loss) and d_loss.item() < loss_threshold:
                if self.use_amp:
                    with torch.amp.autocast(device_type='cuda', enabled=True):
                        self.d_scaler.scale(d_loss).backward()
                        self.d_scaler.step(self.d_optimizer)
                        self.d_scaler.update()
                else:
                    d_loss.backward()
                    # 添加梯度裁剪，防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                    self.d_optimizer.step()
            else:
                print(f"跳过判别器更新，损失值异常: {d_loss.item()}")
                d_loss = torch.tensor(0.0, device=self.device)
                d_real_loss = torch.tensor(0.0, device=self.device)
                d_fake_loss = torch.tensor(0.0, device=self.device)
                gp_loss = torch.tensor(0.0, device=self.device)
        else:
            # 不训练判别器时，设置默认值
            d_loss = torch.tensor(0.0, device=self.device)
            d_real_loss = torch.tensor(0.0, device=self.device)
            d_fake_loss = torch.tensor(0.0, device=self.device)
            gp_loss = torch.tensor(0.0, device=self.device)
        
        # --- Train Generator (VQ-GAN) ---
        # 只有每隔n步才更新一次判别器，这样可以平衡生成器和判别器的训练
        if self.steps % 1 == 0:  # 每步都更新生成器
            self.g_optimizer.zero_grad()
            
            # Forward pass
            vq_output = self.vqgan(real_imgs, return_dict=True, temperature=self.temperature)
            decoded_imgs = vq_output["decoded_imgs"]
            vq_loss = vq_output["commitment_loss"]
            perplexity = vq_output["perplexity"]
            
            # 存储当前批次的索引信息，用于码本利用率监控
            if "indices" in vq_output:
                self.last_batch_indices = vq_output["indices"].detach().flatten()
            
            # 获取熵值，如果没有则使用默认值0
            entropy = vq_output.get("entropy", torch.tensor(0.0, device=self.device))
            normalized_entropy = vq_output.get("normalized_entropy", torch.tensor(0.0, device=self.device))
            
            # Reconstruction loss (L1)
            l1_loss = torch.abs(real_imgs - decoded_imgs).mean()
            
            # Perceptual loss
            p_loss = self.perceptual_loss(real_imgs, decoded_imgs)
            
            # Adversarial loss
            gen_pred = self.discriminator(decoded_imgs)
            adv_loss = -torch.mean(gen_pred)
            
            # 在训练初期逐渐增加对抗损失权重
            effective_adv_weight = self.adversarial_weight
            if self.steps < 1000:
                effective_adv_weight = self.adversarial_weight * (self.steps / 1000.0)
            
            # Entropy regularization - 使用从模型获取的熵值
            entropy_loss = -entropy  # 最大化熵，所以使用负号
            
            # Total generator loss
            g_loss = (
                self.l1_weight * l1_loss + 
                self.perceptual_weight * p_loss + 
                effective_adv_weight * adv_loss +
                vq_loss +
                self.entropy_weight * entropy_loss
            )
            
            # 生成器损失检查
            if not torch.isnan(g_loss) and g_loss.item() < 1000:
                if self.use_amp:
                    with torch.amp.autocast(device_type='cuda', enabled=True):
                        self.g_scaler.scale(g_loss).backward()
                        self.g_scaler.step(self.g_optimizer)
                        self.g_scaler.update()
                else:
                    g_loss.backward()
                    # 添加梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.vqgan.parameters(), max_norm=1.0)
                    self.g_optimizer.step()
            else:
                print(f"跳过生成器更新，损失值异常: {g_loss.item()}")
                g_loss = torch.tensor(1.0, device=self.device)
                l1_loss = torch.tensor(0.5, device=self.device)
                p_loss = torch.tensor(0.5, device=self.device)
                adv_loss = torch.tensor(0.0, device=self.device)
                entropy = torch.tensor(0.0, device=self.device)
                normalized_entropy = torch.tensor(0.0, device=self.device)
                
        else:
            # 如果这步不更新生成器，设置默认值
            l1_loss = torch.tensor(0.0, device=self.device)
            p_loss = torch.tensor(0.0, device=self.device)
            adv_loss = torch.tensor(0.0, device=self.device)
            g_loss = torch.tensor(0.0, device=self.device)
            perplexity = torch.tensor(0.0, device=self.device)
            entropy = torch.tensor(0.0, device=self.device)
            normalized_entropy = torch.tensor(0.0, device=self.device)
            
        self.steps += 1
        
        # Return metrics
        metrics = {
            "L1": l1_loss.item(),
            "Perceptual": p_loss.item(),
            "Adv": adv_loss.item(),
            "Commit": vq_loss.item() if hasattr(vq_loss, 'item') else 0.0,
            "Entropy": entropy.item(),  # 显示原始熵值
            "Normalized_Entropy": normalized_entropy.item(),  # 添加归一化熵值显示
            "Perplexity": perplexity.item() if hasattr(perplexity, 'item') else 0.0,
            "D": d_loss.item()
        }
        
        return metrics

    def _validate_batch(self, batch):
        real_imgs, _ = batch
        real_imgs = real_imgs.to(self.device)
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', enabled=self.use_amp):
                vqgan_output = self.vqgan(real_imgs, return_dict=True, temperature=self.temperature)
                decoded_imgs = vqgan_output["decoded_imgs"]
                commitment_loss = vqgan_output["commitment_loss"]
                perplexity = vqgan_output["perplexity"]
                entropy = vqgan_output.get("entropy", torch.tensor(0.0, device=self.device))
                normalized_entropy = vqgan_output.get("normalized_entropy", torch.tensor(0.0, device=self.device))
                
                l1_loss = F.l1_loss(decoded_imgs, real_imgs)
                
                # 只有当perceptual_weight > 0时才计算感知损失
                if self.perceptual_weight > 0:
                    perceptual_loss = self.perceptual_loss(decoded_imgs, real_imgs).mean()
                else:
                    perceptual_loss = torch.tensor(0.0, device=self.device)
                
        return {
            'L1': l1_loss.item(), 
            'Perceptual': perceptual_loss.item(),
            'Commit': commitment_loss.item(), 
            'Perplexity': perplexity.item(),
            'Entropy': entropy.item(),
            'Normalized_Entropy': normalized_entropy.item(),
            'originals': real_imgs, 
            'reconstructions': decoded_imgs  # 移除clamp操作，保持原始范围
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

    def train(self, train_loader, val_loader, epochs, early_stopping_patience=0, skip_optimizer=False, resume_training=True):
        """
        Train the VQ-GAN model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping. Set to 0 to disable.
            skip_optimizer: Whether to skip loading optimizer state
            resume_training: Whether to resume training from checkpoint
        """
        # Load checkpoint if exists and resume_training is True
        if os.path.exists(self.checkpoint_path) and resume_training:
            self.load_checkpoint(skip_optimizer)
        else:
            print("从头开始训练，或检查点不存在")
            self.start_epoch = 1
            # 对于熵值，越大越好，所以初始值应该是0
            self.best_val_loss = 0.0
        
        # 打印训练配置
        print("\n=== 训练配置 ===")
        print(f"批次大小: {train_loader.batch_size}")
        print(f"训练样本数: {len(train_loader.dataset)}")
        print(f"验证样本数: {len(val_loader.dataset)}")
        print(f"码本大小: {self.vqgan.quantize.num_embeddings}")
        print(f"感知损失权重: {self.perceptual_weight}")
        print(f"对抗损失权重: {self.adversarial_weight}")
        print(f"承诺损失系数: {self.vqgan.config['commitment_loss_beta']}")
        print(f"熵正则化权重: {self.entropy_weight}")
        print(f"EMA衰减率: {self.vqgan.config['ema_decay']}")
        print(f"低使用率码元重置间隔: {self.reset_low_usage_interval}")
        print(f"低使用率码元重置比例: {self.reset_low_usage_percentage}")
        print(f"起始轮次: {self.start_epoch}")
        print(f"总训练轮次: {epochs}")
        print(f"早停耐心值: {early_stopping_patience}")
        print("================\n")
        
        # Early stopping setup
        patience_counter = 0
        
        # Main training loop
        for epoch in range(self.start_epoch, epochs + 1):
            # 检查并重置未使用的码元
            if hasattr(self.vqgan.quantize, 'reset_dead_codes'):
                self.vqgan.quantize.reset_dead_codes(threshold=2, current_epoch=epoch)
                
            # 训练一个epoch
            train_metrics, _ = self._run_epoch(train_loader, is_train=True, epoch=epoch, num_epochs=epochs)
            
            # 验证
            val_metrics, val_samples = self._run_epoch(val_loader, is_train=False, epoch=epoch, num_epochs=epochs)
            # 使用归一化熵作为验证指标，值越大越好
            current_val_loss = val_metrics.get('Normalized_Entropy', 0.0)

            # 学习率调度
            if self.lr_scheduler_g: self.lr_scheduler_g.step()
            if self.lr_scheduler_d: self.lr_scheduler_d.step()

            # --- 码本监控和扩展 ---
            # 每个epoch都检查码本使用情况，并获取当前归一化熵值
            normalized_entropy = self.monitor_codebook(epoch)
            # 使用从monitor_codebook返回的归一化熵值，确保一致性
            current_val_loss = normalized_entropy
            
            # 重置使用频率最低的码元
            if hasattr(self.vqgan.quantize, 'reset_low_usage_codes') and epoch % self.reset_low_usage_interval == 0:
                self.vqgan.quantize.reset_low_usage_codes(
                    percentage=self.reset_low_usage_percentage, 
                    current_epoch=epoch
                )

            # --- Checkpoint and Early Stopping ---
            # 对于归一化熵，值越大越好
            if current_val_loss > self.best_val_loss:
                print(f"归一化熵从 {self.best_val_loss:.4f} 改善到 {current_val_loss:.4f}。保存模型...")
                self.best_val_loss = current_val_loss
                self.save_checkpoint(epoch, is_best=True)
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                print(f"归一化熵未改善。计数器: {self.early_stopping_counter}/{early_stopping_patience}")
            
            if val_samples['originals']:
                self._save_sample_images(epoch, torch.cat(val_samples['originals']), torch.cat(val_samples['reconstructions']))

            if early_stopping_patience > 0 and self.early_stopping_counter >= early_stopping_patience:
                print(f"早停在轮次 {epoch} 触发。")
                break
        
        print("训练完成。")

    def save_checkpoint(self, epoch, is_best=False):
        """
        保存检查点
        
        参数:
            epoch: 当前轮次
            is_best: 是否为最佳模型
        """
        # 确保目录存在
        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        # 获取当前码本统计信息
        if hasattr(self.vqgan.quantize, 'get_codebook_stats'):
            current_stats = self.vqgan.quantize.get_codebook_stats()
            current_val_loss = current_stats.get('normalized_entropy', 0.0)
            # 对于熵值，越大越好，所以如果当前值大于最佳值，则更新最佳值
            if current_val_loss > self.best_val_loss:
                self.best_val_loss = current_val_loss
                is_best = True
        
        # 准备保存的数据
        checkpoint = {
            'epoch': epoch,
            'vqgan_state_dict': self.vqgan.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_scaler_state_dict': self.g_scaler.state_dict(),
            'd_scaler_state_dict': self.d_scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'codebook_stats': self.codebook_stats if hasattr(self, 'codebook_stats') else [],
            'model_config': {
                'vq_num_embed': self.vqgan.quantize.num_embeddings,
                'vq_embed_dim': self.vqgan.config['vq_embed_dim']
            }
        }
        
        # 添加学习率调度器状态（如果存在）
        if self.lr_scheduler_g is not None:
            checkpoint['g_scheduler_state_dict'] = self.lr_scheduler_g.state_dict()
        if self.lr_scheduler_d is not None:
            checkpoint['d_scheduler_state_dict'] = self.lr_scheduler_d.state_dict()
            
        # 保存检查点
        torch.save(checkpoint, self.checkpoint_path)
        print(f"检查点已保存到 {self.checkpoint_path}")
        
        # 如果是最佳模型，复制一份
        if is_best:
            best_path = self.checkpoint_path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)
            print(f"最佳模型已保存到 {best_path}")
            
            # 同时保存一个轻量版本（不包含优化器状态）
            light_checkpoint = {k: v for k, v in checkpoint.items() if 'optimizer' not in k and 'scaler' not in k}
            light_path = self.checkpoint_path.replace('.pt', '_best_light.pt')
            torch.save(light_checkpoint, light_path)
            print(f"轻量版最佳模型已保存到 {light_path}")

    def _save_sample_images(self, epoch, originals, reconstructions):
        # 逆归一化函数，将[-1,1]范围的图像转回[0,1]范围
        def denormalize(images):
            return images * 0.5 + 0.5
        
        # 应用逆归一化
        originals = denormalize(originals)
        reconstructions = denormalize(reconstructions)
        
        num_samples = min(originals.size(0), 8)
        comparison = torch.cat([originals[:num_samples], reconstructions[:num_samples]])
        save_path = os.path.join(self.sample_dir, f'reconstruction_epoch_{epoch:04d}.png')
        save_image(comparison, save_path, nrow=num_samples, normalize=False)

    def _load_checkpoint(self, skip_optimizer=False):
        """
        加载检查点
        
        参数:
            skip_optimizer: 是否跳过加载优化器状态
        """
        if not os.path.isfile(self.checkpoint_path):
            print("未找到检查点，将从头开始训练。")
            return False

        print(f"正在从 {self.checkpoint_path} 加载检查点")
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # 检查模型配置是否匹配
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
                current_config = {
                    'vq_num_embed': self.vqgan.quantize.num_embeddings,
                    'vq_embed_dim': self.vqgan.config['vq_embed_dim']
                }
                
                if config['vq_num_embed'] != current_config['vq_num_embed']:
                    print(f"警告：检查点码本大小 ({config['vq_num_embed']}) 与当前模型 ({current_config['vq_num_embed']}) 不匹配")
                    print("请确保使用正确的 --vq_num_embed 参数")
                    return False
            
            # 加载模型权重
            try:
                self.vqgan.load_state_dict(checkpoint['vqgan_state_dict'])
                self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                print("模型权重加载成功")
            except Exception as e:
                print(f"加载模型权重时出错: {e}")
                return False
            
            # 根据参数决定是否加载优化器状态
            if not skip_optimizer:
                try:
                    self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
                    self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
                    
                    if self.lr_scheduler_g and 'g_scheduler_state_dict' in checkpoint:
                        self.lr_scheduler_g.load_state_dict(checkpoint['g_scheduler_state_dict'])
                    if self.lr_scheduler_d and 'd_scheduler_state_dict' in checkpoint:
                        self.lr_scheduler_d.load_state_dict(checkpoint['d_scheduler_state_dict'])
                    
                    self.g_scaler.load_state_dict(checkpoint['g_scaler_state_dict'])
                    self.d_scaler.load_state_dict(checkpoint['d_scaler_state_dict'])
                    print("优化器状态加载成功")
                except Exception as e:
                    print(f"加载优化器状态时出错: {e}")
                    print("将使用新初始化的优化器")
                    skip_optimizer = True
            else:
                print("跳过加载优化器状态")
            
            # 加载码本统计信息
            if 'codebook_stats' in checkpoint:
                self.codebook_stats = checkpoint['codebook_stats']
            
            # 设置起始轮次
            self.start_epoch = checkpoint.get('epoch', 0) + 1
            # 对于熵值，越大越好，所以初始值应该是负无穷
            self.best_val_loss = checkpoint.get('best_val_loss', 0.0)
            
            print(f"从轮次 {self.start_epoch} 继续训练，最佳归一化熵: {self.best_val_loss:.4f}")
            return True
        except Exception as e:
            print(f"加载检查点时出错: {e}")
            return False

    def _gradient_penalty(self, real_samples, fake_samples):
        """
        计算WGAN-GP中的梯度惩罚项
        添加了额外的稳定性措施
        """
        batch_size = real_samples.size(0)
        device = real_samples.device
        
        # 生成随机插值系数
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        
        # 在真实样本和生成样本之间进行线性插值
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)
        
        # 计算判别器对插值样本的输出
        d_interpolates = self.discriminator(interpolates)
        
        # 创建与d_interpolates形状相同的全1张量
        fake = torch.ones_like(d_interpolates, device=device, requires_grad=False)
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # 展平梯度
        gradients = gradients.view(batch_size, -1)
        
        # 计算梯度的L2范数
        gradient_norm = gradients.norm(2, dim=1)
        
        # 添加一个小的epsilon值以提高数值稳定性
        epsilon = 1e-6
        gradient_penalty = ((gradient_norm - 1) ** 2).mean() + epsilon
        
        return gradient_penalty
    
    def monitor_codebook(self, epoch):
        """监控码本使用情况，并在需要时重置未使用的码元"""
        # 获取码本统计信息
        stats = self.vqgan.quantize.get_codebook_stats()
        
        # 记录统计信息
        if not hasattr(self, 'codebook_stats'):
            self.codebook_stats = []
        
        self.codebook_stats.append(stats)
        
        # 打印码本使用情况
        print(f"\n=== 码本统计 (Epoch {epoch}) ===")
        print(f"活跃码元 (累积EMA): {stats['active_size']}/{self.vqgan.quantize.num_embeddings} ({stats['utilization']*100:.1f}%)")
        
        # 添加: 计算实际码本利用率
        if hasattr(self, 'last_batch_indices') and self.last_batch_indices is not None:
            unique_indices = torch.unique(self.last_batch_indices)
            real_utilization = len(unique_indices) / self.vqgan.quantize.num_embeddings
            print(f"实际码本利用率 (当前批次): {len(unique_indices)}/{self.vqgan.quantize.num_embeddings} ({real_utilization*100:.1f}%)")
            
            # 计算当前批次的熵
            if len(unique_indices) > 0:
                # 计算每个码元的使用频率
                index_counts = torch.zeros(self.vqgan.quantize.num_embeddings, device=self.last_batch_indices.device)
                for idx in range(self.vqgan.quantize.num_embeddings):
                    index_counts[idx] = torch.sum(self.last_batch_indices == idx).item()
                
                # 计算概率分布
                probs = index_counts / torch.sum(index_counts)
                # 只考虑非零概率
                valid_probs = probs[probs > 0]
                # 计算熵
                batch_entropy = -torch.sum(valid_probs * torch.log2(valid_probs + 1e-10)).item()
                max_entropy = torch.log2(torch.tensor(self.vqgan.quantize.num_embeddings, dtype=torch.float, device=probs.device)).item()
                batch_norm_entropy = batch_entropy / max_entropy
                
                print(f"当前批次熵值: {batch_entropy:.4f}")
                print(f"当前批次归一化熵: {batch_norm_entropy:.4f}")
        
        print(f"累积熵值: {stats['entropy']:.4f}")
        print(f"累积归一化熵: {stats['normalized_entropy']:.4f}")
        print(f"总使用次数: {stats['total_usage']:.0f}")
        
        # 如果码本利用率高于阈值，考虑扩展码本
        if not self.disable_codebook_expansion and stats['utilization'] > self.expansion_threshold and self.vqgan.quantize.num_embeddings < self.max_codebook_size:
            # 计算新的码本大小，每次增加25%
            new_size = min(
                int(self.vqgan.quantize.num_embeddings * 1.25),
                self.max_codebook_size
            )
            
            # 只有当新大小大于当前大小时才扩展
            if new_size > self.vqgan.quantize.num_embeddings:
                print(f"码本利用率高 ({stats['utilization']*100:.1f}%)，扩展码本从 {self.vqgan.quantize.num_embeddings} 到 {new_size}")
                
                # 扩展码本
                if self.vqgan.quantize.expand_codebook(new_size):
                    # 更新优化器以包含新参数
                    self._update_optimizers_after_expansion()
        elif self.disable_codebook_expansion:
            print("码本自动扩展已禁用")
        
        # 重置低使用率码元
        if epoch > 0:
            reset_count = self.vqgan.quantize.reset_low_usage_codes(
                percentage=self.reset_low_usage_percentage,
                current_epoch=epoch
            )
            if reset_count > 0:
                print(f"重置了 {reset_count} 个低使用率码元")
        
        # 返回当前的归一化熵值，供train方法使用
        return stats['normalized_entropy']

    def _update_optimizers_after_expansion(self):
        """在码本扩展后更新优化器"""
        # 重新创建生成器优化器，包含新的码本参数
        encoder_decoder_params = list(self.vqgan.encoder.parameters()) + \
                                list(self.vqgan.decoder.parameters()) + \
                                list(self.vqgan.quant_conv.parameters()) + \
                                list(self.vqgan.post_quant_conv.parameters()) + \
                                list(self.vqgan.quantize.parameters())
                                
        # 保存当前状态
        state_dict = self.g_optimizer.state_dict()
        
        # 创建新的优化器
        self.g_optimizer = torch.optim.Adam(
            encoder_decoder_params,
            lr=state_dict['param_groups'][0]['lr'],
            betas=(state_dict['param_groups'][0]['betas'][0], state_dict['param_groups'][0]['betas'][1])
        ) 
