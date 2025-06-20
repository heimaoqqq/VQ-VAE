"""
全新的、重构后的 VQ-GAN 训练器
"""
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from lpips import LPIPS
from torchvision.utils import save_image
import numpy as np

class VQGANTrainer:
    def __init__(self, vqgan, discriminator, g_optimizer, d_optimizer, lr_scheduler_g, lr_scheduler_d, device, use_amp, checkpoint_path, sample_dir, lambda_gp=10.0, l1_weight=1.0, perceptual_weight=0.01, adversarial_weight=0.8, entropy_weight=0.1, log_interval=50):
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
        self.entropy_weight = entropy_weight  # 添加熵正则化权重

        self.checkpoint_path = checkpoint_path
        self.sample_dir = sample_dir
        os.makedirs(self.sample_dir, exist_ok=True)

        self.best_val_loss = float('inf')
        self.start_epoch = 1
        self.log_interval = log_interval
        
        # Early stopping attributes
        self.early_stopping_counter = 0
        
        # 码本监控相关
        self.codebook_stats = []
        self.max_codebook_size = 4096  # 最大码本大小
        self.expansion_threshold = 0.8  # 码本利用率超过此值时扩展
        
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
            
            # 只有当perceptual_weight > 0时才计算感知损失
            if self.perceptual_weight > 0:
                perceptual_loss = self.perceptual_loss(decoded_imgs, real_imgs).mean()
            else:
                perceptual_loss = torch.tensor(0.0, device=self.device)
            
            # Adversarial loss
            g_output = self.discriminator(decoded_imgs)
            g_loss_adv = -g_output.mean()
            
            # 添加码本熵正则化 - 鼓励更均匀的码本使用
            entropy_loss = torch.tensor(0.0, device=self.device)
            if self.entropy_weight > 0 and hasattr(self.vqgan.quantize, 'get_codebook_stats'):
                stats = self.vqgan.quantize.get_codebook_stats()
                # 熵越大越好，所以我们最小化负熵
                entropy_loss = -self.entropy_weight * stats['normalized_entropy']
            
            # Combine all losses for the generator
            # The commitment loss is already scaled by its beta inside the VQGAN model
            g_loss = (self.l1_weight * l1_loss + 
                      self.perceptual_weight * perceptual_loss + 
                      self.adversarial_weight * g_loss_adv + 
                      commitment_loss + 
                      entropy_loss)  # 添加熵损失

        self.g_scaler.scale(g_loss).backward()
        self.g_scaler.step(self.g_optimizer)
        self.g_scaler.update()

        return {
            'L1': l1_loss.item(), 
            'Perceptual': perceptual_loss.item(), 
            'Adv': g_loss_adv.item(),
            'Commit': commitment_loss.item(), 
            'Entropy': entropy_loss.item() if self.entropy_weight > 0 else 0.0,  # 添加熵损失到指标
            'Perplexity': perplexity.item(), 
            'D': d_loss.item()
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

    def train(self, train_loader, val_loader, epochs, early_stopping_patience=0, 
             smoke_test=False, skip_optimizer=False, strict_loading=False, lr_scale=1.0,
             resume_training=True):
        """
        训练模型
        
        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 总训练轮次
            early_stopping_patience: 早停耐心值，0表示禁用早停
            smoke_test: 是否进行冒烟测试（只训练1轮）
            skip_optimizer: 是否跳过加载优化器状态
            strict_loading: 是否严格加载模型权重
            lr_scale: 恢复训练时的学习率缩放因子
            resume_training: 是否从检查点恢复训练，False表示从头开始
        """
        if resume_training:
            load_success = self.load_checkpoint(
                skip_optimizer=skip_optimizer, 
                strict_model_loading=strict_loading,
                lr_scale=lr_scale
            )
            if not load_success:
                print("检查点加载失败，将从头开始训练")
                self.start_epoch = 1
                self.best_val_loss = float('inf')
        else:
            print("从头开始训练，忽略现有检查点")
            self.start_epoch = 1
            self.best_val_loss = float('inf')
        
        if smoke_test:
            epochs = 1
            print("--- 冒烟测试模式：只训练1轮 ---")

        # 打印训练配置
        print("\n=== 训练配置 ===")
        print(f"批次大小: {train_loader.batch_size}")
        print(f"训练样本数: {len(train_loader.dataset)}")
        print(f"验证样本数: {len(val_loader.dataset)}")
        print(f"码本大小: {self.vqgan.quantize.num_embeddings}")
        print(f"感知损失权重: {self.perceptual_weight}")
        print(f"对抗损失权重: {self.adversarial_weight}")
        print(f"承诺损失系数: {self.vqgan.config['commitment_loss_beta']}")
        if hasattr(self, 'entropy_weight'):
            print(f"熵正则化权重: {self.entropy_weight}")
        print(f"EMA衰减率: {self.vqgan.config['ema_decay']}")
        print(f"起始轮次: {self.start_epoch}")
        print(f"总训练轮次: {epochs}")
        print(f"早停耐心值: {early_stopping_patience}")
        print("================\n")

        for epoch in range(self.start_epoch, epochs + 1):
            # 检查并重置未使用的码元
            if hasattr(self.vqgan.quantize, 'reset_dead_codes'):
                self.vqgan.quantize.reset_dead_codes(threshold=2, current_epoch=epoch)
                
            # 训练一个epoch
            train_metrics, _ = self._run_epoch(train_loader, is_train=True, epoch=epoch, num_epochs=epochs)
            
            # 验证
            val_metrics, val_samples = self._run_epoch(val_loader, is_train=False, epoch=epoch, num_epochs=epochs)
            current_val_loss = val_metrics.get('Perceptual', float('inf'))

            # 学习率调度
            if self.lr_scheduler_g: self.lr_scheduler_g.step()
            if self.lr_scheduler_d: self.lr_scheduler_d.step()

            # --- 码本监控和扩展 ---
            if epoch % 5 == 0 or epoch == 1:  # 每5个epoch检查一次
                self.monitor_codebook(epoch)

            # --- Checkpoint and Early Stopping ---
            if current_val_loss < self.best_val_loss:
                print(f"验证损失从 {self.best_val_loss:.4f} 改善到 {current_val_loss:.4f}。保存模型...")
                self.best_val_loss = current_val_loss
                self.save_checkpoint(epoch, is_best=True)
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                print(f"验证损失未改善。计数器: {self.early_stopping_counter}/{early_stopping_patience}")
            
            if val_samples['originals']:
                self._save_sample_images(epoch, torch.cat(val_samples['originals']), torch.cat(val_samples['reconstructions']))

            if early_stopping_patience > 0 and self.early_stopping_counter >= early_stopping_patience:
                print(f"早停在轮次 {epoch} 触发。")
                break
        
        print("训练完成。")

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
            'codebook_stats': self.codebook_stats,
        }
        
        if is_best:
            torch.save(checkpoint, self.checkpoint_path)
            print(f"Saved best model checkpoint to {self.checkpoint_path}")

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

    def load_checkpoint(self, skip_optimizer=False, strict_model_loading=False, lr_scale=1.0):
        """
        加载检查点，支持更灵活的选项
        
        参数:
            skip_optimizer (bool): 是否跳过加载优化器状态
            strict_model_loading (bool): 是否严格加载模型权重（不允许缺失或多余的键）
            lr_scale (float): 学习率缩放因子，用于从中断处恢复时调整学习率
        """
        if not os.path.isfile(self.checkpoint_path):
            print("No checkpoint found, starting from scratch.")
            return

        print(f"Loading checkpoint from {self.checkpoint_path}")
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # 加载模型权重，处理可能的结构变化
            try:
                self.vqgan.load_state_dict(checkpoint['vqgan_state_dict'], strict=strict_model_loading)
                print("VQGAN模型权重加载成功")
            except RuntimeError as e:
                if "size mismatch" in str(e) and "quantize" in str(e):
                    print("检测到码本大小不匹配，尝试调整...")
                    current_codebook_size = self.vqgan.quantize.num_embeddings
                    checkpoint_size = None
                    
                    # 尝试从错误信息中提取码本大小
                    import re
                    size_match = re.search(r'copying a param with shape torch.Size\(\[(\d+)', str(e))
                    if size_match:
                        checkpoint_size = int(size_match.group(1))
                        print(f"检查点码本大小: {checkpoint_size}, 当前码本大小: {current_codebook_size}")
                    
                    if not strict_model_loading:
                        print("使用非严格模式，尝试加载兼容的权重...")
                        # 创建新的状态字典，只包含兼容的键
                        compatible_dict = {}
                        checkpoint_dict = checkpoint['vqgan_state_dict']
                        model_dict = self.vqgan.state_dict()
                        
                        for k, v in checkpoint_dict.items():
                            if k in model_dict and v.shape == model_dict[k].shape:
                                compatible_dict[k] = v
                        
                        # 加载兼容的权重
                        self.vqgan.load_state_dict(compatible_dict, strict=False)
                        print(f"已加载 {len(compatible_dict)}/{len(model_dict)} 个兼容的参数")
                    else:
                        print("严格加载模式下无法处理码本大小不匹配，请调整模型配置或使用非严格模式")
                        return
                else:
                    print(f"加载模型权重时出错: {e}")
                    if strict_model_loading:
                        raise
                    else:
                        print("使用非严格模式继续...")
            
            try:
                self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                print("判别器模型权重加载成功")
            except Exception as e:
                print(f"加载判别器权重时出错: {e}")
                if strict_model_loading:
                    raise
            
            # 根据参数决定是否加载优化器状态
            if not skip_optimizer:
                try:
                    self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
                    self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
                    
                    # 如果需要调整学习率
                    if lr_scale != 1.0:
                        for param_group in self.g_optimizer.param_groups:
                            param_group['lr'] = param_group['lr'] * lr_scale
                        for param_group in self.d_optimizer.param_groups:
                            param_group['lr'] = param_group['lr'] * lr_scale
                        print(f"学习率已按因子 {lr_scale} 调整")
                    
                    print("优化器状态加载成功")
                except Exception as e:
                    print(f"加载优化器状态时出错: {e}")
                    print("将使用新初始化的优化器")
            else:
                print("跳过加载优化器状态")
            
            # 加载学习率调度器状态
            if self.lr_scheduler_g and 'g_scheduler_state_dict' in checkpoint:
                try:
                    self.lr_scheduler_g.load_state_dict(checkpoint['g_scheduler_state_dict'])
                except Exception as e:
                    print(f"加载生成器学习率调度器时出错: {e}")
            
            if self.lr_scheduler_d and 'd_scheduler_state_dict' in checkpoint:
                try:
                    self.lr_scheduler_d.load_state_dict(checkpoint['d_scheduler_state_dict'])
                except Exception as e:
                    print(f"加载判别器学习率调度器时出错: {e}")
            
            # 加载AMP缩放器状态
            try:
                self.g_scaler.load_state_dict(checkpoint['g_scaler_state_dict'])
                self.d_scaler.load_state_dict(checkpoint['d_scaler_state_dict'])
            except Exception as e:
                print(f"加载AMP缩放器状态时出错: {e}")
            
            # 加载码本统计信息
            if 'codebook_stats' in checkpoint:
                self.codebook_stats = checkpoint['codebook_stats']
                print(f"已加载码本统计信息，包含 {len(self.codebook_stats)} 条记录")
            
            # 设置起始轮次
            self.start_epoch = checkpoint.get('epoch', 0) + 1
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            print(f"从轮次 {self.start_epoch} 继续训练，最佳验证损失: {self.best_val_loss:.4f}")
            return True
        except Exception as e:
            print(f"加载检查点时发生未预期的错误: {e}")
            import traceback
            traceback.print_exc()
            return False

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
    
    def monitor_codebook(self, epoch):
        """监控码本使用情况并在必要时扩展码本"""
        if not hasattr(self.vqgan.quantize, 'get_codebook_stats'):
            return
            
        # 获取码本统计信息
        stats = self.vqgan.quantize.get_codebook_stats()
        stats['epoch'] = epoch
        self.codebook_stats.append(stats)
        
        # 打印码本使用情况
        print(f"\nCodebook Stats [Epoch {epoch}]:")
        print(f"- Active codes: {stats['active_size']}/{self.vqgan.quantize.num_embeddings} ({stats['utilization']:.4f})")
        print(f"- Codebook entropy: {stats['entropy']:.4f} (normalized: {stats['normalized_entropy']:.4f})")
        
        # 检查是否需要扩展码本
        current_size = self.vqgan.quantize.num_embeddings
        if stats['utilization'] > self.expansion_threshold and current_size < self.max_codebook_size:
            # 计算新的码本大小 (增加一倍，但不超过最大值)
            new_size = min(current_size * 2, self.max_codebook_size)
            
            # 扩展码本
            if hasattr(self.vqgan.quantize, 'expand_codebook'):
                print(f"Expanding codebook from {current_size} to {new_size}...")
                success = self.vqgan.quantize.expand_codebook(new_size)
                
                if success:
                    # 更新优化器以包含新的码本参数
                    self._update_optimizers_after_expansion()
                    print(f"Codebook successfully expanded to {new_size}")
    
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