"""
VQ-VAE 完整训练器
"""
import os
import torch
import torch.nn.functional as F
import shutil
from tqdm import tqdm
import sys
# 尝试导入wandb
try:
    import wandb
except ImportError:
    wandb = None

from ..models.losses import PerceptualLoss, PositionalLoss
from ..utils.visualization import save_reconstructed_images
from ..utils.training import validate

class VQVAETrainer:
    def __init__(
        self, 
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        args,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.args = args
        self.device = args.device
        
        # 确保输出目录存在
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 图像保存目录
        self.images_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        
        # 初始化混合精度训练
        self.scaler = None
        if args.fp16 and args.device != "cpu":
            from torch.amp import GradScaler
            self.scaler = GradScaler()
            print("使用混合精度训练 (FP16) - GradScaler已初始化")
        else:
            print("使用全精度训练 (FP32)")
            
        # 训练状态
        self.global_step = 0
        self.best_val_recon_loss = float('inf')
        self.early_stop_counter = 0
        self.early_stop_patience = 10  # 早停耐心值
        
        # 初始化损失函数
        # 感知损失设置
        self.use_perceptual = args.use_perceptual
        self.lambda_perceptual = args.lambda_perceptual
        
        # 位置感知损失设置
        self.use_positional = args.use_positional
        self.lambda_positional = args.lambda_positional
        self.lambda_vertical = args.lambda_vertical
        self.lambda_horizontal = args.lambda_horizontal
        
        # 初始化损失模型
        if self.use_perceptual:
            self.perceptual_loss = PerceptualLoss(self.device)
            print(f"启用感知损失，权重为 {self.lambda_perceptual}")
        else:
            self.perceptual_loss = None
            
        if self.use_positional:
            self.positional_loss = PositionalLoss(self.device, self.lambda_vertical, self.lambda_horizontal)
            print(f"启用位置感知损失，权重为 {self.lambda_positional}，垂直权重为 {self.lambda_vertical}，水平权重为 {self.lambda_horizontal}")
        else:
            self.positional_loss = None
        
        # 初始化wandb
        if args.use_wandb and wandb is not None:
            wandb.init(project=args.wandb_project, name=args.wandb_name)
            wandb.config.update(args)

    def train(self):
        """完整训练流程"""
        # 获取训练步数
        if self.args.num_train_steps is None:
            self.args.num_train_steps = self.args.epochs * len(self.train_dataloader)
            
        # 训练循环
        for epoch in range(self.args.epochs):
            self._train_epoch(epoch)
            
            # 验证
            if self.val_dataloader:
                val_results = self._validate()
                val_loss = val_results['loss']
                val_recon_loss = val_results['recon_loss']
                
                print(f"验证损失: {val_loss:.6f}")
                print(f"验证重建损失: {val_recon_loss:.6f}")
                
                if self.args.use_wandb and wandb is not None:
                    wandb.log({
                        "val/loss": val_loss,
                        "val/recon_loss": val_recon_loss,
                        "epoch": epoch
                    }, step=self.global_step)
                
                # 早停检查
                if val_recon_loss < self.best_val_recon_loss:
                    self.best_val_recon_loss = val_recon_loss
                    self.early_stop_counter = 0
                    
                    # 保存最佳模型
                    self._save_checkpoint("best-checkpoint")
                else:
                    self.early_stop_counter += 1
                    print(f"验证损失未改善。早停计数器: {self.early_stop_counter}/{self.early_stop_patience}")
                    
                    if self.early_stop_counter >= self.early_stop_patience:
                        print("早停触发，停止训练")
                        break
            
            # 定期保存检查点
            if (epoch + 1) % self.args.save_epochs == 0:
                self._save_checkpoint(f"checkpoint-epoch-{epoch+1}")
        
        # 保存最终模型
        self._save_checkpoint("final")
        
        # 保存标准化参数
        if self.args.use_adaptive_norm:
            norm_config = {
                'use_adaptive_norm': self.args.use_adaptive_norm,
                'split_ratio': self.args.split_ratio,
                'lower_quantile': self.args.lower_quantile,
                'upper_quantile': self.args.upper_quantile
            }
            torch.save(norm_config, os.path.join(self.output_dir, "final", "norm_config.pt"))
            print("已保存标准化配置")
        
        print("训练完成!")
        print(f"最佳验证重建损失: {self.best_val_recon_loss:.6f}")

    def _train_epoch(self, epoch):
        """训练单个epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_vq_loss = 0.0
        epoch_perplexity = 0.0
        epoch_positional_loss = 0.0
        perplexity_count = 0
        
        # 创建进度条
        progress_bar = tqdm(total=len(self.train_dataloader), 
                           desc=f"Epoch {epoch+1}/{self.args.epochs}", 
                           leave=True,
                           ncols=100)
        
        # 遍历训练数据
        for step, batch in enumerate(self.train_dataloader):
            images = batch.to(self.device)
            
            # 训练步骤
            if self.scaler is not None:
                # 使用混合精度训练
                results, reconstructed = self.train_step_fp16(images, self.scaler)
            else:
                # 常规训练
                results, reconstructed = self.train_step(images)
            
            # 获取损失
            epoch_loss += results['loss']
            epoch_recon_loss += results['recon_loss']
            epoch_vq_loss += results['vq_loss']
            
            # 更新码本利用率统计
            if 'perplexity' in results:
                epoch_perplexity += results['perplexity']
                perplexity_count += 1
            
            # 更新位置损失统计
            if 'positional_loss' in results:
                epoch_positional_loss += results['positional_loss']
            
            # 更新进度条
            status_dict = {
                "loss": f"{results['loss']:.4f}",
                "recon": f"{results['recon_loss']:.4f}",
                "vq": f"{results['vq_loss']:.4f}"
            }
            
            if 'perplexity' in results:
                status_dict["perp"] = f"{results['perplexity']}/{self.args.vq_num_embed}"
                
            if 'perceptual_loss' in results:
                status_dict["percept"] = f"{results['perceptual_loss']:.4f}"
                
            if 'positional_loss' in results:
                status_dict["pos"] = f"{results['positional_loss']:.4f}"
                
            progress_bar.set_postfix(status_dict)
            progress_bar.update(1)
            
            # wandb记录
            if self.args.use_wandb and wandb is not None and self.global_step % self.args.logging_steps == 0:
                log_dict = {
                    "train/loss": results['loss'],
                    "train/recon_loss": results['recon_loss'],
                    "train/vq_loss": results['vq_loss'],
                    "train/step": self.global_step
                }
                
                if 'perplexity' in results:
                    log_dict["train/perplexity"] = results['perplexity']
                    log_dict["train/codebook_usage"] = results['perplexity'] / self.args.vq_num_embed
                    
                if 'perceptual_loss' in results:
                    log_dict["train/perceptual_loss"] = results['perceptual_loss']
                
                if 'positional_loss' in results:
                    log_dict["train/positional_loss"] = results['positional_loss']
                    
                wandb.log(log_dict, step=self.global_step)
            
            # 保存重建图像
            if self.args.save_images and self.global_step % self.args.logging_steps == 0:
                # 保存图像并获取路径
                img_path = save_reconstructed_images(
                    images.detach().cpu(),
                    reconstructed.detach().cpu(),
                    epoch,
                    self.global_step,
                    self.images_dir
                )
                
                if self.args.use_wandb and wandb is not None:
                    wandb.log({
                        "reconstruction": wandb.Image(img_path)
                    }, step=self.global_step)
            
            # 更新全局步数
            self.global_step += 1
        
        # 关闭进度条
        progress_bar.close()
        
        # 计算并显示平均损失
        avg_loss = epoch_loss / len(self.train_dataloader)
        avg_recon_loss = epoch_recon_loss / len(self.train_dataloader)
        avg_vq_loss = epoch_vq_loss / len(self.train_dataloader)
        
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.6f}")
        print(f"重建损失: {avg_recon_loss:.6f}")
        print(f"VQ损失: {avg_vq_loss:.6f}")
        
        if perplexity_count > 0:
            avg_perplexity = epoch_perplexity / perplexity_count
            print(f"码本利用率: {avg_perplexity:.1f}/{self.args.vq_num_embed} ({avg_perplexity/self.args.vq_num_embed*100:.1f}%)")
        
        if self.args.use_positional:
            avg_positional_loss = epoch_positional_loss / len(self.train_dataloader)
            print(f"位置损失: {avg_positional_loss:.6f}")

    def _validate(self):
        """验证模型"""
        return validate(self.model, self.val_dataloader, self.device)

    def _save_checkpoint(self, checkpoint_name):
        """保存检查点"""
        checkpoint_path = os.path.join(self.output_dir, checkpoint_name)
        if os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        print(f"保存检查点到 {checkpoint_path}")
        torch.save(self.model.state_dict(), os.path.join(checkpoint_path, "model.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
        
        # 保存当前训练状态
        torch.save({
            'global_step': self.global_step,
            'best_val_recon_loss': self.best_val_recon_loss,
        }, os.path.join(checkpoint_path, "training_state.pt"))

    def load_checkpoint(self, checkpoint_path):
        """从检查点加载训练状态"""
        print(f"正在从检查点恢复训练: {checkpoint_path}")
        
        # 加载模型权重
        model_path = os.path.join(checkpoint_path, "model.pt")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("模型权重加载成功")
        
        # 加载优化器状态
        optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
            print("优化器状态加载成功")
            
        # 加载训练状态
        state_path = os.path.join(checkpoint_path, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)
            self.global_step = state.get('global_step', 0)
            self.best_val_recon_loss = state.get('best_val_recon_loss', float('inf'))
            print(f"加载训练状态: 步数={self.global_step}, 最佳损失={self.best_val_recon_loss:.6f}")
    
    def compute_vq_loss(self, encoder_output, decoder_output):
        """计算VQ损失，适应diffusers 0.33.1及以上版本API"""
        # 检查encoder_output中是否有loss属性
        if hasattr(encoder_output, "loss"):
            return encoder_output.loss
        
        # 检查decoder_output中的commit_loss (diffusers 0.33.1新API)
        if hasattr(decoder_output, "commit_loss") and decoder_output.commit_loss is not None:
            return decoder_output.commit_loss
        
        # 如果encoder_output有z和z_q属性，手动计算
        if hasattr(encoder_output, "z") and hasattr(encoder_output, "z_q"):
            # 计算commitment loss
            commitment_loss = F.mse_loss(encoder_output.z.detach(), encoder_output.z_q)
            # 计算codebook loss
            codebook_loss = F.mse_loss(encoder_output.z, encoder_output.z_q.detach())
            return codebook_loss + 0.25 * commitment_loss
        
        # 没有可用的VQ损失计算方法，使用默认值
        # 对于diffusers 0.33.1，我们可以使用一个适当的默认损失值或警告信息
        print("警告: 无法计算VQ损失，使用默认值。请检查diffusers版本与模型兼容性。")
        return torch.tensor(0.1, device=self.device)
    
    def get_perplexity(self, encoder_output):
        """获取码本使用情况指标 (perplexity)"""
        # 如果直接提供perplexity属性
        if hasattr(encoder_output, "perplexity"):
            return encoder_output.perplexity
        
        # 在diffusers 0.33.1版本中，尝试通过模型的quantize模块获取
        try:
            quantize = self.model.quantize
            if hasattr(quantize, "embedding") and hasattr(self.model, "quantize"):
                # 计算所有潜在向量的临近码本索引
                with torch.no_grad():
                    # 获取latents并确保是float32类型，防止half和float类型不匹配
                    latents = encoder_output.latents.to(torch.float32)
                    batch = latents.permute(0, 2, 3, 1).reshape(-1, quantize.vq_embed_dim)
                    
                    # 确保embedding权重也是float32类型
                    embedding_weight = quantize.embedding.weight.to(torch.float32)
                    
                    # 计算与码本的距离 (确保都是float32类型)
                    d = torch.sum(batch ** 2, dim=1, keepdim=True) + \
                        torch.sum(embedding_weight ** 2, dim=1) - \
                        2 * torch.matmul(batch, embedding_weight.t())
                        
                    # 获取最近的码本索引
                    encoding_indices = torch.argmin(d, dim=1)
                    
                    # 计算唯一索引数(被使用的码本向量数量)
                    unique_indices = torch.unique(encoding_indices)
                    perplexity = len(unique_indices)
                    
                    return perplexity
        except Exception as e:
            print(f"计算码本利用率时出错: {e}")
        
        # 无法计算perplexity
        return None
    
    def train_step(self, batch):
        """训练步骤"""
        self.optimizer.zero_grad()
        
        # 前向传播
        encoder_output = self.model.encode(batch)
        decoder_output = self.model.decode(encoder_output.latents)
        
        # 计算重建损失
        reconstruction_loss = F.mse_loss(decoder_output.sample, batch)
        
        # 计算VQ损失
        vq_loss = self.compute_vq_loss(encoder_output, decoder_output)
        
        # 初始化总损失
        total_loss = reconstruction_loss + vq_loss
        
        # 如果启用了感知损失，添加到总损失中
        perceptual_loss_val = 0.0
        if self.use_perceptual and self.perceptual_loss is not None:
            perceptual_loss_val = self.perceptual_loss(decoder_output.sample, batch)
            total_loss = total_loss + self.lambda_perceptual * perceptual_loss_val
        
        # 如果启用了位置感知损失，添加到总损失中
        positional_loss_val = 0.0
        if self.use_positional and self.positional_loss is not None:
            positional_loss_val = self.positional_loss(decoder_output.sample, batch)
            total_loss = total_loss + self.lambda_positional * positional_loss_val
        
        # 反向传播
        total_loss.backward()
        self.optimizer.step()
        
        # 返回结果
        result = {
            'loss': total_loss.item(),
            'recon_loss': reconstruction_loss.item(),
            'vq_loss': vq_loss.item(),
        }
        
        if self.use_perceptual:
            result['perceptual_loss'] = perceptual_loss_val.item()
            
        if self.use_positional:
            result['positional_loss'] = positional_loss_val.item()
        
        perplexity = self.get_perplexity(encoder_output)
        if perplexity is not None:
            result['perplexity'] = perplexity
            
        return result, decoder_output.sample
    
    def train_step_fp16(self, batch, scaler):
        """混合精度训练步骤"""
        self.optimizer.zero_grad()
        
        # 使用自动混合精度，尝试兼容新旧版本的API
        try:
            # 尝试使用新版API (需要device_type参数)
            autocast_context = torch.amp.autocast(device_type=self.device.type if hasattr(self.device, 'type') else 'cuda')
        except TypeError:
            # 回退到旧版API (不需要device_type参数)
            autocast_context = torch.amp.autocast()
        
        with autocast_context:
            # 前向传播
            encoder_output = self.model.encode(batch)
            decoder_output = self.model.decode(encoder_output.latents)
            
            # 计算重建损失
            reconstruction_loss = F.mse_loss(decoder_output.sample, batch)
            
            # 计算VQ损失
            vq_loss = self.compute_vq_loss(encoder_output, decoder_output)
            
            # 初始化总损失
            total_loss = reconstruction_loss + vq_loss
            
            # 如果启用了感知损失，添加到总损失中
            perceptual_loss_val = 0.0
            if self.use_perceptual and self.perceptual_loss is not None:
                perceptual_loss_val = self.perceptual_loss(decoder_output.sample, batch)
                total_loss = total_loss + self.lambda_perceptual * perceptual_loss_val
            
            # 如果启用了位置感知损失，添加到总损失中
            positional_loss_val = 0.0
            if self.use_positional and self.positional_loss is not None:
                positional_loss_val = self.positional_loss(decoder_output.sample, batch)
                total_loss = total_loss + self.lambda_positional * positional_loss_val
        
        # 反向传播与优化器步骤
        scaler.scale(total_loss).backward()
        scaler.step(self.optimizer)
        scaler.update()
        
        # 返回结果
        result = {
            'loss': total_loss.item(),
            'recon_loss': reconstruction_loss.item(),
            'vq_loss': vq_loss.item(),
            'using_fp16': True,  # 添加标记以便确认FP16正在使用
        }
        
        if self.use_perceptual:
            result['perceptual_loss'] = perceptual_loss_val.item()
            
        if self.use_positional:
            result['positional_loss'] = positional_loss_val.item()
        
        perplexity = self.get_perplexity(encoder_output)
        if perplexity is not None:
            result['perplexity'] = perplexity
            
        return result, decoder_output.sample
    
    @torch.no_grad()
    def eval_step(self, batch):
        """评估步骤"""
        # 前向传播
        encoder_output = self.model.encode(batch)
        decoder_output = self.model.decode(encoder_output.latents)
        
        # 计算重建损失
        reconstruction_loss = F.mse_loss(decoder_output.sample, batch)
        
        # 计算VQ损失
        vq_loss = self.compute_vq_loss(encoder_output, decoder_output)
        
        # 初始化总损失
        total_loss = reconstruction_loss + vq_loss
        
        # 如果启用了感知损失，添加到总损失中
        perceptual_loss_val = 0.0
        if self.use_perceptual and self.perceptual_loss is not None:
            perceptual_loss_val = self.perceptual_loss(decoder_output.sample, batch)
            total_loss = total_loss + self.lambda_perceptual * perceptual_loss_val
        
        # 如果启用了位置感知损失，添加到总损失中
        positional_loss_val = 0.0
        if self.use_positional and self.positional_loss is not None:
            positional_loss_val = self.positional_loss(decoder_output.sample, batch)
            total_loss = total_loss + self.lambda_positional * positional_loss_val
        
        # 返回结果
        result = {
            'loss': total_loss.item(),
            'recon_loss': reconstruction_loss.item(),
            'vq_loss': vq_loss.item(),
        }
        
        if self.use_perceptual:
            result['perceptual_loss'] = perceptual_loss_val.item()
            
        if self.use_positional:
            result['positional_loss'] = positional_loss_val.item()
        
        perplexity = self.get_perplexity(encoder_output)
        if perplexity is not None:
            result['perplexity'] = perplexity
            
        return result, decoder_output.sample 