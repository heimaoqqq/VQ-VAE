"""
LDM 训练器
"""
import os
import torch
import torch.nn.functional as F
import shutil
from tqdm import tqdm
from torch.optim import AdamW
from accelerate import Accelerator
from diffusers import DDPMScheduler, DDIMScheduler, PNDMScheduler
import wandb
from transformers import get_cosine_schedule_with_warmup  # 添加学习率调度器

from ..utils.visualization import save_generated_images
from ..utils.training import validate

class LDMTrainer:
    def __init__(
        self,
        unet,
        vq_model,
        train_dataloader,
        val_dataloader,
        args,
        beta_schedule="cosine"
    ):
        """
        LDM训练器
        
        参数:
            unet: UNet模型
            vq_model: VQ-VAE模型
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            args: 训练参数
            beta_schedule: beta调度类型
        """
        self.unet = unet
        self.vq_model = vq_model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.args = args
        
        # 创建输出目录
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.images_dir = os.path.join(self.output_dir, "generated_images")
        os.makedirs(self.images_dir, exist_ok=True)
        
        # 初始化accelerator，启用梯度累积
        gradient_accumulation_steps = args.gradient_accumulation_steps if hasattr(args, 'gradient_accumulation_steps') else 4
        print(f"启用梯度累积，累积步数: {gradient_accumulation_steps}")
        
        self.accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            log_with="wandb" if args.use_wandb else None,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        
        # 初始化噪声调度器
        scheduler_type = args.scheduler_type
        
        print(f"使用调度器: {scheduler_type.upper()}")
        print(f"使用Beta调度: {beta_schedule}")
        
        if scheduler_type == "ddpm":
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_schedule=beta_schedule
            )
        elif scheduler_type == "pndm":
            self.noise_scheduler = PNDMScheduler(
                num_train_timesteps=1000,
                beta_schedule=beta_schedule
            )
        else:
            # 默认使用DDIM + squaredcos_cap_v2
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_schedule=beta_schedule,
                prediction_type="v_prediction"  # 明确指定预测类型为v_prediction
            )
        
        # 计算潜在空间大小
        self.latent_size = args.image_size // (2 ** (len(vq_model.config.down_block_types)))
        print(f"图像大小: {args.image_size}x{args.image_size}")
        print(f"VQ-VAE下采样层数: {len(vq_model.config.down_block_types)}")
        print(f"潜在空间大小: {self.latent_size}x{self.latent_size}")
        print(f"使用混合精度: {args.mixed_precision}")
        
        # 初始化优化器，增加weight_decay以增强正则化
        self.optimizer = AdamW(
            unet.parameters(), 
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,  # 增加weight_decay，标准扩散模型通常使用0.01
            eps=1e-8  # 增加epsilon以提高数值稳定性
        )
        
        # 添加学习率调度器
        if args.num_train_steps is None:
            num_training_steps = args.epochs * len(train_dataloader)
        else:
            num_training_steps = args.num_train_steps
            
        num_warmup_steps = int(num_training_steps * 0.05)  # 使用总步数的5%作为预热步数
        
        print(f"设置学习率调度器 - 总训练步数: {num_training_steps}, 预热步数: {num_warmup_steps} (总步数的5%)")
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # 确保模型在accelerator.prepare之前就在正确的设备上
        device = self.accelerator.device
        print(f"手动将模型移动到设备: {device}")
        
        # 首先手动将UNet模型移动到正确设备
        unet = unet.to(device)
        vq_model = vq_model.to(device)
        
        # 使用accelerator准备模型和数据加载器
        (
            self.vq_model, 
            self.unet, 
            self.optimizer, 
            self.train_dataloader, 
            self.val_dataloader
        ) = self.accelerator.prepare(
            self.vq_model, 
            self.unet, 
            self.optimizer, 
            self.train_dataloader, 
            self.val_dataloader
        )
        
        # 设置训练步数
        if args.num_train_steps is None:
            self.num_train_steps = args.epochs * len(train_dataloader)
        else:
            self.num_train_steps = args.num_train_steps
        
        # 初始化早停相关变量
        self.best_val_loss = float('inf')
        self.best_model_path = None
        self.early_stop_counter = 0
        self.early_stop_patience = args.early_stop_patience
        
        # 初始化训练状态跟踪变量
        self.current_epoch = 0
        self.global_step = 0
        
        # 初始化wandb
        if args.use_wandb:
            self.accelerator.init_trackers(
                project_name=args.wandb_project,
                init_kwargs={"wandb": {"name": args.wandb_name}}
            )
            # 记录重要的超参数
            self.accelerator.log({
                "lr": args.lr,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_batch_size": args.batch_size * gradient_accumulation_steps,
                "beta_schedule": beta_schedule,
                "mixed_precision": args.mixed_precision,
                "model_parameters": sum(p.numel() for p in unet.parameters()) / 1e6  # 百万参数数量
            })
    
    def load_checkpoint(self, checkpoint_path):
        """
        从检查点恢复训练状态
        
        参数:
            checkpoint_path: 检查点路径
        """
        try:
            # 检查检查点路径是否存在
            if not os.path.exists(checkpoint_path):
                raise ValueError(f"检查点路径不存在: {checkpoint_path}")
            
            print(f"正在从 {checkpoint_path} 加载检查点...")
            
            # 使用accelerator加载检查点
            self.accelerator.load_state(checkpoint_path)
            
            # 尝试加载训练状态
            training_state_path = os.path.join(checkpoint_path, "training_state.pt")
            if os.path.exists(training_state_path):
                training_state = torch.load(training_state_path)
                self.current_epoch = training_state.get("epoch", 0)
                self.global_step = training_state.get("global_step", 0)
                print(f"已恢复训练状态: 轮次={self.current_epoch}, 全局步数={self.global_step}")
            
            # 尝试加载最佳验证损失
            best_val_loss_path = os.path.join(checkpoint_path, "best_val_loss.pt")
            if os.path.exists(best_val_loss_path):
                self.best_val_loss = torch.load(best_val_loss_path).item()
                print(f"已加载最佳验证损失: {self.best_val_loss:.6f}")
            else:
                # 如果没有保存最佳验证损失，则使用默认值
                print("未找到保存的最佳验证损失，使用默认值")
                self.best_val_loss = float('inf')
            
            print("检查点加载成功！")
            return True
        except Exception as e:
            print(f"加载检查点时出错: {e}")
            return False
    
    def train(self):
        """执行完整的训练流程"""
        start_epoch = self.current_epoch
        global_step = self.global_step
        
        # 监控初始显存使用情况
        if torch.cuda.is_available():
            print(f"训练开始前GPU内存占用: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
            print(f"训练开始前GPU最大内存占用: {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB")
        
        # 使用gradscaler处理混合精度训练中的梯度缩放
        scaler = None
        if self.args.mixed_precision != "no" and not self.accelerator.native_amp:
            print("使用自定义梯度缩放器")
            scaler = torch.cuda.amp.GradScaler()
        
        best_val_loss_epoch = start_epoch
        
        print(f"开始训练，从轮次 {start_epoch + 1} 到 {self.args.epochs}")
        for epoch in range(start_epoch, self.args.epochs):
            self.current_epoch = epoch
            self.unet.train()
            epoch_loss = 0.0
            tqdm_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
            
            for step, batch in enumerate(tqdm_bar):
                # 使用VQ-VAE编码图像到潜在空间
                with torch.no_grad():
                    latents = self.vq_model.encode(batch).latents
                
                # 添加噪声
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, self.noise_scheduler.num_train_timesteps, 
                    (latents.shape[0],), 
                    device=latents.device
                ).long()
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                
                # 预测噪声
                with self.accelerator.accumulate(self.unet):
                    noise_pred = self.unet(noisy_latents, timesteps).sample
                    loss = F.mse_loss(noise_pred, noise)
                    
                    # 反向传播
                    self.accelerator.backward(loss)
                    
                    # 只在梯度累积完成时进行梯度裁剪和优化器步骤
                    if self.accelerator.sync_gradients:
                        # 使用更小的梯度裁剪阈值，增加训练稳定性
                        self.accelerator.clip_grad_norm_(self.unet.parameters(), 0.5)
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()  # 更新学习率
                    self.optimizer.zero_grad()
                
                epoch_loss += loss.detach().item()
                current_lr = self.lr_scheduler.get_last_lr()[0]
                tqdm_bar.set_postfix({"loss": loss.item(), "lr": current_lr})
                
                # 记录日志
                if self.args.use_wandb and global_step % self.args.logging_steps == 0:
                    logs = {
                        "train/loss": loss.detach().item(), 
                        "train/step": global_step,
                        "train/lr": current_lr,
                        "train/epoch": epoch + (step + 1) / len(self.train_dataloader)
                    }
                    
                    if torch.cuda.is_available():
                        logs.update({
                            "system/gpu_memory": torch.cuda.memory_allocated() / (1024**3),  # GB
                            "system/gpu_memory_peak": torch.cuda.max_memory_allocated() / (1024**3)  # GB
                        })
                        
                    self.accelerator.log(logs, step=global_step)
                
                # 评估和生成样本
                if self.args.save_images and global_step % self.args.eval_steps == 0:
                    # 解包装模型
                    unwrapped_model = self.accelerator.unwrap_model(self.unet)
                    unwrapped_vq_model = self.accelerator.unwrap_model(self.vq_model)
                    
                    # 生成样本
                    img_path = save_generated_images(
                        unwrapped_vq_model,
                        unwrapped_model,
                        self.noise_scheduler,
                        self.latent_size,
                        self.accelerator.device,
                        num_images=4,
                        num_inference_steps=self.args.num_inference_steps,
                        output_dir=self.images_dir,
                        step=global_step,
                        dataloader=self.val_dataloader  # 传递数据加载器以便进行三图对比
                    )
                    
                    print(f"生成的样本已保存到: {img_path}")
                    
                    if self.args.use_wandb:
                        self.accelerator.log({
                            "generated_samples": wandb.Image(img_path)
                        }, step=global_step)
                
                global_step += 1
                
                if global_step >= self.num_train_steps:
                    break
            
            avg_loss = epoch_loss / len(self.train_dataloader)
            print(f"Epoch {epoch+1} 平均损失: {avg_loss:.6f}")
            
            # 验证
            if self.val_dataloader:
                val_loss = validate(
                    self.unet, 
                    self.vq_model, 
                    self.val_dataloader, 
                    self.noise_scheduler, 
                    self.accelerator
                )
                print(f"验证损失: {val_loss:.6f}")
                
                if self.args.use_wandb:
                    self.accelerator.log({
                        "val/loss": val_loss,
                        "val/epoch": epoch + 1
                    }, step=global_step)
                
                # 检查是否为最佳验证损失
                if val_loss < self.best_val_loss:
                    previous_best = self.best_val_loss
                    self.best_val_loss = val_loss
                    best_val_loss_epoch = epoch
                    self.early_stop_counter = 0
                    
                    print(f"验证损失改善: {previous_best:.6f} -> {self.best_val_loss:.6f}")
                    
                    # 保存最佳模型
                    # 先删除之前的最佳模型
                    if self.best_model_path:
                        best_checkpoint_path = os.path.join(self.output_dir, "best-checkpoint")
                        best_pipeline_path = os.path.join(self.output_dir, "best-pipeline")
                        
                        if os.path.exists(best_checkpoint_path):
                            shutil.rmtree(best_checkpoint_path)
                        
                        if os.path.exists(best_pipeline_path):
                            shutil.rmtree(best_pipeline_path)
                    
                    # 保存新的最佳模型
                    # 先解除分布式包装
                    unwrapped_model = self.accelerator.unwrap_model(self.unet)
                    
                    # 根据调度器类型选择对应的pipeline
                    scheduler_type = self.args.scheduler_type
                    
                    if scheduler_type == "ddpm":
                        from diffusers import DDPMPipeline
                        pipeline = DDPMPipeline(
                            unet=unwrapped_model,
                            scheduler=self.noise_scheduler
                        )
                    elif scheduler_type == "ddim":
                        from diffusers import DDIMPipeline
                        pipeline = DDIMPipeline(
                            unet=unwrapped_model,
                            scheduler=self.noise_scheduler
                        )
                    elif scheduler_type == "pndm":
                        from diffusers import PNDMPipeline
                        pipeline = PNDMPipeline(
                            unet=unwrapped_model,
                            scheduler=self.noise_scheduler
                        )
                    else:
                        # 默认使用DDIM
                        from diffusers import DDIMPipeline
                        pipeline = DDIMPipeline(
                            unet=unwrapped_model,
                            scheduler=self.noise_scheduler
                        )
                    
                    # 保存模型和pipeline
                    best_checkpoint_path = os.path.join(self.output_dir, "best-checkpoint")
                    best_pipeline_path = os.path.join(self.output_dir, "best-pipeline")
                    self.best_model_path = best_checkpoint_path  # 记录路径
                    
                    self.accelerator.save_state(best_checkpoint_path)
                    pipeline.save_pretrained(best_pipeline_path)
                    
                    # 保存最佳验证损失
                    best_val_loss_path = os.path.join(best_checkpoint_path, "best_val_loss.pt")
                    torch.save(torch.tensor(self.best_val_loss), best_val_loss_path)
                    
                    # 保存训练状态
                    training_state_path = os.path.join(best_checkpoint_path, "training_state.pt")
                    torch.save({
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_val_loss": self.best_val_loss,
                        "best_val_loss_epoch": best_val_loss_epoch,
                    }, training_state_path)
                    
                    print(f"新的最佳模型已保存到 {best_checkpoint_path} (验证损失: {self.best_val_loss:.6f})")
                    
                    if self.args.use_wandb:
                        self.accelerator.log({
                            "best_val_loss": self.best_val_loss,
                            "best_val_loss_epoch": best_val_loss_epoch
                        }, step=global_step)
                else:
                    self.early_stop_counter += 1
                    print(f"验证损失未改善。早停计数器: {self.early_stop_counter}/{self.early_stop_patience}")
                    
                    # 检查是否触发早停
                    if self.early_stop_counter >= self.early_stop_patience:
                        print(f"触发早停！验证损失连续{self.early_stop_patience}轮未改善。")
                        break
            
            # 每save_epochs轮保存一次模型
            if (epoch + 1) % self.args.save_epochs == 0:
                # 删除之前可能存在的轮数保存模型
                previous_epoch_checkpoint = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch + 1 - self.args.save_epochs}")
                previous_epoch_pipeline = os.path.join(self.output_dir, f"pipeline-epoch-{epoch + 1 - self.args.save_epochs}")
                
                if os.path.exists(previous_epoch_checkpoint):
                    shutil.rmtree(previous_epoch_checkpoint)
                
                if os.path.exists(previous_epoch_pipeline):
                    shutil.rmtree(previous_epoch_pipeline)
                
                # 保存新的轮数模型
                # 先解除分布式包装
                unwrapped_model = self.accelerator.unwrap_model(self.unet)
                
                # 根据调度器类型选择对应的pipeline
                scheduler_type = self.args.scheduler_type
                
                if scheduler_type == "ddpm":
                    from diffusers import DDPMPipeline
                    pipeline = DDPMPipeline(
                        unet=unwrapped_model,
                        scheduler=self.noise_scheduler
                    )
                elif scheduler_type == "ddim":
                    from diffusers import DDIMPipeline
                    pipeline = DDIMPipeline(
                        unet=unwrapped_model,
                        scheduler=self.noise_scheduler
                    )
                elif scheduler_type == "pndm":
                    from diffusers import PNDMPipeline
                    pipeline = PNDMPipeline(
                        unet=unwrapped_model,
                        scheduler=self.noise_scheduler
                    )
                else:
                    # 默认使用DDIM
                    from diffusers import DDIMPipeline
                    pipeline = DDIMPipeline(
                        unet=unwrapped_model,
                        scheduler=self.noise_scheduler
                    )
                
                # 保存模型和pipeline
                save_path = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch + 1}")
                pipeline_path = os.path.join(self.output_dir, f"pipeline-epoch-{epoch + 1}")
                
                self.accelerator.save_state(save_path)
                pipeline.save_pretrained(pipeline_path)
                
                # 保存训练状态
                training_state_path = os.path.join(save_path, "training_state.pt")
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val_loss": self.best_val_loss,
                    "best_val_loss_epoch": best_val_loss_epoch,
                }, training_state_path)
                
                print(f"Epoch {epoch+1} 模型已保存到 {save_path}")
            
            if global_step >= self.num_train_steps:
                break
        
        # 保存最终模型
        unwrapped_model = self.accelerator.unwrap_model(self.unet)
        
        # 根据调度器类型选择对应的pipeline
        scheduler_type = self.args.scheduler_type
        
        if scheduler_type == "ddpm":
            from diffusers import DDPMPipeline
            pipeline = DDPMPipeline(
                unet=unwrapped_model,
                scheduler=self.noise_scheduler
            )
        elif scheduler_type == "ddim":
            from diffusers import DDIMPipeline
            pipeline = DDIMPipeline(
                unet=unwrapped_model,
                scheduler=self.noise_scheduler
            )
        elif scheduler_type == "pndm":
            from diffusers import PNDMPipeline
            pipeline = PNDMPipeline(
                unet=unwrapped_model,
                scheduler=self.noise_scheduler
            )
        else:
            # 默认使用DDIM
            from diffusers import DDIMPipeline
            pipeline = DDIMPipeline(
                unet=unwrapped_model,
                scheduler=self.noise_scheduler
            )
        
        pipeline.save_pretrained(self.output_dir)
        
        # 保存最终训练状态
        final_training_state_path = os.path.join(self.output_dir, "training_state.pt")
        torch.save({
            "epoch": self.current_epoch,
            "global_step": global_step,
            "best_val_loss": self.best_val_loss,
            "best_val_loss_epoch": best_val_loss_epoch,
        }, final_training_state_path)
        
        print(f"最终模型已保存到 {self.output_dir}")
        
        # 打印最佳验证结果
        print(f"训练结束。最佳验证损失: {self.best_val_loss:.6f} (轮次 {best_val_loss_epoch+1})")
        
        # 显示最终GPU内存使用情况
        if torch.cuda.is_available():
            print(f"训练结束后GPU内存占用: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
            print(f"训练过程中最大GPU内存占用: {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB") 