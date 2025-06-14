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
        
        # 初始化accelerator
        self.accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            log_with="wandb" if args.use_wandb else None,
            gradient_accumulation_steps=1,
        )
        
        # 初始化噪声调度器
        scheduler_type = args.scheduler_type
        
        print(f"使用调度器: {scheduler_type.upper()}")
        
        if scheduler_type == "ddpm":
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_schedule=beta_schedule
            )
        elif scheduler_type == "ddim":
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_schedule=beta_schedule
            )
        elif scheduler_type == "pndm":
            self.noise_scheduler = PNDMScheduler(
                num_train_timesteps=1000,
                beta_schedule=beta_schedule
            )
        else:
            # 默认使用DDIM
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_schedule=beta_schedule
            )
        
        # 计算潜在空间大小
        self.latent_size = args.image_size // (2 ** (len(vq_model.config.down_block_types)))
        print(f"图像大小: {args.image_size}x{args.image_size}")
        print(f"VQ-VAE下采样层数: {len(vq_model.config.down_block_types)}")
        print(f"潜在空间大小: {self.latent_size}x{self.latent_size}")
        print(f"使用混合精度: {args.mixed_precision}")
        
        # 初始化优化器
        self.optimizer = AdamW(
            unet.parameters(), 
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-5 
        )
        
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
        
        # 初始化wandb
        if args.use_wandb:
            self.accelerator.init_trackers(
                project_name=args.wandb_project,
                init_kwargs={"wandb": {"name": args.wandb_name}}
            )
    
    def train(self):
        """执行完整的训练流程"""
        global_step = 0
        for epoch in range(self.args.epochs):
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
                    
                    # 梯度裁剪
                    if hasattr(self.accelerator, "clip_grad_norm_"):
                        self.accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                epoch_loss += loss.detach().item()
                tqdm_bar.set_postfix({"loss": loss.item()})
                
                # 记录日志
                if self.args.use_wandb and global_step % self.args.logging_steps == 0:
                    logs = {"train/loss": loss.detach().item(), "train/step": global_step}
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
                        step=global_step
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
                    self.accelerator.log({"val/loss": val_loss}, step=global_step)
                
                # 检查是否为最佳验证损失
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.early_stop_counter = 0
                    
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
                    print(f"新的最佳模型已保存到 {best_checkpoint_path} (验证损失: {self.best_val_loss:.6f})")
                    
                    if self.args.use_wandb:
                        self.accelerator.log({"best_val_loss": self.best_val_loss}, step=global_step)
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
        print(f"最终模型已保存到 {self.output_dir}")
        
        # 打印最佳验证结果
        print(f"训练结束。最佳验证损失: {self.best_val_loss:.6f}") 