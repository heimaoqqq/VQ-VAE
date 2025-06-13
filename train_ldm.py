import os
import argparse
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import sys

# 添加版本兼容性检查
try:
    import wandb
    from diffusers import (
        UNet2DModel,
        DDPMScheduler,
        DDPMPipeline,
        VQModel
    )
except ImportError as e:
    if "cannot import name 'cached_download' from 'huggingface_hub'" in str(e):
        print("检测到huggingface_hub版本不兼容。尝试安装兼容版本...")
        os.system("pip install huggingface_hub==0.16.4 diffusers==0.26.3 --force-reinstall")
        print("请重新运行脚本")
        sys.exit(1)
    else:
        raise e

from dataset import get_dataloaders
from accelerate import Accelerator
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

def parse_args():
    parser = argparse.ArgumentParser(description="训练潜在扩散模型(LDM)")
    parser.add_argument("--data_dir", type=str, default="dataset", help="数据集目录")
    parser.add_argument("--vqvae_model_path", type=str, default="vqvae_model", help="VQ-VAE模型路径")
    parser.add_argument("--output_dir", type=str, default="ldm_model", help="模型保存目录")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--image_size", type=int, default=256, help="图像尺寸")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--num_train_steps", type=int, default=None, help="训练步数")
    parser.add_argument("--save_steps", type=int, default=500, help="保存间隔步数")
    parser.add_argument("--eval_steps", type=int, default=1000, help="评估间隔步数")
    parser.add_argument("--logging_steps", type=int, default=100, help="日志间隔步数")
    parser.add_argument("--latent_channels", type=int, default=4, help="潜变量通道数")
    parser.add_argument("--save_images", action="store_true", help="是否保存生成图像")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="推理步数")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb记录训练")
    parser.add_argument("--wandb_project", type=str, default="ldm-microdoppler", help="wandb项目名")
    parser.add_argument("--wandb_name", type=str, default="ldm-training", help="wandb运行名")
    parser.add_argument("--kaggle", action="store_true", help="是否在Kaggle环境中运行")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="混合精度训练")
    return parser.parse_args()

def create_unet_model(args, vq_model):
    """创建UNet模型"""
    # 计算潜在空间分辨率
    latent_size = args.image_size // (2 ** (len(vq_model.config.down_block_types)))
    print(f"潜在空间分辨率: {latent_size}x{latent_size}")
    
    # 根据潜在空间大小调整UNet参数
    if latent_size >= 32:  # 较大的潜在空间
        block_out_channels = (256, 512, 768, 1024)
        layers_per_block = 2
    else:  # 较小的潜在空间
        block_out_channels = (256, 512, 768) 
        layers_per_block = 3
    
    model = UNet2DModel(
        sample_size=latent_size,  # 潜在空间分辨率
        in_channels=args.latent_channels,  # 输入通道数
        out_channels=args.latent_channels,  # 输出通道数
        layers_per_block=layers_per_block,  # 每个块中的层数
        block_out_channels=block_out_channels,  # 每个块的输出通道数
        down_block_types=(
            "DownBlock2D",
        ) * len(block_out_channels),
        up_block_types=(
            "UpBlock2D",
        ) * len(block_out_channels),
    )
    return model

def save_generated_images(vq_model, unet, scheduler, latent_size, device, num_images=4, num_inference_steps=50, output_dir=None, step=0):
    """生成并保存样本图像"""
    # 创建噪声潜变量
    latents = torch.randn(
        (num_images, unet.config.in_channels, latent_size, latent_size),
        device=device
    )
    
    # 设置噪声调度器
    scheduler.set_timesteps(num_inference_steps)
    
    # 去噪过程
    for t in tqdm(scheduler.timesteps, desc="生成样本"):
        # 预测噪声
        with torch.no_grad():
            noise_pred = unet(latents, t).sample
        
        # 更新潜变量
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    # 使用VQ-VAE解码潜变量
    with torch.no_grad():
        images = vq_model.decode(latents).sample
    
    # 归一化到[0,1]
    images = (images / 2 + 0.5).clamp(0, 1)
    
    # 保存图像
    if output_dir:
        grid = make_grid(images, nrow=2)
        save_path = os.path.join(output_dir, f"generated_step{step}.png")
        save_image(grid, save_path)
        return save_path
    
    return None

def train_ldm(args):
    """训练潜在扩散模型"""
    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "generated_images")
    os.makedirs(images_dir, exist_ok=True)
    
    # 初始化accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with="wandb" if args.use_wandb else None,
        gradient_accumulation_steps=1,
    )
    
    if args.use_wandb:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            init_kwargs={"wandb": {"name": args.wandb_name}}
        )
    
    # 创建数据加载器
    train_dataloader, val_dataloader, _ = get_dataloaders(
        args.data_dir, args.batch_size, args.image_size
    )
    
    # 加载预训练的VQ-VAE模型
    vq_model = VQModel.from_pretrained(args.vqvae_model_path)
    vq_model.eval().requires_grad_(False)  # 冻结VQ-VAE参数
    
    # 创建UNet模型和噪声调度器
    model = create_unet_model(args, vq_model)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    
    # 打印潜在空间大小
    latent_size = args.image_size // (2 ** (len(vq_model.config.down_block_types)))
    print(f"图像大小: {args.image_size}x{args.image_size}")
    print(f"VQ-VAE下采样层数: {len(vq_model.config.down_block_types)}")
    print(f"潜在空间大小: {latent_size}x{latent_size}")
    print(f"使用混合精度: {args.mixed_precision}")
    
    # 优化器
    optimizer = AdamW(
        model.parameters(), 
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-5 
    )
    
    # 使用accelerator准备模型和数据加载器
    vq_model, model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        vq_model, model, optimizer, train_dataloader, val_dataloader
    )
    
    # 训练步数
    if args.num_train_steps is None:
        args.num_train_steps = args.epochs * len(train_dataloader)
    
    # 训练循环
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        tqdm_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step, batch in enumerate(tqdm_bar):
            # 使用VQ-VAE编码图像到潜在空间
            with torch.no_grad():
                latents = vq_model.encode(batch).latents
            
            # 添加噪声
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # 预测噪声
            with accelerator.accumulate(model):
                noise_pred = model(noisy_latents, timesteps).sample
                loss = F.mse_loss(noise_pred, noise)
                
                # 反向传播
                accelerator.backward(loss)
                
                # 梯度裁剪
                if hasattr(accelerator, "clip_grad_norm_"):
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.detach().item()
            tqdm_bar.set_postfix({"loss": loss.item()})
            
            # 记录日志
            if args.use_wandb and global_step % args.logging_steps == 0:
                logs = {"train/loss": loss.detach().item(), "train/step": global_step}
                accelerator.log(logs, step=global_step)
            
            # 评估和生成样本
            if args.save_images and global_step % args.eval_steps == 0:
                # 解包装模型
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_vq_model = accelerator.unwrap_model(vq_model)
                
                # 生成样本
                img_path = save_generated_images(
                    unwrapped_vq_model,
                    unwrapped_model,
                    noise_scheduler,
                    latent_size,
                    accelerator.device,
                    num_images=4,
                    num_inference_steps=args.num_inference_steps,
                    output_dir=images_dir,
                    step=global_step
                )
                
                print(f"生成的样本已保存到: {img_path}")
                
                if args.use_wandb:
                    accelerator.log({
                        "generated_samples": wandb.Image(img_path)
                    }, step=global_step)
            
            # 保存模型
            if (global_step > 0 and global_step % args.save_steps == 0) or global_step == args.num_train_steps - 1:
                # 先解除分布式包装
                unwrapped_model = accelerator.unwrap_model(model)
                pipeline = DDPMPipeline(
                    unet=unwrapped_model,
                    scheduler=noise_scheduler
                )
                
                # 保存模型和pipeline
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                pipeline.save_pretrained(os.path.join(args.output_dir, f"pipeline-{global_step}"))
                print(f"保存模型到 {save_path}")
            
            global_step += 1
            
            if global_step >= args.num_train_steps:
                break
        
        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.6f}")
        
        # 验证
        if val_dataloader:
            val_loss = validate(model, vq_model, val_dataloader, noise_scheduler, accelerator)
            print(f"验证损失: {val_loss:.6f}")
            
            if args.use_wandb:
                accelerator.log({"val/loss": val_loss}, step=global_step)
        
        if global_step >= args.num_train_steps:
            break
    
    # 保存最终模型
    unwrapped_model = accelerator.unwrap_model(model)
    pipeline = DDPMPipeline(
        unet=unwrapped_model,
        scheduler=noise_scheduler
    )
    pipeline.save_pretrained(args.output_dir)
    print(f"最终模型已保存到 {args.output_dir}")

def validate(model, vq_model, dataloader, noise_scheduler, accelerator):
    """验证模型"""
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            # 使用VQ-VAE编码图像到潜在空间
            latents = vq_model.encode(batch).latents
            
            # 添加噪声
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # 预测噪声
            noise_pred = model(noisy_latents, timesteps).sample
            loss = F.mse_loss(noise_pred, noise)
            val_loss += loss.item()
    
    val_loss /= len(dataloader)
    return val_loss

if __name__ == "__main__":
    args = parse_args()
    print(f"正在使用设备: {args.device}")
    train_ldm(args) 