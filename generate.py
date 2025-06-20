import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from diffusers import (
    VQModel,
    DDPMScheduler,
    DDPMPipeline
)
from torchvision.utils import make_grid, save_image

def parse_args():
    parser = argparse.ArgumentParser(description="生成微多普勒时频图")
    parser.add_argument("--vqvae_path", type=str, default="vqvae_model", help="VQ-VAE模型路径")
    parser.add_argument("--ldm_path", type=str, default="ldm_model", help="LDM模型路径")
    parser.add_argument("--output_dir", type=str, default="generated_images", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--num_samples", type=int, default=16, help="生成样本数量")
    parser.add_argument("--image_size", type=int, default=256, help="图像尺寸")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="推理步数，DDIM默认50步已足够")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--grid", action="store_true", help="是否生成网格图像")
    parser.add_argument("--scheduler_type", type=str, default="ddim", choices=["ddpm", "ddim", "pndm"], help="采样器类型，默认为DDIM")
    parser.add_argument("--beta_schedule", type=str, default="squaredcos_cap_v2", choices=["linear", "cosine", "squaredcos_cap_v2"], help="beta调度类型，默认为squaredcos_cap_v2")
    return parser.parse_args()

def generate_images(args):
    """使用训练好的模型生成微多普勒时频图"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    print(f"正在加载VQ-VAE模型: {args.vqvae_path}")
    # 加载VQ-VAE模型
    try:
        vq_model = VQModel.from_pretrained(args.vqvae_path).to(args.device)
        vq_model.eval()
    except Exception as e:
        raise ValueError(f"无法加载VQ-VAE模型: {e}")
    
    # 尝试加载Pipeline或者单独的UNet模型
    print(f"正在加载LDM模型: {args.ldm_path}")
    try:
        # 尝试直接加载Pipeline
        pipe = DDPMPipeline.from_pretrained(args.ldm_path).to(args.device)
        print("成功加载DDPMPipeline")
    except Exception as e:
        print(f"无法直接加载Pipeline: {e}")
        print("尝试加载单独的UNet模型...")
        
        from diffusers import UNet2DModel
        
        # 根据参数选择采样器类型
        try:
            if args.scheduler_type.lower() == "ddim":
                from diffusers import DDIMScheduler
                scheduler = DDIMScheduler.from_pretrained(args.ldm_path)
                print("使用DDIM采样器")
            elif args.scheduler_type.lower() == "pndm":
                from diffusers import PNDMScheduler
                scheduler = PNDMScheduler.from_pretrained(args.ldm_path)
                print("使用PNDM采样器")
            else:
                scheduler = DDPMScheduler.from_pretrained(args.ldm_path)
                print("使用DDPM采样器")
        except Exception as e:
            print(f"无法加载特定调度器: {e}，使用默认配置")
            # 使用指定的调度器和beta_schedule
            if args.scheduler_type.lower() == "ddim":
                from diffusers import DDIMScheduler
                scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule=args.beta_schedule)
                print(f"使用DDIM采样器 + {args.beta_schedule} beta调度")
            elif args.scheduler_type.lower() == "pndm":
                from diffusers import PNDMScheduler
                scheduler = PNDMScheduler(num_train_timesteps=1000, beta_schedule=args.beta_schedule)
                print(f"使用PNDM采样器 + {args.beta_schedule} beta调度")
            else:
                scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule=args.beta_schedule)
                print(f"使用DDPM采样器 + {args.beta_schedule} beta调度")
        
        # 尝试不同的路径加载UNet
        possible_unet_paths = [
            os.path.join(args.ldm_path, "unet"),
            args.ldm_path,
            os.path.join(args.ldm_path, "best-pipeline", "unet"),
            os.path.join(args.ldm_path, "best-checkpoint", "unet")
        ]
        
        unet = None
        for path in possible_unet_paths:
            if os.path.exists(path):
                try:
                    unet = UNet2DModel.from_pretrained(path).to(args.device)
                    print(f"成功从 {path} 加载UNet模型")
                    break
                except Exception:
                    continue
        
        if unet is None:
            raise ValueError(f"无法在{args.ldm_path}中找到有效的UNet模型")
        
        # 构建Pipeline
        pipe = DDPMPipeline(unet=unet, scheduler=scheduler)
    
    # 计算潜在空间分辨率
    latent_size = args.image_size // (2 ** len(vq_model.config.down_block_types))
    print(f"图像尺寸: {args.image_size}x{args.image_size}")
    print(f"潜在空间尺寸: {latent_size}x{latent_size}")
    print(f"使用设备: {args.device}")
    print(f"批次大小: {args.batch_size}")
    print(f"生成样本数: {args.num_samples}")
    print(f"推理步数: {args.num_inference_steps}")
    
    # 生成样本
    all_images = []
    for i in tqdm(range(0, args.num_samples, args.batch_size), desc="生成样本"):
        # 确定当前批次的实际大小
        current_batch_size = min(args.batch_size, args.num_samples - i)
        
        # 在潜在空间生成样本
        latents_shape = (current_batch_size, pipe.unet.config.in_channels, latent_size, latent_size)
        latents = torch.randn(latents_shape, device=args.device)
        
        # 使用DDPM去噪过程
        latents = pipe(
            batch_size=current_batch_size,
            generator=torch.Generator(device=args.device).manual_seed(args.seed + i),
            num_inference_steps=args.num_inference_steps,
            output_type="latent",
        ).images
        
        # 使用VQ-VAE解码
        with torch.no_grad():
            images = vq_model.decode(latents).sample.cpu()
            
        # 转换为[0,1]区间
        images = (images / 2 + 0.5).clamp(0, 1)
        
        # 保存单独的图像
        for j, image in enumerate(images):
            if not args.grid:
                save_image(image, os.path.join(args.output_dir, f"sample_{i+j+1:04d}.png"))
            all_images.append(image)
    
    # 生成网格图像
    if args.grid and all_images:
        grid = make_grid(all_images, nrow=int(np.sqrt(len(all_images))))
        grid_path = os.path.join(args.output_dir, f"grid_{args.num_samples}_images.png")
        save_image(grid, grid_path)
        print(f"生成的网格图像已保存到: {grid_path}")
    
    print(f"已成功生成 {min(args.num_samples, len(all_images))} 张图像")

def create_custom_pipeline(args):
    """创建自定义Pipeline，将VQ-VAE和LDM组合在一起"""
    from diffusers import DiffusionPipeline, UNet2DModel
    
    # 加载模型
    try:
        vq_model = VQModel.from_pretrained(args.vqvae_path)
        print(f"成功加载VQ-VAE模型: {args.vqvae_path}")
    except Exception as e:
        raise ValueError(f"无法加载VQ-VAE模型: {e}")
    
    # 检查LDM是否已经是完整的Pipeline
    try:
        # 首先尝试加载UNet
        possible_unet_paths = [
            os.path.join(args.ldm_path, "unet"),
            os.path.join(args.ldm_path, "best-pipeline", "unet"),
            os.path.join(args.ldm_path, "best-checkpoint", "unet"),
            args.ldm_path
        ]
        
        unet = None
        for path in possible_unet_paths:
            if os.path.exists(path):
                try:
                    unet = UNet2DModel.from_pretrained(path)
                    print(f"成功从 {path} 加载UNet模型")
                    break
                except Exception:
                    continue
        
        if unet is None:
            raise ValueError(f"无法在{args.ldm_path}中找到有效的UNet模型")
        
        # 尝试加载调度器
        try:
            if args.scheduler_type.lower() == "ddim":
                from diffusers import DDIMScheduler
                scheduler = DDIMScheduler.from_pretrained(args.ldm_path)
                print(f"从模型加载DDIM调度器")
            elif args.scheduler_type.lower() == "pndm":
                from diffusers import PNDMScheduler
                scheduler = PNDMScheduler.from_pretrained(args.ldm_path)
                print(f"从模型加载PNDM调度器")
            else:
                from diffusers import DDPMScheduler
                scheduler = DDPMScheduler.from_pretrained(args.ldm_path)
                print(f"从模型加载DDPM调度器")
        except Exception:
            print(f"无法加载特定的调度器，使用默认配置{args.scheduler_type.upper()} + {args.beta_schedule}")
            if args.scheduler_type.lower() == "ddim":
                from diffusers import DDIMScheduler
                scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule=args.beta_schedule)
            elif args.scheduler_type.lower() == "pndm":
                from diffusers import PNDMScheduler
                scheduler = PNDMScheduler(num_train_timesteps=1000, beta_schedule=args.beta_schedule)
            else:
                from diffusers import DDPMScheduler
                scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule=args.beta_schedule)
    except Exception as e:
        print(f"加载模型组件时出错: {e}")
        return None
    
    # 创建自定义Pipeline
    class VQDiffusionPipeline(DiffusionPipeline):
        def __init__(self, vqvae, unet, scheduler):
            super().__init__()
            self.register_modules(
                vqvae=vqvae,
                unet=unet,
                scheduler=scheduler,
            )
        
        @torch.no_grad()
        def __call__(self, batch_size=1, generator=None, num_inference_steps=1000, output_type="pil"):
            # 生成随机潜变量
            latent_size = self.unet.config.sample_size
            shape = (batch_size, self.unet.config.in_channels, latent_size, latent_size)
            latents = torch.randn(shape, generator=generator, device=self.device)
            
            # 设置调度器
            self.scheduler.set_timesteps(num_inference_steps)
            
            # 去噪步骤
            for t in tqdm(self.scheduler.timesteps):
                # 扩展时间步到批次大小
                timestep = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                
                # 预测噪声残差
                noise_pred = self.unet(latents, timestep).sample
                
                # 更新样本
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            # 解码潜变量
            images = self.vqvae.decode(latents).sample
            
            # 后处理
            images = (images / 2 + 0.5).clamp(0, 1)
            
            if output_type == "pil":
                images = images.cpu().permute(0, 2, 3, 1).numpy()
                images = (images * 255).astype(np.uint8)
                images = [Image.fromarray(image) for image in images]
            
            return {"images": images, "latents": latents}
    
    # 实例化Pipeline
    pipeline = VQDiffusionPipeline(
        vqvae=vq_model,
        unet=unet,
        scheduler=scheduler,
    )
    
    # 保存Pipeline
    pipeline_path = os.path.join(args.output_dir, "vq_diffusion_pipeline")
    os.makedirs(pipeline_path, exist_ok=True)
    pipeline.save_pretrained(pipeline_path)
    
    print(f"自定义Pipeline已保存到: {pipeline_path}")
    return pipeline

if __name__ == "__main__":
    args = parse_args()
    print(f"正在使用设备: {args.device}")
    print(f"采样器类型: {args.scheduler_type.upper()}")
    print(f"Beta调度: {args.beta_schedule}")
    print(f"推理步数: {args.num_inference_steps}")
    
    # 检查是否需要创建自定义Pipeline
    if args.vqvae_path and args.ldm_path:
        # 自动创建Pipeline而不需要交互式输入
        create_pipeline = True  # 默认创建Pipeline
        # create_pipeline = input("是否创建并保存自定义Pipeline？(y/n): ").lower() == "y"
        if create_pipeline:
            create_custom_pipeline(args)
    
    generate_images(args) 