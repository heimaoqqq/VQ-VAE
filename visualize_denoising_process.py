"""
可视化完整的扩散去噪过程
"""
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from diffusers import VQModel, DDIMScheduler
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torchvision.utils import make_grid, save_image

def parse_args():
    parser = argparse.ArgumentParser(description="可视化扩散模型去噪过程")
    parser.add_argument("--vqvae_path", type=str, default="vqvae_model", help="VQ-VAE模型路径")
    parser.add_argument("--ldm_path", type=str, default="ldm_model", help="LDM模型路径")
    parser.add_argument("--output_dir", type=str, default="denoising_process", help="输出目录")
    parser.add_argument("--num_samples", type=int, default=1, help="生成样本数量")
    parser.add_argument("--image_size", type=int, default=256, help="图像尺寸")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="推理步数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--beta_schedule", type=str, default="squaredcos_cap_v2", choices=["linear", "cosine", "squaredcos_cap_v2"], help="beta调度类型")
    parser.add_argument("--create_gif", action="store_true", help="是否创建GIF动画")
    parser.add_argument("--visualize_steps", type=int, default=10, help="可视化的步骤数")
    return parser.parse_args()

def visualize_denoising_process(vq_model, unet, scheduler, latent_size, device, num_samples=1, num_inference_steps=50, 
                               output_dir=None, seed=None, create_gif=False, visualize_steps=10):
    """
    可视化去噪过程
    
    参数:
        vq_model: VQ-VAE模型
        unet: UNet模型
        scheduler: 噪声调度器
        latent_size: 潜在空间大小
        device: 设备
        num_samples: 生成样本数量
        num_inference_steps: 推理步数
        output_dir: 输出目录
        seed: 随机种子
        create_gif: 是否创建GIF动画
        visualize_steps: 可视化的步骤数
    """
    # 设置随机种子以便复现
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 创建噪声潜变量
    latents = torch.randn(
        (num_samples, unet.config.in_channels, latent_size, latent_size),
        device=device
    )
    
    # 设置噪声调度器并获取步骤
    scheduler.set_timesteps(num_inference_steps)
    
    # 保存初始噪声图像
    init_latents = latents.clone()
    
    # 决定要保存的时间步
    if visualize_steps >= num_inference_steps:
        step_indices = list(range(num_inference_steps))
    else:
        step_indices = np.linspace(0, num_inference_steps-1, visualize_steps, dtype=int)
    
    # 用于存储中间结果
    intermediate_images = []
    
    # 去噪过程
    for idx, t in enumerate(tqdm(scheduler.timesteps, desc="去噪过程")):
        # 预测噪声
        with torch.no_grad():
            noise_pred = unet(latents, t).sample
        
        # 更新潜变量
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        # 如果当前步骤在选定的时间步中，保存图像
        if idx in step_indices or idx == len(scheduler.timesteps) - 1:
            with torch.no_grad():
                # 解码当前潜变量
                current_images = vq_model.decode(latents).sample
                # 归一化
                current_images = (current_images / 2 + 0.5).clamp(0, 1)
                # 保存到列表
                intermediate_images.append(current_images.cpu())
    
    # 解码初始噪声以进行可视化
    with torch.no_grad():
        try:
            noise_images = vq_model.decode(init_latents).sample
            noise_images = (noise_images / 2 + 0.5).clamp(0, 1)
        except:
            # 如果噪声无法直接解码，使用归一化的噪声
            noise_images = init_latents
            # 归一化到[0,1]
            noise_images = (noise_images - noise_images.min()) / (noise_images.max() - noise_images.min())
    
    # 可视化去噪过程
    if output_dir:
        for sample_idx in range(num_samples):
            # 为每个样本创建一个图像网格
            plt.figure(figsize=(20, 4))
            
            # 添加初始噪声
            plt.subplot(1, visualize_steps+1, 1)
            if noise_images.shape[1] == 1:  # 如果是单通道
                plt.imshow(noise_images[sample_idx, 0].cpu().numpy(), cmap='viridis')
            else:  # 如果是3通道
                plt.imshow(noise_images[sample_idx].permute(1, 2, 0).cpu().numpy())
            plt.title(f"初始噪声")
            plt.axis('off')
            
            # 添加中间步骤
            for step_idx, step_image in enumerate(intermediate_images):
                plt.subplot(1, visualize_steps+1, step_idx+2)
                plt.imshow(step_image[sample_idx].permute(1, 2, 0).cpu().numpy())
                step_num = step_indices[step_idx] if step_idx < len(step_indices) else num_inference_steps-1
                plt.title(f"步骤 {step_num}/{num_inference_steps-1}")
                plt.axis('off')
            
            # 保存图像
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"denoising_process_sample_{sample_idx}.png"), dpi=300)
            plt.close()
        
        # 创建GIF动画
        if create_gif and num_samples > 0:
            try:
                import imageio
                
                # 为每个样本创建GIF
                for sample_idx in range(num_samples):
                    gif_frames = []
                    
                    # 添加初始噪声帧
                    noise_frame = noise_images[sample_idx].permute(1, 2, 0).cpu().numpy()
                    if noise_frame.shape[2] == 1:  # 如果是单通道
                        noise_frame = np.concatenate([noise_frame] * 3, axis=2)  # 转换为三通道
                    gif_frames.append((noise_frame * 255).astype(np.uint8))
                    
                    # 添加中间帧
                    for step_image in intermediate_images:
                        frame = step_image[sample_idx].permute(1, 2, 0).cpu().numpy()
                        gif_frames.append((frame * 255).astype(np.uint8))
                    
                    # 保存GIF
                    gif_path = os.path.join(output_dir, f"denoising_process_sample_{sample_idx}.gif")
                    imageio.mimsave(gif_path, gif_frames, duration=0.2)  # 每帧0.2秒
                    print(f"已创建GIF: {gif_path}")
            except ImportError:
                print("未安装imageio库，无法创建GIF。请使用 'pip install imageio' 安装。")
            except Exception as e:
                print(f"创建GIF时出错: {e}")

def main():
    args = parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"正在加载VQ-VAE模型: {args.vqvae_path}")
    # 加载VQ-VAE模型
    try:
        vq_model = VQModel.from_pretrained(args.vqvae_path).to(args.device)
        vq_model.eval()
    except Exception as e:
        raise ValueError(f"无法加载VQ-VAE模型: {e}")
    
    # 加载UNet模型
    print(f"正在加载LDM模型: {args.ldm_path}")
    from diffusers import UNet2DModel
    
    # 尝试不同的路径加载UNet
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
                unet = UNet2DModel.from_pretrained(path).to(args.device)
                print(f"成功从 {path} 加载UNet模型")
                break
            except Exception:
                continue
    
    if unet is None:
        raise ValueError(f"无法在{args.ldm_path}中找到有效的UNet模型")
    
    # 加载DDIM调度器
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule=args.beta_schedule
    )
    
    # 计算潜在空间大小
    latent_size = args.image_size // (2 ** len(vq_model.config.down_block_types))
    print(f"图像尺寸: {args.image_size}x{args.image_size}")
    print(f"潜在空间尺寸: {latent_size}x{latent_size}")
    print(f"使用设备: {args.device}")
    print(f"推理步数: {args.num_inference_steps}")
    print(f"可视化步骤数: {args.visualize_steps}")
    print(f"Beta调度: {args.beta_schedule}")
    
    # 可视化去噪过程
    print("正在可视化去噪过程...")
    visualize_denoising_process(
        vq_model=vq_model,
        unet=unet, 
        scheduler=scheduler,
        latent_size=latent_size,
        device=args.device,
        num_samples=args.num_samples,
        num_inference_steps=args.num_inference_steps,
        output_dir=args.output_dir,
        seed=args.seed,
        create_gif=args.create_gif,
        visualize_steps=args.visualize_steps
    )
    
    print(f"去噪过程可视化已保存到: {args.output_dir}")

if __name__ == "__main__":
    main() 