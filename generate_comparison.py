"""
生成三图对比可视化 - 展示原始图像、加噪图像和去噪生成图像
"""
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from diffusers import VQModel, DDIMScheduler
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

from dataset import get_dataloaders
from ldm.utils.visualization import save_generated_images

def parse_args():
    parser = argparse.ArgumentParser(description="生成三图对比可视化")
    parser.add_argument("--vqvae_path", type=str, default="vqvae_model", help="VQ-VAE模型路径")
    parser.add_argument("--ldm_path", type=str, default="ldm_model", help="LDM模型路径")
    parser.add_argument("--data_dir", type=str, default="dataset", help="数据集目录")
    parser.add_argument("--output_dir", type=str, default="comparison_images", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--num_samples", type=int, default=4, help="生成样本数量")
    parser.add_argument("--image_size", type=int, default=256, help="图像尺寸")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="推理步数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--beta_schedule", type=str, default="squaredcos_cap_v2", choices=["linear", "cosine", "squaredcos_cap_v2"], help="beta调度类型")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据集 - 用于获取原始图像
    _, val_dataloader, _ = get_dataloaders(args.data_dir, args.batch_size, args.image_size)
    
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
    print(f"批次大小: {args.batch_size}")
    print(f"推理步数: {args.num_inference_steps}")
    print(f"Beta调度: {args.beta_schedule}")
    
    # 生成三图对比可视化
    print("正在生成三图对比可视化...")
    save_generated_images(
        vq_model=vq_model,
        unet=unet,
        scheduler=scheduler,
        latent_size=latent_size,
        device=args.device,
        num_images=args.num_samples,
        num_inference_steps=args.num_inference_steps,
        output_dir=args.output_dir,
        step=0,
        seed=args.seed,
        dataloader=val_dataloader
    )
    
    print(f"三图对比可视化已保存到: {args.output_dir}")

if __name__ == "__main__":
    main() 