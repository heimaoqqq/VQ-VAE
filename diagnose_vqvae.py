"""
诊断VQ-VAE重建质量，生成原始图像与重建图像的对比图
"""
import os
import argparse
import torch
import numpy as np
from diffusers import VQModel
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

from dataset import get_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description="诊断VQ-VAE重建质量")
    parser.add_argument("--vqvae_path", type=str, required=True, help="预训练的VQ-VAE模型路径")
    parser.add_argument("--data_dir", type=str, required=True, help="数据集目录，用于获取原始图像")
    parser.add_argument("--output_file", type=str, default="vqvae_reconstruction_comparison.png", help="输出对比图像的文件名")
    parser.add_argument("--num_samples", type=int, default=8, help="用于对比的样本数量")
    parser.add_argument("--image_size", type=int, default=256, help="图像尺寸")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据集 - 只取一个批次用于对比
    _, val_dataloader, _ = get_dataloaders(args.data_dir, args.num_samples, args.image_size)
    original_images = next(iter(val_dataloader))
    original_images = original_images.to(args.device)
    
    # 仅保留所需数量的样本
    original_images = original_images[:args.num_samples]
    
    print(f"正在加载VQ-VAE模型: {args.vqvae_path}")
    # 加载VQ-VAE模型
    try:
        vq_model = VQModel.from_pretrained(args.vqvae_path).to(args.device)
        vq_model.eval()
    except Exception as e:
        raise ValueError(f"无法加载VQ-VAE模型: {e}")

    print("正在对原始图像进行编码和解码...")
    with torch.no_grad():
        # 编码: image -> latent
        latents = vq_model.encode(original_images).latents
        # 解码: latent -> reconstructed_image
        reconstructed_images = vq_model.decode(latents).sample

    # 将原始图像和重建图像拼接在一起进行对比
    # 格式: [原始图1, 重建图1, 原始图2, 重建图2, ...]
    comparison_list = []
    for i in range(args.num_samples):
        comparison_list.append(original_images[i].cpu())
        comparison_list.append(reconstructed_images[i].cpu())
        
    # 创建网格图像
    # nrow=2 表示每行显示2张图 (原始图, 重建图)
    grid = make_grid(comparison_list, nrow=2, padding=2, normalize=True)
    
    # 保存图像
    save_image(grid, args.output_file)
    
    print(f"\n诊断完成！")
    print(f"对比图像已保存到: {args.output_file}")
    print("请检查该文件，对比左右两列图像的细节差异。")
    print("左列是原始图像，右列是经过VQ-VAE重建后的图像。")

if __name__ == "__main__":
    main() 