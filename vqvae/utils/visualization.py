"""
VQ-VAE 可视化工具
"""
import os
import torch
from torchvision.utils import make_grid, save_image

def save_reconstructed_images(original, reconstructed, epoch, step, output_dir):
    """保存原始图像和重建图像对比"""
    # 将张量转换为[0,1]范围的图像
    original = (original + 1) / 2
    reconstructed = (reconstructed + 1) / 2
    
    # 创建网格图像
    batch_size = original.size(0)
    comparison = torch.cat([original, reconstructed], dim=0)
    save_path = os.path.join(output_dir, f"recon_epoch{epoch}_step{step}.png")
    
    # 保存图像
    grid = make_grid(comparison, nrow=batch_size)
    save_image(grid, save_path)
    
    return save_path 