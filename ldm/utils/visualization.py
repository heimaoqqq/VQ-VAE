"""
LDM 可视化工具
"""
import os
import torch
from tqdm import tqdm
from torchvision.utils import make_grid, save_image

def save_generated_images(vq_model, unet, scheduler, latent_size, device, num_images=4, num_inference_steps=50, output_dir=None, step=0):
    """
    生成并保存样本图像
    
    参数:
        vq_model: VQ-VAE模型
        unet: UNet模型
        scheduler: 噪声调度器
        latent_size: 潜在空间大小
        device: 设备
        num_images: 生成图像数量
        num_inference_steps: 推理步数，默认50(适用于DDIM)
        output_dir: 输出目录
        step: 当前步数
        
    返回:
        保存的图像路径
    """
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