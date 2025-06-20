"""
LDM 可视化工具
"""
import os
import torch
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

def save_generated_images(vq_model, unet, scheduler, latent_size, device, num_images=4, num_inference_steps=50, output_dir=None, step=0, guidance_scale=1.0, seed=None, dataloader=None):
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
        guidance_scale: 无条件引导尺度 (目前固定为1.0，为将来实现类别条件生成预留)
        seed: 随机种子，为None时使用随机生成
        dataloader: 数据加载器，用于获取原始图像进行三图对比
        
    返回:
        保存的图像路径
    """
    # 设置随机种子以便复现
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    # 创建噪声潜变量
    latents = torch.randn(
        (num_images, unet.config.in_channels, latent_size, latent_size),
        device=device
    )
    
    # 记录原始潜变量
    orig_latents = latents.clone()
    
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
        # 创建网格布局
        grid = make_grid(images, nrow=2)
        save_path = os.path.join(output_dir, f"generated_step{step}.png")
        save_image(grid, save_path)
        
        # 如果生成的样本数量大于1，也单独保存每个样本
        if num_images > 1:
            samples_dir = os.path.join(output_dir, f"samples_step{step}")
            os.makedirs(samples_dir, exist_ok=True)
            
            for i, img in enumerate(images):
                sample_path = os.path.join(samples_dir, f"sample_{i}.png")
                save_image(img, sample_path)
            
            # 创建一个直观的对比图，显示噪声和生成的图像
            if num_images <= 8:  # 限制显示数量，避免图像过多
                plt.figure(figsize=(18, 4 * (num_images // 2)))
                
                # 尝试获取原始图像进行三图对比
                original_images = None
                if dataloader is not None:
                    try:
                        # 获取一批真实图像样本
                        original_batch = next(iter(dataloader))[:num_images].to(device)
                        
                        # 使用VQ-VAE编码到潜在空间
                        with torch.no_grad():
                            original_latents = vq_model.encode(original_batch).latents
                            
                            # 添加噪声 - 选择最大噪声步骤(T)
                            noise = torch.randn_like(original_latents)
                            timesteps = torch.ones((original_latents.shape[0],), device=device).long() * (scheduler.num_train_timesteps - 1)
                            noisy_latents = scheduler.add_noise(original_latents, noise, timesteps)
                            
                            # 解码噪声潜在变量以可视化
                            noisy_images = vq_model.decode(noisy_latents).sample
                            noisy_images = (noisy_images / 2 + 0.5).clamp(0, 1)
                            
                            # 获取原始图像用于可视化
                            original_images = (original_batch / 2 + 0.5).clamp(0, 1)
                    except Exception as e:
                        print(f"获取原始图像失败: {e}")
                        original_images = None
                
                # 如果成功获取了原始图像，进行三图对比
                if original_images is not None:
                    for i in range(num_images):
                        # 原始图像
                        plt.subplot(num_images, 3, i*3+1)
                        orig_img = original_images[i].detach().cpu().permute(1, 2, 0)
                        plt.imshow(orig_img)
                        plt.title(f"原始图像 {i}")
                        plt.axis('off')
                        
                        # 加噪图像
                        plt.subplot(num_images, 3, i*3+2)
                        noisy_img = noisy_images[i].detach().cpu().permute(1, 2, 0)
                        plt.imshow(noisy_img)
                        plt.title(f"完全加噪 {i}")
                        plt.axis('off')
                        
                        # 生成的图像
                        plt.subplot(num_images, 3, i*3+3)
                        gen_img = images[i].detach().cpu().permute(1, 2, 0)
                        plt.imshow(gen_img)
                        plt.title(f"生成图像 {i}")
                        plt.axis('off')
                else:
                    # 如果没有原始图像，则展示噪声和生成图像的对比
                    for i in range(num_images):
                        # 原始噪声
                        plt.subplot(num_images, 2, i*2+1)
                        noise_img = orig_latents[i].detach().cpu().permute(1, 2, 0)
                        noise_img = (noise_img - noise_img.min()) / (noise_img.max() - noise_img.min())
                        plt.imshow(noise_img[:, :, 0], cmap='viridis')
                        plt.title(f"初始噪声 {i}")
                        plt.axis('off')
                        
                        # 生成的图像
                        plt.subplot(num_images, 2, i*2+2)
                        gen_img = images[i].detach().cpu().permute(1, 2, 0)
                        plt.imshow(gen_img)
                        plt.title(f"生成图像 {i}")
                        plt.axis('off')
                
                plt.tight_layout()
                compare_path = os.path.join(output_dir, f"comparison_step{step}.png")
                plt.savefig(compare_path)
                plt.close()
        
        return save_path
    
    return None 

def visualize_attention_maps(model, latents, timestep, save_path=None):
    """
    可视化注意力图
    
    参数:
        model: UNet模型
        latents: 输入潜变量 [1, C, H, W]
        timestep: 时间步
        save_path: 保存路径
    
    返回:
        None
    """
    # 目前仅作为占位符
    # 未来可以实现从UNet中提取注意力权重并可视化的功能
    pass 