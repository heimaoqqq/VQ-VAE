"""
LDM 训练工具
"""
import torch
import torch.nn.functional as F

@torch.no_grad()
def validate(model, vq_model, dataloader, noise_scheduler, accelerator):
    """
    验证LDM模型
    
    参数:
        model: UNet模型
        vq_model: VQ-VAE模型
        dataloader: 验证数据加载器
        noise_scheduler: 噪声调度器
        accelerator: Accelerator对象
        
    返回:
        验证损失
    """
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            # 使用VQ-VAE编码图像到潜在空间
            latents = vq_model.encode(batch).latents
            
            # 使用固定随机种子生成噪声和时间步
            with torch.random.fork_rng():
                torch.manual_seed(42)  # 固定种子
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