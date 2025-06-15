"""
VQ-VAE 训练工具
"""
import torch
from tqdm import tqdm

@torch.no_grad()
def validate(dataloader, trainer, device, global_step, use_wandb=False, save_images=False, images_dir=None, epoch=0):
    """验证模型"""
    trainer.model.eval()
    val_loss = 0.0
    val_recon_loss = 0.0
    val_vq_loss = 0.0
    val_perceptual_loss = 0.0
    val_perplexity = 0.0
    perplexity_count = 0
    all_results = []
    
    # 只显示一个进度条
    val_bar = tqdm(dataloader, desc="验证中", leave=False)
    
    for batch in val_bar:
        images = batch.to(device)
        results, reconstructed = trainer.eval_step(images)
        
        # 累加损失
        val_loss += results['loss']
        val_recon_loss += results['recon_loss']
        val_vq_loss += results['vq_loss']
        
        if 'perceptual_loss' in results:
            val_perceptual_loss += results['perceptual_loss']
        
        if 'perplexity' in results:
            val_perplexity += results['perplexity']
            perplexity_count += 1
        
        all_results.append(results)
        
        # 更新进度条
        val_bar.set_postfix({
            "loss": f"{results['loss']:.4f}",
            "recon": f"{results['recon_loss']:.4f}"
        })
    
    # 计算平均损失
    num_batches = len(dataloader)
    avg_loss = val_loss / num_batches
    avg_recon_loss = val_recon_loss / num_batches
    avg_vq_loss = val_vq_loss / num_batches
    
    # 计算平均感知损失（如果使用了感知损失）
    results = {
        'loss': avg_loss,
        'recon_loss': avg_recon_loss,
        'vq_loss': avg_vq_loss,
    }
    
    if perplexity_count > 0:
        avg_perplexity = val_perplexity / perplexity_count
        results['perplexity'] = avg_perplexity
    
    if val_perceptual_loss > 0:
        avg_perceptual_loss = val_perceptual_loss / num_batches
        results['perceptual_loss'] = avg_perceptual_loss
    
    # 返回结果
    return results 