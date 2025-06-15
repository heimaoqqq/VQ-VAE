"""
VQ-VAE 训练工具
"""
import torch
from tqdm import tqdm

@torch.no_grad()
def validate(model, dataloader, device):
    """验证模型
    
    参数:
        model: VQ-VAE模型
        dataloader: 验证数据加载器
        device: 设备
        
    返回:
        包含各种损失指标的字典
    """
    model.eval()
    val_loss = 0.0
    val_recon_loss = 0.0
    val_vq_loss = 0.0
    val_perceptual_loss = 0.0
    val_positional_loss = 0.0
    val_perplexity = 0.0
    perplexity_count = 0
    
    # 只显示一个进度条
    val_bar = tqdm(dataloader, desc="验证中", leave=False)
    
    for batch in val_bar:
        images = batch.to(device)
        
        # 前向传播
        outputs = model(images)
        
        # 计算损失
        loss_dict = model.module.loss_function(outputs) if hasattr(model, "module") else model.loss_function(outputs)
        
        # 累加损失
        val_loss += loss_dict['loss']
        val_recon_loss += loss_dict['recon_loss']
        val_vq_loss += loss_dict['vq_loss']
        
        if 'perceptual_loss' in loss_dict:
            val_perceptual_loss += loss_dict['perceptual_loss']
        
        if 'positional_loss' in loss_dict:
            val_positional_loss += loss_dict['positional_loss']
        
        if 'perplexity' in loss_dict:
            val_perplexity += loss_dict['perplexity']
            perplexity_count += 1
        
        # 更新进度条
        val_bar.set_postfix({
            "loss": f"{loss_dict['loss']:.4f}",
            "recon": f"{loss_dict['recon_loss']:.4f}"
        })
    
    # 计算平均损失
    num_batches = len(dataloader)
    avg_loss = val_loss / num_batches
    avg_recon_loss = val_recon_loss / num_batches
    avg_vq_loss = val_vq_loss / num_batches
    
    # 构建结果字典
    results = {
        'loss': avg_loss,
        'recon_loss': avg_recon_loss,
        'vq_loss': avg_vq_loss,
    }
    
    # 添加其他损失（如果存在）
    if perplexity_count > 0:
        avg_perplexity = val_perplexity / perplexity_count
        results['perplexity'] = avg_perplexity
    
    if val_perceptual_loss > 0:
        avg_perceptual_loss = val_perceptual_loss / num_batches
        results['perceptual_loss'] = avg_perceptual_loss
    
    if val_positional_loss > 0:
        avg_positional_loss = val_positional_loss / num_batches
        results['positional_loss'] = avg_positional_loss
    
    # 返回结果
    return results 