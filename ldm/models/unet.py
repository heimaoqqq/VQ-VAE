"""
LDM UNet模型定义 (优化版本)
"""
import torch
from diffusers import UNet2DModel

def create_unet_model(latent_size, latent_channels=4):
    """
    创建UNet模型，增强自注意力机制
    
    参数:
        latent_size: 潜在空间分辨率
        latent_channels: 潜在通道数
        
    返回:
        UNet2DModel实例
    """
    print(f"潜在空间分辨率: {latent_size}x{latent_size}")
    
    # 根据潜在空间大小调整UNet参数 (优化版本)
    if latent_size >= 32:  # 较大的潜在空间
        block_out_channels = (192, 384, 512, 640)  # 优化的四层结构，减少高层通道数以节省显存
        layers_per_block = 2
        # 增强注意力机制，在多个层加入注意力块
        down_block_types = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D")
        up_block_types = ("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")
    else:  # 较小的潜在空间
        block_out_channels = (192, 384, 512)  # 优化的三层结构
        layers_per_block = 2
        # 增强注意力机制，在多个层加入注意力块
        down_block_types = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")
        up_block_types = ("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D")
    
    # 保持8头自注意力，平衡性能和显存使用
    attention_head_dim = 8
    
    print(f"模型结构: 下采样块={down_block_types}, 上采样块={up_block_types}")
    print(f"注意力头数: {attention_head_dim}")
    
    model = UNet2DModel(
        sample_size=latent_size,  # 潜在空间分辨率
        in_channels=latent_channels,  # 输入通道数
        out_channels=latent_channels,  # 输出通道数
        layers_per_block=layers_per_block,  # 每个块中的层数
        block_out_channels=block_out_channels,  # 每个块的输出通道数
        down_block_types=down_block_types,  # 下采样块类型，添加增强的注意力
        up_block_types=up_block_types,  # 上采样块类型，添加增强的注意力
        attention_head_dim=attention_head_dim,  # 设置8头自注意力
        use_memory_efficient_attention=True,  # 使用内存高效的注意力实现
    )
    print(f"通道配置: {block_out_channels}")
    print(f"注意力分布: 下采样={down_block_types}, 上采样={up_block_types}")
    print(f"启用内存高效注意力: True")
    return model
