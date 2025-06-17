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
    
    # 根据潜在空间大小调整UNet参数
    if latent_size >= 32:  # 例如 32x32 或 64x64 的潜在空间
        block_out_channels = (128, 256, 512, 512)
        layers_per_block = 2
        # 混合注意力：仅在低分辨率层 (16x16, 8x8) 使用注意力
        down_block_types = (
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        )
        up_block_types = (
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        )
    else:  # 较小的潜在空间, 例如 16x16
        block_out_channels = (256, 512, 512)
        layers_per_block = 2
        # 混合注意力：仅在低分辨率层 (8x8) 使用注意力
        down_block_types = (
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        )
        up_block_types = (
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        )

    # 标准化注意力头维度
    attention_head_dim = 8
    
    print(f"模型结构: 下采样块={down_block_types}, 上采样块={up_block_types}")
    print(f"每头注意力维度: {attention_head_dim}")
    
    model = UNet2DModel(
        sample_size=latent_size,  # 潜在空间分辨率
        in_channels=latent_channels,  # 输入通道数
        out_channels=latent_channels,  # 输出通道数
        layers_per_block=layers_per_block,  # 每个块中的层数
        block_out_channels=block_out_channels,  # 每个块的输出通道数
        down_block_types=down_block_types,  # 下采样块类型，混合标准块和注意力块
        up_block_types=up_block_types,  # 上采样块类型，混合标准块和注意力块
        attention_head_dim=attention_head_dim,  # 设置每个注意力头的维度
    )
    print(f"通道配置: {block_out_channels}")
    print(f"注意力分布: 下采样={down_block_types}, 上采样={up_block_types}")
    return model 
