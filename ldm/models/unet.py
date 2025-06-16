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
    
    # 根据潜在空间大小调整UNet参数 (标准扩散模型配置)
    if latent_size >= 32:  # 较大的潜在空间
        block_out_channels = (256, 512, 768, 1024)  # 标准扩散模型通道配置
        layers_per_block = 2
        # 全覆盖注意力机制，所有层都加入注意力块
        down_block_types = ("AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")
        up_block_types = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D")
    else:  # 较小的潜在空间
        block_out_channels = (256, 512, 768)  # 标准三层结构
        layers_per_block = 2
        # 全覆盖注意力机制，所有层都加入注意力块
        down_block_types = ("AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")
        up_block_types = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D")
    
    # 增强注意力头数到16，提升注意力机制表达能力
    attention_head_dim = 16
    
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
    )
    print(f"通道配置: {block_out_channels}")
    print(f"注意力分布: 下采样={down_block_types}, 上采样={up_block_types}")
    return model
