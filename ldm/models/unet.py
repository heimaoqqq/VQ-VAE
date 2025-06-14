"""
LDM UNet模型定义
"""
import torch
from diffusers import UNet2DModel

def create_unet_model(latent_size, latent_channels=4):
    """
    创建UNet模型
    
    参数:
        latent_size: 潜在空间分辨率
        latent_channels: 潜在通道数
        
    返回:
        UNet2DModel实例
    """
    print(f"潜在空间分辨率: {latent_size}x{latent_size}")
    
    # 根据潜在空间大小调整UNet参数 - 增加模型容量
    if latent_size >= 32:  # 较大的潜在空间
        block_out_channels = (192, 384, 576, 768)  # 四层结构，增强层次特征表示
        layers_per_block = 2
    else:  # 较小的潜在空间
        block_out_channels = (192, 384, 576)  # 增加通道数
        layers_per_block = 2  # 保持每个块的层数
    
    model = UNet2DModel(
        sample_size=latent_size,  # 潜在空间分辨率
        in_channels=latent_channels,  # 输入通道数
        out_channels=latent_channels,  # 输出通道数
        layers_per_block=layers_per_block,  # 每个块中的层数
        block_out_channels=block_out_channels,  # 每个块的输出通道数
        down_block_types=(
            "DownBlock2D",
        ) * len(block_out_channels),
        up_block_types=(
            "UpBlock2D",
        ) * len(block_out_channels),
    )
    return model 