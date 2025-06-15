"""
VQ-VAE 模型定义
"""
import torch
from diffusers import VQModel

def create_vq_model(args):
    """创建VQModel实例"""
    # 根据参数设置层数和通道数
    n_layers = args.n_layers
    
    # 动态构建层配置
    down_block_types = ["DownEncoderBlock2D"] * n_layers
    up_block_types = ["UpDecoderBlock2D"] * n_layers
    
    # 构建通道配置，每下采样一层通道数翻倍，最多到512
    # 为微多普勒时频图优化的通道配置
    block_out_channels = []
    current_channels = 64  # 起始通道数增加到64（原为32）
    for i in range(n_layers):
        current_channels = min(current_channels * 2, 512)  # 最大通道数增加到512（原为256）
        block_out_channels.append(current_channels)
    
    # 确保vq_embed_dim与latent_channels一致
    # 在VQModel中，vq_embed_dim控制潜在表示中的通道数
    latent_channels = args.latent_channels
    vq_embed_dim = latent_channels  # 确保一致
    
    # 在diffusers中的VQModel，每层下采样2倍，总下采样倍数是2^n_layers
    actual_downscale = 2 ** n_layers
    
    print(f"模型配置:")
    print(f"- 下采样层数: {n_layers}")
    print(f"- 通道配置: {block_out_channels}")
    print(f"- 潜在通道数/嵌入维度: {latent_channels}")
    print(f"- 码本大小: {args.vq_num_embed}")
    print(f"- 下采样倍数: {actual_downscale}x")
    print(f"- 预期潜在空间尺寸: {args.image_size//actual_downscale}x{args.image_size//actual_downscale}")
    
    model = VQModel(
        in_channels=3,  # RGB图像
        out_channels=3,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(block_out_channels),
        latent_channels=latent_channels,
        num_vq_embeddings=args.vq_num_embed,
        vq_embed_dim=vq_embed_dim,  # 设置为与latent_channels相同
        norm_num_groups=8,  # 对小batch_size更友好
        norm_type="group"  # 使用组归一化
    )
    return model 