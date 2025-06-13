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
    
    # 构建通道配置，从128开始，每层翻倍，最多到512
    block_out_channels = []
    current_channels = 128
    for i in range(n_layers):
        current_channels = min(current_channels * 2, 512)
        block_out_channels.append(current_channels)
    
    model = VQModel(
        in_channels=3,  # RGB图像
        out_channels=3,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(block_out_channels),
        latent_channels=args.latent_channels,
        num_vq_embeddings=args.vq_num_embed,
        vq_embed_dim=args.vq_embed_dim,
    )
    return model 