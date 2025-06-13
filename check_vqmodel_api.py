import os
import torch
import torch.nn.functional as F
from diffusers import VQModel
import numpy as np

def main():
    # 打印diffusers版本
    import diffusers
    print(f"Diffusers版本: {diffusers.__version__}")
    
    # 创建一个简单的VQModel实例
    print("创建VQModel实例...")
    model = VQModel(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(128, 256, 512, 512),
        latent_channels=4,
        num_vq_embeddings=256,
        vq_embed_dim=128,
    )
    
    # 检查模型结构
    print("模型结构概述:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    print(f"编码器类型: {type(model.encoder)}")
    print(f"量化器类型: {type(model.quant_conv)} -> {type(model.quantize)}")
    print(f"解码器类型: {type(model.decoder)}")

    # 检查量化器内部结构
    print("\n量化器内部结构:")
    if hasattr(model, "quantize"):
        quant = model.quantize
        print(f"量化器类型: {type(quant)}")
        print(f"量化器属性: {dir(quant)}")
        
        # 检查特定API属性
        embedding_attrs = ["embedding", "embedding_weight", "_embedding", "_embedding.weight"]
        for attr in embedding_attrs:
            if hasattr(quant, attr) or (hasattr(quant, "_parameters") and attr in quant._parameters):
                print(f"发现嵌入权重: {attr}")
        
        print(f"是否有原始的VQ实现中的commitment_cost属性: {hasattr(quant, 'commitment_cost')}")
        print(f"是否实现了forward方法: {hasattr(quant, 'forward') and callable(getattr(quant, 'forward'))}")

    # 生成随机输入进行测试
    print("\n测试前向传递...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # 创建随机输入 (1, 3, 256, 256)
    x = torch.randn(1, 3, 256, 256, device=device)
    
    with torch.no_grad():
        # 编码
        print("编码过程:")
        encoder_output = model.encode(x)
        print(f"编码器输出类型: {type(encoder_output)}")
        print(f"编码器输出属性: {dir(encoder_output)}")
        
        if hasattr(encoder_output, "latents"):
            latents = encoder_output.latents
            print(f"潜在表示形状: {latents.shape}")
        
        # 检查是否有VQ相关属性
        vq_attrs = ["loss", "z", "z_q", "commitment_loss", "codebook_loss", "perplexity"]
        for attr in vq_attrs:
            if hasattr(encoder_output, attr):
                val = getattr(encoder_output, attr)
                if isinstance(val, torch.Tensor):
                    print(f"找到属性 {attr}: {val.item() if val.numel() == 1 else val.shape}")
                else:
                    print(f"找到属性 {attr}: {val}")
        
        # 解码
        print("\n解码过程:")
        decoder_output = model.decode(encoder_output.latents)
        print(f"解码器输出类型: {type(decoder_output)}")
        print(f"解码器输出属性: {dir(decoder_output)}")
        
        if hasattr(decoder_output, "sample"):
            sample = decoder_output.sample
            print(f"重建样本形状: {sample.shape}")
    
    # 测试VQ损失计算的不同方式
    print("\n测试VQ损失计算:")
    
    # 方式1: 直接使用encoder_output.loss
    if hasattr(encoder_output, "loss"):
        print(f"直接使用encoder_output.loss: {encoder_output.loss.item()}")
    else:
        print("encoder_output没有loss属性")
    
    # 方式2: 手动计算
    if hasattr(encoder_output, "z") and hasattr(encoder_output, "z_q"):
        commitment_loss = F.mse_loss(encoder_output.z.detach(), encoder_output.z_q)
        codebook_loss = F.mse_loss(encoder_output.z, encoder_output.z_q.detach())
        total_loss = codebook_loss + 0.25 * commitment_loss
        print(f"手动计算 - commitment_loss: {commitment_loss.item()}")
        print(f"手动计算 - codebook_loss: {codebook_loss.item()}")
        print(f"手动计算 - 总VQ损失: {total_loss.item()}")
    else:
        print("encoder_output没有z或z_q属性，无法手动计算VQ损失")

if __name__ == "__main__":
    main() 