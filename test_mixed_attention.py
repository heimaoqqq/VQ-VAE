"""
测试混合Window+Axial注意力模型
"""
import torch
from diffusers import VQModel

from ldm.models.mixed_attention_unet import create_mixed_attention_unet

def test_mixed_attention_model():
    """测试混合注意力模型是否正常工作"""
    # 设置随机种子以提高可重复性
    torch.manual_seed(42)
    
    # 参数设置
    latent_size = 32
    latent_channels = 4
    batch_size = 2
    
    # 创建模型
    model = create_mixed_attention_unet(
        latent_size=latent_size,
        latent_channels=latent_channels,
        window_size=8
    )
    
    # 打印模型信息
    print(f"模型参数量: {sum(p.numel() for p in model.model.parameters())/1e6:.2f}M")
    
    # 创建随机输入
    sample = torch.randn(batch_size, latent_channels, latent_size, latent_size)
    timestep = torch.ones((batch_size,), dtype=torch.long)
    
    # 测试前向传播
    print("测试前向传播...")
    with torch.no_grad():
        try:
            output = model(sample, timestep)
            print(f"输入形状: {sample.shape}")
            print(f"输出形状: {output.sample.shape}")
            print("前向传播测试成功!")
        except Exception as e:
            print(f"前向传播失败: {e}")
            raise

if __name__ == "__main__":
    test_mixed_attention_model() 