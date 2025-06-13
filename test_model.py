"""
VQ-VAE模型创建和前向传播测试
"""
import argparse
import torch
from vqvae.models import create_vq_model
from vqvae.trainers import VQModelTrainer
from torch.optim import AdamW

def test_model():
    """测试模型创建和前向传播"""
    print("开始测试VQ-VAE模型...")
    
    # 创建参数对象
    args = argparse.Namespace(
        batch_size=8,
        image_size=256,
        n_layers=2,
        latent_channels=4,
        vq_embed_dim=4,  # 与latent_channels一致
        vq_num_embed=128,
        use_perceptual=True,
        lambda_perceptual=0.1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"使用设备: {args.device}")
    
    # 创建模型
    model = create_vq_model(args)
    model.to(args.device)
    
    # 创建优化器和训练器
    optimizer = AdamW(model.parameters(), lr=1e-4)
    trainer = VQModelTrainer(model, optimizer, args.device,
                           use_perceptual=args.use_perceptual,
                           lambda_perceptual=args.lambda_perceptual)
    
    # 创建随机输入数据
    batch = torch.randn(args.batch_size, 3, args.image_size, args.image_size).to(args.device)
    
    # 测试编码和解码
    with torch.no_grad():
        encoder_output = model.encode(batch)
        print(f"编码输出形状: {encoder_output.latents.shape}")
        
        decoder_output = model.decode(encoder_output.latents)
        print(f"解码输出形状: {decoder_output.sample.shape}")
    
    # 测试训练步骤
    try:
        results, reconstructed = trainer.train_step(batch)
        print("\n训练步骤结果:")
        print(f"- 总损失: {results['loss']:.4f}")
        print(f"- 重建损失: {results['recon_loss']:.4f}")
        print(f"- VQ损失: {results['vq_loss']:.4f}")
        
        if 'perceptual_loss' in results:
            print(f"- 感知损失: {results['perceptual_loss']:.4f}")
            
        if 'perplexity' in results:
            print(f"- 码本利用率: {results['perplexity']:.2f}/{args.vq_num_embed} ({results['perplexity']/args.vq_num_embed*100:.1f}%)")
        
        print("\n✓ 测试完成，模型工作正常!")
        return True
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        return False

if __name__ == "__main__":
    test_model() 