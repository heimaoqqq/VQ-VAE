"""
VQ-VAE训练流程测试
"""
import argparse
import torch
import os
from torch.optim import AdamW
from tqdm import tqdm
from dataset import get_dataloaders
from vqvae.models import create_vq_model
from vqvae.trainers import VQModelTrainer
from vqvae.utils import validate, save_reconstructed_images

def test_training():
    """测试VQ-VAE训练流程"""
    print("开始测试VQ-VAE训练流程...")
    
    # 创建命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset", help="数据集目录路径")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--image_size", type=int, default=256, help="图像尺寸")
    parser.add_argument("--n_layers", type=int, default=3, help="下采样层数")
    parser.add_argument("--latent_channels", type=int, default=4, help="潜在通道数")
    parser.add_argument("--vq_embed_dim", type=int, default=4, help="VQ嵌入维度")
    parser.add_argument("--vq_num_embed", type=int, default=128, help="码本大小")
    parser.add_argument("--use_perceptual", action="store_true", help="使用感知损失")
    parser.add_argument("--lambda_perceptual", type=float, default=0.1, help="感知损失权重")
    
    # 解析参数，使用命令行传入的参数而非硬编码的参数
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"使用设备: {args.device}")
    
    # 检查数据集目录是否存在
    if not os.path.exists(args.data_dir):
        print(f"警告: 数据集目录 '{args.data_dir}' 不存在，将创建测试数据")
        os.makedirs(args.data_dir, exist_ok=True)
        
        # 创建伪数据集用于测试
        def create_dummy_dataset(size=10):
            os.makedirs(os.path.join(args.data_dir, "images"), exist_ok=True)
            for i in range(size):
                # 生成随机图像
                img = torch.rand(3, args.image_size, args.image_size) * 255
                img = img.byte().numpy()
                
                # 如果没有PIL库，这里会报错，需要安装Pillow
                try:
                    from PIL import Image
                    img = Image.fromarray(img.transpose(1, 2, 0))
                    img.save(os.path.join(args.data_dir, "images", f"test_{i}.png"))
                    print(f"创建测试图像: test_{i}.png")
                except ImportError:
                    print("未找到PIL库，无法创建测试图像")
                    return False
                except Exception as e:
                    print(f"创建测试图像失败: {e}")
                    return False
            return True
            
        if not create_dummy_dataset():
            print("创建测试数据失败，退出测试")
            return False
    
    # 加载数据
    try:
        train_dataloader, val_dataloader, _ = get_dataloaders(
            args.data_dir, args.batch_size, args.image_size
        )
        print(f"数据加载成功:")
        print(f"- 训练数据: {len(train_dataloader.dataset)}个样本, {len(train_dataloader)}个批次")
        print(f"- 验证数据: {len(val_dataloader.dataset)}个样本, {len(val_dataloader)}个批次")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return False
    
    # 创建模型
    model = create_vq_model(args)
    model.to(args.device)
    
    # 创建优化器和训练器
    optimizer = AdamW(model.parameters(), lr=1e-4)
    trainer = VQModelTrainer(model, optimizer, args.device,
                           use_perceptual=args.use_perceptual,
                           lambda_perceptual=args.lambda_perceptual)
    
    # 运行几个训练步骤
    print("\n运行测试训练...")
    model.train()
    test_steps = min(2, len(train_dataloader))  # 最多运行2个批次
    
    for step, batch in enumerate(train_dataloader):
        if step >= test_steps:
            break
            
        images = batch.to(args.device)
        results, reconstructed = trainer.train_step(images)
        
        print(f"步骤 {step+1}/{test_steps}:")
        print(f"- 总损失: {results['loss']:.4f}")
        print(f"- 重建损失: {results['recon_loss']:.4f}")
        print(f"- VQ损失: {results['vq_loss']:.4f}")
        
        if 'perceptual_loss' in results:
            print(f"- 感知损失: {results['perceptual_loss']:.4f}")
            
        if 'perplexity' in results:
            print(f"- 码本利用率: {results['perplexity']:.2f}/{args.vq_num_embed} ({results['perplexity']/args.vq_num_embed*100:.1f}%)")
    
    # 运行一次验证
    print("\n运行测试验证...")
    try:
        val_results = validate(val_dataloader, trainer, args.device, global_step=0)
        
        print(f"验证结果:")
        print(f"- 总损失: {val_results['loss']:.4f}")
        print(f"- 重建损失: {val_results['recon_loss']:.4f}")
        print(f"- VQ损失: {val_results['vq_loss']:.4f}")
        
        if 'perceptual_loss' in val_results:
            print(f"- 感知损失: {val_results['perceptual_loss']:.4f}")
        
        if 'perplexity' in val_results:
            print(f"- 码本利用率: {val_results['perplexity']:.2f}/{args.vq_num_embed} ({val_results['perplexity']/args.vq_num_embed*100:.1f}%)")
    except Exception as e:
        print(f"验证失败: {e}")
    
    print("\n✓ 训练流程测试完成!")
    return True

if __name__ == "__main__":
    test_training() 
