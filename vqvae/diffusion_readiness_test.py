import torch
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os

# 导入自定义模块
from vqvae.custom_vqgan import CustomVQGAN
from vqvae.latent_space_analyzer import LatentSpaceAnalyzer

def parse_args():
    parser = argparse.ArgumentParser(description='VQ-VAE潜在空间分析')
    parser.add_argument('--model_path', type=str, default='/kaggle/input/vq-gan/vqgan_model_best_best.pt', 
                        help='模型检查点路径')
    parser.add_argument('--data_dir', type=str, required=True, help='数据集目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--max_samples', type=int, default=1000, help='分析的最大样本数')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='analysis_results', help='输出目录')
    
    # 模型配置参数
    parser.add_argument('--in_channels', type=int, default=3, help='输入通道数')
    parser.add_argument('--out_channels', type=int, default=3, help='输出通道数')
    parser.add_argument('--block_out_channels', type=int, nargs='+', default=[64, 128, 256], help='每个块的输出通道数')
    parser.add_argument('--layers_per_block', type=int, default=2, help='每个块的层数')
    parser.add_argument('--latent_channels', type=int, default=256, help='潜在通道数')
    parser.add_argument('--n_embed', type=int, default=8192, help='码本大小')
    parser.add_argument('--embed_dim', type=int, default=256, help='嵌入维度')
    
    return parser.parse_args()

def create_dataloader(data_dir, batch_size):
    """创建数据加载器"""
    # 根据您的实际数据集类型修改此函数
    # 这里假设使用ImageFolder格式的数据集
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 检查路径是否存在
    if not os.path.exists(data_dir):
        raise ValueError(f"数据目录不存在: {data_dir}")
        
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    print(f"数据集加载完成，共 {len(dataset)} 个样本")
    return dataloader

def main():
    args = parse_args()
    
    # 检查CUDA是否可用
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，切换到CPU")
        args.device = 'cpu'
    else:
        print(f"使用设备: {args.device}")
    
    # 检查模型路径
    if not os.path.exists(args.model_path):
        raise ValueError(f"模型文件不存在: {args.model_path}")
    
    # 创建模型配置
    model_config = {
        'in_channels': args.in_channels,
        'out_channels': args.out_channels,
        'block_out_channels': args.block_out_channels,
        'layers_per_block': args.layers_per_block,
        'latent_channels': args.latent_channels,
        'n_embed': args.n_embed,
        'embed_dim': args.embed_dim
    }
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    model = CustomVQGAN(**model_config)
    
    # 加载检查点
    try:
        checkpoint = torch.load(args.model_path, map_location=args.device)
        print(f"成功加载检查点，检查点类型: {type(checkpoint)}")
        
        # 打印检查点的键以便调试
        if isinstance(checkpoint, dict):
            print(f"检查点包含以下键: {list(checkpoint.keys())}")
    except Exception as e:
        print(f"加载检查点时出错: {e}")
        return
    
    # 智能地寻找正确的state_dict
    state_dict_key = None
    if isinstance(checkpoint, dict):
        if 'generator_state_dict' in checkpoint:
            state_dict_key = 'generator_state_dict'
        elif 'vqgan_state_dict' in checkpoint:
            state_dict_key = 'vqgan_state_dict'
        elif 'model_state_dict' in checkpoint:
            state_dict_key = 'model_state_dict'

        if state_dict_key:
            print(f"使用键 '{state_dict_key}' 加载模型权重")
            try:
                model.load_state_dict(checkpoint[state_dict_key])
            except Exception as e:
                print(f"加载模型权重时出错: {e}")
                return
        else:
            # 如果上面都找不到，就尝试直接加载整个文件
            try:
                model.load_state_dict(checkpoint)
            except Exception as e:
                print(f"尝试直接加载模型权重时出错: {e}")
                return
    else:
        print("检查点不是字典类型，无法加载模型")
        return
    
    # 创建数据加载器
    print(f"加载数据集: {args.data_dir}")
    try:
        dataloader = create_dataloader(args.data_dir, args.batch_size)
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        return
    
    # 创建潜在空间分析器
    analyzer = LatentSpaceAnalyzer(model, device=args.device)
    
    # 运行综合分析
    report = analyzer.analyze_for_diffusion(dataloader, max_samples=args.max_samples)
    
    # 保存分析报告
    output_dir = Path(args.output_dir)
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        # 在Kaggle上使用特定的输出路径
        output_dir = Path('/kaggle/working') / args.output_dir
    
    output_dir.mkdir(exist_ok=True)
    print(f"将结果保存到: {output_dir}")
    
    # 创建雷达图可视化分析结果
    labels = ['分布正态性', '码本结构', '潜在平滑度', '潜在一致性', '重建质量']
    scores = [
        report['distribution_score'],
        report['codebook_score'],
        report['smoothness_score'],
        report['coherence_score'],
        report['reconstruction_score']
    ]
    
    # 创建雷达图
    angles = [n / float(len(labels)) * 2 * 3.14159 for n in range(len(labels))]
    angles += angles[:1]  # 闭合图形
    scores += scores[:1]  # 闭合图形
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, scores, 'o-', linewidth=2)
    ax.fill(angles, scores, alpha=0.25)
    ax.set_thetagrids([a * 180 / 3.14159 for a in angles[:-1]], labels)
    ax.set_ylim(0, 1)
    plt.title(f'VQ-VAE扩散适应性分析 (总分: {report["diffusion_readiness"]:.2f}/1.00)')
    
    # 保存图表
    plt.savefig(output_dir / 'diffusion_readiness_radar.png')
    print(f"分析结果已保存到 {output_dir}")
    
    # 保存详细报告
    with open(output_dir / 'diffusion_readiness_report.txt', 'w') as f:
        f.write(f"VQ-VAE潜在空间分析报告\n")
        f.write(f"====================\n\n")
        f.write(f"模型路径: {args.model_path}\n")
        f.write(f"数据集路径: {args.data_dir}\n")
        f.write(f"分析样本数: {args.max_samples}\n\n")
        f.write(f"分布正态性得分: {report['distribution_score']:.4f}/1.00\n")
        f.write(f"码本结构得分: {report['codebook_score']:.4f}/1.00\n")
        f.write(f"潜在平滑度得分: {report['smoothness_score']:.4f}/1.00\n")
        f.write(f"潜在一致性得分: {report['coherence_score']:.4f}/1.00\n")
        f.write(f"重建质量得分: {report['reconstruction_score']:.4f}/1.00\n")
        f.write(f"扩散适应性总得分: {report['diffusion_readiness']:.4f}/1.00\n")

if __name__ == "__main__":
    main() 