import torch
import argparse
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np

# 导入自定义模块
from vqvae.custom_vqgan import CustomVQGAN
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def parse_args():
    parser = argparse.ArgumentParser(description='VQ-VAE码本利用率分析')
    parser.add_argument('--model_path', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--data_dir', type=str, required=True, help='数据集目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    
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

def analyze_codebook_utilization(model, dataloader, device):
    """
    分析码本利用率
    
    参数:
        model: VQ-VAE模型
        dataloader: 数据加载器
        device: 设备
    """
    model.eval()
    model.to(device)
    
    # 初始化码本使用计数
    codebook_size = model.quantize.num_embeddings
    code_usage = torch.zeros(codebook_size, dtype=torch.int64, device=device)
    total_codes = 0
    
    print(f"分析码本利用率 (码本大小: {codebook_size})...")
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="处理批次"):
            images = images.to(device)
            
            # 编码图像
            z = model.encode(images)
            
            # 获取量化索引
            _, _, quantize_info = model.quantize(z)
            indices = quantize_info[2]  # 获取量化索引
            
            # 统计每个码元的使用次数
            for idx in range(codebook_size):
                code_usage[idx] += torch.sum(indices == idx).item()
            
            # 统计总码元使用次数
            total_codes += indices.numel()
    
    # 计算每个码元的使用频率
    code_freq = code_usage.float() / total_codes
    
    # 计算使用过的码元数量
    used_codes = torch.sum(code_usage > 0).item()
    utilization_rate = used_codes / codebook_size
    
    # 计算熵
    probs = code_freq.cpu().numpy()
    probs = probs[probs > 0]  # 只考虑非零概率
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(codebook_size)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    # 计算前10个最常用的码元
    top_k = 10
    if used_codes > 0:
        top_indices = torch.topk(code_usage, min(top_k, used_codes))
        top_usage = [(idx.item(), count.item(), (count.item() / total_codes) * 100) 
                     for idx, count in zip(top_indices.indices, top_indices.values)]
    else:
        top_usage = []
    
    # 打印结果
    print("\n===== 码本利用率分析结果 =====")
    print(f"总码本大小: {codebook_size}")
    print(f"使用过的码元数量: {used_codes}")
    print(f"码本利用率: {utilization_rate * 100:.2f}%")
    print(f"码本熵值: {entropy:.4f}")
    print(f"归一化熵: {normalized_entropy:.4f}")
    
    print("\n最常用的码元:")
    for i, (idx, count, percentage) in enumerate(top_usage):
        print(f"  {i+1}. 码元 {idx}: {count} 次 ({percentage:.2f}%)")
    
    # 计算使用频率的分布
    if used_codes > 0:
        non_zero_freqs = code_freq[code_usage > 0].cpu().numpy()
        percentiles = np.percentile(non_zero_freqs, [25, 50, 75, 90, 95, 99])
        
        print("\n使用频率分布:")
        print(f"  25%分位数: {percentiles[0]*100:.6f}%")
        print(f"  中位数: {percentiles[1]*100:.6f}%")
        print(f"  75%分位数: {percentiles[2]*100:.6f}%")
        print(f"  90%分位数: {percentiles[3]*100:.6f}%")
        print(f"  95%分位数: {percentiles[4]*100:.6f}%")
        print(f"  99%分位数: {percentiles[5]*100:.6f}%")
        print(f"  最大值: {torch.max(code_freq).item()*100:.6f}%")
    
    return {
        "codebook_size": codebook_size,
        "used_codes": used_codes,
        "utilization_rate": utilization_rate,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "top_usage": top_usage
    }

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
    
    # 分析码本利用率
    analyze_codebook_utilization(model, dataloader, args.device)

if __name__ == "__main__":
    main() 