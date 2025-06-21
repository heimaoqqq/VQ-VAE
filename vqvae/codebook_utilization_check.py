"""
码本利用率检查工具 - 用于分析VQ-VAE模型的码本使用情况
"""

import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

def load_model(model_path, device):
    """加载训练好的VQ-VAE模型"""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # 从检查点中获取模型配置
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            print(f"模型配置: {config}")
        else:
            print("警告: 检查点中没有找到模型配置")
            
        # 从检查点中获取VQGAN状态字典
        if 'vqgan_state_dict' in checkpoint:
            vqgan_state_dict = checkpoint['vqgan_state_dict']
            print(f"成功加载VQGAN状态字典，包含 {len(vqgan_state_dict)} 个键")
            
            # 获取码本大小
            if 'quantize.embedding.weight' in vqgan_state_dict:
                codebook_shape = vqgan_state_dict['quantize.embedding.weight'].shape
                print(f"码本形状: {codebook_shape}")
                n_embed = codebook_shape[0]
                embed_dim = codebook_shape[1]
            else:
                print("警告: 未找到码本权重")
                n_embed = 512  # 默认值
                embed_dim = 256  # 默认值
                
            # 导入模型类
            from vqvae.custom_vqgan import CustomVQGAN
            
            # 创建模型实例
            vqgan = CustomVQGAN(
                n_embed=n_embed,
                embed_dim=embed_dim
            ).to(device)
            
            # 加载状态字典
            vqgan.load_state_dict(vqgan_state_dict)
            print("模型加载成功")
            
            return vqgan
        else:
            print("错误: 检查点中没有找到VQGAN状态字典")
            return None
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None

def analyze_codebook_utilization(vqgan, dataloader, device, temperature=1.0):
    """分析码本利用率"""
    vqgan.eval()
    all_indices = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="处理批次"):
            images, _ = batch
            images = images.to(device)
            
            # 前向传播
            output = vqgan(images, return_dict=True, temperature=temperature)
            
            # 获取索引
            if "indices" in output:
                indices = output["indices"].flatten().cpu().numpy()
                all_indices.extend(indices)
    
    # 转换为numpy数组
    all_indices = np.array(all_indices)
    
    # 计算唯一索引
    unique_indices = np.unique(all_indices)
    
    # 计算码本利用率
    n_embed = vqgan.quantize.num_embeddings
    utilization = len(unique_indices) / n_embed
    
    # 计算每个索引的使用频率
    index_counts = np.zeros(n_embed)
    for idx in range(n_embed):
        index_counts[idx] = np.sum(all_indices == idx)
    
    # 计算概率分布
    probs = index_counts / len(all_indices)
    
    # 计算熵
    valid_probs = probs[probs > 0]
    entropy = -np.sum(valid_probs * np.log2(valid_probs + 1e-10))
    max_entropy = np.log2(n_embed)
    normalized_entropy = entropy / max_entropy
    
    # 计算码本使用的不均匀性
    non_zero_probs = probs[probs > 0]
    gini = 1 - np.sum((non_zero_probs / np.mean(non_zero_probs)) ** 2) / len(non_zero_probs)
    
    # 返回结果
    return {
        "n_embed": n_embed,
        "unique_indices": unique_indices,
        "utilization": utilization,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "gini": gini,
        "index_counts": index_counts,
        "probs": probs
    }

def plot_codebook_usage(results, save_path=None):
    """绘制码本使用情况"""
    n_embed = results["n_embed"]
    probs = results["probs"]
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制码本使用频率
    plt.subplot(2, 1, 1)
    plt.bar(range(n_embed), probs)
    plt.title(f"码本使用频率 (利用率: {results['utilization']*100:.2f}%)")
    plt.xlabel("码元索引")
    plt.ylabel("使用频率")
    
    # 绘制码本使用频率（对数尺度）
    plt.subplot(2, 1, 2)
    plt.bar(range(n_embed), probs)
    plt.yscale("log")
    plt.title(f"码本使用频率（对数尺度）(熵: {results['entropy']:.2f}, 归一化熵: {results['normalized_entropy']:.2f})")
    plt.xlabel("码元索引")
    plt.ylabel("使用频率（对数尺度）")
    
    plt.tight_layout()
    
    # 保存图形
    if save_path:
        plt.savefig(save_path)
        print(f"图形已保存到 {save_path}")
    
    plt.show()

def main(args):
    # 设置设备
    device = torch.device(args.device)
    
    # 加载模型
    vqgan = load_model(args.model_path, device)
    if vqgan is None:
        return
    
    # 创建数据变换
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 归一化到[-1, 1]
    ])
    
    # 加载数据集
    dataset = ImageFolder(root=args.data_dir, transform=transform)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 分析码本利用率
    results = analyze_codebook_utilization(vqgan, dataloader, device, args.temperature)
    
    # 打印结果
    print("\n=== 码本利用率分析 ===")
    print(f"码本大小: {results['n_embed']}")
    print(f"使用的唯一码元数量: {len(results['unique_indices'])}")
    print(f"码本利用率: {results['utilization']*100:.2f}%")
    print(f"熵: {results['entropy']:.4f}")
    print(f"归一化熵: {results['normalized_entropy']:.4f}")
    print(f"基尼系数: {results['gini']:.4f}")
    
    # 绘制码本使用情况
    if args.plot:
        os.makedirs(args.output_dir, exist_ok=True)
        plot_path = os.path.join(args.output_dir, "codebook_usage.png")
        plot_codebook_usage(results, plot_path)
    
    # 保存详细结果
    if args.save_details:
        import json
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 将numpy数组转换为列表
        save_results = {
            "n_embed": int(results["n_embed"]),
            "unique_indices": results["unique_indices"].tolist(),
            "utilization": float(results["utilization"]),
            "entropy": float(results["entropy"]),
            "normalized_entropy": float(results["normalized_entropy"]),
            "gini": float(results["gini"]),
            "index_counts": results["index_counts"].tolist(),
            "probs": results["probs"].tolist()
        }
        
        # 保存为JSON文件
        json_path = os.path.join(args.output_dir, "codebook_stats.json")
        with open(json_path, "w") as f:
            json.dump(save_results, f, indent=2)
        
        print(f"详细结果已保存到 {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQ-VAE码本利用率分析工具")
    
    # 模型相关参数
    parser.add_argument("--model_path", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--data_dir", type=str, required=True, help="数据集目录")
    parser.add_argument("--output_dir", type=str, default="./codebook_analysis", help="输出目录")
    
    # 数据相关参数
    parser.add_argument("--image_size", type=int, default=256, help="图像大小")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器工作线程数")
    
    # 分析相关参数
    parser.add_argument("--temperature", type=float, default=1.0, help="温度参数")
    parser.add_argument("--plot", action="store_true", help="是否绘制码本使用情况")
    parser.add_argument("--save_details", action="store_true", help="是否保存详细结果")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    
    args = parser.parse_args()
    main(args) 