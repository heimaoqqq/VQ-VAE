"""
测试分段自适应标准化方法

此脚本用于测试和可视化分段自适应标准化方法对微多普勒时频图的效果
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from vqvae.utils.normalization import MicroDopplerNormalizer

def parse_args():
    parser = argparse.ArgumentParser(description="测试分段自适应标准化方法")
    parser.add_argument("--image_path", type=str, required=True, help="输入图像路径")
    parser.add_argument("--output_dir", type=str, default="normalization_test", help="输出目录")
    parser.add_argument("--split_ratio", type=float, default=0.5, help="图像分割比例")
    parser.add_argument("--lower_quantile", type=float, default=0.01, help="下半部分的下分位数")
    parser.add_argument("--upper_quantile", type=float, default=0.99, help="下半部分的上分位数")
    parser.add_argument("--upper_contrast_factor", type=float, default=0.7, help="上半部分对比度增强因子")
    return parser.parse_args()

def visualize_normalization(image_path, output_dir, split_ratio=0.5, 
                           lower_quantile=0.01, upper_quantile=0.99, 
                           upper_contrast_factor=0.7):
    """可视化分段自适应标准化的效果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 转换为张量
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(image)
    
    # 创建标准化器
    normalizer = MicroDopplerNormalizer(
        split_ratio=split_ratio,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        upper_contrast_factor=upper_contrast_factor
    )
    
    # 应用标准化
    normalized_tensor = normalizer.normalize(image_tensor)
    
    # 反归一化
    denormalized_tensor = normalizer.denormalize(normalized_tensor)
    
    # 转换为NumPy数组进行可视化
    image_np = image_tensor.permute(1, 2, 0).numpy()
    normalized_np = normalized_tensor.permute(1, 2, 0).numpy()
    denormalized_np = denormalized_tensor.permute(1, 2, 0).numpy()
    
    # 确保值在[0,1]范围内
    normalized_np = (normalized_np + 1) / 2  # 从[-1,1]转换到[0,1]
    
    # 创建可视化图像
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原始图像
    axes[0].imshow(image_np)
    axes[0].set_title("原始图像")
    axes[0].axis('off')
    
    # 标准化后的图像
    axes[1].imshow(normalized_np)
    axes[1].set_title("标准化后的图像")
    axes[1].axis('off')
    
    # 反归一化后的图像
    axes[2].imshow(denormalized_np)
    axes[2].set_title("反归一化后的图像")
    axes[2].axis('off')
    
    # 添加标题
    plt.suptitle(f"分段自适应标准化 (分割比例={split_ratio}, 下分位数={lower_quantile}, 上分位数={upper_quantile})", fontsize=16)
    
    # 保存图像
    plt.tight_layout()
    output_path = os.path.join(output_dir, "normalization_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存到: {output_path}")
    
    # 显示图像
    plt.show()
    
    # 保存垂直分布图
    plt.figure(figsize=(10, 8))
    
    # 计算垂直分布
    height = image_tensor.shape[1]
    split_idx = int(height * split_ratio)
    
    # 计算原始图像的垂直分布
    vertical_profile_orig = torch.mean(image_tensor, dim=2)[0].numpy()  # 只取第一个通道
    
    # 计算标准化后的垂直分布
    vertical_profile_norm = torch.mean((normalized_tensor + 1) / 2, dim=2)[0].numpy()
    
    # 计算反归一化后的垂直分布
    vertical_profile_denorm = torch.mean(denormalized_tensor, dim=2)[0].numpy()
    
    # 绘制垂直分布
    plt.plot(vertical_profile_orig, range(height), 'b-', label='原始')
    plt.plot(vertical_profile_norm, range(height), 'r-', label='标准化')
    plt.plot(vertical_profile_denorm, range(height), 'g-', label='反归一化')
    
    # 添加分割线
    plt.axhline(y=split_idx, color='k', linestyle='--', alpha=0.5)
    plt.text(0.02, split_idx + 5, f'分割线 ({split_ratio})', fontsize=12)
    
    # 设置坐标轴
    plt.gca().invert_yaxis()  # 反转Y轴，使得图像顶部在上方
    plt.xlabel('平均强度')
    plt.ylabel('垂直位置')
    plt.title('微多普勒时频图垂直分布对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存垂直分布图
    profile_path = os.path.join(output_dir, "vertical_profile_comparison.png")
    plt.savefig(profile_path, dpi=300, bbox_inches='tight')
    print(f"垂直分布对比图已保存到: {profile_path}")
    
    return normalized_tensor, denormalized_tensor

def test_batch_normalization():
    """测试批处理标准化"""
    # 创建随机批次数据
    batch_size = 4
    batch = torch.rand(batch_size, 3, 256, 256)
    
    # 创建标准化器
    normalizer = MicroDopplerNormalizer(split_ratio=0.5)
    
    # 应用标准化
    normalized_batch = normalizer.normalize(batch)
    
    # 检查形状和值范围
    print(f"批处理标准化测试:")
    print(f"输入形状: {batch.shape}")
    print(f"输出形状: {normalized_batch.shape}")
    print(f"输入范围: [{batch.min().item():.4f}, {batch.max().item():.4f}]")
    print(f"输出范围: [{normalized_batch.min().item():.4f}, {normalized_batch.max().item():.4f}]")
    
    # 反归一化测试
    denormalized_batch = normalizer.denormalize(normalized_batch)
    print(f"反归一化后范围: [{denormalized_batch.min().item():.4f}, {denormalized_batch.max().item():.4f}]")
    
    # 计算误差
    error = torch.abs(batch - denormalized_batch).mean().item()
    print(f"平均绝对误差: {error:.6f}")
    
    return error < 0.1  # 如果误差小于0.1，则测试通过

if __name__ == "__main__":
    args = parse_args()
    
    # 可视化标准化效果
    visualize_normalization(
        args.image_path, 
        args.output_dir,
        args.split_ratio,
        args.lower_quantile,
        args.upper_quantile,
        args.upper_contrast_factor
    )
    
    # 测试批处理标准化
    test_passed = test_batch_normalization()
    print(f"批处理标准化测试: {'通过' if test_passed else '失败'}") 