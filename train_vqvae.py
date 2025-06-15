"""
VQ-VAE 训练脚本 - 模块化版本
"""
import os
import torch
import sys

# 尝试导入必要的库
try:
    import wandb
except ImportError:
    print("未找到wandb库，将不使用wandb进行可视化")
    wandb = None

from dataset import get_dataloaders
from vqvae.models import create_vq_model
from vqvae.trainers import VQVAETrainer
from vqvae.utils.config import parse_args

def train_vqvae(args):
    """训练VQ-VAE模型主函数"""
    # 设置全局随机种子以提高可重复性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建数据加载器，使用分段自适应标准化
    train_dataloader, val_dataloader, _ = get_dataloaders(
        args.data_dir, 
        args.batch_size, 
        args.image_size,
        use_adaptive_norm=args.use_adaptive_norm,
        split_ratio=args.split_ratio,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile
    )
    
    # 打印数据集划分信息
    total_samples = len(train_dataloader.dataset) + len(val_dataloader.dataset)
    
    print(f"数据集统计信息:")
    print(f"总样本数: {total_samples}")
    print(f"训练集: {len(train_dataloader.dataset)}个样本 ({len(train_dataloader.dataset)/total_samples*100:.1f}%), {len(train_dataloader)}个批次")
    print(f"验证集: {len(val_dataloader.dataset)}个样本 ({len(val_dataloader.dataset)/total_samples*100:.1f}%), {len(val_dataloader)}个批次")
    print(f"批次大小: {args.batch_size}")
    print(f"使用分段自适应标准化: {'是' if args.use_adaptive_norm else '否'}")
    if args.use_adaptive_norm:
        print(f"  - 分割比例: {args.split_ratio}")
        print(f"  - 下半部分分位数范围: [{args.lower_quantile}, {args.upper_quantile}]")
    
    # 创建模型
    model = create_vq_model(args)
    model.to(args.device)
    
    # 打印模型结构
    print(f"模型下采样层数: {args.n_layers}")
    print(f"码本大小: {args.vq_num_embed}")
    print(f"嵌入维度: {args.vq_embed_dim}")
    print(f"潜在空间尺寸: {args.image_size // (2 ** args.n_layers)}x{args.image_size // (2 ** args.n_layers)}")
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # 创建训练器
    trainer = VQVAETrainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        args=args
    )
    
    # 如果指定了检查点路径，则从检查点恢复训练
    if args.resume_from_checkpoint:
        trainer.load_checkpoint(args.resume_from_checkpoint)
    
    # 开始训练
    trainer.train()
    
    return model

if __name__ == "__main__":
    args = parse_args()
    print(f"正在使用设备: {args.device}")
    model = train_vqvae(args) 