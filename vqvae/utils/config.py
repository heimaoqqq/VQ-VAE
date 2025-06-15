"""
VQ-VAE 配置工具
"""
import argparse
import torch

def parse_args():
    """
    解析命令行参数
    
    返回:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(description="训练VQ-VAE模型")
    parser.add_argument("--data_dir", type=str, default="dataset", help="数据集目录")
    parser.add_argument("--output_dir", type=str, default="vqvae_model", help="模型保存目录")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--image_size", type=int, default=256, help="图像尺寸")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--num_train_steps", type=int, default=None, help="训练步数")
    parser.add_argument("--save_epochs", type=int, default=5, help="每多少个epoch保存一次模型")
    parser.add_argument("--logging_steps", type=int, default=100, help="日志间隔步数")
    parser.add_argument("--latent_channels", type=int, default=6, help="潜变量通道数")
    parser.add_argument("--vq_embed_dim", type=int, default=6, help="VQ嵌入维度（应与latent_channels一致）")
    parser.add_argument("--vq_num_embed", type=int, default=256, help="VQ嵌入数量")
    parser.add_argument("--n_layers", type=int, default=3, help="下采样层数")
    parser.add_argument("--save_images", action="store_true", help="是否保存重建图像")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb记录训练")
    parser.add_argument("--wandb_project", type=str, default="vq-vae-microdoppler", help="wandb项目名")
    parser.add_argument("--wandb_name", type=str, default="vqvae-training", help="wandb运行名")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--fp16", action="store_true", help="是否使用半精度训练")
    parser.add_argument("--kaggle", action="store_true", help="是否在Kaggle环境中运行")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")
    # 感知损失相关参数
    parser.add_argument("--use_perceptual", action="store_true", help="是否使用感知损失")
    parser.add_argument("--lambda_perceptual", type=float, default=0.1, help="感知损失权重")
    # 位置感知损失相关参数
    parser.add_argument("--use_positional", action="store_true", help="是否使用位置感知损失")
    parser.add_argument("--lambda_positional", type=float, default=0.7, help="位置感知损失权重")
    parser.add_argument("--lambda_vertical", type=float, default=1.0, help="垂直方向权重")
    parser.add_argument("--lambda_horizontal", type=float, default=0.5, help="水平方向权重")
    # 分段自适应标准化相关参数
    parser.add_argument("--use_adaptive_norm", action="store_true", help="是否使用分段自适应标准化")
    parser.add_argument("--split_ratio", type=float, default=0.5, help="图像分割比例，用于分段标准化")
    parser.add_argument("--lower_quantile", type=float, default=0.01, help="下半部分的下分位数")
    parser.add_argument("--upper_quantile", type=float, default=0.99, help="下半部分的上分位数")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从检查点恢复训练的路径")
    return parser.parse_args()