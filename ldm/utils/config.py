"""
LDM 配置工具
"""
import argparse
import torch

def parse_args():
    """
    解析命令行参数
    
    返回:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(description="生成微多普勒时频图")
    parser.add_argument("--data_dir", type=str, default="dataset", help="数据集目录")
    parser.add_argument("--vqvae_path", type=str, default="vqvae_model", help="VQ-VAE模型路径")
    parser.add_argument("--ldm_path", type=str, default="ldm_model", help="LDM模型路径")
    parser.add_argument("--output_dir", type=str, default="generated_images", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--num_samples", type=int, default=16, help="生成样本数量")
    parser.add_argument("--image_size", type=int, default=256, help="图像尺寸")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--num_train_steps", type=int, default=None, help="训练步数")
    parser.add_argument("--save_epochs", type=int, default=5, help="每多少个epoch保存一次模型")
    parser.add_argument("--eval_steps", type=int, default=1000, help="评估间隔步数")
    parser.add_argument("--logging_steps", type=int, default=100, help="日志间隔步数")
    parser.add_argument("--latent_channels", type=int, default=4, help="潜变量通道数")
    parser.add_argument("--save_images", action="store_true", help="是否保存生成图像")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="推理步数，默认50步(适用于DDIM)")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb记录训练")
    parser.add_argument("--wandb_project", type=str, default="ldm-microdoppler", help="wandb项目名")
    parser.add_argument("--wandb_name", type=str, default="ldm-training", help="wandb运行名")
    parser.add_argument("--kaggle", action="store_true", help="是否在Kaggle环境中运行")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="混合精度训练")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="早停耐心值，默认10轮")
    parser.add_argument("--beta_schedule", type=str, default="cosine", choices=["linear", "cosine", "squaredcos_cap_v2"], help="beta调度类型")
    parser.add_argument("--grid", action="store_true", help="是否生成网格图像")
    parser.add_argument("--scheduler_type", type=str, default="ddim", choices=["ddpm", "ddim", "pndm"], help="采样器类型，默认为DDIM")
    return parser.parse_args() 