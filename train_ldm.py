"""
LDM (潜在扩散模型) 训练脚本 - 模块化版本
"""
import os
import torch
import sys

# 尝试导入必要的库
try:
    import wandb
    from diffusers import VQModel
except ImportError as e:
    if "cannot import name 'cached_download' from 'huggingface_hub'" in str(e):
        print("检测到huggingface_hub版本不兼容。尝试安装兼容版本...")
        os.system("pip install huggingface_hub>=0.20.2")
        os.system("pip install diffusers>=0.26.3 --no-deps")
        print("请重新运行脚本")
        sys.exit(1)
    else:
        raise e

from dataset import get_dataloaders
from ldm.models.unet import create_unet_model
from ldm.trainers.ldm_trainer import LDMTrainer
from ldm.utils.config import parse_args

def train_ldm(args):
    """训练潜在扩散模型主函数"""
    # 设置全局随机种子以提高可重复性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建数据加载器
    train_dataloader, val_dataloader, _ = get_dataloaders(
        args.data_dir, args.batch_size, args.image_size
    )
    
    total_samples = len(train_dataloader.dataset) + len(val_dataloader.dataset)
    print(f"数据集总样本数: {total_samples}")
    print(f"训练集样本数: {len(train_dataloader.dataset)} ({len(train_dataloader.dataset)/total_samples*100:.1f}%)")
    print(f"验证集样本数: {len(val_dataloader.dataset)} ({len(val_dataloader.dataset)/total_samples*100:.1f}%)")
    print(f"数据集分割比例: 训练集 80% : 验证集 20%")
    print("-" * 50)
    
    # 加载预训练的VQ-VAE模型
    vq_model = VQModel.from_pretrained(args.vqvae_path)
    vq_model.eval().requires_grad_(False)  # 冻结VQ-VAE参数
    
    # 计算潜在空间大小
    latent_size = args.image_size // (2 ** (len(vq_model.config.down_block_types)))
    
    # 创建UNet模型
    model = create_unet_model(latent_size, args.latent_channels)
    
    # 创建LDM训练器
    trainer = LDMTrainer(
        unet=model,
        vq_model=vq_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        args=args,
        beta_schedule=args.beta_schedule
    )
    
    # 如果指定了检查点路径，则从检查点恢复训练
    if args.resume_from_checkpoint:
        print(f"正在从检查点恢复训练: {args.resume_from_checkpoint}")
        trainer.load_checkpoint(args.resume_from_checkpoint)
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    args = parse_args()
    print(f"正在使用设备: {args.device}")
    train_ldm(args) 
