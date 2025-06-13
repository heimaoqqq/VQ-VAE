"""
VQ-VAE 训练脚本

使用向量量化自编码器（VQ-VAE）训练微多普勒时频图的生成模型
"""

import os
import argparse
import torch
import shutil
from torch.optim import AdamW
from tqdm import tqdm

# 导入数据集
from dataset import get_dataloaders

# 导入自定义模块
from vqvae.models import create_vq_model, PerceptualLoss
from vqvae.trainers import VQModelTrainer
from vqvae.utils import save_reconstructed_images, validate

# 导入其他库
try:
    import wandb
except ImportError:
    print("未找到wandb库，将不使用wandb进行可视化")
    wandb = None

def parse_args():
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
    parser.add_argument("--latent_channels", type=int, default=4, help="潜变量通道数")
    parser.add_argument("--vq_embed_dim", type=int, default=4, help="VQ嵌入维度（应与latent_channels一致）")
    parser.add_argument("--vq_num_embed", type=int, default=128, help="VQ嵌入数量")
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
    return parser.parse_args()

def train_vqvae(args):
    """训练VQ-VAE模型"""
    # 创建输出目录和可视化目录
    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # 初始化wandb
    if args.use_wandb and wandb is not None:
        wandb.init(project=args.wandb_project, name=args.wandb_name)
        wandb.config.update(args)

    # 创建数据加载器
    train_dataloader, val_dataloader, _ = get_dataloaders(
        args.data_dir, args.batch_size, args.image_size
    )
    
    # 打印数据集划分信息
    total_samples = len(train_dataloader.dataset) + len(val_dataloader.dataset)
    
    print(f"数据集统计信息:")
    print(f"总样本数: {total_samples}")
    print(f"训练集: {len(train_dataloader.dataset)}个样本 ({len(train_dataloader.dataset)/total_samples*100:.1f}%), {len(train_dataloader)}个批次")
    print(f"验证集: {len(val_dataloader.dataset)}个样本 ({len(val_dataloader.dataset)/total_samples*100:.1f}%), {len(val_dataloader)}个批次")
    print(f"批次大小: {args.batch_size}")
    
    # 创建模型
    model = create_vq_model(args)
    model.to(args.device)
    
    # 打印模型结构
    print(f"模型下采样层数: {args.n_layers}")
    print(f"码本大小: {args.vq_num_embed}")
    print(f"潜在空间尺寸: {args.image_size // (2 ** args.n_layers)}x{args.image_size // (2 ** args.n_layers)}")
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # 创建训练器
    trainer = VQModelTrainer(model, optimizer, args.device, 
                            use_perceptual=args.use_perceptual, 
                            lambda_perceptual=args.lambda_perceptual)
    
    # 使用混合精度训练（用于16G显存）
    if args.fp16 and args.device != "cpu":
        from torch.amp import GradScaler, autocast
        scaler = GradScaler()  # 移除device_type参数，兼容旧版PyTorch
        print("使用混合精度训练 (FP16) - GradScaler已初始化")
        # 检查autocast能否工作
        try:
            with autocast():
                print("自动混合精度(autocast)测试成功")
        except Exception as e:
            print(f"警告: autocast测试失败, 原因: {e}")
    else:
        scaler = None
        print("使用全精度训练 (FP32)")
    
    # 调试模式
    if args.debug:
        print("调试模式: 检查模型输出...")
        test_batch = next(iter(train_dataloader)).to(args.device)
        encoder_output = model.encode(test_batch)
        decoder_output = model.decode(encoder_output.latents)
        print(f"Encoder output type: {type(encoder_output)}")
        print(f"Encoder output attributes: {dir(encoder_output)}")
        print(f"Decoder output type: {type(decoder_output)}")
        print(f"Decoder output attributes: {dir(decoder_output)}")
    
    # 训练步数
    if args.num_train_steps is None:
        args.num_train_steps = args.epochs * len(train_dataloader)
    
    # 初始化早停相关变量
    best_val_recon_loss = float('inf')
    best_model_path = None
    early_stop_counter = 0
    
    # 训练循环
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_vq_loss = 0.0
        epoch_perplexity = 0.0
        perplexity_count = 0
        
        # 检查训练精度模式
        precision_mode = "FP16" if scaler is not None else "FP32"
        print(f"训练精度模式: {precision_mode}")
        
        # 创建单个动态进度条
        progress_bar = tqdm(total=len(train_dataloader), 
                           desc=f"Epoch {epoch+1}/{args.epochs}", 
                           leave=True,
                           ncols=100)
        
        # 遍历训练数据
        for step, batch in enumerate(train_dataloader):
            images = batch.to(args.device)
            
            # 训练步骤
            if scaler is not None:
                # 使用混合精度训练
                results, reconstructed = trainer.train_step_fp16(images, scaler)
            else:
                # 常规训练
                results, reconstructed = trainer.train_step(images)
            
            # 获取损失
            epoch_loss += results['loss']
            epoch_recon_loss += results['recon_loss']
            epoch_vq_loss += results['vq_loss']
            
            # 更新码本利用率统计
            if 'perplexity' in results:
                epoch_perplexity += results['perplexity']
                perplexity_count += 1
            
            # 更新进度条，包含关键指标
            status_dict = {
                "loss": f"{results['loss']:.4f}",
                "recon": f"{results['recon_loss']:.4f}",
                "vq": f"{results['vq_loss']:.4f}"
            }
            
            if 'perplexity' in results:
                status_dict["perp"] = f"{results['perplexity']}/{args.vq_num_embed}"
                
            if 'perceptual_loss' in results:
                status_dict["percept"] = f"{results['perceptual_loss']:.4f}"
                
            # 添加FP16显示
            if 'using_fp16' in results:
                status_dict["fp16"] = "✓"
                
            progress_bar.set_postfix(status_dict)
            progress_bar.update(1)
            
            # wandb记录
            if args.use_wandb and wandb is not None and global_step % args.logging_steps == 0:
                log_dict = {
                    "train/loss": results['loss'],
                    "train/recon_loss": results['recon_loss'],
                    "train/vq_loss": results['vq_loss'],
                    "train/step": global_step
                }
                
                if 'perplexity' in results:
                    log_dict["train/perplexity"] = results['perplexity']
                    log_dict["train/codebook_usage"] = results['perplexity'] / args.vq_num_embed
                    
                if 'perceptual_loss' in results:
                    log_dict["train/perceptual_loss"] = results['perceptual_loss']
                    
                # 记录FP16状态
                if 'using_fp16' in results:
                    log_dict["train/using_fp16"] = results['using_fp16']
                    
                wandb.log(log_dict, step=global_step)
            
            # 保存重建图像
            if args.save_images and global_step % args.logging_steps == 0:
                # 保存图像并获取路径
                img_path = save_reconstructed_images(
                    images.detach().cpu(),
                    reconstructed.detach().cpu(),
                    epoch,
                    global_step,
                    images_dir
                )
                
                if args.use_wandb and wandb is not None:
                    wandb.log({
                        "reconstruction": wandb.Image(img_path)
                    }, step=global_step)
            
            # 更新全局步数
            global_step += 1
        
        # 关闭进度条
        progress_bar.close()
        
        # 计算并显示平均损失
        avg_loss = epoch_loss / len(train_dataloader)
        avg_recon_loss = epoch_recon_loss / len(train_dataloader)
        avg_vq_loss = epoch_vq_loss / len(train_dataloader)
        
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.6f}, 重建损失: {avg_recon_loss:.6f}, VQ损失: {avg_vq_loss:.6f}")
        
        # 显示平均码本利用率
        if perplexity_count > 0:
            avg_perplexity = epoch_perplexity / perplexity_count
            usage_percent = avg_perplexity / args.vq_num_embed * 100
            print(f"平均码本利用率: {avg_perplexity:.1f}/{args.vq_num_embed} ({usage_percent:.2f}%)")
        
        # 每save_epochs轮保存一次模型
        if (epoch + 1) % args.save_epochs == 0:
            # 删除之前的轮数保存
            previous_epoch_path = os.path.join(args.output_dir, f"epoch-{epoch + 1 - args.save_epochs}")
            if os.path.exists(previous_epoch_path):
                shutil.rmtree(previous_epoch_path)
                
            # 保存新模型
            model_path = os.path.join(args.output_dir, f"epoch-{epoch + 1}")
            model.save_pretrained(model_path)
            print(f"Epoch {epoch+1} 模型已保存到 {model_path}")
        
        # 执行全面验证 - 每轮结束后验证一次
        val_results = validate(
            val_dataloader, 
            trainer, 
            args.device, 
            global_step,
            args.use_wandb and wandb is not None,
            args.save_images,
            images_dir,
            epoch
        )
        
        val_recon_loss = val_results['recon_loss']
        print(f"Epoch {epoch+1} 验证结果: 总损失={val_results['loss']:.6f}, 重建损失={val_recon_loss:.6f}, VQ损失={val_results['vq_loss']:.6f}")
        
        if args.use_wandb and wandb is not None:
            wandb.log({
                "epoch": epoch + 1,
                "val/epoch_loss": val_results['loss'],
                "val/epoch_recon_loss": val_recon_loss,
                "val/epoch_vq_loss": val_results['vq_loss'],
                "train/epoch_loss": avg_loss,
                "train/epoch_recon_loss": avg_recon_loss,
                "train/epoch_vq_loss": avg_vq_loss
            }, step=global_step)
        
        # 检查是否为最佳验证损失
        if val_recon_loss < best_val_recon_loss:
            best_val_recon_loss = val_recon_loss
            early_stop_counter = 0
            
            # 保存最佳模型
            if best_model_path:
                # 删除之前的最佳模型
                if os.path.exists(best_model_path):
                    shutil.rmtree(best_model_path)
            
            # 保存新的最佳模型
            best_model_path = os.path.join(args.output_dir, "best-model")
            model.save_pretrained(best_model_path)
            print(f"新的最佳模型已保存到 {best_model_path} (验证重建损失: {best_val_recon_loss:.6f})")
            
            if args.use_wandb and wandb is not None:
                wandb.log({
                    "best_val_recon_loss": best_val_recon_loss
                }, step=global_step)
        else:
            early_stop_counter += 1
            print(f"验证损失未改善，当前最佳: {best_val_recon_loss:.6f}, 早停计数: {early_stop_counter}/3")
            
            # 检查是否触发早停
            if early_stop_counter >= 3:
                print(f"触发早停！验证损失连续3次评估未改善。")
                break
    
    # 保存最终模型
    model.save_pretrained(args.output_dir)
    print(f"最终模型已保存到 {args.output_dir}")
    
    # 训练结束
    print(f"训练结束，最佳验证重建损失: {best_val_recon_loss:.6f}")
    if args.use_wandb and wandb is not None:
        wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    print(f"正在使用设备: {args.device}")
    train_vqvae(args) 