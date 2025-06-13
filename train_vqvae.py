import os
import argparse
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from diffusers import VQModel, DDIMScheduler
import wandb
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from dataset import get_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description="训练VQ-VAE模型")
    parser.add_argument("--data_dir", type=str, default="dataset", help="数据集目录")
    parser.add_argument("--output_dir", type=str, default="vqvae_model", help="模型保存目录")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--image_size", type=int, default=256, help="图像尺寸")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率")
    parser.add_argument("--num_train_steps", type=int, default=None, help="训练步数")
    parser.add_argument("--save_steps", type=int, default=500, help="保存间隔步数")
    parser.add_argument("--logging_steps", type=int, default=100, help="日志间隔步数")
    parser.add_argument("--latent_channels", type=int, default=3, help="潜变量通道数")
    parser.add_argument("--vq_embed_dim", type=int, default=64, help="VQ嵌入维度")
    parser.add_argument("--vq_num_embed", type=int, default=128, help="VQ嵌入数量")
    parser.add_argument("--n_layers", type=int, default=3, help="下采样层数")
    parser.add_argument("--save_images", action="store_true", help="是否保存重建图像")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb记录训练")
    parser.add_argument("--wandb_project", type=str, default="vq-vae-microdoppler", help="wandb项目名")
    parser.add_argument("--wandb_name", type=str, default="vqvae-training", help="wandb运行名")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")
    return parser.parse_args()

def create_vq_model(args):
    """创建VQModel实例"""
    # 根据参数设置层数和通道数
    n_layers = args.n_layers
    
    # 动态构建层配置
    down_block_types = ["DownEncoderBlock2D"] * n_layers
    up_block_types = ["UpDecoderBlock2D"] * n_layers
    
    # 构建通道配置，从128开始，每层翻倍，最多到512
    block_out_channels = []
    current_channels = 64
    for i in range(n_layers):
        current_channels = min(current_channels * 2, 512)
        block_out_channels.append(current_channels)
    
    model = VQModel(
        in_channels=3,  # RGB图像
        out_channels=3,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(block_out_channels),
        latent_channels=args.latent_channels,
        num_vq_embeddings=args.vq_num_embed,
        vq_embed_dim=args.vq_embed_dim,
    )
    return model

def save_reconstructed_images(original, reconstructed, epoch, step, output_dir):
    """保存原始图像和重建图像对比"""
    # 将张量转换为[0,1]范围的图像
    original = (original + 1) / 2
    reconstructed = (reconstructed + 1) / 2
    
    # 创建网格图像
    batch_size = original.size(0)
    comparison = torch.cat([original, reconstructed], dim=0)
    save_path = os.path.join(output_dir, f"recon_epoch{epoch}_step{step}.png")
    
    # 保存图像
    grid = make_grid(comparison, nrow=batch_size)
    save_image(grid, save_path)
    
    return save_path

class VQModelTrainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
    
    def compute_vq_loss(self, encoder_output, decoder_output):
        """计算VQ损失，适应不同版本的diffusers API"""
        if hasattr(encoder_output, "loss"):
            return encoder_output.loss
        
        # 如果没有直接提供loss属性，手动计算一个简单的VQ损失
        # 这是一个备选方案，实际损失计算应该根据VQ-VAE原理来设计
        if hasattr(encoder_output, "z_q") and hasattr(encoder_output, "z"):
            # 计算commitment loss
            commitment_loss = F.mse_loss(encoder_output.z.detach(), encoder_output.z_q)
            # 计算codebook loss
            codebook_loss = F.mse_loss(encoder_output.z, encoder_output.z_q.detach())
            return codebook_loss + 0.25 * commitment_loss
        
        # 如果连z和z_q都没有，返回0作为VQ损失部分
        return torch.tensor(0.0, device=self.device)
    
    def get_perplexity(self, encoder_output):
        """尝试获取perplexity (码本使用情况)"""
        if hasattr(encoder_output, "perplexity"):
            return encoder_output.perplexity
        return None
    
    def train_step(self, batch):
        """训练步骤"""
        self.optimizer.zero_grad()
        
        # 前向传播
        encoder_output = self.model.encode(batch)
        decoder_output = self.model.decode(encoder_output.latents)
        
        # 计算重建损失
        reconstruction_loss = F.mse_loss(decoder_output.sample, batch)
        
        # 计算VQ损失
        vq_loss = self.compute_vq_loss(encoder_output, decoder_output)
        
        # 总损失
        loss = reconstruction_loss + vq_loss
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        # 返回结果
        result = {
            'loss': loss.item(),
            'recon_loss': reconstruction_loss.item(),
            'vq_loss': vq_loss.item(),
        }
        
        perplexity = self.get_perplexity(encoder_output)
        if perplexity is not None:
            result['perplexity'] = perplexity
            
        return result, decoder_output.sample
    
    @torch.no_grad()
    def eval_step(self, batch):
        """评估步骤"""
        # 前向传播
        encoder_output = self.model.encode(batch)
        decoder_output = self.model.decode(encoder_output.latents)
        
        # 计算重建损失
        reconstruction_loss = F.mse_loss(decoder_output.sample, batch)
        
        # 计算VQ损失
        vq_loss = self.compute_vq_loss(encoder_output, decoder_output)
        
        # 总损失
        loss = reconstruction_loss + vq_loss
        
        # 返回结果
        result = {
            'loss': loss.item(),
            'recon_loss': reconstruction_loss.item(),
            'vq_loss': vq_loss.item(),
        }
        
        perplexity = self.get_perplexity(encoder_output)
        if perplexity is not None:
            result['perplexity'] = perplexity
            
        return result, decoder_output.sample

def train_vqvae(args):
    """训练VQ-VAE模型"""
    # 创建输出目录和可视化目录
    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # 初始化wandb
    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name)
        wandb.config.update(args)

    # 创建数据加载器
    train_dataloader, val_dataloader, _ = get_dataloaders(
        args.data_dir, args.batch_size, args.image_size
    )
    
    # 创建模型
    model = create_vq_model(args)
    model.to(args.device)
    
    # 打印模型结构
    print(f"模型下采样层数: {args.n_layers}")
    print(f"码本大小: {args.vq_num_embed}")
    print(f"潜在空间大小: {args.image_size // (2 ** args.n_layers)}")
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # 创建训练器
    trainer = VQModelTrainer(model, optimizer, args.device)
    
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
    
    # 训练循环
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_vq_loss = 0.0
        tqdm_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step, batch in enumerate(tqdm_bar):
            images = batch.to(args.device)
            
            # 训练步骤
            results, reconstructed = trainer.train_step(images)
            
            # 获取损失
            epoch_loss += results['loss']
            epoch_recon_loss += results['recon_loss']
            epoch_vq_loss += results['vq_loss']
            
            # 更新进度条
            tqdm_bar.set_postfix({
                "loss": results['loss'],
                "recon_loss": results['recon_loss'],
                "vq_loss": results['vq_loss']
            })
            
            # 监控码本使用情况
            if 'perplexity' in results:
                perplexity = results['perplexity']
                print(f"码本利用率: {perplexity}/{args.vq_num_embed} ({perplexity/args.vq_num_embed*100:.2f}%)")
            
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
                
                print(f"重建图像已保存到: {img_path}")
                
                if args.use_wandb:
                    wandb.log({
                        "reconstruction": wandb.Image(img_path)
                    }, step=global_step)
            
            # 记录日志
            if args.use_wandb and global_step % args.logging_steps == 0:
                log_dict = {
                    "train/loss": results['loss'],
                    "train/recon_loss": results['recon_loss'],
                    "train/vq_loss": results['vq_loss'],
                    "train/epoch": epoch,
                    "train/step": global_step,
                }
                if 'perplexity' in results:
                    log_dict["train/perplexity"] = results['perplexity']
                wandb.log(log_dict, step=global_step)
            
            # 保存模型
            if global_step > 0 and global_step % args.save_steps == 0:
                model_path = os.path.join(args.output_dir, f"vqmodel_step_{global_step}")
                model.save_pretrained(model_path)
                print(f"模型已保存到 {model_path}")
            
            global_step += 1
            
            if global_step >= args.num_train_steps:
                break
        
        # 计算平均损失
        avg_loss = epoch_loss / len(train_dataloader)
        avg_recon_loss = epoch_recon_loss / len(train_dataloader)
        avg_vq_loss = epoch_vq_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.6f}, 重建: {avg_recon_loss:.6f}, VQ: {avg_vq_loss:.6f}")
        
        # 验证
        if val_dataloader:
            val_results = validate(val_dataloader, trainer, args.device, global_step, args.use_wandb, args.save_images, images_dir, epoch)
            print(f"验证损失: {val_results['loss']:.6f}, 重建损失: {val_results['recon_loss']:.6f}, VQ损失: {val_results['vq_loss']:.6f}")
        
        # 每个epoch结束保存模型
        model_path = os.path.join(args.output_dir, f"vqmodel_epoch_{epoch+1}")
        model.save_pretrained(model_path)
        print(f"Epoch {epoch+1} 模型已保存到 {model_path}")
        
        if global_step >= args.num_train_steps:
            break

    # 保存最终模型
    model.save_pretrained(args.output_dir)
    print(f"最终模型已保存到 {args.output_dir}")

def validate(dataloader, trainer, device, global_step, use_wandb=False, save_images=False, images_dir=None, epoch=0):
    """验证模型"""
    trainer.model.eval()
    val_loss = 0.0
    val_recon_loss = 0.0
    val_vq_loss = 0.0
    val_perplexity = 0.0
    perplexity_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch.to(device)
            results, reconstructed = trainer.eval_step(images)
            
            val_loss += results['loss']
            val_recon_loss += results['recon_loss']
            val_vq_loss += results['vq_loss']
            
            if 'perplexity' in results:
                val_perplexity += results['perplexity']
                perplexity_count += 1
            
            # 只保存第一个批次的重建图像
            if save_images and batch_idx == 0 and images_dir:
                img_path = save_reconstructed_images(
                    images.detach().cpu(),
                    reconstructed.detach().cpu(),
                    epoch,
                    f"val_{global_step}",
                    images_dir
                )
                
                if use_wandb:
                    wandb.log({
                        "validation_reconstruction": wandb.Image(img_path)
                    }, step=global_step)
    
    # 计算平均值
    results = {
        'loss': val_loss / len(dataloader),
        'recon_loss': val_recon_loss / len(dataloader),
        'vq_loss': val_vq_loss / len(dataloader)
    }
    
    # 打印码本使用情况
    if perplexity_count > 0:
        avg_perplexity = val_perplexity / perplexity_count
        print(f"验证码本利用率: {avg_perplexity:.2f}")
        results['perplexity'] = avg_perplexity
        
        if use_wandb:
            wandb.log({
                "val/perplexity": avg_perplexity,
            }, step=global_step)
    
    if use_wandb:
        log_dict = {
            "val/loss": results['loss'],
            "val/recon_loss": results['recon_loss'],
            "val/vq_loss": results['vq_loss']
        }
        wandb.log(log_dict, step=global_step)
    
    return results

if __name__ == "__main__":
    args = parse_args()
    print(f"正在使用设备: {args.device}")
    train_vqvae(args) 