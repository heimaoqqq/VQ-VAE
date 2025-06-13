import os
import argparse
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import sys
import shutil

# 添加版本兼容性检查
try:
    from diffusers import VQModel, DDIMScheduler
except ImportError as e:
    if "cannot import name 'cached_download' from 'huggingface_hub'" in str(e):
        print("检测到huggingface_hub版本不兼容。尝试安装兼容版本...")
        os.system("pip install huggingface_hub>=0.20.2")
        os.system("pip install diffusers>=0.26.3 --no-deps")
        print("请重新运行脚本")
        sys.exit(1)
    else:
        raise e

import wandb
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from dataset import get_dataloaders

# 添加感知损失相关导入
import torchvision.models as models

class PerceptualLoss(torch.nn.Module):
    """基于VGG的感知损失"""
    def __init__(self, device, resize=True):
        super(PerceptualLoss, self).__init__()
        # 加载预训练的VGG16模型
        vgg = models.vgg16(pretrained=True).features.to(device)
        vgg.eval()
        # 冻结VGG参数
        for param in vgg.parameters():
            param.requires_grad = False
            
        # 选择需要提取特征的层
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        
        # VGG的不同层分组
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg[x])
            
        self.device = device
        self.resize = resize
        self.normalize = torch.nn.functional.normalize
        
    def forward(self, x, y):
        # 确保输入是正确的范围[0,1]
        if x.min() < 0 or y.min() < 0:
            x = (x + 1) / 2  # 从[-1,1]转换到[0,1]
            y = (y + 1) / 2
        
        # 从[0,1]转换到归一化的VGG输入
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        x = (x - mean) / std
        y = (y - mean) / std
        
        # 提取特征
        h_x1 = self.slice1(x)
        h_x2 = self.slice2(h_x1)
        h_x3 = self.slice3(h_x2)
        h_x4 = self.slice4(h_x3)
        
        h_y1 = self.slice1(y)
        h_y2 = self.slice2(h_y1)
        h_y3 = self.slice3(h_y2)
        h_y4 = self.slice4(h_y3)
        
        # 计算各层特征的损失
        loss = F.mse_loss(h_x1, h_y1) + \
               F.mse_loss(h_x2, h_y2) + \
               F.mse_loss(h_x3, h_y3) + \
               F.mse_loss(h_x4, h_y4)
               
        return loss

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
    parser.add_argument("--vq_embed_dim", type=int, default=128, help="VQ嵌入维度")
    parser.add_argument("--vq_num_embed", type=int, default=256, help="VQ嵌入数量")
    parser.add_argument("--n_layers", type=int, default=4, help="下采样层数")
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

def create_vq_model(args):
    """创建VQModel实例"""
    # 根据参数设置层数和通道数
    n_layers = args.n_layers
    
    # 动态构建层配置
    down_block_types = ["DownEncoderBlock2D"] * n_layers
    up_block_types = ["UpDecoderBlock2D"] * n_layers
    
    # 构建通道配置，从128开始，每层翻倍，最多到512
    block_out_channels = []
    current_channels = 128
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
    def __init__(self, model, optimizer, device, use_perceptual=False, lambda_perceptual=0.1):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.use_perceptual = use_perceptual
        self.lambda_perceptual = lambda_perceptual
        
        # 初始化感知损失模型
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss(device)
            print(f"启用感知损失，权重为 {self.lambda_perceptual}")
        else:
            self.perceptual_loss = None
    
    def compute_vq_loss(self, encoder_output, decoder_output):
        """计算VQ损失，适应diffusers 0.33.1及以上版本API"""
        # 检查encoder_output中是否有loss属性
        if hasattr(encoder_output, "loss"):
            return encoder_output.loss
        
        # 检查decoder_output中的commit_loss (diffusers 0.33.1新API)
        if hasattr(decoder_output, "commit_loss") and decoder_output.commit_loss is not None:
            return decoder_output.commit_loss
        
        # 如果encoder_output有z和z_q属性，手动计算
        if hasattr(encoder_output, "z") and hasattr(encoder_output, "z_q"):
            # 计算commitment loss
            commitment_loss = F.mse_loss(encoder_output.z.detach(), encoder_output.z_q)
            # 计算codebook loss
            codebook_loss = F.mse_loss(encoder_output.z, encoder_output.z_q.detach())
            return codebook_loss + 0.25 * commitment_loss
        
        # 没有可用的VQ损失计算方法，使用默认值
        # 对于diffusers 0.33.1，我们可以使用一个适当的默认损失值或警告信息
        print("警告: 无法计算VQ损失，使用默认值。请检查diffusers版本与模型兼容性。")
        return torch.tensor(0.1, device=self.device)
    
    def get_perplexity(self, encoder_output):
        """获取码本使用情况指标 (perplexity)"""
        # 如果直接提供perplexity属性
        if hasattr(encoder_output, "perplexity"):
            return encoder_output.perplexity
        
        # 在diffusers 0.33.1版本中，尝试通过模型的quantize模块获取
        try:
            quantize = self.model.quantize
            if hasattr(quantize, "embedding") and hasattr(self.model, "quantize"):
                # 计算所有潜在向量的临近码本索引
                with torch.no_grad():
                    batch = encoder_output.latents.permute(0, 2, 3, 1).reshape(-1, quantize.vq_embed_dim)
                    
                    # 计算与码本的距离
                    d = torch.sum(batch ** 2, dim=1, keepdim=True) + \
                        torch.sum(quantize.embedding.weight ** 2, dim=1) - \
                        2 * torch.matmul(batch, quantize.embedding.weight.t())
                        
                    # 获取最近的码本索引
                    encoding_indices = torch.argmin(d, dim=1)
                    
                    # 计算唯一索引数(被使用的码本向量数量)
                    unique_indices = torch.unique(encoding_indices)
                    perplexity = len(unique_indices)
                    
                    return perplexity
        except Exception as e:
            print(f"计算码本利用率时出错: {e}")
        
        # 无法计算perplexity
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
        
        # 初始化总损失
        total_loss = reconstruction_loss + vq_loss
        
        # 如果启用了感知损失，添加到总损失中
        perceptual_loss_val = 0.0
        if self.use_perceptual and self.perceptual_loss is not None:
            perceptual_loss_val = self.perceptual_loss(decoder_output.sample, batch)
            total_loss = total_loss + self.lambda_perceptual * perceptual_loss_val
        
        # 反向传播
        total_loss.backward()
        self.optimizer.step()
        
        # 返回结果
        result = {
            'loss': total_loss.item(),
            'recon_loss': reconstruction_loss.item(),
            'vq_loss': vq_loss.item(),
        }
        
        if self.use_perceptual:
            result['perceptual_loss'] = perceptual_loss_val.item()
        
        perplexity = self.get_perplexity(encoder_output)
        if perplexity is not None:
            result['perplexity'] = perplexity
            
        return result, decoder_output.sample
    
    def train_step_fp16(self, batch, scaler):
        """混合精度训练步骤"""
        self.optimizer.zero_grad()
        
        # 使用自动混合精度
        with torch.cuda.amp.autocast():
            # 前向传播
            encoder_output = self.model.encode(batch)
            decoder_output = self.model.decode(encoder_output.latents)
            
            # 计算重建损失
            reconstruction_loss = F.mse_loss(decoder_output.sample, batch)
            
            # 计算VQ损失
            vq_loss = self.compute_vq_loss(encoder_output, decoder_output)
            
            # 初始化总损失
            total_loss = reconstruction_loss + vq_loss
            
            # 如果启用了感知损失，添加到总损失中
            perceptual_loss_val = 0.0
            if self.use_perceptual and self.perceptual_loss is not None:
                perceptual_loss_val = self.perceptual_loss(decoder_output.sample, batch)
                total_loss = total_loss + self.lambda_perceptual * perceptual_loss_val
        
        # 反向传播与优化器步骤
        scaler.scale(total_loss).backward()
        scaler.step(self.optimizer)
        scaler.update()
        
        # 返回结果
        result = {
            'loss': total_loss.item(),
            'recon_loss': reconstruction_loss.item(),
            'vq_loss': vq_loss.item(),
        }
        
        if self.use_perceptual:
            result['perceptual_loss'] = perceptual_loss_val.item()
        
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
        
        # 初始化总损失
        total_loss = reconstruction_loss + vq_loss
        
        # 如果启用了感知损失，添加到总损失中
        perceptual_loss_val = 0.0
        if self.use_perceptual and self.perceptual_loss is not None:
            perceptual_loss_val = self.perceptual_loss(decoder_output.sample, batch)
            total_loss = total_loss + self.lambda_perceptual * perceptual_loss_val
        
        # 返回结果
        result = {
            'loss': total_loss.item(),
            'recon_loss': reconstruction_loss.item(),
            'vq_loss': vq_loss.item(),
        }
        
        if self.use_perceptual:
            result['perceptual_loss'] = perceptual_loss_val.item()
        
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
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        args.data_dir, args.batch_size, args.image_size
    )
    
    # 打印数据集划分信息
    total_samples = len(train_dataloader.dataset) + len(val_dataloader.dataset)
    if test_dataloader is not None:
        total_samples += len(test_dataloader.dataset)
    
    print(f"数据集统计信息:")
    print(f"总样本数: {total_samples}")
    print(f"训练集: {len(train_dataloader.dataset)}个样本 ({len(train_dataloader.dataset)/total_samples*100:.1f}%), {len(train_dataloader)}个批次")
    print(f"验证集: {len(val_dataloader.dataset)}个样本 ({len(val_dataloader.dataset)/total_samples*100:.1f}%), {len(val_dataloader)}个批次")
    if test_dataloader is not None:
        print(f"测试集: {len(test_dataloader.dataset)}个样本 ({len(test_dataloader.dataset)/total_samples*100:.1f}%), {len(test_dataloader)}个批次")
    print(f"批次大小: {args.batch_size}")
    
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
    trainer = VQModelTrainer(model, optimizer, args.device, 
                            use_perceptual=args.use_perceptual, 
                            lambda_perceptual=args.lambda_perceptual)
    
    # 使用混合精度训练（用于16G显存）
    if args.fp16 and args.device != "cpu":
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler()
        print("使用混合精度训练 (FP16)")
    else:
        scaler = None
    
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
        tqdm_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step, batch in enumerate(tqdm_bar):
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
            
            # 更新进度条
            tqdm_bar.set_postfix({
                "loss": results['loss'],
                "recon_loss": results['recon_loss'],
                "vq_loss": results['vq_loss'],
                "perceptual": results.get('perceptual_loss', 0.0)
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
                if 'perceptual_loss' in results:
                    log_dict["train/perceptual_loss"] = results['perceptual_loss']
                if 'perplexity' in results:
                    log_dict["train/perplexity"] = results['perplexity']
                wandb.log(log_dict, step=global_step)
            
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
            val_recon_loss = val_results['recon_loss']
            print(f"验证损失: {val_results['loss']:.6f}, 重建损失: {val_recon_loss:.6f}, VQ损失: {val_results['vq_loss']:.6f}")
            
            # 检查是否为最佳验证重建损失
            if val_recon_loss < best_val_recon_loss:
                best_val_recon_loss = val_recon_loss
                early_stop_counter = 0
                
                # 保存最佳模型
                # 先删除之前的最佳模型
                if best_model_path and os.path.exists(best_model_path):
                    if os.path.isdir(best_model_path):
                        shutil.rmtree(best_model_path)
                
                # 保存新的最佳模型
                best_model_path = os.path.join(args.output_dir, "best_model")
                model.save_pretrained(best_model_path)
                print(f"新的最佳模型已保存到 {best_model_path} (验证重建损失: {best_val_recon_loss:.6f})")
                
                if args.use_wandb:
                    wandb.log({"best_val_recon_loss": best_val_recon_loss}, step=global_step)
            else:
                early_stop_counter += 1
                print(f"验证重建损失未改善。早停计数器: {early_stop_counter}/3")
                
                # 检查是否触发早停
                if early_stop_counter >= 3:
                    print(f"触发早停！验证重建损失连续3轮未改善。")
                    break
        
        # 每save_epochs轮保存一次模型
        if (epoch + 1) % args.save_epochs == 0:
            # 删除之前可能存在的轮数保存模型
            previous_epoch_model = os.path.join(args.output_dir, f"vqmodel_epoch_{epoch + 1 - args.save_epochs}")
            if os.path.exists(previous_epoch_model):
                if os.path.isdir(previous_epoch_model):
                    shutil.rmtree(previous_epoch_model)
            
            # 保存新的轮数模型
            model_path = os.path.join(args.output_dir, f"vqmodel_epoch_{epoch + 1}")
            model.save_pretrained(model_path)
            print(f"Epoch {epoch+1} 模型已保存到 {model_path}")
        
        if global_step >= args.num_train_steps:
            break

    # 保存最终模型
    model.save_pretrained(args.output_dir)
    print(f"最终模型已保存到 {args.output_dir}")
    
    # 打印最佳验证结果
    print(f"训练结束。最佳验证重建损失: {best_val_recon_loss:.6f}")

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