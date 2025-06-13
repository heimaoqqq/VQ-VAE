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
        try:
            # 新版API
            from torchvision.models import VGG16_Weights
            vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device)
        except ImportError:
            # 旧版API兼容
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
                    # 获取latents并确保是float32类型，防止half和float类型不匹配
                    latents = encoder_output.latents.to(torch.float32)
                    batch = latents.permute(0, 2, 3, 1).reshape(-1, quantize.vq_embed_dim)
                    
                    # 确保embedding权重也是float32类型
                    embedding_weight = quantize.embedding.weight.to(torch.float32)
                    
                    # 计算与码本的距离 (确保都是float32类型)
                    d = torch.sum(batch ** 2, dim=1, keepdim=True) + \
                        torch.sum(embedding_weight ** 2, dim=1) - \
                        2 * torch.matmul(batch, embedding_weight.t())
                        
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
        epoch_perplexity = 0.0
        perplexity_count = 0
        
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
                
            progress_bar.set_postfix(status_dict)
            progress_bar.update(1)
            
            # wandb记录
            if args.use_wandb and global_step % args.logging_steps == 0:
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
                
                if args.use_wandb:
                    wandb.log({
                        "reconstruction": wandb.Image(img_path)
                    }, step=global_step)
            
            # 验证
            if global_step > 0 and global_step % args.eval_steps == 0:
                val_results = validate(
                    val_dataloader, 
                    trainer, 
                    args.device, 
                    global_step,
                    args.use_wandb,
                    args.save_images,
                    images_dir,
                    epoch
                )
                
                # 获取验证重建损失
                val_recon_loss = val_results['recon_loss']
                progress_bar.write(f"验证损失: {val_results['loss']:.4f}, 重建损失: {val_recon_loss:.4f}, VQ损失: {val_results['vq_loss']:.4f}")
                
                if args.use_wandb:
                    wandb.log({
                        "val/loss": val_results['loss'],
                        "val/recon_loss": val_recon_loss,
                        "val/vq_loss": val_results['vq_loss'],
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
                    progress_bar.write(f"新的最佳模型已保存到 {best_model_path} (验证重建损失: {best_val_recon_loss:.6f})")
                    
                    if args.use_wandb:
                        wandb.log({
                            "best_val_recon_loss": best_val_recon_loss
                        }, step=global_step)
                else:
                    early_stop_counter += 1
                    progress_bar.write(f"验证损失未改善，当前最佳: {best_val_recon_loss:.6f}, 早停计数: {early_stop_counter}/3")
                    
                    # 检查是否触发早停
                    if early_stop_counter >= 3:
                        progress_bar.write(f"触发早停！验证损失连续3次评估未改善。")
                        break
            
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
        
        # 执行全面验证
        val_results = validate(
            val_dataloader, 
            trainer, 
            args.device, 
            global_step,
            args.use_wandb,
            args.save_images,
            images_dir,
            epoch
        )
        
        val_recon_loss = val_results['recon_loss']
        print(f"Epoch {epoch+1} 验证结果: 总损失={val_results['loss']:.6f}, 重建损失={val_recon_loss:.6f}, VQ损失={val_results['vq_loss']:.6f}")
        
        if args.use_wandb:
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
            
            if args.use_wandb:
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
    if args.use_wandb:
        wandb.finish()

@torch.no_grad()
def validate(dataloader, trainer, device, global_step, use_wandb=False, save_images=False, images_dir=None, epoch=0):
    """验证模型"""
    trainer.model.eval()
    val_loss = 0.0
    val_recon_loss = 0.0
    val_vq_loss = 0.0
    val_perceptual_loss = 0.0
    val_perplexity = 0.0
    perplexity_count = 0
    all_results = []
    
    # 只显示一个进度条
    val_bar = tqdm(dataloader, desc="验证中", leave=False)
    
    for batch in val_bar:
        images = batch.to(device)
        results, reconstructed = trainer.eval_step(images)
        
        # 累加损失
        val_loss += results['loss']
        val_recon_loss += results['recon_loss']
        val_vq_loss += results['vq_loss']
        
        if 'perceptual_loss' in results:
            val_perceptual_loss += results['perceptual_loss']
        
        if 'perplexity' in results:
            val_perplexity += results['perplexity']
            perplexity_count += 1
        
        all_results.append(results)
        
        # 更新进度条
        val_bar.set_postfix({
            "loss": f"{results['loss']:.4f}",
            "recon": f"{results['recon_loss']:.4f}"
        })
    
    # 计算平均损失
    num_batches = len(dataloader)
    avg_loss = val_loss / num_batches
    avg_recon_loss = val_recon_loss / num_batches
    avg_vq_loss = val_vq_loss / num_batches
    
    # 计算平均感知损失（如果使用了感知损失）
    results = {
        'loss': avg_loss,
        'recon_loss': avg_recon_loss,
        'vq_loss': avg_vq_loss,
    }
    
    if perplexity_count > 0:
        avg_perplexity = val_perplexity / perplexity_count
        results['perplexity'] = avg_perplexity
    
    if val_perceptual_loss > 0:
        avg_perceptual_loss = val_perceptual_loss / num_batches
        results['perceptual_loss'] = avg_perceptual_loss
    
    # 返回结果
    return results

if __name__ == "__main__":
    args = parse_args()
    print(f"正在使用设备: {args.device}")
    train_vqvae(args) 