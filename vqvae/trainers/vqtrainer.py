"""
VQ-VAE 训练器
"""
import torch
import torch.nn.functional as F
from ..models.losses import PerceptualLoss

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
        
        # 使用自动混合精度，尝试兼容新旧版本的API
        try:
            # 尝试使用新版API (需要device_type参数)
            autocast_context = torch.amp.autocast(device_type=self.device.type if hasattr(self.device, 'type') else 'cuda')
        except TypeError:
            # 回退到旧版API (不需要device_type参数)
            autocast_context = torch.amp.autocast()
        
        with autocast_context:
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
            'using_fp16': True,  # 添加标记以便确认FP16正在使用
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