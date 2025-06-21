import torch
from torch import nn
import torch.nn.functional as F

class EMAVectorQuantizer(nn.Module):
    """
    A custom, self-contained Vector Quantizer with Exponential Moving Average (EMA) updates.
    This implementation is based on the standard VQ-VAE-2 and SoundStream papers to
    ensure stable codebook learning and prevent collapse.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.95, epsilon=1e-5):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # Initialize the codebook embeddings
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
        
        # Buffers for EMA updates, not part of the model's parameters
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', torch.zeros(num_embeddings, self.embedding_dim))

        # 添加一个计数器用于跟踪码元使用情况
        self.register_buffer('usage_count', torch.zeros(num_embeddings))
        self.register_buffer('last_reset_epoch', torch.zeros(1))
        self.register_buffer('last_low_usage_reset_epoch', torch.zeros(1))

    def forward(self, inputs):
        # inputs: (B, C, H, W) -> (B, H, W, C)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # 确保嵌入权重在与输入相同的设备上
        embedding_weight = self.embedding.weight.to(inputs.device)
        
        # Calculate distances: (z - e)^2 = z^2 + e^2 - 2ze
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True) 
            + torch.sum(embedding_weight**2, dim=1)
            - 2 * torch.matmul(flat_input, embedding_weight.t())
        )
            
        # Find closest encodings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and un-flatten
        quantized = torch.matmul(encodings, embedding_weight).view(inputs.shape)
        
        # Use EMA to update the embedding vectors if in training mode
        if self.training:
            # 更新使用计数
            with torch.no_grad():
                # 增加本批次使用的码元计数
                batch_usage = torch.sum(encodings, dim=0)
                self.usage_count = self.usage_count.to(batch_usage.device)
                self.usage_count += batch_usage
                
                # 确保ema_cluster_size在正确的设备上
                self.ema_cluster_size = self.ema_cluster_size.to(encodings.device)
                self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                      (1 - self.decay) * torch.sum(encodings, 0)
                
                # Laplace smoothing to avoid zero counts
                n = torch.sum(self.ema_cluster_size.data)
                self.ema_cluster_size = (
                    (self.ema_cluster_size + self.epsilon)
                    / (n + self.num_embeddings * self.epsilon)
                    * n
                )
                
                # 确保ema_w在正确的设备上
                self.ema_w = self.ema_w.to(flat_input.device)
                dw = torch.matmul(encodings.t(), flat_input)
                self.ema_w = self.ema_w * self.decay + (1 - self.decay) * dw
                
                # 更新嵌入权重
                self.embedding.weight.data = self.embedding.weight.data.to(self.ema_w.device)
                self.embedding.weight.data.copy_(self.ema_w / self.ema_cluster_size.unsqueeze(1))
            
        # Calculate commitment loss
        commitment_loss = self.commitment_cost * F.mse_loss(quantized.detach(), inputs)
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        # Reshape back to (B, C, H, W)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        # The trainer expects a tuple with commitment_loss and the indices
        # We wrap indices in a structure to be compatible
        # (quant_states, commitment_loss, (perplexity, indices, one_hot_encodings))
        
        # Perplexity calculation for monitoring (optional but good practice)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, commitment_loss, (perplexity, encoding_indices, encodings) 
    
    def expand_codebook(self, new_size):
        """扩展码本大小，保留现有码元并添加新码元"""
        if new_size <= self.num_embeddings:
            return False
            
        # 保存旧权重
        old_weight = self.embedding.weight.data
        old_size = self.num_embeddings
        
        # 创建新嵌入层
        self.num_embeddings = new_size
        new_embedding = nn.Embedding(new_size, self.embedding_dim)
        
        # 复制旧权重
        new_embedding.weight.data[:old_size] = old_weight
        
        # 初始化新码元 - 基于现有码元添加随机扰动
        if old_size > 0:
            # 找出使用最频繁的码元
            usage = self.usage_count[:old_size]
            device = old_weight.device  # 获取old_weight的设备
            _, top_indices = torch.topk(usage, min(100, old_size))
            
            # 随机选择这些热门码元进行复制和扰动
            if len(top_indices) > 0:
                indices = top_indices[torch.randint(0, len(top_indices), (new_size - old_size,), device=device)]
            else:
                indices = torch.randint(0, old_size, (new_size - old_size,), device=device)
                
            new_embeddings = old_weight[indices].clone()
            # 增加随机扰动大小，从0.1增加到0.2
            new_embeddings += torch.randn_like(new_embeddings) * 0.2
            new_embedding.weight.data[old_size:] = new_embeddings
        
        # 替换嵌入层
        self.embedding = new_embedding
        
        # 扩展EMA缓冲区
        device = self.ema_cluster_size.device
        new_ema_cluster_size = torch.zeros(new_size, device=device)
        new_ema_cluster_size[:old_size] = self.ema_cluster_size
        self.register_buffer('ema_cluster_size', new_ema_cluster_size)
        
        new_ema_w = torch.zeros(new_size, self.embedding_dim, device=device)
        new_ema_w[:old_size] = self.ema_w
        self.register_buffer('ema_w', new_ema_w)
        
        # 扩展使用计数
        new_usage_count = torch.zeros(new_size, device=device)
        new_usage_count[:old_size] = self.usage_count
        self.register_buffer('usage_count', new_usage_count)
        
        print(f"Codebook expanded from {old_size} to {new_size}")
        return True
    
    def reset_dead_codes(self, threshold=2, current_epoch=None):
        """重置长期未使用的码元"""
        if not self.training or current_epoch is None:
            return 0
            
        # 每3个epoch检查一次（原来是10个epoch）
        if current_epoch - self.last_reset_epoch.item() < 3:
            return 0
            
        with torch.no_grad():
            # 找出使用次数低于阈值的码元
            device = self.embedding.weight.device
            dead_indices = torch.where(self.ema_cluster_size < threshold)[0]
            n_dead = len(dead_indices)
            
            if n_dead > 0:
                # 找出使用最频繁的码元
                _, top_indices = torch.topk(self.ema_cluster_size, min(100, self.num_embeddings - n_dead))
                
                # 为每个死码元随机选择一个活跃码元
                for i, dead_idx in enumerate(dead_indices):
                    # 随机选择一个活跃码元
                    live_idx = top_indices[torch.randint(0, len(top_indices), (1,), device=device).item()]
                    
                    # 复制活跃码元的权重并添加随机扰动
                    self.embedding.weight.data[dead_idx] = self.embedding.weight.data[live_idx].clone()
                    self.embedding.weight.data[dead_idx] += torch.randn_like(self.embedding.weight.data[dead_idx]) * 0.2
                    
                    # 重置EMA统计
                    self.ema_cluster_size[dead_idx] = self.ema_cluster_size[live_idx] * 0.1
                    self.ema_w[dead_idx] = self.ema_w[live_idx] * 0.1
                    self.usage_count[dead_idx] = 0
                
                # 更新最后重置时间
                self.last_reset_epoch.fill_(current_epoch)
                
                print(f"Reset {n_dead} dead codes at epoch {current_epoch}")
            
            return n_dead
    
    def get_codebook_stats(self):
        """获取码本使用统计信息"""
        with torch.no_grad():
            # 确保所有张量在同一设备上
            device = self.embedding.weight.device
            ema_cluster_size = self.ema_cluster_size.to(device)
            
            # 计算活跃码元数量
            active_size = torch.sum(ema_cluster_size > 0).item()
            utilization = active_size / self.num_embeddings
            
            # 计算熵
            if active_size > 0:
                probs = ema_cluster_size / torch.sum(ema_cluster_size)
                entropy = -torch.sum(probs * torch.log2(probs + 1e-10)).item()
                max_entropy = torch.log2(torch.tensor(self.num_embeddings, dtype=torch.float, device=device)).item()
                normalized_entropy = entropy / max_entropy
            else:
                entropy = 0
                normalized_entropy = 0
            
            return {
                "active_size": active_size,
                "utilization": utilization,
                "entropy": entropy,
                "normalized_entropy": normalized_entropy,
                "total_usage": self.usage_count.sum().item()
            } 
    
    def reset_low_usage_codes(self, percentage=0.1, current_epoch=None):
        """重置使用频率最低的一部分码元
        
        参数:
            percentage: 要重置的码元比例，默认为10%
            current_epoch: 当前训练轮次
        """
        if not self.training or current_epoch is None:
            return 0
            
        # 每5个epoch检查一次低使用率码元
        if current_epoch - self.last_low_usage_reset_epoch.item() < 5:
            return 0
            
        with torch.no_grad():
            device = self.embedding.weight.device
            
            # 计算要重置的码元数量
            n_reset = max(1, int(self.num_embeddings * percentage))
            
            # 找出使用频率最低的n_reset个码元
            _, low_usage_indices = torch.topk(self.ema_cluster_size, n_reset, largest=False)
            
            if len(low_usage_indices) > 0:
                # 找出使用最频繁的码元
                _, top_indices = torch.topk(self.ema_cluster_size, min(100, self.num_embeddings - n_reset))
                
                # 为每个低使用率码元随机选择一个高使用率码元
                for i, low_idx in enumerate(low_usage_indices):
                    # 随机选择一个高使用率码元
                    high_idx = top_indices[torch.randint(0, len(top_indices), (1,), device=device).item()]
                    
                    # 复制高使用率码元的权重并添加随机扰动
                    self.embedding.weight.data[low_idx] = self.embedding.weight.data[high_idx].clone()
                    # 增加扰动幅度，促进多样性
                    self.embedding.weight.data[low_idx] += torch.randn_like(self.embedding.weight.data[low_idx]) * 0.4
                    
                    # 重置EMA统计
                    self.ema_cluster_size[low_idx] = self.ema_cluster_size[high_idx] * 0.2
                    self.ema_w[low_idx] = self.ema_w[high_idx] * 0.2
                    self.usage_count[low_idx] = 0
                
                # 更新最后重置时间
                self.last_low_usage_reset_epoch.fill_(current_epoch)
                
                print(f"Reset {len(low_usage_indices)} low usage codes at epoch {current_epoch}")
            
            return len(low_usage_indices) 