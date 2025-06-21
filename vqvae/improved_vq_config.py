"""
改进的VQ-VAE配置，专门用于解决码本坍塌问题
"""

# 码本配置
CODEBOOK_CONFIG = {
    # 码本大小 - 考虑减小码本大小，确保充分利用
    "n_embed": 512,  # 原来是8192或512，如果是512建议减小到256或128
    
    # 嵌入维度 - 保持不变
    "embed_dim": 256,
    
    # EMA衰减率 - 降低以使码本更新更快响应新的数据
    "ema_decay": 0.85,  # 原来是0.95，降低到0.85
    
    # 承诺损失权重 - 增加以鼓励编码器输出接近码本向量
    "commitment_loss_beta": 8.0,  # 原来是3.0，增加到8.0
}

# 训练器配置
TRAINER_CONFIG = {
    # 熵正则化权重 - 增加以鼓励码本使用的均匀性
    "entropy_weight": 1.0,  # 原来是0.3，增加到1.0
    
    # 重建损失权重 - 可以适当降低以平衡其他损失
    "l1_weight": 0.8,  # 原来是1.0，降低到0.8
    
    # 感知损失权重 - 保持不变
    "perceptual_weight": 0.005,
    
    # 对抗损失权重 - 可以适当降低以减少对抗性影响
    "adversarial_weight": 0.6,  # 原来是0.8，降低到0.6
    
    # 低使用率码元重置间隔 - 减小以更频繁地重置
    "reset_low_usage_interval": 1,  # 原来是5，减小到1，每个epoch都重置
    
    # 低使用率码元重置比例 - 增加以重置更多的低使用率码元
    "reset_low_usage_percentage": 0.2,  # 原来是0.1，增加到0.2
}

# 优化器配置
OPTIMIZER_CONFIG = {
    # 学习率 - 使用较小的学习率
    "learning_rate": 1e-4,  # 如果原来是1e-3或更高，降低到1e-4
    
    # Adam优化器参数
    "beta1": 0.5,  # 原来可能是0.9，降低到0.5以增加稳定性
    "beta2": 0.999,  # 保持不变
    
    # 权重衰减 - 添加轻微的权重衰减以防止过拟合
    "weight_decay": 1e-5,  # 如果原来没有，添加轻微的权重衰减
}

# 训练策略配置
TRAINING_STRATEGY = {
    # 批次大小 - 使用较大的批次以增加码本使用多样性
    "batch_size": 64,  # 如果原来是32或更小，增加到64
    
    # 温度参数 - 添加温度参数使码本选择更加随机化
    "temperature": 1.2,  # 大于1.0使选择更随机
    
    # 码本初始化 - 使用更好的初始化方法
    "codebook_init": "uniform",  # 均匀初始化
    
    # 预热阶段 - 添加预热阶段，逐渐增加承诺损失权重
    "commitment_warmup_epochs": 5,  # 前5个epoch逐渐增加承诺损失权重
    
    # 码本重置策略 - 更积极地重置未使用的码元
    "reset_dead_codes_threshold": 1,  # 原来是2，降低到1
}

# 使用说明
"""
使用方法:

1. 在训练脚本中导入此配置:
   from vqvae.improved_vq_config import CODEBOOK_CONFIG, TRAINER_CONFIG, OPTIMIZER_CONFIG, TRAINING_STRATEGY

2. 创建模型时使用改进的码本配置:
   model = CustomVQGAN(
       n_embed=CODEBOOK_CONFIG["n_embed"],
       embed_dim=CODEBOOK_CONFIG["embed_dim"],
       ema_decay=CODEBOOK_CONFIG["ema_decay"],
       commitment_loss_beta=CODEBOOK_CONFIG["commitment_loss_beta"],
       ...
   )

3. 创建训练器时使用改进的训练器配置:
   trainer = VQGANTrainer(
       vqgan=model,
       discriminator=discriminator,
       entropy_weight=TRAINER_CONFIG["entropy_weight"],
       l1_weight=TRAINER_CONFIG["l1_weight"],
       perceptual_weight=TRAINER_CONFIG["perceptual_weight"],
       adversarial_weight=TRAINER_CONFIG["adversarial_weight"],
       reset_low_usage_interval=TRAINER_CONFIG["reset_low_usage_interval"],
       reset_low_usage_percentage=TRAINER_CONFIG["reset_low_usage_percentage"],
       ...
   )

4. 创建优化器时使用改进的优化器配置:
   g_optimizer = torch.optim.Adam(
       model.parameters(),
       lr=OPTIMIZER_CONFIG["learning_rate"],
       betas=(OPTIMIZER_CONFIG["beta1"], OPTIMIZER_CONFIG["beta2"]),
       weight_decay=OPTIMIZER_CONFIG["weight_decay"]
   )

5. 训练时使用改进的训练策略:
   - 使用TRAINING_STRATEGY["batch_size"]作为批次大小
   - 实现温度参数(在EMAVectorQuantizer的forward方法中)
   - 实现承诺损失预热
"""

# 修改EMAVectorQuantizer实现温度参数的示例代码
"""
def forward(self, inputs, temperature=1.0):
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
    
    # 应用温度参数 - 温度越高，选择越随机
    if temperature != 1.0:
        distances = distances / temperature
        
    # Find closest encodings
    encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
    
    # ... 其余代码保持不变 ...
""" 