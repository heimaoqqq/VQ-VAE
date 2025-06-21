"""
改进的VQ-VAE配置，专门用于解决码本坍塌问题
"""

# 码本配置
CODEBOOK_CONFIG = {
    # 码本大小 - 大幅减小码本大小，确保充分利用
    "n_embed": 64,  # 原来是512，减小到64以确保更高的利用率
    
    # 嵌入维度 - 保持不变
    "embed_dim": 256,
    
    # EMA衰减率 - 大幅降低以使码本更新更快响应新的数据
    "ema_decay": 0.7,  # 原来是0.85，降低到0.7
    
    # 承诺损失权重 - 进一步增加以鼓励编码器输出接近码本向量
    "commitment_loss_beta": 10.0,  # 原来是8.0，增加到10.0
}

# 训练器配置
TRAINER_CONFIG = {
    # 熵正则化权重 - 大幅增加以鼓励码本使用的均匀性
    "entropy_weight": 2.0,  # 原来是1.0，增加到2.0
    
    # 重建损失权重 - 可以适当降低以平衡其他损失
    "l1_weight": 0.7,  # 原来是0.8，降低到0.7
    
    # 感知损失权重 - 保持不变
    "perceptual_weight": 0.005,
    
    # 对抗损失权重 - 可以适当降低以减少对抗性影响
    "adversarial_weight": 0.4,  # 原来是0.6，降低到0.4
    
    # 低使用率码元重置间隔 - 保持每个epoch都重置
    "reset_low_usage_interval": 1,
    
    # 低使用率码元重置比例 - 大幅增加以重置更多的低使用率码元
    "reset_low_usage_percentage": 0.3,  # 原来是0.2，增加到0.3
}

# 优化器配置
OPTIMIZER_CONFIG = {
    # 学习率 - 使用较小的学习率
    "learning_rate": 1e-4,
    
    # Adam优化器参数
    "beta1": 0.5,
    "beta2": 0.999,
    
    # 权重衰减 - 添加轻微的权重衰减以防止过拟合
    "weight_decay": 1e-5,
}

# 训练策略配置
TRAINING_STRATEGY = {
    # 批次大小 - 使用较大的批次以增加码本使用多样性
    "batch_size": 64,
    
    # 温度参数 - 大幅提高温度参数使码本选择更加随机化
    "temperature": 2.5,  # 原来是1.2，大幅提高到2.5
    
    # 码本初始化 - 使用更好的初始化方法
    "codebook_init": "uniform",
    
    # 预热阶段 - 添加预热阶段，逐渐增加承诺损失权重
    "commitment_warmup_epochs": 5,
    
    # 码本重置策略 - 更积极地重置未使用的码元
    "reset_dead_codes_threshold": 1,
    
    # 禁用码本自动扩展 - 添加新参数
    "disable_codebook_expansion": True,
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
       temperature=TRAINING_STRATEGY["temperature"],  # 确保传入温度参数
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
   - 确保温度参数正确传递给VQGANTrainer
   - 实现承诺损失预热
   - 如果启用了disable_codebook_expansion，确保在trainer中禁用自动扩展
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