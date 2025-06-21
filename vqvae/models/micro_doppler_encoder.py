import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyAttention(nn.Module):
    """
    专门为微多普勒时频图设计的频域注意力机制
    特别关注频率维度的能量分布
    """
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        
        # 特别关注频率维度
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # 假设x的形状为[B, C, T, F]，其中T是时间维度，F是频率维度
        batch_size, C, T, F = x.size()
        
        # 自注意力计算
        proj_query = self.query(x).view(batch_size, -1, T * F).permute(0, 2, 1)  # [B, T*F, C//8]
        proj_key = self.key(x).view(batch_size, -1, T * F)  # [B, C//8, T*F]
        energy = torch.bmm(proj_query, proj_key)  # [B, T*F, T*F]
        attention = F.softmax(energy, dim=-1)
        
        proj_value = self.value(x).view(batch_size, -1, T * F)  # [B, C, T*F]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, T, F)
        
        return self.gamma * out + x

class MicroDopplerEncoder(nn.Module):
    """
    专为微多普勒时频图设计的编码器
    结合了周期感知、频域注意力和信号-噪声分离
    """
    def __init__(self, in_channels=1, latent_dim=256, base_channels=64):
        super().__init__()
        
        # 基础特征提取
        self.base_encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2)
        )
        
        # 周期性感知分支 - 使用较大卷积核捕捉周期性
        self.periodic_branch = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=(7, 3), stride=1, padding=(3, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=(7, 3), stride=2, padding=(3, 1)),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2)
        )
        
        # 频域注意力分支
        self.freq_attention = FrequencyAttention(base_channels * 2)
        
        # 信号分支 - 关注高能量区域
        self.signal_branch = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2)
        )
        
        # 噪声分支 - 平滑处理低能量区域
        self.noise_branch = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=1),
            nn.LeakyReLU(0.2)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels * 2 + base_channels * 2 + base_channels, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2)
        )
        
        # 最终输出层，保持与原编码器输出一致
        self.output_conv = nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        # 基础特征提取
        base_features = self.base_encoder(x)
        
        # 周期性特征提取
        periodic_features = self.periodic_branch(base_features)
        
        # 频域注意力
        attention_features = self.freq_attention(base_features)
        
        # 信号-噪声分离
        signal_features = self.signal_branch(base_features)
        noise_features = self.noise_branch(base_features)
        
        # 融合所有特征
        combined = torch.cat([periodic_features, signal_features, noise_features], dim=1)
        latent = self.fusion(combined)
        
        # 最终输出
        output = self.output_conv(latent)
        
        return output 