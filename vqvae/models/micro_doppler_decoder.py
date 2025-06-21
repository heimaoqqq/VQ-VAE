import torch
import torch.nn as nn
import torch.nn.functional as F

class MicroDopplerDecoder(nn.Module):
    """
    专为微多普勒时频图设计的解码器
    结合了频率增强和波形保持特性
    """
    def __init__(self, in_channels=256, out_channels=1, base_channels=64):
        super().__init__()
        
        # 初始处理
        self.init_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        
        # 第一次上采样块 - 从特征图到中等分辨率
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2)
        )
        
        # 频率增强模块 - 增强频率特征
        self.freq_enhance = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=(3, 5), stride=1, padding=(1, 2)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=(3, 5), stride=1, padding=(1, 2)),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2)
        )
        
        # 第二次上采样块
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2)
        )
        
        # 波形保持模块 - 保持周期性波形特征
        self.wave_preserve = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=(7, 3), stride=1, padding=(3, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels, base_channels, kernel_size=(7, 3), stride=1, padding=(3, 1)),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2)
        )
        
        # 第三次上采样块
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.LeakyReLU(0.2)
        )
        
        # 细节增强模块
        self.detail_enhance = nn.Sequential(
            nn.Conv2d(base_channels // 2, base_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        # 输出层
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_channels // 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # 输出范围[-1, 1]
        )
        
        # 残差连接
        self.skip1 = nn.Conv2d(in_channels, base_channels * 2, kernel_size=1)
        self.skip2 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)
        
    def forward(self, x):
        # 初始处理
        x = self.init_conv(x)
        
        # 第一次上采样
        skip1 = F.interpolate(self.skip1(x), scale_factor=2, mode='nearest')
        up1 = self.up1(x) + skip1
        
        # 频率增强
        freq = self.freq_enhance(up1)
        
        # 第二次上采样
        skip2 = F.interpolate(self.skip2(freq), scale_factor=2, mode='nearest')
        up2 = self.up2(freq) + skip2
        
        # 波形保持
        wave = self.wave_preserve(up2)
        
        # 第三次上采样
        up3 = self.up3(wave)
        
        # 细节增强
        detail = self.detail_enhance(up3)
        
        # 输出
        out = self.output_conv(detail)
        
        return out 