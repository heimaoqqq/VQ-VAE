"""
用于VQ-GAN的PatchGAN判别器模型
"""

import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    """
    一个简单的PatchGAN判别器.
    将输入图像映射到一个N x N的patch网格, 每个patch的值代表其为'真'的概率.
    """
    def __init__(self, input_channels=3, n_filters_start=64, n_layers=3):
        """
        初始化判别器.
        
        参数:
            input_channels (int): 输入图像的通道数 (例如, 3 for RGB).
            n_filters_start (int): 第一层卷积的滤波器数量.
            n_layers (int): 判别器中的卷积层数量.
        """
        super().__init__()
        
        layers = [
            nn.Conv2d(input_channels, n_filters_start, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        n_filters1 = n_filters_start
        n_filters2 = n_filters_start
        
        for i in range(n_layers):
            n_filters1 = n_filters2
            n_filters2 = n_filters_start * (2 ** (i + 1))
            layers.extend([
                nn.Conv2d(n_filters1, n_filters2, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(n_filters2),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            
        # 最终输出层, 输出一个单通道的概率图
        layers.append(
            nn.Conv2d(n_filters2, 1, kernel_size=4, stride=1, padding=1)
        )
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播.
        
        参数:
            x (Tensor): 输入图像.
        
        返回:
            Tensor: N x N 的patch概率图.
        """
        return self.model(x)

def weights_init(m):
    """
    自定义权重初始化函数.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0) 