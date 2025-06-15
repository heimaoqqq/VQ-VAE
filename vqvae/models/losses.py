"""
VQ-VAE 损失函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
import torchvision.transforms as transforms
import numpy as np

class PerceptualLoss(nn.Module):
    """基于VGG16的感知损失"""
    def __init__(self, device, resize=True):
        super(PerceptualLoss, self).__init__()
        # 尝试使用新版API加载预训练VGG16模型
        try:
            self.vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.to(device).eval()
        except:
            # 兼容旧版torchvision
            self.vgg = vgg16(pretrained=True).features.to(device).eval()
            
        # 冻结VGG16参数
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        self.device = device
        self.resize = resize
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
        # 选择VGG层进行对比
        self.layers = {'3': 'relu1_2', 
                      '8': 'relu2_2', 
                      '15': 'relu3_3', 
                      '22': 'relu4_3'}
                      
        self.weights = {
            'relu1_2': 0.1,
            'relu2_2': 0.2,
            'relu3_3': 0.4,
            'relu4_3': 0.3
        }
    
    def extract_features(self, x):
        """从VGG16模型中提取特征"""
        features = {}
        # 确保输入在[0,1]范围内
        x = (x - self.mean) / self.std
        
        # 提取各层特征
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
                
        return features
    
    def forward(self, x, y):
        # 确保输入是正确的范围[0,1]
        if x.min() < 0 or x.max() > 1:
            x = (x + 1) / 2.0  # 从[-1,1]转换为[0,1]
        if y.min() < 0 or y.max() > 1:
            y = (y + 1) / 2.0  # 从[-1,1]转换为[0,1]
        
        # 提取特征
        x_features = self.extract_features(x)
        y_features = self.extract_features(y)
        
        # 计算各层特征的损失
        loss = 0
        for layer in self.layers.values():
            h_x = x_features[layer]
            h_y = y_features[layer]
            loss += self.weights[layer] * F.mse_loss(h_x, h_y)
            
        return loss

class PositionalLoss(nn.Module):
    """位置感知损失，特别关注垂直方向上的分布一致性"""
    def __init__(self, device, lambda_vertical=1.0, lambda_horizontal=0.5):
        super(PositionalLoss, self).__init__()
        self.device = device
        self.lambda_vertical = lambda_vertical  # 垂直方向权重
        self.lambda_horizontal = lambda_horizontal  # 水平方向权重
    
    def compute_vertical_profile(self, x):
        """计算图像的垂直分布轮廓"""
        # 计算每一行的平均强度 [B, C, H, W] -> [B, C, H]
        return torch.mean(x, dim=3)
    
    def compute_horizontal_profile(self, x):
        """计算图像的水平分布轮廓"""
        # 计算每一列的平均强度 [B, C, H, W] -> [B, C, W]
        return torch.mean(x, dim=2)
    
    def compute_frequency_distribution(self, x):
        """计算图像在频率域的分布特征"""
        # 对每个通道应用傅里叶变换
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        # 计算幅度谱
        x_magnitude = torch.abs(x_fft)
        # 对幅度谱进行对数变换，使其更符合人类视觉感知
        x_log_magnitude = torch.log(x_magnitude + 1e-10)
        return x_log_magnitude
    
    def forward(self, x, y):
        """计算位置感知损失"""
        # 确保输入是正确的范围[0,1]
        if x.min() < 0 or x.max() > 1:
            x = (x + 1) / 2.0  # 从[-1,1]转换为[0,1]
        if y.min() < 0 or y.max() > 1:
            y = (y + 1) / 2.0  # 从[-1,1]转换为[0,1]
        
        # 计算垂直分布轮廓
        x_vertical = self.compute_vertical_profile(x)
        y_vertical = self.compute_vertical_profile(y)
        vertical_loss = F.mse_loss(x_vertical, y_vertical)
        
        # 计算水平分布轮廓
        x_horizontal = self.compute_horizontal_profile(x)
        y_horizontal = self.compute_horizontal_profile(y)
        horizontal_loss = F.mse_loss(x_horizontal, y_horizontal)
        
        # 计算频率域分布差异
        x_freq = self.compute_frequency_distribution(x)
        y_freq = self.compute_frequency_distribution(y)
        freq_loss = F.mse_loss(x_freq, y_freq)
        
        # 组合损失，特别强调垂直方向
        total_loss = (self.lambda_vertical * vertical_loss + 
                      self.lambda_horizontal * horizontal_loss + 
                      0.5 * freq_loss)
        
        return total_loss 