"""
VQ-VAE 损失函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
import torchvision.transforms as transforms

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