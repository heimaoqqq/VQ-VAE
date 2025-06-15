"""
标准化工具 - 专为微多普勒时频图设计
"""
import torch
import numpy as np

class MicroDopplerNormalizer:
    """
    微多普勒时频图的分段自适应标准化器
    
    这个类提供了专为微多普勒时频图设计的归一化和反归一化方法，
    使用分段自适应标准化技术来保留垂直方向的分布特征。
    """
    
    def __init__(self, split_ratio=0.5, lower_quantile=0.01, upper_quantile=0.99, 
                 upper_contrast_factor=0.7, clamp_lower=[-1, 1], clamp_upper=[-1, 1]):
        """
        初始化标准化器
        
        参数:
            split_ratio: 图像分割比例，默认在中间(0.5)分割
            lower_quantile: 下半部分的下分位数，用于分位数标准化
            upper_quantile: 下半部分的上分位数，用于分位数标准化
            upper_contrast_factor: 上半部分的对比度增强因子
            clamp_lower: 下半部分的裁剪范围
            clamp_upper: 上半部分的裁剪范围
        """
        self.split_ratio = split_ratio
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.upper_contrast_factor = upper_contrast_factor
        self.clamp_lower = clamp_lower
        self.clamp_upper = clamp_upper
        
        # 保存标准化参数，用于反归一化
        self.params = {}
    
    def normalize(self, image):
        """
        对微多普勒时频图进行分段自适应标准化
        
        参数:
            image: 输入图像张量，形状为 [B, C, H, W]
            
        返回:
            标准化后的图像张量，范围在 [-1, 1]
        """
        # 确保输入是张量
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)
        
        # 保存原始形状
        original_shape = image.shape
        
        # 如果输入是单张图像，添加批次维度
        if len(original_shape) == 3:
            image = image.unsqueeze(0)
        
        # 获取图像尺寸
        batch_size, channels, height, width = image.shape
        split_idx = int(height * self.split_ratio)
        
        # 分割图像为上下两部分
        lower_part = image[:, :, split_idx:, :]  # 下半部分（强信号区域）
        upper_part = image[:, :, :split_idx, :]  # 上半部分（弱信号区域）
        
        # 下半部分使用分位数标准化
        q_low_lower = torch.quantile(lower_part.reshape(batch_size, -1), 
                                    self.lower_quantile, dim=1).reshape(batch_size, 1, 1, 1)
        q_high_lower = torch.quantile(lower_part.reshape(batch_size, -1), 
                                     self.upper_quantile, dim=1).reshape(batch_size, 1, 1, 1)
        norm_lower = (lower_part - q_low_lower) / (q_high_lower - q_low_lower + 1e-8)
        
        # 缩放到指定范围
        lower_range = self.clamp_lower[1] - self.clamp_lower[0]
        norm_lower = torch.clamp(norm_lower, 0, 1) * lower_range + self.clamp_lower[0]
        
        # 上半部分使用增强对比度的标准化
        mean_upper = upper_part.mean(dim=[2, 3], keepdim=True)
        std_upper = upper_part.std(dim=[2, 3], keepdim=True)
        norm_upper = (upper_part - mean_upper) / (std_upper * self.upper_contrast_factor + 1e-8)
        
        # 缩放到指定范围
        upper_range = self.clamp_upper[1] - self.clamp_upper[0]
        norm_factor = 2.0  # 假设标准化后的值大部分在 [-2, 2] 范围内
        norm_upper = torch.clamp(norm_upper / norm_factor, -1, 1) * upper_range/2 + (self.clamp_upper[0] + self.clamp_upper[1])/2
        
        # 保存标准化参数，用于反归一化
        self.params = {
            'split_idx': split_idx,
            'q_low_lower': q_low_lower,
            'q_high_lower': q_high_lower,
            'mean_upper': mean_upper,
            'std_upper': std_upper,
            'upper_contrast_factor': self.upper_contrast_factor,
            'norm_factor': norm_factor
        }
        
        # 重新组合图像
        normalized = torch.zeros_like(image)
        normalized[:, :, :split_idx, :] = norm_upper
        normalized[:, :, split_idx:, :] = norm_lower
        
        # 恢复原始形状
        if len(original_shape) == 3:
            normalized = normalized.squeeze(0)
            
        return normalized
    
    def denormalize(self, normalized_image):
        """
        将标准化后的图像反归一化回原始范围
        
        参数:
            normalized_image: 标准化后的图像张量，范围在 [-1, 1]
            
        返回:
            反归一化后的图像张量
        """
        # 确保输入是张量
        if not isinstance(normalized_image, torch.Tensor):
            normalized_image = torch.tensor(normalized_image, dtype=torch.float32)
        
        # 保存原始形状
        original_shape = normalized_image.shape
        
        # 如果输入是单张图像，添加批次维度
        if len(original_shape) == 3:
            normalized_image = normalized_image.unsqueeze(0)
        
        # 检查是否有标准化参数
        if not self.params:
            raise ValueError("没有可用的标准化参数，请先调用normalize方法")
        
        # 获取参数
        split_idx = self.params['split_idx']
        q_low_lower = self.params['q_low_lower']
        q_high_lower = self.params['q_high_lower']
        mean_upper = self.params['mean_upper']
        std_upper = self.params['std_upper']
        upper_contrast_factor = self.params['upper_contrast_factor']
        norm_factor = self.params['norm_factor']
        
        # 分割图像为上下两部分
        norm_upper = normalized_image[:, :, :split_idx, :]
        norm_lower = normalized_image[:, :, split_idx:, :]
        
        # 反归一化下半部分
        lower_range = self.clamp_lower[1] - self.clamp_lower[0]
        denorm_lower = (norm_lower - self.clamp_lower[0]) / lower_range
        denorm_lower = denorm_lower * (q_high_lower - q_low_lower) + q_low_lower
        
        # 反归一化上半部分
        upper_range = self.clamp_upper[1] - self.clamp_upper[0]
        denorm_upper = (norm_upper - (self.clamp_upper[0] + self.clamp_upper[1])/2) / (upper_range/2) * norm_factor
        denorm_upper = denorm_upper * (std_upper * upper_contrast_factor) + mean_upper
        
        # 重新组合图像
        denormalized = torch.zeros_like(normalized_image)
        denormalized[:, :, :split_idx, :] = denorm_upper
        denormalized[:, :, split_idx:, :] = denorm_lower
        
        # 恢复原始形状
        if len(original_shape) == 3:
            denormalized = denormalized.squeeze(0)
            
        return denormalized


def adaptive_segmented_normalize(image, split_ratio=0.5):
    """
    分段自适应标准化函数，专为微多普勒时频图设计
    
    参数:
        image: 输入图像张量，形状为 [B, C, H, W]
        split_ratio: 图像分割比例，默认在中间(0.5)分割
        
    返回:
        标准化后的图像张量，范围在 [-1, 1]
    """
    # 确保输入是张量
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image, dtype=torch.float32)
    
    # 获取图像尺寸
    if len(image.shape) == 4:
        batch_size, channels, height, width = image.shape
    else:
        channels, height, width = image.shape
        batch_size = 1
        image = image.unsqueeze(0)
    
    split_idx = int(height * split_ratio)
    
    # 分割图像为上下两部分
    lower_part = image[:, :, split_idx:, :]  # 下半部分（强信号区域）
    upper_part = image[:, :, :split_idx, :]  # 上半部分（弱信号区域）
    
    # 下半部分使用分位数标准化
    q_low_lower = torch.quantile(lower_part.reshape(batch_size, -1), 0.01, dim=1).reshape(batch_size, 1, 1, 1)
    q_high_lower = torch.quantile(lower_part.reshape(batch_size, -1), 0.99, dim=1).reshape(batch_size, 1, 1, 1)
    norm_lower = (lower_part - q_low_lower) / (q_high_lower - q_low_lower + 1e-8)
    norm_lower = torch.clamp(norm_lower, 0, 1) * 2 - 1  # 缩放到[-1,1]
    
    # 上半部分使用增强对比度的标准化
    mean_upper = upper_part.mean(dim=[2, 3], keepdim=True)
    std_upper = upper_part.std(dim=[2, 3], keepdim=True)
    norm_upper = (upper_part - mean_upper) / (std_upper * 0.7 + 1e-8)  # 减小除数增强对比度
    norm_upper = torch.clamp(norm_upper, -2, 2) / 2  # 范围限制在[-1,1]
    
    # 重新组合图像
    normalized = torch.zeros_like(image)
    normalized[:, :, :split_idx, :] = norm_upper
    normalized[:, :, split_idx:, :] = norm_lower
    
    # 恢复原始形状
    if len(image.shape) != 4:
        normalized = normalized.squeeze(0)
    
    return normalized 