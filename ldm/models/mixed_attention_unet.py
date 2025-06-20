"""
应用混合Window+Axial注意力策略的UNet包装器
"""

import torch
from diffusers import UNet2DModel
from diffusers.models.attention_processor import AttnProcessor

from .attention import MixedAttentionProcessor, get_attention_processor_for_block


class MixedAttentionUNetWrapper:
    """
    包装UNet模型，应用混合Window+Axial注意力策略
    """
    def __init__(self, unet_model):
        """
        初始化包装器
        
        参数:
            unet_model: 原始UNet2DModel实例
        """
        self.model = unet_model
        self._apply_mixed_attention()
    
    def _apply_mixed_attention(self):
        """应用混合注意力策略到模型的各层"""
        # 获取下采样块总数
        down_blocks_count = len(self.model.down_blocks)
        
        # 应用到下采样块
        for i, block in enumerate(self.model.down_blocks):
            if hasattr(block, 'attentions'):
                for j, attn_block in enumerate(block.attentions):
                    # 获取隐藏维度大小
                    hidden_size = attn_block.to_q.out_features
                    num_heads = attn_block.heads
                    
                    # 应用窗口注意力到前半部分下采样块
                    attn_processor = get_attention_processor_for_block(
                        block_idx=i,
                        total_blocks=down_blocks_count,
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        window_size=8
                    )
                    attn_block.set_processor(attn_processor)
        
        # 获取上采样块总数
        up_blocks_count = len(self.model.up_blocks)
        
        # 应用到上采样块
        for i, block in enumerate(self.model.up_blocks):
            if hasattr(block, 'attentions'):
                for j, attn_block in enumerate(block.attentions):
                    # 获取隐藏维度大小
                    hidden_size = attn_block.to_q.out_features
                    num_heads = attn_block.heads
                    
                    # 应用轴注意力到后半部分上采样块
                    attn_processor = get_attention_processor_for_block(
                        block_idx=up_blocks_count - i - 1,  # 反向索引
                        total_blocks=up_blocks_count,
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        window_size=8
                    )
                    attn_block.set_processor(attn_processor)
        
        # 应用到中间块
        if hasattr(self.model.mid_block, 'attentions'):
            for attn_block in self.model.mid_block.attentions:
                # 获取隐藏维度大小
                hidden_size = attn_block.to_q.out_features
                num_heads = attn_block.heads
                
                # 中间块使用轴分解注意力
                attn_processor = MixedAttentionProcessor(
                    hidden_size=hidden_size,
                    num_attention_heads=num_heads,
                    attention_type="axial"
                )
                attn_block.set_processor(attn_processor)
    
    def __call__(self, *args, **kwargs):
        """委托给原始模型"""
        return self.model(*args, **kwargs)
    
    def parameters(self):
        """委托给原始模型的parameters方法"""
        return self.model.parameters()
    
    def named_parameters(self):
        """委托给原始模型的named_parameters方法"""
        return self.model.named_parameters()
    
    def to(self, *args, **kwargs):
        """委托给原始模型的to方法，并确保所有注意力处理器也正确地移动到设备上"""
        # 获取目标设备
        device = None
        if args:
            device = args[0]
        elif 'device' in kwargs:
            device = kwargs['device']
        
        # 如果没有指定设备，直接返回
        if device is None:
            return self
            
        # 移动基础模型
        self.model = self.model.to(device)
        
        # 确保所有注意力处理器都在正确的设备上
        # 下采样块
        for block in self.model.down_blocks:
            if hasattr(block, 'attentions'):
                for attn_block in block.attentions:
                    if hasattr(attn_block, 'processor'):
                        processor = attn_block.processor
                        if hasattr(processor, 'attention'):
                            processor.attention = processor.attention.to(device)
        
        # 上采样块
        for block in self.model.up_blocks:
            if hasattr(block, 'attentions'):
                for attn_block in block.attentions:
                    if hasattr(attn_block, 'processor'):
                        processor = attn_block.processor
                        if hasattr(processor, 'attention'):
                            processor.attention = processor.attention.to(device)
        
        # 中间块
        if hasattr(self.model.mid_block, 'attentions'):
            for attn_block in self.model.mid_block.attentions:
                if hasattr(attn_block, 'processor'):
                    processor = attn_block.processor
                    if hasattr(processor, 'attention'):
                        processor.attention = processor.attention.to(device)
        
        return self
    
    @property
    def config(self):
        """访问内部模型的配置"""
        return self.model.config
    
    # 添加其他可能需要的属性访问方法
    @property
    def in_channels(self):
        """访问内部模型的输入通道数"""
        return self.model.in_channels
        
    @property
    def dtype(self):
        """访问内部模型的数据类型"""
        return self.model.dtype
        
    @property
    def device(self):
        """访问内部模型的设备"""
        return self.model.device
    
    def __getattr__(self, name):
        """对于未定义的属性，尝试从内部模型获取"""
        try:
            return getattr(self.model, name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def eval(self):
        """委托给原始模型的eval方法"""
        self.model.eval()
        return self
    
    def train(self, mode=True):
        """委托给原始模型的train方法"""
        self.model.train(mode)
        return self


def create_mixed_attention_unet(latent_size, latent_channels=4, window_size=8):
    """
    创建应用混合Window+Axial注意力的UNet模型
    
    参数:
        latent_size: 潜在空间分辨率
        latent_channels: 潜在通道数
        window_size: 窗口注意力的窗口大小
    
    返回:
        应用混合注意力策略的UNet模型
    """
    from .unet import create_unet_model
    
    # 创建标准UNet模型
    model = create_unet_model(latent_size, latent_channels)
    
    # 包装并应用混合注意力
    wrapped_model = MixedAttentionUNetWrapper(model)
    
    print(f"已应用混合Window+Axial注意力策略")
    print(f"窗口大小: {window_size}")
    
    return wrapped_model 