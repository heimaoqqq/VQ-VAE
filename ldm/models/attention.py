"""
微多普勒时频图的自定义注意力机制实现
包括窗口注意力和轴分解注意力
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class WindowAttention(nn.Module):
    """
    窗口自注意力机制，类似Swin Transformer中的实现
    将输入分割成固定大小的窗口，在窗口内计算自注意力
    """
    def __init__(self, dim, num_heads=8, window_size=8, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        
        # QKV投影
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        """
        输入x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 将空间维度划分为窗口
        # 调整H和W为窗口大小的整数倍
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        _, _, H_pad, W_pad = x.shape
        
        # 转换为 [B, H, W, C] 格式，便于窗口划分
        x = x.permute(0, 2, 3, 1)
        
        # 划分窗口：[B, num_windows_h, num_windows_w, window_size, window_size, C]
        x = x.view(B, H_pad // self.window_size, self.window_size, W_pad // self.window_size, self.window_size, C)
        # 调整维度: [B, num_windows_h, num_windows_w, window_size*window_size, C]
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(
            B, (H_pad // self.window_size) * (W_pad // self.window_size), self.window_size * self.window_size, C)
        
        # 在每个窗口内计算自注意力
        B_, N_win, N_tokens, _ = x.shape
        
        # 生成qkv: [B_, N_win, N_tokens, 3*C]
        qkv = self.qkv(x)
        qkv = qkv.reshape(B_, N_win, N_tokens, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(3, 0, 1, 4, 2, 5)  # [3, B_, N_win, num_heads, N_tokens, head_dim]
        q, k, v = qkv.unbind(0)
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B_, N_win, num_heads, N_tokens, N_tokens]
        attn = attn.softmax(dim=-1)
        
        # 输出: [B_, N_win, N_tokens, C]
        x = (attn @ v).transpose(2, 3).reshape(B_, N_win, N_tokens, C)
        x = self.proj(x)
        
        # 重构为原始图像格式
        x = x.view(B, H_pad // self.window_size, W_pad // self.window_size, self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H_pad, W_pad, C)
        
        # 转回 [B, C, H, W] 格式
        x = x.permute(0, 3, 1, 2)
        
        # 如果有padding，去除
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]
            
        return x


class AxialAttention(nn.Module):
    """
    轴分解注意力:
    将2D自注意力分解为两个1D自注意力,分别在行和列方向计算
    显著降低计算复杂度
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        # 水平(行)方向的QKV投影
        self.qkv_h = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_h = nn.Linear(dim, dim)
        
        # 垂直(列)方向的QKV投影
        self.qkv_v = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim)
    
    def forward(self, x):
        """
        输入x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 首先计算行方向注意力 (在W维度)
        # 将x转为[B*H, W, C]格式
        x_h = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
        
        # 生成行方向qkv
        qkv_h = self.qkv_h(x_h)
        qkv_h = qkv_h.reshape(B * H, W, 3, self.num_heads, C // self.num_heads)
        qkv_h = qkv_h.permute(2, 0, 3, 1, 4)  # [3, B*H, heads, W, head_dim]
        q_h, k_h, v_h = qkv_h.unbind(0)
        
        # 计算行方向注意力
        attn_h = (q_h @ k_h.transpose(-2, -1)) * self.scale  # [B*H, heads, W, W]
        attn_h = attn_h.softmax(dim=-1)
        
        # 应用行注意力
        x_h = (attn_h @ v_h).transpose(1, 2).reshape(B * H, W, C)
        x_h = self.proj_h(x_h).reshape(B, H, W, C)
        
        # 然后计算列方向注意力 (在H维度)
        # 将x_h转为[B*W, H, C]格式
        x_v = x_h.permute(0, 2, 1, 3).reshape(B * W, H, C)
        
        # 生成列方向qkv
        qkv_v = self.qkv_v(x_v)
        qkv_v = qkv_v.reshape(B * W, H, 3, self.num_heads, C // self.num_heads)
        qkv_v = qkv_v.permute(2, 0, 3, 1, 4)  # [3, B*W, heads, H, head_dim]
        q_v, k_v, v_v = qkv_v.unbind(0)
        
        # 计算列方向注意力
        attn_v = (q_v @ k_v.transpose(-2, -1)) * self.scale  # [B*W, heads, H, H]
        attn_v = attn_v.softmax(dim=-1)
        
        # 应用列注意力
        x_v = (attn_v @ v_v).transpose(1, 2).reshape(B * W, H, C)
        x_v = self.proj_v(x_v).reshape(B, W, H, C).permute(0, 2, 1, 3)
        
        # 转回[B, C, H, W]格式
        x = x_v.permute(0, 3, 1, 2)
        
        return x


class MixedAttentionProcessor(nn.Module):
    """
    混合注意力处理器，根据层级选择使用窗口注意力或轴分解注意力
    用于替换diffusers中的注意力处理器
    """
    def __init__(self, hidden_size, num_attention_heads=8, attention_type="window", window_size=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_type = attention_type
        
        if attention_type == "window":
            self.attention = WindowAttention(
                dim=hidden_size,
                num_heads=num_attention_heads,
                window_size=window_size
            )
        elif attention_type == "axial":
            self.attention = AxialAttention(
                dim=hidden_size,
                num_heads=num_attention_heads
            )
        else:
            raise ValueError(f"注意力类型 {attention_type} 不支持。请选择 'window' 或 'axial'")
    
    def __call__(self, attn_output, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        
        # 将hidden_states从[b, seq_len, dim]转换为[b, dim, h, w]格式
        # 假设序列长度是h*w的形式
        h = w = int(math.sqrt(sequence_length))
        assert h * w == sequence_length, "序列长度必须是平方数"
        
        x = hidden_states.reshape(batch_size, h, w, self.hidden_size).permute(0, 3, 1, 2)
        
        # 应用自定义注意力
        x = self.attention(x)
        
        # 转回原始形状
        hidden_states = x.permute(0, 2, 3, 1).reshape(batch_size, sequence_length, self.hidden_size)
        
        return hidden_states, None


def get_attention_processor_for_block(block_idx, total_blocks, hidden_size, num_heads, window_size=8):
    """
    根据块索引决定使用哪种注意力类型
    前半部分块使用窗口注意力，后半部分使用轴分解注意力
    """
    if block_idx < total_blocks // 2:
        return MixedAttentionProcessor(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            attention_type="window",
            window_size=window_size
        )
    else:
        return MixedAttentionProcessor(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            attention_type="axial"
        ) 