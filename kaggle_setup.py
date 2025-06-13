#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kaggle环境配置脚本，用于安装所需依赖并准备环境
"""

import os
import sys

def setup_kaggle_environment():
    """配置Kaggle环境"""
    print("正在配置Kaggle环境...")
    
    # 安装指定版本依赖
    print("安装兼容版本依赖...")
    os.system("pip install huggingface_hub==0.16.4 diffusers==0.26.3 torch==2.0.0 torchvision==0.15.1 --force-reinstall")
    
    # 环境测试
    print("测试环境配置...")
    try:
        import torch
        import diffusers
        from huggingface_hub import hf_hub_download  # 测试是否能正确导入
        
        print(f"PyTorch版本: {torch.__version__}")
        print(f"Diffusers版本: {diffusers.__version__}")
        print("环境配置成功!")
        return True
    except ImportError as e:
        print(f"环境配置失败: {e}")
        return False

if __name__ == "__main__":
    # 检查是否在Kaggle环境
    is_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
    
    if is_kaggle:
        print("检测到Kaggle环境")
        success = setup_kaggle_environment()
        
        if success:
            print("\n您可以使用以下命令训练模型:")
            print("python train_vqvae.py --data_dir /kaggle/input/your-dataset --kaggle --fp16 --batch_size 16")
            print("python train_ldm.py --data_dir /kaggle/input/your-dataset --kaggle --vqvae_model_path vqvae_model")
        else:
            print("\n环境配置失败，请手动安装依赖")
    else:
        print("非Kaggle环境，无需特殊配置") 