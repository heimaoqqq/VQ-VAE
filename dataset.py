import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MicroDopplerDataset(Dataset):
    def __init__(self, data_dir, transform=None, image_size=256):
        """
        微多普勒时频图数据集
        
        参数:
            data_dir (str): 数据集根目录
            image_size (int): 图像尺寸
        """
        self.data_dir = data_dir
        self.transform = transform
        
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        # 获取所有子目录下的图像路径
        self.image_paths = []
        for subdir in os.listdir(data_dir):
            subdir_path = os.path.join(data_dir, subdir)
            if os.path.isdir(subdir_path):
                image_files = glob.glob(os.path.join(subdir_path, '*.jpg'))
                self.image_paths.extend(image_files)
        
        print(f"找到 {len(self.image_paths)} 张图像")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image

def get_dataloaders(data_dir, batch_size=32, image_size=256, train_ratio=0.9, val_ratio=0.1):
    """
    创建训练、验证和测试数据加载器
    
    参数:
        data_dir: 数据集目录
        batch_size: 批次大小
        image_size: 图像尺寸
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    
    返回:
        train_dataloader, val_dataloader, test_dataloader (测试集为None)
    """
    # 定义数据转换 - 移除了RandomHorizontalFlip()，只保留基本的预处理
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 创建数据集
    dataset = MicroDopplerDataset(data_dir, transform=transform, image_size=image_size)
    
    # 分割数据集为9:1 (训练集:验证集)
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=4, pin_memory=True)
    
    # 返回训练集和验证集，测试集为None
    return train_dataloader, val_dataloader, None

if __name__ == "__main__":
    # 测试数据集
    dataset = MicroDopplerDataset("dataset")
    print(f"数据集大小: {len(dataset)}")
    
    # 测试第一张图像
    sample = dataset[0]
    print(f"图像张量形状: {sample.shape}")
    
    # 测试数据加载器
    train_loader, val_loader, _ = get_dataloaders("dataset", batch_size=16)
    print(f"训练集: {len(train_loader.dataset)}个样本 (90%)")
    print(f"验证集: {len(val_loader.dataset)}个样本 (10%)")
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}") 