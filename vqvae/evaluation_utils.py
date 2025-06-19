import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torchvision.utils as vutils
from .custom_vqgan import CustomVQGAN
import torch.nn.functional as F

def postprocess_for_display(tensors):
    """
    接收一个-1到1范围的张量，返回一个0到1范围的numpy数组。
    """
    tensors = torch.clamp(tensors, -1.0, 1.0) # 确保范围
    return ((tensors + 1.0) / 2.0).permute(0, 2, 3, 1).cpu().numpy()

class VQGAN_Evaluator:
    """
    一个用于评估 VQ-GAN 模型性能的封装类。
    专为在 Kaggle 或 Jupyter Notebooks 中使用而设计。

    使用说明:
    1. 确保您的 'vqvae' 文件夹已上传到 Kaggle 并添加到 sys.path。
       import sys
       sys.path.append('/kaggle/working/your_project_folder') # 替换为实际路径

    2. 创建一个数据加载器 (dataloader)。
    3. 按如下方式实例化并使用该类：
       evaluator = VQGAN_Evaluator(
           model_path='/path/to/your/best_model.pt',
           model_config={...}, # 提供与训练时完全相同的模型配置
           device='cuda' if torch.cuda.is_available() else 'cpu'
       )
       evaluator.plot_reconstructions(dataloader)
       evaluator.calculate_metrics(dataloader)
       evaluator.analyze_codebook_usage(dataloader)
       evaluator.plot_latent_interpolations(dataloader)
    """
    def __init__(self, model_path: str, model_config: dict, device: str = 'cuda'):
        """
        初始化评估器。

        参数:
            model_path (str): 已训练模型检查点 (.pt) 的路径。
            model_config (dict): 用于实例化模型的配置字典。
            device (str): 运行模型的设备 ('cuda' 或 'cpu')。
        """
        self.device = torch.device(device)
        self.model = CustomVQGAN(**model_config).to(self.device)
        
        # 加载检查点。我们的训练器保存的是一个字典。
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 智能地寻找正确的 state_dict
        state_dict_key = None
        if 'generator_state_dict' in checkpoint:
            state_dict_key = 'generator_state_dict'
        elif 'vqgan_state_dict' in checkpoint:
            state_dict_key = 'vqgan_state_dict'
        elif 'model_state_dict' in checkpoint:
            state_dict_key = 'model_state_dict'

        if state_dict_key:
            self.model.load_state_dict(checkpoint[state_dict_key])
        else:
            # 如果上面都找不到，就尝试直接加载整个文件，作为最后的兼容手段
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        print(f"模型已从 {model_path} 加载并设置为评估模式。")

    @torch.no_grad()
    def plot_reconstructions(self, dataloader, num_images: int = 8):
        """
        可视化原始图像及其重建版本。
        """
        print("正在生成重建图像对比图...")
        # 从数据加载器中获取一个批次
        real_images, _ = next(iter(dataloader))
        real_images = real_images.to(self.device)
        
        # 选择 N 张图像进行可视化
        real_images = real_images[:num_images]
        
        # 获取重建
        model_output = self.model(real_images, return_dict=True)
        reconstructed_images = model_output["decoded_imgs"]

        # 将原始图像和重建图像拼接在一起
        comparison = torch.cat([real_images.cpu(), reconstructed_images.cpu()])
        grid = vutils.make_grid(comparison, nrow=num_images, padding=2, normalize=True)
        
        # 显示图像
        plt.figure(figsize=(15, 4))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.title('Top Row: Original Images | Bottom Row: Reconstructed Images')
        plt.show()

    @torch.no_grad()
    def calculate_metrics(self, dataloader):
        """
        在整个数据集上计算并报告 PSNR 和 SSIM。
        """
        total_psnr = 0.0
        total_ssim = 0.0
        total_images = 0

        print("正在计算整个验证集的 PSNR 和 SSIM (这可能需要一些时间)...")
        for images, _ in tqdm(dataloader, desc="Metric Calculation"):
            images = images.to(self.device)
            
            model_output = self.model(images, return_dict=True)
            reconstructed = model_output["decoded_imgs"]

            # 转换图像以进行度量计算
            originals_np = postprocess_for_display(images)
            reconstructed_np = postprocess_for_display(reconstructed)

            for i in range(originals_np.shape[0]):
                # data_range 是原始未归一化图像的最大值减去最小值。
                # 由于我们的数据是[0,1]，所以是1.0。
                total_psnr += psnr(originals_np[i], reconstructed_np[i], data_range=1.0)
                total_ssim += ssim(originals_np[i], reconstructed_np[i], data_range=1.0, channel_axis=-1, win_size=7)

            total_images += images.size(0)

        avg_psnr = total_psnr / total_images
        avg_ssim = total_ssim / total_images

        print(f"\n--- 评估结果 ---")
        print(f"在 {total_images} 张图像上计算:")
        print(f"平均 PSNR: {avg_psnr:.4f} dB")
        print(f"平均 SSIM: {avg_ssim:.4f}")
        print("--------------------")

    @torch.no_grad()
    def analyze_codebook_usage(self, dataloader, plot: bool = True):
        """
        分析码本中向量的使用频率。
        """
        num_embeddings = self.model.quantize.num_embeddings
        codebook_counts = torch.zeros(num_embeddings, dtype=torch.long, device=self.device)

        print("正在分析码本利用率...")
        for images, _ in tqdm(dataloader, desc="Analyzing Codebook"):
            images = images.to(self.device)
            model_output = self.model(images, return_dict=True)
            indices = model_output["indices"].flatten()
            
            # 使用bincount高效地计算每个索引的出现次数
            # 强制转换为long类型以解决 "not implemented for 'Float'" 错误
            codebook_counts += torch.bincount(indices.long(), minlength=num_embeddings)

        used_codes = torch.sum(codebook_counts > 0).item()
        usage_percentage = (used_codes / num_embeddings) * 100

        print(f"\n--- 码本利用率分析 ---")
        print(f"码本大小: {num_embeddings}")
        print(f"使用的码本向量数量: {used_codes}")
        print(f"利用率: {usage_percentage:.2f}%")
        print("--------------------------")

        if plot:
            plt.figure(figsize=(15, 5))
            plt.bar(range(num_embeddings), codebook_counts.cpu().numpy())
            plt.title('Codebook Usage Frequency')
            plt.xlabel('Codebook Index')
            plt.ylabel('Frequency')
            plt.show()
            
        return usage_percentage, codebook_counts

    @torch.no_grad()
    def plot_latent_interpolations(self, dataloader, num_pairs: int = 4, num_steps: int = 8):
        """
        可视化潜在空间中的插值。
        """
        print("正在生成潜在空间插值图像...")
        # 从数据加载器中获取足够的图像
        images, _ = next(iter(dataloader))
        if len(images) < num_pairs * 2:
            print(f"警告: Dataloader批次大小 ({len(images)}) 小于进行插值所需的图像数 ({num_pairs * 2})。")
            return
            
        images = images.to(self.device)

        fig, axes = plt.subplots(num_pairs, num_steps + 2, figsize=(num_steps * 2, num_pairs * 2))
        
        for i in range(num_pairs):
            img1 = images[i*2].unsqueeze(0)
            img2 = images[i*2 + 1].unsqueeze(0)

            # 编码到连续的潜在空间
            z1 = self.model.encode(img1)
            z2 = self.model.encode(img2)

            # 存储插值后的图像
            interp_imgs = []
            
            # 添加起始图像
            interp_imgs.append(postprocess_for_display(img1)[0])

            for j in range(num_steps):
                # 线性插值
                alpha = j / (num_steps - 1)
                z_interp = torch.lerp(z1, z2, alpha)

                # 量化和解码
                quant_interp, _, _ = self.model.quantize(z_interp)
                decoded_interp = self.model.decode(quant_interp)
                interp_imgs.append(postprocess_for_display(decoded_interp)[0])

            # 添加结束图像
            interp_imgs.append(postprocess_for_display(img2)[0])
            
            # 可视化这一对的插值
            for k, img in enumerate(interp_imgs):
                ax = axes[i, k] if num_pairs > 1 else axes[k]
                ax.imshow(img)
                ax.axis('off')
        
        fig.suptitle("Latent Space Interpolations (Left: Start, Right: End)", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show() 