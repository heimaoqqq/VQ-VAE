import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from scipy.stats import wasserstein_distance
import torch.nn.functional as F
from .custom_vqgan import CustomVQGAN

class LatentSpaceAnalyzer:
    """
    VQ-VAE潜在空间分析工具，专注于评估潜在空间的质量和特性，
    特别是针对下游扩散模型的应用。
    
    该类提供了一系列方法来分析潜在空间的结构、分布和连续性，
    这些特性对于在潜在空间中训练扩散模型至关重要。
    
    使用方法:
    1. 初始化分析器，提供训练好的VQ-VAE模型
    2. 使用各种分析方法评估潜在空间
    3. 可视化结果以了解潜在空间的质量
    """
    
    def __init__(self, model: CustomVQGAN, device: str = 'cuda'):
        """
        初始化潜在空间分析器
        
        参数:
            model: 训练好的VQ-VAE模型
            device: 运行设备 ('cuda' 或 'cpu')
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        
        # 获取码本信息
        self.num_embeddings = model.quantize.num_embeddings
        self.embedding_dim = model.quantize.embedding_dim
        
        print(f"初始化潜在空间分析器")
        print(f"码本大小: {self.num_embeddings}")
        print(f"嵌入维度: {self.embedding_dim}")
    
    @torch.no_grad()
    def collect_latent_representations(self, dataloader, max_samples=10000):
        """
        收集数据集的潜在表示
        
        参数:
            dataloader: 数据加载器
            max_samples: 最大样本数
            
        返回:
            latents: 潜在向量 (N, C)
            indices: 量化索引 (N)
            original_images: 原始图像 (N, C, H, W)
        """
        print("收集潜在表示...")
        latents = []
        indices_list = []
        original_images = []
        sample_count = 0
        
        for images, _ in tqdm(dataloader, desc="收集潜在表示"):
            if sample_count >= max_samples:
                break
                
            batch_size = images.shape[0]
            if sample_count + batch_size > max_samples:
                # 只取需要的样本数
                images = images[:max_samples - sample_count]
                
            images = images.to(self.device)
            original_images.append(images.cpu())
            
            # 编码图像
            z = self.model.encode(images)
            
            # 量化
            _, _, quantize_info = self.model.quantize(z)
            indices = quantize_info[2]  # 获取量化索引
            
            # 将编码结果添加到列表中
            latents.append(z.reshape(z.shape[0], z.shape[1], -1).permute(0, 2, 1).reshape(-1, z.shape[1]).cpu())
            indices_list.append(indices.flatten().cpu())
            
            sample_count += images.shape[0]
        
        # 合并所有批次的结果
        latents = torch.cat(latents, dim=0)
        indices = torch.cat(indices_list, dim=0)
        original_images = torch.cat(original_images, dim=0)
        
        print(f"收集了 {latents.shape[0]} 个潜在向量")
        return latents, indices, original_images
    
    def analyze_latent_distribution(self, latents, plot=True):
        """
        分析潜在空间的分布特性
        
        参数:
            latents: 潜在向量 (N, C)
            plot: 是否绘制分布图
            
        返回:
            stats: 分布统计信息字典
        """
        print("分析潜在空间分布...")
        
        # 计算基本统计量
        mean = torch.mean(latents, dim=0).numpy()
        std = torch.std(latents, dim=0).numpy()
        min_val = torch.min(latents, dim=0)[0].numpy()
        max_val = torch.max(latents, dim=0)[0].numpy()
        
        # 计算通道间相关性
        corr_matrix = np.corrcoef(latents.numpy().T)
        
        # 计算与标准正态分布的Wasserstein距离
        normal_samples = np.random.normal(0, 1, size=latents.shape)
        w_distances = []
        for i in range(min(10, latents.shape[1])):  # 只计算前10个维度以节省时间
            w_distances.append(wasserstein_distance(latents[:, i].numpy(), normal_samples[:, i]))
        avg_w_distance = np.mean(w_distances)
        
        stats = {
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val,
            "avg_wasserstein_distance": avg_w_distance
        }
        
        if plot:
            # 绘制分布图
            plt.figure(figsize=(15, 10))
            
            # 1. 均值和标准差
            plt.subplot(2, 2, 1)
            plt.errorbar(range(len(mean[:20])), mean[:20], yerr=std[:20], fmt='o')
            plt.title('前20个维度的均值和标准差')
            plt.xlabel('维度')
            plt.ylabel('值')
            
            # 2. 直方图
            plt.subplot(2, 2, 2)
            plt.hist(latents[:, 0].numpy(), bins=50, alpha=0.5, label='维度0')
            plt.hist(latents[:, 1].numpy(), bins=50, alpha=0.5, label='维度1')
            plt.hist(latents[:, 2].numpy(), bins=50, alpha=0.5, label='维度2')
            plt.title('前3个维度的分布直方图')
            plt.xlabel('值')
            plt.ylabel('频率')
            plt.legend()
            
            # 3. 相关性热图
            plt.subplot(2, 2, 3)
            sns.heatmap(corr_matrix[:20, :20], cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('前20个维度的相关性矩阵')
            
            # 4. 与正态分布的Q-Q图
            plt.subplot(2, 2, 4)
            from scipy import stats
            stats.probplot(latents[:, 0].numpy(), dist="norm", plot=plt)
            plt.title('维度0的Q-Q图 (vs 正态分布)')
            
            plt.tight_layout()
            plt.show()
            
            print(f"平均Wasserstein距离 (vs 正态分布): {avg_w_distance:.4f}")
            print(f"潜在向量均值范围: [{np.min(mean):.4f}, {np.max(mean):.4f}]")
            print(f"潜在向量标准差范围: [{np.min(std):.4f}, {np.max(std):.4f}]")
        
        return stats
    
    def visualize_latent_space(self, latents, indices, method='tsne', perplexity=30):
        """
        可视化潜在空间
        
        参数:
            latents: 潜在向量 (N, C)
            indices: 量化索引 (N)
            method: 降维方法 ('tsne' 或 'pca')
            perplexity: t-SNE的困惑度参数
        """
        print(f"使用{method}可视化潜在空间...")
        
        # 随机采样以加速计算（如果样本数太多）
        max_points = 5000
        if latents.shape[0] > max_points:
            idx = np.random.choice(latents.shape[0], max_points, replace=False)
            latents_sample = latents[idx]
            indices_sample = indices[idx]
        else:
            latents_sample = latents
            indices_sample = indices
        
        # 降维
        if method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        else:
            reducer = PCA(n_components=2, random_state=42)
            
        embedded = reducer.fit_transform(latents_sample.numpy())
        
        # 可视化
        plt.figure(figsize=(12, 10))
        
        # 1. 按量化索引着色
        plt.subplot(1, 1, 1)
        scatter = plt.scatter(embedded[:, 0], embedded[:, 1], c=indices_sample.numpy(), 
                   cmap='tab20', alpha=0.5, s=5)
        plt.colorbar(scatter, label='量化索引')
        plt.title(f'{method.upper()}降维后的潜在空间 (按量化索引着色)')
        plt.xlabel('维度1')
        plt.ylabel('维度2')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_codebook_structure(self):
        """
        分析码本结构
        
        返回:
            stats: 码本统计信息字典
        """
        print("分析码本结构...")
        
        # 获取码本嵌入
        codebook = self.model.quantize.embedding.weight.detach().cpu()
        
        # 计算码本向量之间的距离矩阵
        distances = torch.cdist(codebook, codebook)
        
        # 计算每个码本向量到最近邻的距离
        min_distances = []
        for i in range(self.num_embeddings):
            # 排除自身
            d = distances[i]
            d[i] = float('inf')
            min_distances.append(torch.min(d).item())
        
        min_distances = torch.tensor(min_distances)
        
        # 计算统计量
        stats = {
            "mean_min_distance": torch.mean(min_distances).item(),
            "std_min_distance": torch.std(min_distances).item(),
            "min_min_distance": torch.min(min_distances).item(),
            "max_min_distance": torch.max(min_distances).item(),
        }
        
        # 可视化码本结构
        plt.figure(figsize=(15, 5))
        
        # 1. 最近邻距离分布
        plt.subplot(1, 2, 1)
        plt.hist(min_distances.numpy(), bins=30)
        plt.title('码本向量到最近邻的距离分布')
        plt.xlabel('距离')
        plt.ylabel('频率')
        
        # 2. 码本向量的PCA可视化
        plt.subplot(1, 2, 2)
        pca = PCA(n_components=2)
        codebook_2d = pca.fit_transform(codebook.numpy())
        plt.scatter(codebook_2d[:, 0], codebook_2d[:, 1], alpha=0.7)
        plt.title('码本向量的PCA可视化')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        
        plt.tight_layout()
        plt.show()
        
        print(f"码本向量到最近邻的平均距离: {stats['mean_min_distance']:.4f}")
        print(f"码本向量到最近邻的最小距离: {stats['min_min_distance']:.4f}")
        
        return stats
    
    def analyze_latent_continuity(self, dataloader, num_pairs=5, num_steps=10):
        """
        分析潜在空间的连续性
        
        参数:
            dataloader: 数据加载器
            num_pairs: 图像对数量
            num_steps: 插值步数
        """
        print("分析潜在空间连续性...")
        
        # 获取一批图像
        images, _ = next(iter(dataloader))
        if len(images) < num_pairs * 2:
            print(f"警告: 数据加载器批次大小 ({len(images)}) 小于所需的图像数量 ({num_pairs * 2})。")
            num_pairs = len(images) // 2
            
        images = images.to(self.device)
        
        # 计算插值质量指标
        smoothness_scores = []
        coherence_scores = []
        
        fig, axes = plt.subplots(num_pairs, num_steps + 2, figsize=(num_steps * 2, num_pairs * 2))
        
        for i in range(num_pairs):
            img1 = images[i*2].unsqueeze(0)
            img2 = images[i*2 + 1].unsqueeze(0)

            # 编码到连续的潜在空间
            z1 = self.model.encode(img1)
            z2 = self.model.encode(img2)

            # 存储插值后的图像和潜在向量
            interp_imgs = []
            interp_zs = []
            quant_zs = []
            
            # 添加起始图像
            interp_imgs.append(img1.cpu())

            for j in range(num_steps):
                # 线性插值
                alpha = j / (num_steps - 1)
                z_interp = torch.lerp(z1, z2, alpha)
                interp_zs.append(z_interp.cpu())
                
                # 量化和解码
                quant_interp, _, _ = self.model.quantize(z_interp)
                quant_zs.append(quant_interp.cpu())
                decoded_interp = self.model.decode(quant_interp)
                interp_imgs.append(decoded_interp.cpu())

            # 添加结束图像
            interp_imgs.append(img2.cpu())
            
            # 计算平滑度得分 (连续潜在向量之间的平均欧氏距离)
            interp_zs = torch.cat(interp_zs, dim=0)
            smoothness = 0
            for j in range(len(interp_zs) - 1):
                smoothness += F.mse_loss(interp_zs[j], interp_zs[j+1]).item()
            smoothness /= (len(interp_zs) - 1)
            smoothness_scores.append(smoothness)
            
            # 计算一致性得分 (量化前后的平均欧氏距离)
            quant_zs = torch.cat(quant_zs, dim=0)
            coherence = F.mse_loss(interp_zs, quant_zs).item()
            coherence_scores.append(coherence)
            
            # 可视化这一对的插值
            for k, img in enumerate(interp_imgs):
                ax = axes[i, k] if num_pairs > 1 else axes[k]
                img_np = ((img[0] + 1.0) / 2.0).permute(1, 2, 0).numpy()
                ax.imshow(np.clip(img_np, 0, 1))
                ax.axis('off')
        
        fig.suptitle("潜在空间插值 (左: 起点, 右: 终点)", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
        # 报告平滑度和一致性得分
        avg_smoothness = np.mean(smoothness_scores)
        avg_coherence = np.mean(coherence_scores)
        
        print(f"平均平滑度得分 (越低越好): {avg_smoothness:.6f}")
        print(f"平均一致性得分 (越低越好): {avg_coherence:.6f}")
        
        return {
            "smoothness_scores": smoothness_scores,
            "coherence_scores": coherence_scores,
            "avg_smoothness": avg_smoothness,
            "avg_coherence": avg_coherence
        }
    
    def analyze_reconstruction_error_distribution(self, dataloader, num_samples=100):
        """
        分析重建误差的分布
        
        参数:
            dataloader: 数据加载器
            num_samples: 样本数量
        """
        print("分析重建误差分布...")
        
        pixel_errors = []
        feature_errors = []
        sample_count = 0
        
        for images, _ in tqdm(dataloader, desc="计算重建误差"):
            if sample_count >= num_samples:
                break
                
            batch_size = images.shape[0]
            if sample_count + batch_size > num_samples:
                # 只取需要的样本数
                images = images[:num_samples - sample_count]
                
            images = images.to(self.device)
            
            # 获取重建
            model_output = self.model(images, return_dict=True)
            reconstructed = model_output["decoded_imgs"]
            
            # 计算像素级误差
            pixel_error = F.mse_loss(images, reconstructed, reduction='none')
            pixel_errors.append(pixel_error.cpu())
            
            # 计算特征级误差 (使用编码器的中间层)
            # 这里我们简单地使用编码后的向量
            z_orig = self.model.encode(images)
            z_recon = self.model.encode(reconstructed)
            feature_error = F.mse_loss(z_orig, z_recon, reduction='none')
            feature_errors.append(feature_error.cpu())
            
            sample_count += images.shape[0]
        
        # 合并所有批次的结果
        pixel_errors = torch.cat(pixel_errors, dim=0)
        feature_errors = torch.cat(feature_errors, dim=0)
        
        # 计算每张图像的平均误差
        pixel_errors_mean = pixel_errors.mean(dim=[1, 2, 3]).numpy()
        feature_errors_mean = feature_errors.mean(dim=[1, 2, 3]).numpy()
        
        # 可视化误差分布
        plt.figure(figsize=(15, 5))
        
        # 1. 像素级误差分布
        plt.subplot(1, 2, 1)
        plt.hist(pixel_errors_mean, bins=30)
        plt.title('像素级重建误差分布')
        plt.xlabel('MSE')
        plt.ylabel('频率')
        
        # 2. 特征级误差分布
        plt.subplot(1, 2, 2)
        plt.hist(feature_errors_mean, bins=30)
        plt.title('特征级重建误差分布')
        plt.xlabel('MSE')
        plt.ylabel('频率')
        
        plt.tight_layout()
        plt.show()
        
        # 报告统计量
        print(f"像素级重建误差: 均值={np.mean(pixel_errors_mean):.6f}, 标准差={np.std(pixel_errors_mean):.6f}")
        print(f"特征级重建误差: 均值={np.mean(feature_errors_mean):.6f}, 标准差={np.std(feature_errors_mean):.6f}")
        
        return {
            "pixel_errors": pixel_errors_mean,
            "feature_errors": feature_errors_mean,
            "pixel_error_mean": np.mean(pixel_errors_mean),
            "feature_error_mean": np.mean(feature_errors_mean)
        }
    
    def analyze_for_diffusion(self, dataloader, max_samples=1000):
        """
        综合分析潜在空间是否适合扩散模型
        
        参数:
            dataloader: 数据加载器
            max_samples: 最大样本数
            
        返回:
            report: 分析报告字典
        """
        print("\n========== VQ-VAE潜在空间扩散适应性分析 ==========")
        
        # 1. 收集潜在表示
        latents, indices, original_images = self.collect_latent_representations(dataloader, max_samples)
        
        # 2. 分析潜在分布
        dist_stats = self.analyze_latent_distribution(latents)
        
        # 3. 可视化潜在空间
        self.visualize_latent_space(latents, indices)
        
        # 4. 分析码本结构
        codebook_stats = self.analyze_codebook_structure()
        
        # 5. 分析潜在连续性
        continuity_stats = self.analyze_latent_continuity(dataloader)
        
        # 6. 分析重建误差
        error_stats = self.analyze_reconstruction_error_distribution(dataloader)
        
        # 7. 计算扩散适应性得分
        # 理想情况: 
        # - 潜在分布接近正态分布 (低Wasserstein距离)
        # - 码本向量分布均匀 (高平均最小距离)
        # - 高连续性 (低平滑度和一致性得分)
        # - 低重建误差
        
        w_dist_score = max(0, 1 - dist_stats["avg_wasserstein_distance"] / 2)
        codebook_score = min(1, codebook_stats["mean_min_distance"] * 5)
        smoothness_score = max(0, 1 - continuity_stats["avg_smoothness"] * 10)
        coherence_score = max(0, 1 - continuity_stats["avg_coherence"] * 10)
        recon_score = max(0, 1 - error_stats["pixel_error_mean"] * 10)
        
        # 综合得分 (加权平均)
        diffusion_readiness = (
            w_dist_score * 0.2 +
            codebook_score * 0.2 +
            smoothness_score * 0.2 +
            coherence_score * 0.2 +
            recon_score * 0.2
        )
        
        # 生成报告
        report = {
            "distribution_score": w_dist_score,
            "codebook_score": codebook_score,
            "smoothness_score": smoothness_score,
            "coherence_score": coherence_score,
            "reconstruction_score": recon_score,
            "diffusion_readiness": diffusion_readiness
        }
        
        print("\n========== 扩散适应性评分 ==========")
        print(f"分布正态性得分: {w_dist_score:.2f}/1.00")
        print(f"码本结构得分: {codebook_score:.2f}/1.00")
        print(f"潜在平滑度得分: {smoothness_score:.2f}/1.00")
        print(f"潜在一致性得分: {coherence_score:.2f}/1.00")
        print(f"重建质量得分: {recon_score:.2f}/1.00")
        print(f"扩散适应性总得分: {diffusion_readiness:.2f}/1.00")
        
        if diffusion_readiness < 0.5:
            print("\n⚠️ 警告: 潜在空间可能不适合扩散模型。考虑以下改进:")
            if w_dist_score < 0.6:
                print("- 改进潜在分布: 尝试添加KL散度损失或正则化")
            if codebook_score < 0.6:
                print("- 改进码本结构: 增加码本大小或调整承诺损失权重")
            if smoothness_score < 0.6 or coherence_score < 0.6:
                print("- 改进潜在连续性: 尝试添加感知损失或对抗损失")
            if recon_score < 0.6:
                print("- 改进重建质量: 增加模型容量或训练更长时间")
        else:
            print("\n✅ 潜在空间可能适合扩散模型。")
        
        return report


# 使用示例
"""
from vqvae.latent_space_analyzer import LatentSpaceAnalyzer

# 加载模型
model = CustomVQGAN(**model_config).to(device)
model.load_state_dict(torch.load('path/to/model.pt')['generator_state_dict'])

# 创建分析器
analyzer = LatentSpaceAnalyzer(model, device='cuda')

# 运行综合分析
analyzer.analyze_for_diffusion(val_dataloader)
""" 