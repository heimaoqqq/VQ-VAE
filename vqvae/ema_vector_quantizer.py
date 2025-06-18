import torch
from torch import nn
import torch.nn.functional as F

class EMAVectorQuantizer(nn.Module):
    """
    A custom, self-contained Vector Quantizer with Exponential Moving Average (EMA) updates.
    This implementation is based on the standard VQ-VAE-2 and SoundStream papers to
    ensure stable codebook learning and prevent collapse.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # Initialize the codebook embeddings
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
        
        # Buffers for EMA updates, not part of the model's parameters
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', torch.zeros(num_embeddings, self.embedding_dim))

    def forward(self, inputs):
        # inputs: (B, C, H, W) -> (B, H, W, C)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances: (z - e)^2 = z^2 + e^2 - 2ze
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True) 
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )
            
        # Find closest encodings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and un-flatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)
        
        # Use EMA to update the embedding vectors if in training mode
        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                  (1 - self.decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing to avoid zero counts
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )
            
            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w = self.ema_w * self.decay + (1 - self.decay) * dw
            
            self.embedding.weight.data.copy_(self.ema_w / self.ema_cluster_size.unsqueeze(1))
            
        # Calculate commitment loss
        commitment_loss = self.commitment_cost * F.mse_loss(quantized.detach(), inputs)
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        # Reshape back to (B, C, H, W)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        # The trainer expects a tuple with commitment_loss and the indices
        # We wrap indices in a structure to be compatible
        # (quant_states, commitment_loss, (perplexity, indices, one_hot_encodings))
        
        # Perplexity calculation for monitoring (optional but good practice)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, commitment_loss, (perplexity, encoding_indices, encodings) 