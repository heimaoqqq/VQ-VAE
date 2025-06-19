import torch
import torch.nn as nn
from diffusers.models.autoencoders.vae import Encoder, Decoder
from .ema_vector_quantizer import EMAVectorQuantizer
from dataclasses import dataclass
from typing import Tuple

class CustomVQGAN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: list = [64, 128, 256],
        layers_per_block: int = 2,
        latent_channels: int = 256,
        n_embed: int = 8192,
        embed_dim: int = 256,
        ema_decay: float = 0.999,
        commitment_loss_beta: float = 0.25,
    ):
        super().__init__()
        
        if embed_dim != latent_channels:
             raise ValueError(f"Latent channels from encoder ({latent_channels}) must match VQ embedding dimension ({embed_dim}).")

        # Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=["DownEncoderBlock2D"] * len(block_out_channels),
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
        )

        # Our new, reliable EMA Vector Quantizer
        self.quantize = EMAVectorQuantizer(
            num_embeddings=n_embed,
            embedding_dim=embed_dim,
            commitment_cost=commitment_loss_beta,
            decay=ema_decay
        )

        # Decoder
        self.decoder = Decoder(
            in_channels=embed_dim,
            out_channels=out_channels,
            up_block_types=["UpDecoderBlock2D"] * len(block_out_channels),
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
        )

        self.quant_conv = nn.Conv2d(latent_channels, embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, embed_dim, 1)

        # To access config values in the trainer
        self.config = {
            "num_vq_embeddings": n_embed,
            "vq_embed_dim": embed_dim,
            "ema_decay": ema_decay,
            "commitment_loss_beta": commitment_loss_beta
        }


    def encode(self, x):
        h = self.encoder(x)
        # The diffusers VAE encoder is designed to output 2*latent_channels
        # (mean and logvar). We only need the first half for VQ-GAN.
        if h.shape[1] == 2 * self.quant_conv.in_channels:
            h, _ = torch.chunk(h, 2, dim=1)
        h = self.quant_conv(h)
        return h

    def decode(self, h):
        h = self.post_quant_conv(h)
        h = self.decoder(h)
        return h

    def forward(self, x, return_dict=False):
        # 1. Encode
        h = self.encode(x)
        
        # 2. Quantize
        quant_states, vq_loss, perplexity_info = self.quantize(h)
        
        # The returned vq_loss is often a dictionary in newer diffusers
        if isinstance(vq_loss, dict):
            commitment_loss = vq_loss.get('commitment_loss', torch.tensor(0.0, device=x.device))
        else: # Or a simple tensor
            commitment_loss = vq_loss

        # Perplexity should be available in the third element of the tuple
        perplexity = perplexity_info[0] if perplexity_info is not None and len(perplexity_info) > 0 else torch.tensor(0.0)
        # We also need the indices for codebook usage calculation
        indices = perplexity_info[2] if perplexity_info is not None and len(perplexity_info) > 2 else torch.tensor([], dtype=torch.long)
        
        # 3. Decode
        reconstructed_x = self.decode(quant_states)

        if not return_dict:
            # This tuple format is for simple inference, not training
            return (reconstructed_x, commitment_loss, perplexity)

        # The dictionary format is more descriptive and is what the trainer will now use
        return {
            "decoded_imgs": reconstructed_x,
            "commitment_loss": commitment_loss,
            "indices": indices,
            "perplexity": perplexity,
        }

@dataclass
class VQGANConfig:
    """VQGAN 配置"""
    in_channels: int = 3
    out_channels: int = 3
    down_block_types: Tuple[str] = ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D")
    up_block_types: Tuple[str] = ("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D")
    block_out_channels: Tuple[int] = (128, 256, 256, 512)
    layers_per_block: int = 2
    latent_channels: int = 256
    num_vq_embeddings: int = 1024
    vq_embed_dim: int = 256
    scaling_factor: float = 0.18215
    ema_decay: float = 0.999
    commitment_loss_beta: float = 2.0
    # Discriminator
    # ... existing code ... 