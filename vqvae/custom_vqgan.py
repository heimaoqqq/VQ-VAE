import torch
import torch.nn as nn
from diffusers.models.autoencoders.vae import Encoder, Decoder
from diffusers.models.autoencoders.vq_model import VectorQuantizer
from dataclasses import dataclass
from typing import Tuple

class CustomVQGAN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels = [128, 256, 512],
        layers_per_block: int = 2,
        latent_channels: int = 256,
        num_vq_embeddings: int = 8192,
        vq_embed_dim: int = 256,
        use_ema: bool = True,
        ema_decay: float = 0.995,
        # Legacy parameter name for older diffusers versions
        n_e: int = None,
    ):
        super().__init__()
        
        # Handle legacy parameter name for num_vq_embeddings
        if n_e is not None:
            num_vq_embeddings = n_e

        # The dimension of the VQ embedding (codebook vectors) must match the latent channels from the encoder.
        # This was a source of error previously.
        if vq_embed_dim != latent_channels:
             raise ValueError(f"Latent channels from encoder ({latent_channels}) must match VQ embedding dimension ({vq_embed_dim}).")

        # Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=["DownEncoderBlock2D"] * len(block_out_channels),
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
        )

        # Vector Quantizer with EMA enabled
        # We are switching to legacy parameter names (`n_e`, `e_dim`) to match
        # the version of the diffusers library on your server. EMA is typically
        # enabled by just providing a decay value in these older versions.
        self.quantize = VectorQuantizer(
            n_e=num_vq_embeddings,
            e_dim=vq_embed_dim,
            beta=0.25,
            decay=ema_decay
        )

        # Decoder
        self.decoder = Decoder(
            in_channels=vq_embed_dim, # Decoder input must be vq_embed_dim
            out_channels=out_channels,
            up_block_types=["UpDecoderBlock2D"] * len(block_out_channels),
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
        )

        self.quant_conv = nn.Conv2d(latent_channels, vq_embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(vq_embed_dim, vq_embed_dim, 1)

        # To access config values in the trainer
        self.config = {
            "num_vq_embeddings": num_vq_embeddings,
            "vq_embed_dim": vq_embed_dim
        }


    def encode(self, x):
        h = self.encoder(x)
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
    num_vq_embeddings: int = 1024,
    vq_embed_dim: int = 256
    scaling_factor: float = 0.18215
    # Discriminator
    # ... existing code ... 