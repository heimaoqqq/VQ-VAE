import torch
import torch.nn as nn
from diffusers.models.autoencoders.vae import Encoder, Decoder
from diffusers.models.autoencoders.vq_model import VectorQuantizer

class CustomVQGAN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels = [64, 128, 256],
        layers_per_block: int = 2,
        latent_channels: int = 4,
        num_vq_embeddings: int = 8192,
    ):
        super().__init__()

        # Pass latent_channels to vq_embed_dim
        vq_embed_dim = latent_channels

        # Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=["DownEncoderBlock2D"] * len(block_out_channels),
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
        )

        # Vector Quantizer
        self.quantize = VectorQuantizer(
            n_e=num_vq_embeddings,
            vq_embed_dim=vq_embed_dim,
            beta=0.25,
        )

        # Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=["UpDecoderBlock2D"] * len(block_out_channels),
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
        )

        # The VQModel from diffusers has these two conv layers before and after quantization
        self.quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)

    def encode(self, x):
        h = self.encoder(x)
        
        # The underlying Encoder class from diffusers seems to be designed for AutoencoderKL
        # and doubles the output channels (for mean and logvar). We only need the "mean"
        # part for VQ-GAN, so we split the channels and take the first half.
        if h.shape[1] == 2 * self.quant_conv.in_channels:
             h, _ = h.chunk(2, dim=1)

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
        # The diffusers VQ returns a tuple: (quant_states, vq_loss, (perplexity, min_encoding_indices, min_encodings))
        quant_states, vq_loss, perplexity_info = self.quantize(h)
        # From diagnostics, we know perplexity is the first element of the inner tuple
        perplexity = perplexity_info[0]
        
        # 3. Decode
        reconstructed_x = self.decode(quant_states)

        # 4. Calculate losses
        reconstruction_loss = nn.functional.l1_loss(reconstructed_x, x)
        total_loss = reconstruction_loss + vq_loss

        if not return_dict:
            return (reconstructed_x, total_loss, perplexity)

        return {
            "sample": reconstructed_x,
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "vq_loss": vq_loss,
            "perplexity": perplexity
        } 