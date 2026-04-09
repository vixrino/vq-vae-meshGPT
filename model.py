"""
Full VQ-VAE model for 3D mesh tokenization.

Combines the GNN encoder, vector quantizer, and MLP decoder
into a single end-to-end trainable module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from decoder import MLPDecoder
from encoder import GraphEncoder
from vector_quantizer import VectorQuantizer


class MeshVQVAE(nn.Module):
    """
    VQ-VAE for triangle meshes.

    Pipeline:
        1. Encoder: triangle graph → per-triangle embeddings z_e
        2. VQ: z_e → discrete codebook indices → z_q
        3. Decoder: z_q → reconstructed 9D coordinates
    """

    def __init__(
        self,
        in_channels: int = 9,
        latent_dim: int = 256,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
    ):
        super().__init__()

        self.encoder = GraphEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
        )
        self.decoder = MLPDecoder(latent_dim=latent_dim, out_channels=in_channels)

    def forward(
        self, data: Data
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            data: PyG Data with x [N, 9], edge_index [2, E], y [N, 9]

        Returns:
            Dictionary with:
                - recon: Reconstructed coordinates [N, 9]
                - vq_loss: VQ objective scalar
                - recon_loss: L1 reconstruction loss scalar
                - total_loss: vq_loss + recon_loss
                - indices: Codebook indices [N]
                - z_e: Pre-quantization embeddings [N, D]
                - z_q: Post-quantization embeddings [N, D]
        """
        z_e = self.encoder(data.x, data.edge_index)
        z_q, vq_loss, indices = self.quantizer(z_e)
        recon = self.decoder(z_q)

        recon_loss = F.l1_loss(recon, data.y)

        total_loss = recon_loss + vq_loss

        return {
            "recon": recon,
            "vq_loss": vq_loss,
            "recon_loss": recon_loss,
            "total_loss": total_loss,
            "indices": indices,
            "z_e": z_e,
            "z_q": z_q,
        }

    def encode(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode mesh to codebook indices."""
        z_e = self.encoder(data.x, data.edge_index)
        _, _, indices = self.quantizer(z_e)
        return indices, z_e

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode triangle coordinates from codebook indices."""
        z_q = self.quantizer.embedding(indices)
        return self.decoder(z_q)

    def count_parameters(self) -> dict[str, int]:
        """Count trainable parameters per component."""
        counts = {}
        for name, module in [
            ("encoder", self.encoder),
            ("quantizer", self.quantizer),
            ("decoder", self.decoder),
        ]:
            counts[name] = sum(p.numel() for p in module.parameters() if p.requires_grad)
        counts["total"] = sum(counts.values())
        return counts
