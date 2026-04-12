"""
Full VQ-VAE model for 3D mesh tokenization.

Combines the GNN encoder, vector quantizer, and MLP decoder
into a single end-to-end trainable module.
Includes vertex consistency loss to keep shared vertices aligned.
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

    The vertex consistency loss ensures that when two triangles share
    a vertex, they predict the same 3D coordinates for it.
    """

    def __init__(
        self,
        in_channels: int = 9,
        latent_dim: int = 128,
        num_embeddings: int = 512,
        commitment_cost: float = 1.0,
        diversity_weight: float = 0.1,
        consistency_weight: float = 1.0,
    ):
        super().__init__()

        self.consistency_weight = consistency_weight

        self.encoder = GraphEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
            diversity_weight=diversity_weight,
        )
        self.decoder = MLPDecoder(latent_dim=latent_dim, out_channels=in_channels)

    def forward(
        self, data: Data
    ) -> dict[str, torch.Tensor]:
        z_e = self.encoder(data.x, data.edge_index)
        z_q, vq_losses, indices = self.quantizer(z_e)
        recon = self.decoder(z_q)

        recon_loss = F.l1_loss(recon, data.y)

        consistency_loss = self._vertex_consistency_loss(recon, data)

        total_loss = (
            recon_loss
            + vq_losses["total_vq_loss"]
            + self.consistency_weight * consistency_loss
        )

        return {
            "recon": recon,
            "recon_loss": recon_loss,
            "vq_loss": vq_losses["vq_loss"],
            "diversity_loss": vq_losses["diversity_loss"],
            "consistency_loss": consistency_loss,
            "total_loss": total_loss,
            "indices": indices,
            "z_e": z_e,
            "z_q": z_q,
        }

    def _vertex_consistency_loss(self, recon: torch.Tensor, data: Data) -> torch.Tensor:
        """
        Penalize when two triangles sharing a vertex predict different
        coordinates for that vertex.

        recon is [N, 9] = N triangles × (v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z)
        sv_tri_a[k], sv_local_a[k] says: pair k, triangle A index, local vertex index (0/1/2)
        sv_tri_b[k], sv_local_b[k] says: pair k, triangle B index, local vertex index (0/1/2)
        These two local vertices are the SAME global vertex → should have same coords.
        """
        if not hasattr(data, "sv_tri_a") or len(data.sv_tri_a) == 0:
            return torch.tensor(0.0, device=recon.device)

        recon_3d = recon.view(-1, 3, 3)

        coords_a = recon_3d[data.sv_tri_a, data.sv_local_a]
        coords_b = recon_3d[data.sv_tri_b, data.sv_local_b]

        return F.mse_loss(coords_a, coords_b)

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
