"""
Full VQ-VAE model for 3D mesh tokenization.

Supports a warmup phase where the VQ is bypassed,
allowing the encoder/decoder to learn as a plain autoencoder first.
After warmup, the codebook is initialized from encoder outputs
and VQ is activated.

Uses GNN decoder (mirror of encoder) so neighboring triangles
coordinate their vertex predictions via message passing.

Quantizer is a Residual VQ (RVQ) with num_vq_levels levels.
Setting num_vq_levels=1 recovers plain VQ.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from decoder import GraphDecoder
from encoder import GraphEncoder
from residual_vector_quantizer import ResidualVectorQuantizer


class MeshVQVAE(nn.Module):

    def __init__(
        self,
        in_channels: int = 9,
        latent_dim: int = 128,
        num_embeddings: int = 512,
        num_vq_levels: int = 3,
        commitment_cost: float = 0.25,
        warmup_epochs: int = 30,
    ):
        super().__init__()

        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

        self.encoder = GraphEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.quantizer = ResidualVectorQuantizer(
            num_levels=num_vq_levels,
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
        )
        self.decoder = GraphDecoder(latent_dim=latent_dim, out_channels=in_channels)

    @property
    def in_warmup(self) -> bool:
        return self.current_epoch < self.warmup_epochs

    def _cyclic_permutation_loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Permutation-invariant L1 loss over the 3 cyclic orderings of triangle vertices.
        """
        r = recon.view(-1, 3, 3)
        t = target.view(-1, 3, 3)

        perm0 = F.l1_loss(r, t, reduction="none").mean(dim=(1, 2))
        perm1 = F.l1_loss(r, t[:, [1, 2, 0], :], reduction="none").mean(dim=(1, 2))
        perm2 = F.l1_loss(r, t[:, [2, 0, 1], :], reduction="none").mean(dim=(1, 2))

        min_loss = torch.stack([perm0, perm1, perm2], dim=1).min(dim=1).values
        return min_loss.mean()

    def _vertex_consistency_loss(self, recon: torch.Tensor, data: Data) -> torch.Tensor:
        """
        Penalize when two triangles sharing a vertex predict different coords for it.
        Uses precomputed shared vertex pair indices from the graph.
        """
        if not hasattr(data, "sv_tri_a") or len(data.sv_tri_a) == 0:
            return torch.tensor(0.0, device=recon.device)

        recon_3d = recon.view(-1, 3, 3)
        coords_a = recon_3d[data.sv_tri_a, data.sv_local_a]
        coords_b = recon_3d[data.sv_tri_b, data.sv_local_b]
        return F.mse_loss(coords_a, coords_b)

    def forward(self, data: Data) -> dict[str, torch.Tensor]:
        z_e = self.encoder(data.x, data.edge_index)

        bypass = self.in_warmup
        z_q, vq_losses, indices = self.quantizer(z_e, bypass=bypass)

        recon = self.decoder(z_q, data.edge_index)

        recon_loss = self._cyclic_permutation_loss(recon, data.y)
        cons_loss = self._vertex_consistency_loss(recon, data)
        total_loss = recon_loss + vq_losses["vq_loss"] + 0.3 * cons_loss

        return {
            "recon": recon,
            "recon_loss": recon_loss,
            "vq_loss": vq_losses["vq_loss"],
            "cons_loss": cons_loss,
            "total_loss": total_loss,
            "indices": indices,
            "z_e": z_e,
            "z_q": z_q,
        }

    def encode(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        z_e = self.encoder(data.x, data.edge_index)
        _, _, indices = self.quantizer(z_e)
        return indices, z_e

    def decode_from_indices(self, indices: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        z_q = self.quantizer.lookup(indices)
        return self.decoder(z_q, edge_index)

    def count_parameters(self) -> dict[str, int]:
        counts = {}
        for name, module in [
            ("encoder", self.encoder),
            ("quantizer", self.quantizer),
            ("decoder", self.decoder),
        ]:
            counts[name] = sum(p.numel() for p in module.parameters() if p.requires_grad)
        counts["total"] = sum(counts.values())
        return counts
