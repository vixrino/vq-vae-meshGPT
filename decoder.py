"""
MLP decoder that reconstructs triangle vertex coordinates
from quantized embeddings.

Each triangle is decoded independently — the graph context
is already captured by the GNN encoder + VQ codebook.
"""

import torch
import torch.nn as nn


class MLPDecoder(nn.Module):
    """
    Decodes quantized embeddings back to 9D triangle coordinates
    (3 vertices × 3 coordinates).

    Architecture:
        Linear(latent → 256) + ReLU + LayerNorm
        Linear(256 → 128)    + ReLU + LayerNorm
        Linear(128 → 9)      + Tanh (vertices are in [-1, 1])
    """

    def __init__(self, latent_dim: int = 128, out_channels: int = 9):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, out_channels),
            nn.Tanh(),
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_q: Quantized embeddings [N, latent_dim]

        Returns:
            coords: Reconstructed triangle coords [N, 9]
        """
        return self.network(z_q)
