"""
Graph Neural Network decoder for triangle mesh reconstruction.

Mirror architecture of the encoder: uses SAGEConv layers so each triangle's
reconstruction is informed by its neighbors, producing consistent shared
vertex predictions without post-processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphDecoder(nn.Module):
    """
    GNN decoder that reconstructs 9D triangle coordinates using
    neighborhood context via message passing.

    Architecture mirrors the encoder:
        latent_dim → 256 → 256 → 128 → 9
    """

    def __init__(self, latent_dim: int = 128, out_channels: int = 9):
        super().__init__()

        self.conv1 = SAGEConv(latent_dim, 256)
        self.conv2 = SAGEConv(256, 256)
        self.conv3 = SAGEConv(256, 128)
        self.conv4 = SAGEConv(128, out_channels)

        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(256)
        self.norm3 = nn.LayerNorm(128)

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.norm1(self.conv1(z, edge_index)))
        x = F.relu(self.norm2(self.conv2(x, edge_index)))
        x = F.relu(self.norm3(self.conv3(x, edge_index)))
        x = self.conv4(x, edge_index)
        return x
