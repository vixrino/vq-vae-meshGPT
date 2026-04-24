# VQ-VAE for 3D Meshes (MeshGPT-style)

A from-scratch reimplementation of the Vector Quantized Variational Autoencoder (VQ-VAE) used in [MeshGPT](https://nihalsid.github.io/mesh-gpt/) for tokenizing 3D triangle meshes.

## Architecture

```
.obj mesh
    │
    ▼
┌──────────────┐
│  Triangle     │   Each triangle → node, shared edges → graph connections
│  Graph Builder│
└──────┬───────┘
       │  PyG Data (node features = 9D coords, edge_index)
       ▼
┌──────────────┐
│  Encoder      │   4 × SAGEConv layers
│  (GNN)        │   9D → 64 → 128 → 256 → 256
└──────┬───────┘
       │  z_e ∈ ℝ^(N_triangles × 256)
       ▼
┌──────────────┐
│  Residual VQ  │   3 levels × 512 codes (dim=128)
│  (RVQ)        │   z_q = q_L0 + q_L1 + q_L2 (sum of codebook vectors)
└──────┬───────┘
       │  z_q ∈ ℝ^(N_triangles × 128)
       ▼
┌──────────────┐
│  Decoder      │   4 × SAGEConv layers
│  (GNN)        │   128 → 256 → 256 → 128 → 9
└──────────────┘
       │
       ▼
  Reconstructed triangle coordinates
```

## Project Structure

```
vq-vae/
├── README.md
├── requirements.txt
├── .gitignore
├── mesh_dataset.py      # ShapeNet .obj loading + triangle graph construction
├── encoder.py           # 4-layer SAGEConv GNN encoder
├── vector_quantizer.py  # Vector Quantization with EMA updates
├── residual_vector_quantizer.py  # Residual VQ (N stacked VQ levels)
├── decoder.py           # GNN decoder for coordinate reconstruction
├── model.py             # Full VQ-VAE model
├── train.py             # Training loop
└── evaluate.py          # Evaluation and mesh visualization
```

## Setup

```bash
pip install -r requirements.txt
```

## Data

Place ShapeNet `.obj` files under `data/shapenet/`. The dataset loader expects directories of `.obj` files organized by category.

```bash
mkdir -p data/shapenet
# Copy or symlink your .obj files here
```

## Training

```bash
python train.py --data_dir data/shapenet --epochs 100 --batch_size 8 --lr 3e-4
```

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --data_dir data/shapenet
```

## Key Design Decisions

- **Graph construction**: Each triangle is a node. Two triangles sharing an edge are connected. Node features are the 9 flattened vertex coordinates (3 vertices × 3 coords).
- **Encoder**: GraphSAGE convolutions propagate information across neighboring triangles, building a contextual embedding per triangle.
- **Residual VQ**: Stack of 3 codebooks of 512 entries each (dim=128). Each level quantizes the residual of the previous one, and the final latent is the sum of codebook vectors. Rare triangles use the later levels for detail.
- **Decoder**: Mirror GraphSAGE GNN so neighboring triangles coordinate their vertex predictions via message passing.
- **Losses**: cyclic-permutation L1 reconstruction (handles triangle vertex ordering) + VQ commitment + vertex consistency loss (pulls coords of shared vertices together).

## Results

Trained on 425 low-poly meshes across 8 categories for 300 epochs
(30 epoch autoencoder warmup + 270 epoch RVQ phase).

![Per-category reconstructions](visualizations/per_category.png)

Stitched reconstruction L1 per category with 3-level Residual VQ
(20 meshes each, mean over all categories = **0.050**):

| Category   | Plain VQ (1 level) | **RVQ (3 levels)** | Speedup |
|------------|-------------------:|-------------------:|--------:|
| icosphere  |              0.034 |          **0.013** |    2.6× |
| torus      |              0.036 |          **0.014** |    2.5× |
| sphere     |              0.052 |          **0.022** |    2.3× |
| capsule    |              0.082 |          **0.033** |    2.4× |
| cylinder   |              0.100 |          **0.044** |    2.3× |
| cone       |              0.127 |          **0.062** |    2.1× |
| deformed   |              0.146 |          **0.084** |    1.7× |
| box        |              0.218 |          **0.125** |    1.7× |
| **Mean**   |          **0.099** |          **0.050** | **2.0×** |

Moving from plain VQ to 3-level Residual VQ halves the reconstruction
error across every category. The first level captures the coarse shape,
while levels 1 and 2 absorb the residuals - this is exactly the design
choice that helps the hardest irregular meshes (boxes: 0.218 → 0.125).

Codebook utilization (across all eval meshes): 96.9% / 96.1% / 85.2% at
levels 0/1/2 - all three levels are actively used.

See [RESULTS.md](RESULTS.md) for the full breakdown including raw L1 and
token-reuse statistics.

## References

- [MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers](https://nihalsid.github.io/mesh-gpt/)
- [Neural Discrete Representation Learning (VQ-VAE)](https://arxiv.org/abs/1711.00937)
- [MeshAnything V2](https://arxiv.org/abs/2408.02555)
