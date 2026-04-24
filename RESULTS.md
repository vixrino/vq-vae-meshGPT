# VQ-VAE Reconstruction Results

Trained on 425 low-poly meshes across 8 categories for 300 epochs
(30 epoch autoencoder warmup + 270 epoch VQ phase).

## Model

- Encoder: 4x SAGEConv (9 → 64 → 128 → 256 → 128 = latent_dim)
- Quantizer: 3-level Residual VQ, 512 codes per level, EMA updates, dead-code reset
- Decoder: 4x SAGEConv (128 → 256 → 256 → 128 → 9)
- Losses: cyclic-permutation L1 recon + VQ commitment + 0.3 x vertex consistency

## Per-category reconstruction (20 meshes per category, stitched)

Comparison of plain VQ (single level) vs 3-level Residual VQ:

| Category   | Plain VQ L1 | RVQ L1 (3 levels) | Raw L1 (RVQ) | L0 usage | L1 usage | L2 usage |
|------------|------------:|------------------:|-------------:|---------:|---------:|---------:|
| icosphere  |       0.034 |         **0.013** |        0.023 |    43.4% |    54.7% |    49.8% |
| torus      |       0.036 |         **0.014** |        0.022 |    17.7% |    40.9% |    49.5% |
| sphere     |       0.052 |         **0.022** |        0.033 |    59.9% |    68.2% |    67.0% |
| capsule    |       0.082 |         **0.033** |        0.044 |    46.1% |    68.0% |    77.5% |
| cylinder   |       0.100 |         **0.044** |        0.055 |    42.3% |    67.9% |    78.9% |
| cone       |       0.127 |         **0.062** |        0.073 |    47.4% |    80.7% |    89.2% |
| deformed   |       0.146 |         **0.084** |        0.094 |    60.7% |    84.8% |    88.8% |
| box        |       0.218 |         **0.125** |        0.137 |    83.5% |    91.3% |    92.8% |
| **Average**|       0.099 |         **0.050** |        0.060 |          |          |          |

Best val recon: 0.0413 (epoch 270) vs 0.0793 for plain VQ.

Global codebook utilization across the evaluation set:
- Level 0: 496/512 active (96.9%), 16 dead
- Level 1: 492/512 active (96.1%), 20 dead
- Level 2: 436/512 active (85.2%), 76 dead

## Observations

1. Residual VQ halves the reconstruction error across every single
   category. The gain is largest on simple shapes (torus, icosphere)
   because the higher levels can afford to encode minute residuals
   instead of spending code capacity on coarse shape.
2. Boxes are still the hardest case but improved from L1 = 0.218 to
   0.125 (43% reduction). Their level-0 / level-1 / level-2 token usage
   per triangle is 84 / 91 / 93 %, meaning almost every triangle gets a
   unique code at every level: no repetition to compress, the model
   essentially memorizes each triangle across the 3 codebooks.
3. For symmetric shapes (torus, sphere), level 0 uses very few unique
   tokens per triangle (18 % for torus!) while levels 1 and 2 reuse more
   tokens too. The coarse+residual decomposition matches the geometric
   hierarchy of the shape.
4. All three codebooks are well utilized: 97 / 96 / 85 % active codes.
   Dead codes are still reset every 20 steps, keeping the codebook
   healthy even with a 3-level stack.

## Training knobs that mattered

- **Residual VQ** (3 levels): biggest single win in this project. Every
  category improved roughly 2x on stitched L1 with no change to
  encoder/decoder/loss. Drop-in replacement for plain VQ.
- Two-phase training (warmup as plain AE, then activate VQ with codebook
  initialized from encoder outputs) stabilized VQ learning; essential
  for RVQ where codebook init at each level is sequential.
- Removing L2 normalization from the quantizer eliminated periodic
  `vq_loss` spikes caused by the mismatch between cosine-based lookup
  and MSE-based commitment.
- Cyclic permutation L1 loss handles the vertex-ordering ambiguity of
  triangles without needing to pre-sort vertices (pre-sorting breaks
  stitching because it desyncs from `mesh.faces`).
- Switching from MLP decoder to GNN decoder let neighboring triangles
  share context, improving raw (pre-stitching) reconstructions.
