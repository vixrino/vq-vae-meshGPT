# VQ-VAE Reconstruction Results

Trained on 425 low-poly meshes across 8 categories for 300 epochs
(30 epoch autoencoder warmup + 270 epoch VQ phase).

## Model

- Encoder: 4x SAGEConv (9 → 64 → 128 → 256 → 128 = latent_dim)
- Quantizer: VQ with 512 codes, EMA updates, dead-code reset
- Decoder: 4x SAGEConv (128 → 256 → 256 → 128 → 9)
- Losses: cyclic-permutation L1 recon + VQ commitment + 0.3 x vertex consistency

## Per-category reconstruction (20 meshes per category, stitched)

| Category   |  N  | Raw L1  | Stitched L1 | Tokens / triangle |
|------------|----:|--------:|------------:|------------------:|
| torus      |  20 |  0.0483 |  **0.0355** |             19.3% |
| icosphere  |  19 |  0.0568 |  **0.0343** |             37.6% |
| sphere     |  20 |  0.0762 |      0.0523 |             56.0% |
| capsule    |  20 |  0.0942 |      0.0815 |             46.4% |
| cylinder   |  20 |  0.1140 |      0.0996 |             41.6% |
| cone       |  20 |  0.1420 |      0.1274 |             46.8% |
| deformed   |  20 |  0.1623 |      0.1463 |             59.8% |
| box        |  20 |  0.2363 |      0.2181 |             86.5% |
| **Average**|     |  0.1163 |  **0.0994** |                   |

Best val loss: 0.0793 (epoch 295).

## Observations

1. The model excels on shapes with repetitive local geometry (torus,
   icosphere, sphere) - exactly what VQ is designed to exploit. Tokens
   are reused heavily (only 19% unique tokens per triangle on the torus)
   which means strong compression.
2. Boxes are the pathological case: 12 triangles that are all different
   from one another, no repetition to exploit. 86% of triangles get a
   unique token and reconstruction error is highest.
3. Stitching always helps. Raw per-triangle predictions are noisy at
   shared edges; averaging across triangles sharing a vertex produces
   watertight, visually clean results.
4. The vertex consistency loss (weight 0.3) improves overall val recon
   from 0.084 to 0.079 and drastically reduces the gap between raw and
   stitched reconstructions for most categories.

## Training knobs that mattered

- Two-phase training (warmup as plain AE, then activate VQ with codebook
  initialized from encoder outputs) stabilized VQ learning.
- Removing L2 normalization from the quantizer eliminated periodic
  `vq_loss` spikes caused by the mismatch between cosine-based lookup
  and MSE-based commitment.
- Cyclic permutation L1 loss handles the vertex-ordering ambiguity of
  triangles without needing to pre-sort vertices (pre-sorting breaks
  stitching because it desyncs from `mesh.faces`).
- Switching from MLP decoder to GNN decoder let neighboring triangles
  share context, improving raw (pre-stitching) reconstructions.
