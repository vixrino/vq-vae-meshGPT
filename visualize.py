"""
Visualization of VQ-VAE reconstruction quality.

Side-by-side 3D plots of original vs reconstructed meshes,
codebook usage histogram, and token assignment maps.

Includes post-processing that stitches shared vertices by averaging
predictions, producing watertight reconstructions without needing
a training loss.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mesh_dataset import MeshDataset, build_triangle_graph, load_mesh, normalize_vertices
from model import MeshVQVAE


def stitch_vertices(recon_9d: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Post-process reconstruction by averaging shared vertex predictions.

    Each triangle independently predicts 3 vertex positions. When triangles
    share a vertex (same global vertex index), we average all their predictions
    for that vertex, then rebuild the 9D representation.

    Args:
        recon_9d: [N, 9] raw reconstructed coordinates per triangle
        faces: [N, 3] original face indices (global vertex IDs)

    Returns:
        [N, 9] stitched coordinates where shared vertices are consistent
    """
    recon_tris = recon_9d.reshape(-1, 3, 3)
    num_vertices = faces.max() + 1

    vertex_sum = np.zeros((num_vertices, 3), dtype=np.float64)
    vertex_count = np.zeros(num_vertices, dtype=np.float64)

    for tri_idx, face in enumerate(faces):
        for local_idx, global_vid in enumerate(face):
            vertex_sum[global_vid] += recon_tris[tri_idx, local_idx]
            vertex_count[global_vid] += 1

    vertex_count = np.maximum(vertex_count, 1)
    avg_vertices = vertex_sum / vertex_count[:, None]

    stitched = avg_vertices[faces].reshape(-1, 9)
    return stitched.astype(np.float32)


def plot_triangles(ax, coords_9d: np.ndarray, title: str, color: str = "skyblue", token_indices: np.ndarray | None = None):
    """Plot triangles in 3D from 9D flattened coordinates."""
    triangles = coords_9d.reshape(-1, 3, 3)

    if token_indices is not None:
        cmap = plt.cm.tab20
        unique_tokens = np.unique(token_indices)
        norm = plt.Normalize(vmin=unique_tokens.min(), vmax=unique_tokens.max())
        face_colors = [cmap(norm(t)) for t in token_indices]
    else:
        face_colors = [color] * len(triangles)

    poly = Poly3DCollection(triangles, alpha=0.7, linewidths=0.3, edgecolors="gray")
    poly.set_facecolor(face_colors)
    ax.add_collection3d(poly)

    all_verts = triangles.reshape(-1, 3)
    margin = 0.1
    for i, label in enumerate(["X", "Y", "Z"]):
        lo, hi = all_verts[:, i].min() - margin, all_verts[:, i].max() + margin
        getattr(ax, f"set_{label.lower()}lim")(lo, hi)
        getattr(ax, f"set_{label.lower()}label")(label)

    ax.set_title(title, fontsize=11, fontweight="bold")


@torch.no_grad()
def visualize_reconstruction(
    model: MeshVQVAE,
    mesh_path: str,
    device: torch.device,
    save_path: str | None = None,
):
    """Plot original, raw reconstruction, stitched reconstruction, and token map."""
    mesh = load_mesh(mesh_path)
    graph = build_triangle_graph(mesh)
    faces = mesh.faces
    graph = graph.to(device)

    out = model(graph)

    original = graph.y.cpu().numpy()
    raw_recon = out["recon"].cpu().numpy()
    stitched_recon = stitch_vertices(raw_recon, faces)
    indices = out["indices"].cpu().numpy()
    level0_indices = indices[:, 0] if indices.ndim == 2 else indices

    raw_l1 = np.abs(original - raw_recon).mean()
    stitched_l1 = np.abs(original - stitched_recon).mean()
    unique_codes = len(np.unique(level0_indices))

    fig = plt.figure(figsize=(24, 6))

    ax1 = fig.add_subplot(141, projection="3d")
    plot_triangles(ax1, original, f"Original\n({len(original)} triangles)")

    ax2 = fig.add_subplot(142, projection="3d")
    plot_triangles(ax2, raw_recon, f"Raw Recon\nL1={raw_l1:.4f}", color="salmon")

    ax3 = fig.add_subplot(143, projection="3d")
    plot_triangles(ax3, stitched_recon, f"Stitched Recon\nL1={stitched_l1:.4f}", color="lightgreen")

    ax4 = fig.add_subplot(144, projection="3d")
    plot_triangles(ax4, original, f"Token Map (level 0)\n({unique_codes} unique codes)", token_indices=level0_indices)

    name = os.path.basename(mesh_path)
    fig.suptitle(f"{name}", fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)
    return stitched_l1, unique_codes


@torch.no_grad()
def visualize_codebook_usage(
    model: MeshVQVAE,
    dataset: MeshDataset,
    device: torch.device,
    max_meshes: int = 100,
    save_path: str | None = None,
):
    """Histogram of codebook usage across the dataset."""
    all_indices = []

    for i in range(min(len(dataset), max_meshes)):
        graph = dataset[i]
        if graph is None:
            continue
        graph = graph.to(device)
        out = model(graph)
        idx = out["indices"].cpu().numpy()
        if idx.ndim == 1:
            idx = idx[:, None]
        all_indices.append(idx)

    if not all_indices:
        print("No valid meshes to visualize.")
        return

    all_idx = np.concatenate(all_indices, axis=0)
    num_codes = model.quantizer.num_embeddings
    num_levels = all_idx.shape[1]

    fig, axes = plt.subplots(num_levels, 2, figsize=(14, 4 * num_levels), squeeze=False)

    for level in range(num_levels):
        counts = np.bincount(all_idx[:, level], minlength=num_codes)
        used = (counts > 0).sum()
        dead = num_codes - used

        axes[level, 0].bar(range(num_codes), counts, width=1.0, color="steelblue", alpha=0.8)
        axes[level, 0].set_xlabel("Code Index")
        axes[level, 0].set_ylabel("Usage Count")
        axes[level, 0].set_title(f"Level {level}: {used}/{num_codes} active, {dead} dead")
        active_counts = counts[counts > 0]
        if len(active_counts) > 0:
            axes[level, 0].axhline(y=active_counts.mean(), color="red", linestyle="--", label=f"Mean={active_counts.mean():.0f}")
            axes[level, 0].legend()

            sorted_counts = np.sort(active_counts)[::-1]
            axes[level, 1].bar(range(len(sorted_counts)), sorted_counts, width=1.0, color="coral", alpha=0.8)
            axes[level, 1].set_xlabel("Rank")
            axes[level, 1].set_ylabel("Usage Count")
            axes[level, 1].set_title(f"Level {level} distribution (sorted)")
            axes[level, 1].set_yscale("log")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


@torch.no_grad()
def visualize_grid(
    model: MeshVQVAE,
    dataset: MeshDataset,
    device: torch.device,
    n_samples: int = 8,
    save_path: str | None = None,
):
    """Grid of original vs stitched reconstructed meshes."""
    samples = []
    for i in range(len(dataset)):
        if len(samples) >= n_samples:
            break
        graph = dataset[i]
        if graph is not None:
            mesh = load_mesh(dataset.file_paths[i])
            samples.append((dataset.file_paths[i], graph, mesh.faces))

    n = len(samples)
    fig = plt.figure(figsize=(6 * 2, 4 * n))

    for row, (path, graph, faces) in enumerate(samples):
        graph = graph.to(device)
        out = model(graph)

        original = graph.y.cpu().numpy()
        raw_recon = out["recon"].cpu().numpy()
        stitched = stitch_vertices(raw_recon, faces)
        l1 = np.abs(original - stitched).mean()

        ax1 = fig.add_subplot(n, 2, row * 2 + 1, projection="3d")
        plot_triangles(ax1, original, f"Original: {os.path.basename(path)}")

        ax2 = fig.add_subplot(n, 2, row * 2 + 2, projection="3d")
        plot_triangles(ax2, stitched, f"Stitched (L1={l1:.3f})", color="lightgreen")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize VQ-VAE reconstructions")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--data_dir", type=str, default="data/shapenet")
    parser.add_argument("--output_dir", type=str, default="visualizations")
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--max_faces", type=int, default=800)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})

    model = MeshVQVAE(
        latent_dim=saved_args.get("latent_dim", 128),
        num_embeddings=saved_args.get("num_embeddings", 512),
        num_vq_levels=saved_args.get("num_vq_levels", 1),
        commitment_cost=saved_args.get("commitment_cost", 0.25),
        warmup_epochs=0,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}, val_loss={checkpoint.get('val_loss', '?'):.4f}")

    dataset = MeshDataset(args.data_dir, max_faces=args.max_faces)
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n1/3 Generating reconstruction grid...")
    visualize_grid(
        model, dataset, device,
        n_samples=args.n_samples,
        save_path=os.path.join(args.output_dir, "reconstruction_grid.png"),
    )

    print("2/3 Generating codebook usage plot...")
    visualize_codebook_usage(
        model, dataset, device,
        save_path=os.path.join(args.output_dir, "codebook_usage.png"),
    )

    print("3/3 Generating individual reconstruction plots...")
    indiv_dir = os.path.join(args.output_dir, "individual")
    os.makedirs(indiv_dir, exist_ok=True)

    errors = []
    for i in range(min(len(dataset), args.n_samples)):
        path = dataset.file_paths[i]
        graph = dataset[i]
        if graph is None:
            continue
        l1, codes = visualize_reconstruction(
            model, path, device,
            save_path=os.path.join(indiv_dir, f"recon_{i:03d}.png"),
        )
        errors.append(l1)

    if errors:
        print(f"\nMean L1 across {len(errors)} samples: {np.mean(errors):.4f}")

    print(f"\nAll visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
