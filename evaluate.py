"""
Evaluation script for the Mesh VQ-VAE.

Loads a trained model and evaluates reconstruction quality:
- Per-mesh L1 and L2 error on vertex coordinates
- Codebook utilization statistics
- Optional: exports reconstructed meshes as .obj for visual inspection
"""

import argparse
import os

import numpy as np
import torch
import trimesh

from mesh_dataset import MeshDataset, build_triangle_graph, load_mesh, normalize_vertices
from model import MeshVQVAE
from visualize import stitch_vertices


@torch.no_grad()
def evaluate_mesh(
    model: MeshVQVAE, mesh_path: str, device: torch.device
) -> dict:
    """Evaluate reconstruction quality on a single mesh with stitched vertices."""
    mesh = load_mesh(mesh_path)
    graph = build_triangle_graph(mesh)
    graph = graph.to(device)

    out = model(graph)

    original = graph.y.cpu().numpy()
    raw_recon = out["recon"].cpu().numpy()
    stitched = stitch_vertices(raw_recon, mesh.faces)

    l1_error = np.abs(original - stitched).mean()
    l2_error = np.sqrt(((original - stitched) ** 2).sum(axis=1)).mean()

    indices = out["indices"].cpu()
    # indices may be [N] (plain VQ) or [N, num_levels] (RVQ)
    unique_codes = torch.unique(indices.reshape(-1)).shape[0]

    return {
        "mesh_path": mesh_path,
        "num_triangles": graph.num_nodes,
        "l1_error": float(l1_error),
        "l2_error": float(l2_error),
        "num_unique_codes": int(unique_codes),
        "total_codes_used": int(indices.numel()),
        "original": original,
        "reconstructed": stitched,
        "indices": indices.numpy(),
        "faces": mesh.faces,
        "original_vertices": normalize_vertices(mesh.vertices.copy()),
    }


def export_reconstructed_obj(
    result: dict, output_path: str
) -> None:
    """Export the reconstructed mesh as .obj using stitched vertices."""
    recon_coords = result["reconstructed"].reshape(-1, 3, 3)
    faces = result["faces"]

    vertex_map = {}
    new_vertices = []
    new_faces = []
    tolerance = 1e-4

    for tri_idx in range(len(recon_coords)):
        face_indices = []
        for v_idx in range(3):
            v = tuple(recon_coords[tri_idx, v_idx].round(5))

            matched = None
            for existing_key, existing_idx in vertex_map.items():
                if all(abs(a - b) < tolerance for a, b in zip(v, existing_key)):
                    matched = existing_idx
                    break

            if matched is not None:
                face_indices.append(matched)
            else:
                new_idx = len(new_vertices)
                vertex_map[v] = new_idx
                new_vertices.append(list(v))
                face_indices.append(new_idx)

        new_faces.append(face_indices)

    mesh = trimesh.Trimesh(
        vertices=np.array(new_vertices),
        faces=np.array(new_faces),
        process=False,
    )
    mesh.export(output_path)
    print(f"Exported reconstructed mesh: {output_path}")
    print(f"  Vertices: {len(new_vertices)} (original: {len(result['original_vertices'])})")
    print(f"  Faces: {len(new_faces)}")


def print_codebook_stats(model: MeshVQVAE, all_indices: list[np.ndarray]) -> None:
    """Print codebook usage statistics across the evaluation set."""
    total_codes = model.quantizer.num_embeddings
    num_levels = getattr(model.quantizer, "num_levels", 1)

    print(f"\n{'='*60}")
    print(f"Codebook Statistics ({num_levels} RVQ levels x {total_codes} codes)")
    print(f"{'='*60}")

    stacked = np.concatenate([idx.reshape(idx.shape[0], -1) for idx in all_indices], axis=0)
    if stacked.ndim == 1:
        stacked = stacked[:, None]

    for level in range(stacked.shape[1]):
        level_idx = stacked[:, level]
        unique, counts = np.unique(level_idx, return_counts=True)
        used = len(unique)
        print(
            f"  Level {level}: {used}/{total_codes} active ({used/total_codes:.1%}), "
            f"dead={total_codes - used}, mean={counts.mean():.1f}, std={counts.std():.1f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mesh VQ-VAE")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to .obj mesh directory")
    parser.add_argument("--export_dir", type=str, default=None, help="Directory to export reconstructed .obj files")
    parser.add_argument("--max_meshes", type=int, default=50, help="Max meshes to evaluate")
    parser.add_argument("--max_faces", type=int, default=800)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}")
    print(f"  Val loss at save: {checkpoint.get('val_loss', '?')}")
    print(f"  Parameters: {model.count_parameters()}")

    dataset = MeshDataset(args.data_dir, max_faces=args.max_faces)

    if args.export_dir:
        os.makedirs(args.export_dir, exist_ok=True)

    results = []
    all_indices = []

    n_eval = min(len(dataset), args.max_meshes)
    print(f"\nEvaluating {n_eval} meshes...")
    print("-" * 60)

    for i in range(n_eval):
        path = dataset.file_paths[i]
        try:
            result = evaluate_mesh(model, path, device)
            results.append(result)
            all_indices.append(result["indices"])

            print(
                f"[{i+1:3d}/{n_eval}] {os.path.basename(path):30s} | "
                f"triangles={result['num_triangles']:4d} | "
                f"L1={result['l1_error']:.4f} | "
                f"L2={result['l2_error']:.4f} | "
                f"codes={result['num_unique_codes']}"
            )

            if args.export_dir:
                out_name = f"recon_{i:04d}_{os.path.basename(path)}"
                export_reconstructed_obj(
                    result, os.path.join(args.export_dir, out_name)
                )
        except Exception as e:
            print(f"[{i+1:3d}/{n_eval}] ERROR on {path}: {e}")

    if results:
        avg_l1 = np.mean([r["l1_error"] for r in results])
        avg_l2 = np.mean([r["l2_error"] for r in results])
        print(f"\n{'='*60}")
        print("Reconstruction Quality")
        print(f"{'='*60}")
        print(f"  Meshes evaluated: {len(results)}")
        print(f"  Mean L1 error: {avg_l1:.4f}")
        print(f"  Mean L2 error: {avg_l2:.4f}")

        if all_indices:
            print_codebook_stats(model, all_indices)


if __name__ == "__main__":
    main()
