"""
Training script for the Mesh VQ-VAE.

Trains the model on .obj meshes from a specified directory,
with periodic logging and checkpoint saving.
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Batch
from tqdm import tqdm

from mesh_dataset import MeshDataset, collate_skip_none
from model import MeshVQVAE


def pyg_collate(batch: list) -> Batch | None:
    """Collate PyG Data objects into a Batch, skipping None."""
    filtered = [item for item in batch if item is not None]
    if len(filtered) == 0:
        return None
    return Batch.from_data_list(filtered)


def train_epoch(
    model: MeshVQVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals = {"recon_loss": 0.0, "vq_loss": 0.0, "diversity_loss": 0.0, "total_loss": 0.0, "codebook_usage": 0.0}
    num_batches = 0

    for batch in loader:
        if batch is None:
            continue

        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch)
        out["total_loss"].backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        totals["recon_loss"] += out["recon_loss"].item()
        totals["vq_loss"] += out["vq_loss"].item()
        totals["diversity_loss"] += out["diversity_loss"].item()
        totals["total_loss"] += out["total_loss"].item()
        totals["codebook_usage"] += model.quantizer.get_codebook_usage(out["indices"])
        num_batches += 1

    if num_batches == 0:
        return {k: 0.0 for k in totals}

    return {k: v / num_batches for k, v in totals.items()}


@torch.no_grad()
def eval_epoch(
    model: MeshVQVAE,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = {"recon_loss": 0.0, "vq_loss": 0.0, "diversity_loss": 0.0, "total_loss": 0.0, "codebook_usage": 0.0}
    num_batches = 0

    for batch in loader:
        if batch is None:
            continue

        batch = batch.to(device)
        out = model(batch)

        totals["recon_loss"] += out["recon_loss"].item()
        totals["vq_loss"] += out["vq_loss"].item()
        totals["diversity_loss"] += out["diversity_loss"].item()
        totals["total_loss"] += out["total_loss"].item()
        totals["codebook_usage"] += model.quantizer.get_codebook_usage(out["indices"])
        num_batches += 1

    if num_batches == 0:
        return {k: 0.0 for k in totals}

    return {k: v / num_batches for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="Train Mesh VQ-VAE")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to .obj mesh directory")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--num_embeddings", type=int, default=512)
    parser.add_argument("--commitment_cost", type=float, default=1.0)
    parser.add_argument("--diversity_weight", type=float, default=0.1)
    parser.add_argument("--max_faces", type=int, default=800)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = MeshDataset(args.data_dir, max_faces=args.max_faces)
    if len(dataset) == 0:
        print("No meshes found. Check your data_dir path.")
        return

    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    print(f"Train: {train_size} meshes | Val: {val_size} meshes")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=pyg_collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pyg_collate,
        num_workers=0,
    )

    model = MeshVQVAE(
        latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings,
        commitment_cost=args.commitment_cost,
        diversity_weight=args.diversity_weight,
    ).to(device)

    param_counts = model.count_parameters()
    print(f"Model parameters: {param_counts}")
    print(f"Config: latent_dim={args.latent_dim}, codebook={args.num_embeddings}, "
          f"commitment={args.commitment_cost}, diversity={args.diversity_weight}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")

    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 100)

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training"):
        t0 = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = eval_epoch(model, val_loader, device)

        scheduler.step()

        elapsed = time.time() - t0

        if epoch % args.log_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d} | "
                f"Train L={train_metrics['total_loss']:.4f} "
                f"(recon={train_metrics['recon_loss']:.4f}, "
                f"vq={train_metrics['vq_loss']:.4f}, "
                f"div={train_metrics['diversity_loss']:.4f}) | "
                f"Val recon={val_metrics['recon_loss']:.4f} | "
                f"CB={train_metrics['codebook_usage']:.1%} | "
                f"LR={scheduler.get_last_lr()[0]:.2e} | "
                f"{elapsed:.1f}s"
            )

        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "args": vars(args),
                },
                os.path.join(args.save_dir, "best_model.pth"),
            )

        if epoch % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["total_loss"],
                    "args": vars(args),
                },
                os.path.join(args.save_dir, f"model_epoch_{epoch}.pth"),
            )

    print("-" * 100)
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to {args.save_dir}/best_model.pth")


if __name__ == "__main__":
    main()
