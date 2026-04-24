"""
Training script for the Mesh VQ-VAE.

Two-phase training:
  Phase 1 (warmup): Train as plain autoencoder (VQ bypassed)
  Phase 2: Activate VQ, init codebook from encoder outputs, train end-to-end
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Batch
from tqdm import tqdm

from mesh_dataset import MeshDataset
from model import MeshVQVAE


def pyg_collate(batch: list) -> Batch | None:
    filtered = [item for item in batch if item is not None]
    if len(filtered) == 0:
        return None
    return Batch.from_data_list(filtered)


def train_epoch(model, loader, optimizer, device):
    model.train()
    totals = {"recon_loss": 0.0, "vq_loss": 0.0, "cons_loss": 0.0, "total_loss": 0.0, "codebook_usage": 0.0}
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
        totals["cons_loss"] += out["cons_loss"].item()
        totals["total_loss"] += out["total_loss"].item()
        if not model.in_warmup:
            totals["codebook_usage"] += model.quantizer.get_codebook_usage(out["indices"])
        num_batches += 1

    if num_batches == 0:
        return {k: 0.0 for k in totals}
    return {k: v / num_batches for k, v in totals.items()}


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    totals = {"recon_loss": 0.0, "vq_loss": 0.0, "cons_loss": 0.0, "total_loss": 0.0, "codebook_usage": 0.0}
    num_batches = 0

    for batch in loader:
        if batch is None:
            continue
        batch = batch.to(device)
        out = model(batch)

        totals["recon_loss"] += out["recon_loss"].item()
        totals["vq_loss"] += out["vq_loss"].item()
        totals["cons_loss"] += out["cons_loss"].item()
        totals["total_loss"] += out["total_loss"].item()
        if not model.in_warmup:
            totals["codebook_usage"] += model.quantizer.get_codebook_usage(out["indices"])
        num_batches += 1

    if num_batches == 0:
        return {k: 0.0 for k in totals}
    return {k: v / num_batches for k, v in totals.items()}


@torch.no_grad()
def init_codebook(model, loader, device):
    """Collect encoder outputs and initialize codebook."""
    model.eval()
    all_z = []
    for batch in loader:
        if batch is None:
            continue
        batch = batch.to(device)
        z_e = model.encoder(batch.x, batch.edge_index)
        all_z.append(z_e)
        if sum(z.shape[0] for z in all_z) >= model.quantizer.num_embeddings * 2:
            break

    all_z = torch.cat(all_z, dim=0)
    model.quantizer.init_codebook_from_data(all_z)
    print(f"  Codebook initialized from {all_z.shape[0]} encoder outputs")
    print(f"  Encoder output stats: mean={all_z.mean():.4f}, std={all_z.std():.4f}, "
          f"min={all_z.min():.4f}, max={all_z.max():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train Mesh VQ-VAE")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--num_embeddings", type=int, default=512)
    parser.add_argument("--num_vq_levels", type=int, default=3, help="Number of residual VQ levels")
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument("--warmup_epochs", type=int, default=30)
    parser.add_argument("--max_faces", type=int, default=800)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=25)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = MeshDataset(args.data_dir, max_faces=args.max_faces)
    if len(dataset) == 0:
        print("No meshes found.")
        return

    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"Train: {train_size} | Val: {val_size}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pyg_collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pyg_collate, num_workers=0)

    model = MeshVQVAE(
        latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings,
        num_vq_levels=args.num_vq_levels,
        commitment_cost=args.commitment_cost,
        warmup_epochs=args.warmup_epochs,
    ).to(device)

    print(f"Parameters: {model.count_parameters()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_recon = float("inf")

    print(f"\n=== Phase 1: Autoencoder warmup ({args.warmup_epochs} epochs) ===")
    print("-" * 100)

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training"):
        model.current_epoch = epoch - 1
        t0 = time.time()

        # At the end of warmup, initialize codebook from learned encoder
        if epoch == args.warmup_epochs + 1:
            print(f"\n=== Phase 2: VQ activated (epoch {epoch}) ===")
            init_codebook(model, train_loader, device)
            print("-" * 100)

        train_m = train_epoch(model, train_loader, optimizer, device)
        val_m = eval_epoch(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        phase = "AE" if model.in_warmup else "VQ"

        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.warmup_epochs + 1:
            cb_str = f"CB={train_m['codebook_usage']:.1%}" if not model.in_warmup else "CB=n/a"
            print(
                f"[{phase}] Epoch {epoch:4d} | "
                f"Train L={train_m['total_loss']:.4f} "
                f"(recon={train_m['recon_loss']:.4f}, vq={train_m['vq_loss']:.4f}, cons={train_m['cons_loss']:.4f}) | "
                f"Val recon={val_m['recon_loss']:.4f} | "
                f"{cb_str} | {elapsed:.1f}s"
            )

        if not model.in_warmup and val_m["recon_loss"] < best_val_recon:
            best_val_recon = val_m["recon_loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_recon,
                "args": vars(args),
            }, os.path.join(args.save_dir, "best_model.pth"))

        if epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_m["recon_loss"],
                "args": vars(args),
            }, os.path.join(args.save_dir, f"model_epoch_{epoch}.pth"))

    print("-" * 100)
    print(f"Done. Best val recon: {best_val_recon:.4f}")


if __name__ == "__main__":
    main()
