"""
Download ShapeNet meshes or generate synthetic training meshes.

Two modes:
  1. --source shapenet : Download from ShapeNet via PyG (needs Stanford server up)
  2. --source synthetic : Generate diverse low-poly meshes locally (instant, no network)
  3. --source modelnet  : Download ModelNet10 via PyG (reliable mirror)
"""

import argparse
import os
import random

import numpy as np
import trimesh


def generate_synthetic_meshes(
    output_dir: str = "data/shapenet",
    num_meshes: int = 200,
    seed: int = 42,
):
    """
    Generate diverse synthetic low-poly meshes for VQ-VAE training.
    Creates varied shapes: spheres, boxes, cylinders, tori, cones,
    icospheres, with random deformations and noise.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    shape_types = [
        "sphere", "box", "cylinder", "torus",
        "cone", "icosphere", "capsule", "deformed",
    ]

    categories = {}
    for stype in shape_types:
        cat_dir = os.path.join(output_dir, stype)
        os.makedirs(cat_dir, exist_ok=True)
        categories[stype] = cat_dir

    count = 0
    meshes_per_type = num_meshes // len(shape_types)

    for stype in shape_types:
        for i in range(meshes_per_type):
            mesh = _create_shape(stype)
            if mesh is None or len(mesh.faces) < 10:
                continue

            mesh = _random_transform(mesh)

            if len(mesh.faces) > 800:
                continue

            out_path = os.path.join(categories[stype], f"{stype}_{i:04d}.obj")
            mesh.export(out_path)
            count += 1

    print(f"Generated {count} synthetic meshes in {output_dir}/")
    for stype in shape_types:
        n = len([f for f in os.listdir(categories[stype]) if f.endswith(".obj")])
        print(f"  {stype}: {n} meshes")


def _create_shape(shape_type: str) -> trimesh.Trimesh:
    if shape_type == "sphere":
        subdivisions = random.randint(1, 3)
        mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=random.uniform(0.5, 1.5))
    elif shape_type == "box":
        extents = [random.uniform(0.3, 2.0) for _ in range(3)]
        mesh = trimesh.creation.box(extents=extents)
        if random.random() > 0.5:
            mesh = mesh.subdivide()
    elif shape_type == "cylinder":
        mesh = trimesh.creation.cylinder(
            radius=random.uniform(0.3, 1.2),
            height=random.uniform(0.5, 3.0),
            sections=random.randint(6, 20),
        )
    elif shape_type == "torus":
        mesh = trimesh.creation.torus(
            major_radius=random.uniform(0.8, 1.5),
            minor_radius=random.uniform(0.1, 0.4),
            major_sections=random.randint(8, 20),
            minor_sections=random.randint(6, 14),
        )
    elif shape_type == "cone":
        mesh = trimesh.creation.cone(
            radius=random.uniform(0.3, 1.5),
            height=random.uniform(0.5, 3.0),
            sections=random.randint(6, 20),
        )
    elif shape_type == "icosphere":
        mesh = trimesh.creation.icosphere(
            subdivisions=random.randint(2, 4),
            radius=random.uniform(0.5, 1.5),
        )
    elif shape_type == "capsule":
        mesh = trimesh.creation.capsule(
            height=random.uniform(0.5, 2.0),
            radius=random.uniform(0.2, 0.8),
            count=[random.randint(4, 12), random.randint(4, 12)],
        )
    elif shape_type == "deformed":
        base = random.choice(["sphere", "box", "cylinder"])
        mesh = _create_shape(base)
        noise = np.random.randn(*mesh.vertices.shape) * random.uniform(0.02, 0.15)
        mesh.vertices += noise
    else:
        mesh = trimesh.creation.icosphere(subdivisions=2)

    return mesh


def _random_transform(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Apply random scale and rotation."""
    scale = [random.uniform(0.5, 2.0) for _ in range(3)]
    mesh.vertices *= scale

    angle = random.uniform(0, 2 * np.pi)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    rot = trimesh.transformations.rotation_matrix(angle, axis)
    mesh.apply_transform(rot)

    return mesh


def download_modelnet(
    output_dir: str = "data/shapenet",
    max_per_category: int = 50,
):
    """Download ModelNet10 via PyG and export as .obj via convex hull."""
    from torch_geometric.datasets import ModelNet
    from torch_geometric.transforms import SamplePoints

    print("Downloading ModelNet10...")
    dataset = ModelNet(
        root="data/modelnet_raw",
        name="10",
        train=True,
        pre_transform=SamplePoints(1024),
    )

    categories = {}
    count = 0

    for i, data in enumerate(dataset):
        cat_name = f"cat_{data.y.item()}"
        if cat_name not in categories:
            categories[cat_name] = 0
            os.makedirs(os.path.join(output_dir, cat_name), exist_ok=True)

        if categories[cat_name] >= max_per_category:
            continue

        try:
            pos = data.pos.numpy()
            cloud = trimesh.PointCloud(pos)
            mesh = cloud.convex_hull
            if mesh is None or len(mesh.faces) < 10 or len(mesh.faces) > 800:
                continue

            out_path = os.path.join(
                output_dir, cat_name, f"model_{i:05d}.obj"
            )
            mesh.export(out_path)
            categories[cat_name] += 1
            count += 1
        except Exception:
            continue

    print(f"Exported {count} meshes from ModelNet10 to {output_dir}/")
    for cat, n in categories.items():
        print(f"  {cat}: {n} meshes")


def download_shapenet_pyg(
    output_dir: str = "data/shapenet",
    categories: list[str] | None = None,
    max_per_category: int = 50,
):
    """Download ShapeNet via PyG (requires Stanford server to be up)."""
    from torch_geometric.datasets import ShapeNet

    CATEGORIES = [
        "Airplane", "Bag", "Cap", "Car", "Chair", "Earphone",
        "Guitar", "Knife", "Lamp", "Laptop", "Motorbike", "Mug",
        "Pistol", "Rocket", "Skateboard", "Table",
    ]

    if categories is None:
        categories = ["Chair", "Table", "Airplane", "Car", "Lamp"]

    for cat in categories:
        if cat not in CATEGORIES:
            print(f"Warning: '{cat}' not available, skipping.")
            continue

        print(f"Downloading ShapeNet category: {cat}")
        dataset = ShapeNet(root="data/shapenet_raw", categories=[cat], split="train")

        cat_dir = os.path.join(output_dir, cat.lower())
        os.makedirs(cat_dir, exist_ok=True)

        count = 0
        for i, data in enumerate(dataset):
            if count >= max_per_category:
                break
            try:
                pos = data.pos.numpy()
                cloud = trimesh.PointCloud(pos)
                mesh = cloud.convex_hull
                if mesh is None or len(mesh.faces) < 10 or len(mesh.faces) > 800:
                    continue
                mesh.export(os.path.join(cat_dir, f"{cat.lower()}_{i:04d}.obj"))
                count += 1
            except Exception:
                continue

        print(f"  Exported {count} meshes to {cat_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Download/generate training meshes")
    parser.add_argument(
        "--source", type=str, default="synthetic",
        choices=["shapenet", "modelnet", "synthetic"],
        help="Data source: shapenet (Stanford), modelnet (reliable), synthetic (local)",
    )
    parser.add_argument("--output_dir", type=str, default="data/shapenet")
    parser.add_argument("--num_meshes", type=int, default=500)
    parser.add_argument("--max_per_category", type=int, default=50)
    parser.add_argument(
        "--categories", nargs="+",
        default=["Chair", "Table", "Airplane", "Car", "Lamp"],
    )
    args = parser.parse_args()

    if args.source == "synthetic":
        generate_synthetic_meshes(args.output_dir, args.num_meshes)
    elif args.source == "modelnet":
        download_modelnet(args.output_dir, args.max_per_category)
    elif args.source == "shapenet":
        download_shapenet_pyg(args.output_dir, args.categories, args.max_per_category)


if __name__ == "__main__":
    main()
