"""
Dataset for loading 3D meshes (.obj) and converting them to triangle graphs.

Each triangle becomes a graph node with features = 9D flattened vertex coordinates.
Two triangles sharing an edge are connected in the graph.
Shared vertex pairs are tracked for the consistency loss.
"""

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset
from torch_geometric.data import Data


def load_mesh(path: str) -> trimesh.Trimesh:
    """Load a mesh and ensure it's triangulated."""
    mesh = trimesh.load(path, force="mesh", process=False)
    return mesh


def normalize_vertices(vertices: np.ndarray) -> np.ndarray:
    """Center and scale vertices to fit in [-1, 1]."""
    centroid = vertices.mean(axis=0)
    vertices = vertices - centroid
    scale = np.abs(vertices).max()
    if scale > 0:
        vertices = vertices / scale
    return vertices


def normalize_triangle_vertex_order(tri_coords: np.ndarray) -> np.ndarray:
    """
    Sort the 3 vertices of each triangle in lexicographic order (x, then y, then z).

    This ensures that the same geometric triangle always produces the same 9D
    representation regardless of original vertex winding. Critical for the
    decoder to learn a consistent mapping.
    """
    triangles = tri_coords.reshape(-1, 3, 3)
    for i in range(len(triangles)):
        order = np.lexsort((triangles[i][:, 2], triangles[i][:, 1], triangles[i][:, 0]))
        triangles[i] = triangles[i][order]
    return triangles.reshape(-1, 9)


def build_triangle_graph(mesh: trimesh.Trimesh) -> Data:
    """
    Convert a triangle mesh into a PyG graph.

    Each triangle → one node with 9D features (3 vertices × 3 coords, flattened).
    Two triangles sharing an edge → connected by an undirected edge in the graph.
    """
    vertices = normalize_vertices(mesh.vertices.copy())
    faces = mesh.faces

    num_triangles = len(faces)

    node_features = vertices[faces].reshape(num_triangles, 9)

    edge_to_triangles = defaultdict(list)
    for tri_idx, face in enumerate(faces):
        v0, v1, v2 = sorted(face)
        edges = [
            (v0, v1),
            (v0, v2),
            (v1, v2),
        ]
        for edge in edges:
            edge_to_triangles[edge].append(tri_idx)

    src, dst = [], []
    for _edge, tri_indices in edge_to_triangles.items():
        for i in range(len(tri_indices)):
            for j in range(i + 1, len(tri_indices)):
                src.append(tri_indices[i])
                dst.append(tri_indices[j])
                src.append(tri_indices[j])
                dst.append(tri_indices[i])

    if len(src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)

    vertex_to_triangles = defaultdict(list)
    for tri_idx, face in enumerate(faces):
        for local_idx, global_vid in enumerate(face):
            vertex_to_triangles[global_vid].append((tri_idx, local_idx))

    sv_tri_a, sv_local_a, sv_tri_b, sv_local_b = [], [], [], []
    for _vid, tri_local_list in vertex_to_triangles.items():
        for i in range(len(tri_local_list)):
            for j in range(i + 1, len(tri_local_list)):
                ta, la = tri_local_list[i]
                tb, lb = tri_local_list[j]
                sv_tri_a.append(ta)
                sv_local_a.append(la)
                sv_tri_b.append(tb)
                sv_local_b.append(lb)

    x = torch.tensor(node_features, dtype=torch.float32)
    target_coords = x.clone()

    data = Data(
        x=x,
        edge_index=edge_index,
        y=target_coords,
        num_nodes=num_triangles,
    )

    if len(sv_tri_a) > 0:
        data.sv_tri_a = torch.tensor(sv_tri_a, dtype=torch.long)
        data.sv_local_a = torch.tensor(sv_local_a, dtype=torch.long)
        data.sv_tri_b = torch.tensor(sv_tri_b, dtype=torch.long)
        data.sv_local_b = torch.tensor(sv_local_b, dtype=torch.long)
    else:
        data.sv_tri_a = torch.zeros(0, dtype=torch.long)
        data.sv_local_a = torch.zeros(0, dtype=torch.long)
        data.sv_tri_b = torch.zeros(0, dtype=torch.long)
        data.sv_local_b = torch.zeros(0, dtype=torch.long)

    return data


class MeshDataset(Dataset):
    """
    PyTorch dataset that loads .obj meshes from a directory
    and returns PyG triangle graphs.

    Supports flat directory or nested category directories:
        data_dir/
        ├── chair/
        │   ├── model_001.obj
        │   └── model_002.obj
        └── table/
            └── model_003.obj
    """

    def __init__(
        self,
        data_dir: str,
        max_faces: int = 800,
        min_faces: int = 10,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.max_faces = max_faces
        self.min_faces = min_faces
        self.file_paths = self._collect_obj_files()
        print(f"Found {len(self.file_paths)} valid mesh files in {data_dir}")

    def _collect_obj_files(self) -> list[str]:
        paths = []
        for root, _dirs, files in os.walk(self.data_dir):
            for f in files:
                if f.endswith(".obj"):
                    paths.append(os.path.join(root, f))
        return sorted(paths)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Data | None:
        path = self.file_paths[idx]
        try:
            mesh = load_mesh(path)

            if len(mesh.faces) > self.max_faces or len(mesh.faces) < self.min_faces:
                return None

            graph = build_triangle_graph(mesh)
            graph.mesh_path = path
            return graph
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None


def collate_skip_none(batch: list) -> list:
    """Collate function that filters out None entries."""
    return [item for item in batch if item is not None]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python mesh_dataset.py <path_to_obj_or_directory>")
        sys.exit(1)

    target = sys.argv[1]

    if os.path.isfile(target):
        mesh = load_mesh(target)
        print(f"Mesh: {target}")
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Faces: {len(mesh.faces)}")
        graph = build_triangle_graph(mesh)
        print(f"  Graph nodes (triangles): {graph.num_nodes}")
        print(f"  Graph edges: {graph.edge_index.shape[1]}")
        print(f"  Node feature shape: {graph.x.shape}")
        print(f"  Shared vertex pairs: {len(graph.sv_tri_a)}")
        print(f"  Avg edges per node: {graph.edge_index.shape[1] / graph.num_nodes:.1f}")
    else:
        ds = MeshDataset(target)
        print(f"\nDataset size: {len(ds)}")
        if len(ds) > 0:
            sample = ds[0]
            if sample is not None:
                print(f"Sample graph:")
                print(f"  Nodes: {sample.num_nodes}")
                print(f"  Edges: {sample.edge_index.shape[1]}")
                print(f"  Features: {sample.x.shape}")
                print(f"  Shared vertex pairs: {len(sample.sv_tri_a)}")
