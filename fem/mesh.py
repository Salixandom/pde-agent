"""
Structured triangular mesh generator for 2D rectangular domains.

Each cell of an Nx x Ny structured grid is split into 2 triangles,
giving P1-compatible linear-triangle meshes without any external
meshing library (Gmsh in the base paper is replaced by this, since
the sandbox has no mesh-generator binary available).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class Mesh2D:
    nodes: np.ndarray          # (n_nodes, 2) coordinates
    triangles: np.ndarray      # (n_tri, 3) node indices (CCW)
    Lx: float
    Ly: float
    Nx: int
    Ny: int

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_tri(self) -> int:
        return self.triangles.shape[0]

    def boundary_nodes(self, side: str, tol: float = 1e-9) -> np.ndarray:
        """side in {'left','right','bottom','top'}"""
        x, y = self.nodes[:, 0], self.nodes[:, 1]
        if side == "left":
            return np.where(np.isclose(x, 0.0, atol=tol))[0]
        if side == "right":
            return np.where(np.isclose(x, self.Lx, atol=tol))[0]
        if side == "bottom":
            return np.where(np.isclose(y, 0.0, atol=tol))[0]
        if side == "top":
            return np.where(np.isclose(y, self.Ly, atol=tol))[0]
        raise ValueError(f"unknown side {side!r}")

    def boundary_edges(self, side: str) -> np.ndarray:
        """Ordered (n_edges, 2) node-index pairs along one side, for
        Neumann / Robin flux integration."""
        idx = self.boundary_nodes(side)
        if side in ("left", "right"):
            order = np.argsort(self.nodes[idx, 1])
        else:
            order = np.argsort(self.nodes[idx, 0])
        idx = idx[order]
        return np.stack([idx[:-1], idx[1:]], axis=1)


def rectangle_mesh(Lx: float, Ly: float, Nx: int, Ny: int) -> Mesh2D:
    """Uniform structured mesh on [0,Lx] x [0,Ly] with Nx x Ny cells
    (each cell -> 2 triangles). Matches the paper's `nx`, `ny` mesh
    parameters and its COARSE_MESH_2D rule (nx < 10 or ny < 10)."""
    xs = np.linspace(0.0, Lx, Nx + 1)
    ys = np.linspace(0.0, Ly, Ny + 1)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    nodes = np.stack([xx.ravel(), yy.ravel()], axis=1)

    def nid(i, j):  # i: column (x), j: row (y)
        return j * (Nx + 1) + i

    tris = []
    for j in range(Ny):
        for i in range(Nx):
            n00, n10 = nid(i, j), nid(i + 1, j)
            n01, n11 = nid(i, j + 1), nid(i + 1, j + 1)
            # two CCW triangles per cell
            tris.append([n00, n10, n11])
            tris.append([n00, n11, n01])
    return Mesh2D(nodes=nodes, triangles=np.array(tris, dtype=int),
                   Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)
