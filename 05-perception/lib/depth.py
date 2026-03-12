from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def backproject_depth_frame(
    depth_m: np.ndarray,
    K: np.ndarray,
    T_cw: np.ndarray | None = None,
    max_depth: float = 5.0,
    subsample: int = 4,
) -> np.ndarray:
    """Back-project a depth image to a 3D point cloud.

    Parameters
    ----------
    depth_m  : (H, W) float32 — depth in metres
    K        : (3, 3) intrinsics
    T_cw     : (4, 4) world-to-camera transform; if given, points are
               returned in world frame, otherwise in camera frame
    max_depth : clip depths beyond this value
    subsample : keep every Nth pixel row and column

    Returns
    -------
    (N, 3) float64 point cloud
    """
    H, W = depth_m.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    v_idx = np.arange(0, H, subsample)
    u_idx = np.arange(0, W, subsample)
    uu, vv = np.meshgrid(u_idx, v_idx)
    zz = depth_m[vv, uu].astype(np.float64)

    valid = (zz > 0) & (zz < max_depth)
    z = zz[valid]
    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)

    X_c = np.stack([
        (u - cx) / fx * z,
        (v - cy) / fy * z,
        z,
    ], axis=1)

    if T_cw is not None:
        # T_cw maps world -> camera, so T_wc = inv(T_cw) maps camera -> world
        R = T_cw[:3, :3]
        t = T_cw[:3, 3]
        X_w = (R.T @ (X_c - t).T).T
        return X_w
    return X_c


@dataclass
class ToFModel:
    modulation_freq: float = 20e6
    c: float = 3e8

    @property
    def max_unambiguous_range(self) -> float:
        return self.c / (2 * self.modulation_freq)

    def phase_to_depth(self, phi: np.ndarray) -> np.ndarray:
        return self.c / (4 * np.pi * self.modulation_freq) * phi

    def depth_noise(self, amplitude: np.ndarray, sigma_base: float = 0.005) -> np.ndarray:
        return sigma_base / np.maximum(amplitude, 1e-6)


class TSDFFusion:
    def __init__(
        self,
        grid_size: tuple[int, int, int] = (100, 100, 100),
        voxel_size: float = 0.02,
        trunc_dist: float = 0.06,
    ):
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.trunc_dist = trunc_dist
        self.tsdf = np.zeros(grid_size, dtype=np.float32)
        self.weight = np.zeros(grid_size, dtype=np.float32)
        self.w_max = 50.0

    def integrate(self, sdf_new: np.ndarray, w_new: np.ndarray) -> None:
        """Bayesian weighted average update."""
        mask = w_new > 0
        w_old = self.weight[mask]
        w_sum = w_old + w_new[mask]
        self.tsdf[mask] = (w_old * self.tsdf[mask] + w_new[mask] * sdf_new[mask]) / w_sum
        self.weight[mask] = np.minimum(w_sum, self.w_max)

    def extract_surface(self, level: float = 0.0) -> np.ndarray:
        """Return voxel centres where TSDF crosses the given level (simple zero-crossing)."""
        mask = (self.weight > 0) & (np.abs(self.tsdf) < self.trunc_dist * 0.5)
        indices = np.argwhere(mask)
        return indices.astype(np.float32) * self.voxel_size

    def integrate_depth_frame(
        self,
        depth_m: np.ndarray,
        K: np.ndarray,
        T_cw: np.ndarray,
    ) -> None:
        """Fuse a real depth frame (H×W float32 metres) into the TSDF volume.

        Voxel world positions are computed from the grid origin, then projected
        into the depth image to look up the measured depth and compute the SDF.
        """
        gx, gy, gz = self.grid_size
        xs = np.arange(gx) * self.voxel_size
        ys = np.arange(gy) * self.voxel_size
        zs = np.arange(gz) * self.voxel_size
        # World-space voxel centres
        X_w = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
        # Transform to camera frame: X_c = R * X_w + t
        R = T_cw[:3, :3]
        t = T_cw[:3, 3]
        X_c = (R @ X_w.T).T + t
        valid = X_c[:, 2] > 0
        if not np.any(valid):
            return
        # Project to image
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        H, W = depth_m.shape
        u = (fx * X_c[valid, 0] / X_c[valid, 2] + cx).astype(int)
        v = (fy * X_c[valid, 1] / X_c[valid, 2] + cy).astype(int)
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx_all = np.where(valid)[0]
        idx = idx_all[in_bounds]
        u_b = u[in_bounds]
        v_b = v[in_bounds]
        z_meas = depth_m[v_b, u_b].astype(np.float64)
        z_pred = X_c[idx, 2]
        sdf_vals = np.clip(z_meas - z_pred, -self.trunc_dist, self.trunc_dist)
        w_new_vals = np.where(z_meas > 0, 1.0, 0.0).astype(np.float32)
        sdf_full = np.zeros(gx * gy * gz, dtype=np.float32)
        w_full = np.zeros(gx * gy * gz, dtype=np.float32)
        sdf_full[idx] = sdf_vals.astype(np.float32)
        w_full[idx] = w_new_vals
        self.integrate(sdf_full.reshape(self.grid_size), w_full.reshape(self.grid_size))

    @staticmethod
    def generate_synthetic_frame(
        grid_size: tuple[int, int, int],
        voxel_size: float,
        trunc_dist: float,
        surface_z: float,
        noise_std: float = 0.005,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a synthetic SDF + weight volume for a planar surface at z=surface_z."""
        if rng is None:
            rng = np.random.default_rng()
        nz = grid_size[2]
        z_coords = np.arange(nz).astype(np.float32) * voxel_size
        sdf_1d = z_coords - surface_z + rng.normal(0, noise_std, nz)
        sdf = np.broadcast_to(sdf_1d[np.newaxis, np.newaxis, :], grid_size).copy()
        sdf = np.clip(sdf, -trunc_dist, trunc_dist)
        weight = np.ones(grid_size, dtype=np.float32)
        weight[np.abs(sdf) >= trunc_dist] = 0
        return sdf, weight
