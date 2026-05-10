"""Visualization functions for the Perception lecture notebook."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

if TYPE_CHECKING:
    from lib.depth import TSDFFusion
    from lib.imu import ImuDriftResult, ImuLiveDriftTracker
    from lib.lidar import BeamModel, OccupancyGrid
    from lib.live import DepthFrame, ServoSample
    from lib.sfm import BundleAdjustmentResult, SfmResult

# ---------------------------------------------------------------------------
# V0 / V18: Factor graph diagrams
# ---------------------------------------------------------------------------

def _draw_factor_graph(ax: plt.Axes, annotated: bool = False) -> None:
    poses = [(0.5 + i * 1.5, 3.0) for i in range(4)]
    landmarks = [(1.25 + i * 2.0, 0.5) for i in range(3)]

    for i, (x, y) in enumerate(poses):
        ax.add_patch(plt.Circle((x, y), 0.25, fc="#4C72B0", ec="k", zorder=3))
        ax.text(x, y, f"$x_{i}$", ha="center", va="center", color="w", fontsize=9, zorder=4)

    for i, (x, y) in enumerate(landmarks):
        ax.add_patch(plt.Circle((x, y), 0.2, fc="#55A868", ec="k", zorder=3))
        ax.text(x, y, f"$l_{i}$", ha="center", va="center", color="w", fontsize=8, zorder=4)

    factor_colors = {"odom": "#DD8452", "cam": "#C44E52", "lidar": "#8172B3", "imu": "#937860"}
    # odometry factors
    for i in range(len(poses) - 1):
        mx = (poses[i][0] + poses[i + 1][0]) / 2
        ax.plot([poses[i][0], poses[i + 1][0]], [poses[i][1], poses[i + 1][1]],
                "k-", lw=1, zorder=1)
        ax.add_patch(plt.Rectangle((mx - 0.1, 2.85), 0.2, 0.3,
                                   fc=factor_colors["odom"], ec="k", zorder=2))

    # camera factors
    connections = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)]
    for pi, li in connections:
        ax.plot([poses[pi][0], landmarks[li][0]],
                [poses[pi][1], landmarks[li][1]],
                "k-", lw=0.8, alpha=0.5, zorder=1)
        mx = (poses[pi][0] + landmarks[li][0]) / 2
        my = (poses[pi][1] + landmarks[li][1]) / 2
        ax.add_patch(plt.Rectangle((mx - 0.08, my - 0.08), 0.16, 0.16,
                                   fc=factor_colors["cam"], ec="k", zorder=2))

    legend_items = [
        mpatches.Patch(fc="#4C72B0", label="Pose node"),
        mpatches.Patch(fc="#55A868", label="Landmark node"),
        mpatches.Patch(fc=factor_colors["odom"], label="Odometry / IMU factor"),
        mpatches.Patch(fc=factor_colors["cam"], label="Camera factor"),
    ]

    if annotated:
        # LiDAR factors between consecutive poses
        for i in range(len(poses) - 1):
            mx = (poses[i][0] + poses[i + 1][0]) / 2
            ax.add_patch(plt.Rectangle((mx - 0.1, 3.25), 0.2, 0.3,
                                       fc=factor_colors["lidar"], ec="k", zorder=2))
        legend_items.append(mpatches.Patch(fc=factor_colors["lidar"], label="LiDAR scan factor"))
        legend_items.append(mpatches.Patch(fc=factor_colors["imu"], label="IMU preint. factor"))

    ax.legend(handles=legend_items, loc="upper right", fontsize=7, framealpha=0.9)
    ax.set_xlim(-0.5, 7)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_factor_graph_overview() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.set_title("Factor Graph — Sensor Factors Preview", fontsize=11)
    _draw_factor_graph(ax, annotated=False)
    plt.tight_layout()
    return fig


def plot_factor_graph_annotated() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.set_title("Factor Graph — All Sensor Factors", fontsize=11)
    _draw_factor_graph(ax, annotated=True)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# V1: Encoder quantization
# ---------------------------------------------------------------------------

def plot_encoder_quantization(
    t: np.ndarray,
    true_angle: np.ndarray,
    quantized_angle: np.ndarray,
    delta: float,
) -> plt.Figure:
    error = quantized_angle - true_angle
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))

    axes[0].plot(t, true_angle, label="true", lw=1)
    axes[0].plot(t, quantized_angle, label="quantized", lw=1, alpha=0.8)
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("angle [rad]")
    axes[0].set_title("Encoder Staircase")
    axes[0].legend(fontsize=8)

    axes[1].plot(t, error, lw=0.5, alpha=0.7)
    axes[1].axhline(delta / 2, ls="--", color="r", lw=0.8, label=f"±Δ/2 = ±{delta/2:.4f}")
    axes[1].axhline(-delta / 2, ls="--", color="r", lw=0.8)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("error [rad]")
    axes[1].set_title("Quantization Error")
    axes[1].legend(fontsize=8)

    axes[2].hist(error, bins=50, density=True, alpha=0.7)
    axes[2].axhline(1.0 / delta, color="r", ls="--", lw=1, label=f"Uniform 1/Δ = {1/delta:.1f}")
    axes[2].set_xlabel("error [rad]")
    axes[2].set_ylabel("density")
    axes[2].set_title("Error Distribution")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# V2: IMU drift
# ---------------------------------------------------------------------------

def plot_imu_drift(data: ImuDriftResult) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))

    for i, label in enumerate(["x", "y", "z"]):
        axes[0].plot(data.t, data.accel_bias[:, i], label=f"bias {label}", lw=0.8)
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("accel bias [m/s²]")
    axes[0].set_title("Accelerometer Bias Drift")
    axes[0].legend(fontsize=7)

    for i, label in enumerate(["x", "y", "z"]):
        axes[1].plot(data.t, data.velocity[:, i], label=label, lw=0.8)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("velocity [m/s]")
    axes[1].set_title("Integrated Velocity (should be 0)")
    axes[1].legend(fontsize=7)

    pos_norm = np.linalg.norm(data.position, axis=1)
    axes[2].plot(data.t, pos_norm, lw=1, color="#C44E52")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("position drift [m]")
    axes[2].set_title("Position Drift from Double Integration")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# V3: Interactive pinhole projection
# ---------------------------------------------------------------------------

def plot_projection_interactive() -> plt.Figure:
    f_vals = [200, 500, 1000]
    fig, axes = plt.subplots(1, len(f_vals), figsize=(14, 4))
    fig.suptitle("Pinhole Projection — Effect of Focal Length", fontsize=11)

    rng = np.random.default_rng(0)
    points_3d = rng.uniform(-1, 1, (30, 3))
    points_3d[:, 2] = rng.uniform(2, 6, 30)

    for ax, f in zip(axes, f_vals):
        K = np.array([[f, 0, 320], [0, f, 240], [0, 0, 1.0]])
        uv = np.column_stack([
            K[0, 0] * points_3d[:, 0] / points_3d[:, 2] + K[0, 2],
            K[1, 1] * points_3d[:, 1] / points_3d[:, 2] + K[1, 2],
        ])
        sc = ax.scatter(uv[:, 0], uv[:, 1], c=points_3d[:, 2], cmap="viridis",
                        s=30, edgecolors="k", linewidths=0.5)
        ax.set_xlim(0, 640)
        ax.set_ylim(480, 0)
        ax.set_aspect("equal")
        ax.set_title(f"f = {f} px")
        ax.set_xlabel("u [px]")
        if ax == axes[0]:
            ax.set_ylabel("v [px]")
    plt.colorbar(sc, ax=axes[-1], label="Z [m]", shrink=0.8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# V4: Distortion grid
# ---------------------------------------------------------------------------

def plot_distortion_grid() -> plt.Figure:
    from lib.camera import apply_distortion

    configs = [
        {"k1": 0, "k2": 0, "label": "No distortion"},
        {"k1": -0.3, "k2": 0.1, "label": "Barrel (k₁=-0.3)"},
        {"k1": 0.3, "k2": -0.1, "label": "Pincushion (k₁=0.3)"},
        {"k1": -0.5, "k2": 0.3, "label": "Mustache (k₁=-0.5, k₂=0.3)"},
        {"p1": 0.08, "p2": -0.04, "label": "Tangential (p₁,p₂)"},
    ]
    gx, gy = np.meshgrid(np.linspace(-1, 1, 15), np.linspace(-1, 1, 15))
    gx_flat, gy_flat = gx.ravel(), gy.ravel()

    fig, axes = plt.subplots(1, len(configs), figsize=(14, 3.5))
    fig.suptitle("Brown-Conrady Distortion Examples", fontsize=11)
    for ax, cfg in zip(axes, configs):
        xd, yd = apply_distortion(
            gx_flat,
            gy_flat,
            k1=cfg.get("k1", 0.0),
            k2=cfg.get("k2", 0.0),
            p1=cfg.get("p1", 0.0),
            p2=cfg.get("p2", 0.0),
        )
        xd = xd.reshape(gx.shape)
        yd = yd.reshape(gy.shape)
        for i in range(xd.shape[0]):
            ax.plot(xd[i], yd[i], "b-", lw=0.6)
        for j in range(xd.shape[1]):
            ax.plot(xd[:, j], yd[:, j], "b-", lw=0.6)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.set_title(cfg["label"], fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# V5: Calibration result
# ---------------------------------------------------------------------------

def plot_calibration_result(
    image_paths: list,
    K: np.ndarray,
    dist: np.ndarray,
    rvecs: list,
    tvecs: list,
    reproj_err: float,
    calib_data: dict[str, object] | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    per_view_rms = None if calib_data is None else calib_data.get("per_view_rms")
    obj_points = None if calib_data is None else calib_data.get("obj_points")
    img_points = None if calib_data is None else calib_data.get("img_points")

    if per_view_rms is None:
        per_view_rms = [reproj_err for _ in rvecs]
    axes[0].bar(range(len(per_view_rms)), per_view_rms, color="#4C72B0", alpha=0.7)
    axes[0].set_xlabel("View index")
    axes[0].set_ylabel("RMS reproj. error [px]")
    axes[0].set_title(f"Per-view reprojection error (mean = {reproj_err:.3f} px)")

    residuals = []
    if obj_points is not None and img_points is not None:
        for obj, img, rvec, tvec in zip(obj_points, img_points, rvecs, tvecs):
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
            residual = img.reshape(-1, 2) - proj.reshape(-1, 2)
            residuals.append(residual)
    if residuals:
        residual = np.concatenate(residuals, axis=0)
        axes[1].scatter(residual[:, 0], residual[:, 1], s=6, alpha=0.4)
        axes[1].axhline(0.0, color="k", lw=0.8, alpha=0.6)
        axes[1].axvline(0.0, color="k", lw=0.8, alpha=0.6)
        axes[1].set_xlabel("Δu [px]")
        axes[1].set_ylabel("Δv [px]")
        axes[1].set_title("Reprojection residuals")
        axes[1].set_aspect("equal")
    else:
        coeffs = dist.ravel()
        names = [f"$d_{i}$" for i in range(len(coeffs))]
        axes[1].bar(names, coeffs, color="#55A868", alpha=0.7)
        axes[1].set_title("Estimated distortion coefficients")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# V6: ArUco detection
# ---------------------------------------------------------------------------

def plot_aruco_detection(
    img: np.ndarray,
    corners: list[np.ndarray],
    ids: np.ndarray | None,
    poses: list | None = None,
    K: np.ndarray | None = None,
    dist: np.ndarray | None = None,
) -> plt.Figure:
    vis = img.copy()
    if img.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    if corners and ids is not None:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        if poses is not None and K is not None:
            if dist is None:
                dist = np.zeros(5)
            for rvec, tvec in poses:
                cv2.drawFrameAxes(vis, K, dist, rvec, tvec, 0.03)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    n_det = 0 if ids is None else len(ids)
    ax.set_title(f"ArUco Detection — {n_det} markers found")
    ax.axis("off")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# V7: Harris steps
# ---------------------------------------------------------------------------

def plot_harris_steps(
    img: np.ndarray,
    response: np.ndarray,
    eigenvalues: np.ndarray,
) -> plt.Figure:
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 3, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img, cmap="gray")
    ax0.set_title("Input image")
    ax0.axis("off")

    grad_mag = np.sqrt(
        cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)**2 +
        cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)**2
    )
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(grad_mag, cmap="hot")
    ax1.set_title("Gradient magnitude")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 2])
    lam1 = eigenvalues[:, :, 0].ravel()
    lam2 = eigenvalues[:, :, 1].ravel()
    subsample = np.random.default_rng(0).choice(len(lam1), min(5000, len(lam1)), replace=False)
    ax2.scatter(lam1[subsample], lam2[subsample], s=1, alpha=0.3)
    ax2.set_xlabel("λ₁")
    ax2.set_ylabel("λ₂")
    ax2.set_title("Eigenvalue scatter")
    ax2.set_aspect("equal")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(response, cmap="jet")
    ax3.set_title("Cornerness R")
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, 1:])
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    thresh = response.max() * 0.01
    dilated = cv2.dilate(response, np.ones((5, 5), np.float32))
    mask = (response >= dilated) & (response > thresh)
    corners_y, corners_x = np.where(mask)
    ax4.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB) if vis.ndim == 3 else vis, cmap="gray")
    ax4.scatter(corners_x, corners_y, s=3, c="red", alpha=0.6)
    ax4.set_title(f"Detected corners after NMS ({len(corners_x)})")
    ax4.axis("off")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# V8: Scale space + keypoints
# ---------------------------------------------------------------------------

def plot_scale_space_keypoints(
    img: np.ndarray, keypoints: list[cv2.KeyPoint],
) -> plt.Figure:
    vis = img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    cv2.drawKeypoints(vis, keypoints, vis,
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.set_title(f"ORB Keypoints with Scale ({len(keypoints)} features)")
    ax.axis("off")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# V9: Feature matching
# ---------------------------------------------------------------------------

def plot_feature_matching(
    img1: np.ndarray,
    kp1: list[cv2.KeyPoint],
    img2: np.ndarray,
    kp2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    inlier_mask: np.ndarray,
) -> plt.Figure:
    n_inliers = int(inlier_mask.sum()) if len(inlier_mask) > 0 else 0
    n_outliers = len(inlier_mask) - n_inliers

    vis_in = cv2.drawMatches(
        img1, kp1, img2, kp2,
        [m for m, ok in zip(matches, inlier_mask) if ok],
        None, matchColor=(0, 200, 0), flags=2,
    )
    vis_out = cv2.drawMatches(
        img1, kp1, img2, kp2,
        [m for m, ok in zip(matches, inlier_mask) if not ok],
        None, matchColor=(200, 0, 0), flags=2,
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].imshow(cv2.cvtColor(vis_in, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Inlier matches ({n_inliers})", fontsize=10)
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(vis_out, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Outlier matches ({n_outliers})", fontsize=10)
    axes[1].axis("off")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# V10a: Epipolar geometry — 3-D scene diagram
# ---------------------------------------------------------------------------

def _camera_frustum_lines(center: np.ndarray, R_cw: np.ndarray, f: float, hw: float):
    """Return (N,2,3) segment array for a camera frustum in world coords."""
    corners_cam = np.array([
        [-hw, -hw, f], [hw, -hw, f], [hw, hw, f], [-hw, hw, f],
    ])
    # camera-to-world rotation = R_cw.T
    Rwc = R_cw.T
    corners_w = corners_cam @ Rwc.T + center
    segments = []
    # pyramid edges
    for c in corners_w:
        segments.append([center, c])
    # image-plane rectangle
    for i in range(4):
        segments.append([corners_w[i], corners_w[(i + 1) % 4]])
    return np.array(segments)


def plot_epipolar_geometry_3d(
    R: np.ndarray | None = None,
    t: np.ndarray | None = None,
    points_3d: np.ndarray | None = None,
    f: float = 0.6,
    hw: float = 0.4,
) -> plt.Figure:
    """3-D epipolar geometry diagram.

    Camera 1 is at the world origin looking along +Z.
    Camera 2 is placed at C2 = -R.T @ t with orientation R (world-to-cam).
    Shows frustums, 3-D points, epipolar planes, and projected rays on each
    image plane.
    """
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    if R is None:
        angle = np.deg2rad(20)
        R = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ])
    if t is None:
        t = np.array([0.8, 0.0, 0.2])
    if points_3d is None:
        points_3d = np.array([
            [0.0,  0.2, 2.5],
            [0.5, -0.3, 3.0],
            [-0.4, 0.4, 2.0],
            [0.3,  0.1, 3.5],
        ])

    C1 = np.zeros(3)
    C2 = -R.T @ t          # world-frame position of camera 2
    R1 = np.eye(3)          # world-to-cam1
    R2 = R                  # world-to-cam2

    colors = plt.cm.tab10(np.linspace(0, 0.5, len(points_3d)))

    fig = plt.figure(figsize=(13, 6))
    ax3d = fig.add_subplot(121, projection="3d")
    ax1  = fig.add_subplot(222)
    ax2  = fig.add_subplot(224)

    # --- frustums ---
    for center, Rcw, label, col in [(C1, R1, "Cam 1", "#4C72B0"),
                                     (C2, R2, "Cam 2", "#DD8452")]:
        segs = _camera_frustum_lines(center, Rcw, f, hw)
        lc = Line3DCollection(segs, colors=col, linewidths=1.0, alpha=0.7)
        ax3d.add_collection3d(lc)
        ax3d.scatter(*center, color=col, s=50, zorder=5)
        ax3d.text(*center, f"  {label}", fontsize=8, color=col)

    # --- baseline ---
    ax3d.plot(*zip(C1, C2), "k--", lw=1, alpha=0.5)

    # --- image-plane helpers ---
    def _image_plane_origin(center, Rcw):
        return center + Rcw.T @ np.array([0., 0., f])

    def _project_to_image(X_w, center, Rcw):
        X_c = Rcw @ (X_w - center)
        if X_c[2] <= 0:
            return None
        return X_c[:2] / X_c[2]   # normalised coords

    def _epipole(other_center, center, Rcw):
        return _project_to_image(other_center, center, Rcw)

    e1 = _epipole(C2, C1, R1)
    e2 = _epipole(C1, C2, R2)

    # image-plane axis limits
    lim = hw * 1.1
    for ax, ep, title in [(ax1, e1, "Image 1"), (ax2, e2, "Image 2")]:
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal"); ax.set_title(title, fontsize=9)
        ax.axhline(0, color="gray", lw=0.5, alpha=0.4)
        ax.axvline(0, color="gray", lw=0.5, alpha=0.4)
        ax.add_patch(plt.Rectangle((-hw, -hw), 2*hw, 2*hw,
                                   fc="none", ec="gray", lw=1.5, ls="--"))
        ax.tick_params(labelsize=7)
        if ep is not None:
            ax.scatter(*ep, marker="x", s=60, color="k", linewidths=1.5, zorder=5)
            ax.text(ep[0]+0.02, ep[1]+0.02, "e", fontsize=8)

    # --- per-point geometry ---
    for X, col in zip(points_3d, colors):
        c = tuple(col[:3])

        # 3D point
        ax3d.scatter(*X, color=c, s=40, zorder=5)

        # epipolar plane (triangle C1–X–C2)
        verts = [[C1, X, C2]]
        poly = Poly3DCollection(verts, alpha=0.08, facecolor=c, edgecolor=c, linewidths=0.5)
        ax3d.add_collection3d(poly)

        # rays from camera centers through X (extended slightly beyond X)
        far = X + 0.3 * (X - C1)
        ax3d.plot(*zip(C1, far), color=c, lw=1.2, alpha=0.7)
        far2 = X + 0.3 * (X - C2)
        ax3d.plot(*zip(C2, far2), color=c, lw=1.2, alpha=0.7, ls="--")

        # epipolar lines in image planes
        p1 = _project_to_image(X, C1, R1)
        p2 = _project_to_image(X, C2, R2)

        for ax, pt, ep in [(ax1, p1, e1), (ax2, p2, e2)]:
            if pt is None:
                continue
            ax.scatter(*pt, color=c, s=25, zorder=4)
            if ep is not None:
                # epipolar line: through pt and ep
                direction = pt - ep
                norm = np.linalg.norm(direction) + 1e-12
                t_vals = np.linspace(-2, 2, 2)
                line_pts = np.outer(t_vals, direction / norm) + pt
                ax.plot(line_pts[:, 0], line_pts[:, 1], color=c, lw=1.0, alpha=0.7)

    ax3d.set_xlabel("X", fontsize=8); ax3d.set_ylabel("Y", fontsize=8)
    ax3d.set_zlabel("Z", fontsize=8)
    ax3d.set_title("Epipolar geometry — 3-D", fontsize=9)
    ax3d.tick_params(labelsize=7)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# V10b: Epipolar lines
# ---------------------------------------------------------------------------

def plot_epipolar_lines(
    imgL: np.ndarray,
    imgR: np.ndarray,
    kpL: list[cv2.KeyPoint],
    kpR: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    inlier_mask: np.ndarray,
    n_lines: int = 15,
) -> plt.Figure:
    if len(matches) < 8:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Not enough matches for F-matrix", ha="center", va="center")
        return fig

    inlier_matches = [m for m, ok in zip(matches, inlier_mask) if ok]
    pts1 = np.float32([kpL[m.queryIdx].pt for m in inlier_matches])
    pts2 = np.float32([kpR[m.trainIdx].pt for m in inlier_matches])

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    if F is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "F-matrix estimation failed", ha="center", va="center")
        return fig

    idx = np.random.default_rng(0).choice(len(pts1), min(n_lines, len(pts1)), replace=False)

    def _draw_lines(img, lines, pts):
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
        h, w = vis.shape[:2]
        colors = plt.cm.tab10(np.linspace(0, 1, len(lines)))
        for l, pt, c in zip(lines, pts, colors):
            c_bgr = (int(c[2] * 255), int(c[1] * 255), int(c[0] * 255))
            x0, y0 = 0, int(-l[2] / (l[1] + 1e-12))
            x1, y1 = w, int(-(l[2] + l[0] * w) / (l[1] + 1e-12))
            cv2.line(vis, (x0, y0), (x1, y1), c_bgr, 1)
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 5, c_bgr, -1)
        return vis

    lines_R = cv2.computeCorrespondEpilines(pts1[idx].reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    lines_L = cv2.computeCorrespondEpilines(pts2[idx].reshape(-1, 1, 2), 2, F).reshape(-1, 3)

    visL = _draw_lines(imgL, lines_L, pts1[idx])
    visR = _draw_lines(imgR, lines_R, pts2[idx])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(cv2.cvtColor(visL, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Left image + epipolar lines")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(visR, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Right image + epipolar lines")
    axes[1].axis("off")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# V11: Stereo pipeline
# ---------------------------------------------------------------------------

def plot_stereo_pipeline(
    imgL: np.ndarray,
    imgR: np.ndarray,
    imgL_rect: np.ndarray,
    imgR_rect: np.ndarray,
    disparity: np.ndarray,
    depth: np.ndarray,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].imshow(imgL, cmap="gray" if imgL.ndim == 2 else None)
    axes[0, 0].set_title("Left (raw)")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(imgR, cmap="gray" if imgR.ndim == 2 else None)
    axes[0, 1].set_title("Right (raw)")
    axes[0, 1].axis("off")

    vis_rect_left = imgL_rect.copy()
    vis_rect_right = imgR_rect.copy()
    if vis_rect_left.ndim == 2:
        vis_rect_left = cv2.cvtColor(vis_rect_left, cv2.COLOR_GRAY2BGR)
    if vis_rect_right.ndim == 2:
        vis_rect_right = cv2.cvtColor(vis_rect_right, cv2.COLOR_GRAY2BGR)
    for y in range(0, vis_rect_left.shape[0], 30):
        cv2.line(vis_rect_left, (0, y), (vis_rect_left.shape[1], y), (0, 255, 0), 1)
        cv2.line(vis_rect_right, (0, y), (vis_rect_right.shape[1], y), (0, 255, 0), 1)
    stereo_strip = np.hstack([vis_rect_left, vis_rect_right])
    axes[0, 2].imshow(cv2.cvtColor(stereo_strip, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("Rectified pair + horizontal epipolar lines")
    axes[0, 2].axis("off")

    disp_valid = disparity.copy()
    disp_valid[disp_valid <= 0] = np.nan
    im1 = axes[1, 0].imshow(disp_valid, cmap="magma")
    axes[1, 0].set_title("Disparity")
    axes[1, 0].axis("off")
    plt.colorbar(im1, ax=axes[1, 0], shrink=0.7)

    depth_clip = np.clip(depth, 0, np.nanpercentile(depth[np.isfinite(depth)], 95) if np.any(np.isfinite(depth)) else 10)
    im2 = axes[1, 1].imshow(depth_clip, cmap="viridis_r")
    axes[1, 1].set_title("Depth [m]")
    axes[1, 1].axis("off")
    plt.colorbar(im2, ax=axes[1, 1], shrink=0.7)

    axes[1, 2].axis("off")
    axes[1, 2].text(0.5, 0.5, "3D point cloud\n(use Open3D for\ninteractive view)",
                    ha="center", va="center", fontsize=10, style="italic", alpha=0.5)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# V11b / V14a: Depth noise curves
# ---------------------------------------------------------------------------

def plot_depth_noise_curves(
    f: float = 500, baselines: list[float] | None = None, sigma_d: float = 0.5,
) -> plt.Figure:
    if baselines is None:
        baselines = [0.06, 0.12, 0.25]
    Z = np.linspace(0.5, 20, 200)
    fig, ax = plt.subplots(figsize=(8, 4))
    for B in baselines:
        sigma_Z = Z**2 / (f * B) * sigma_d
        ax.plot(Z, sigma_Z, label=f"B = {B:.2f} m")
    ax.set_xlabel("Depth Z [m]")
    ax.set_ylabel("σ_Z [m]")
    ax.set_title(f"Stereo Depth Uncertainty (f={f} px, σ_d={sigma_d} px)")
    ax.legend()
    ax.set_ylim(0, min(5, ax.get_ylim()[1]))
    plt.tight_layout()
    return fig


def plot_depth_comparison(
    f: float = 500,
    baselines: list[float] | None = None,
    sigma_d: float = 0.5,
    tof_sigma: float = 0.01,
    tof_max: float = 10.0,
    lidar_sigma: float = 0.02,
    lidar_max: float = 100.0,
) -> plt.Figure:
    if baselines is None:
        baselines = [0.12]
    Z = np.linspace(0.5, 30, 300)
    fig, ax = plt.subplots(figsize=(10, 5))
    for B in baselines:
        sigma_Z = Z**2 / (f * B) * sigma_d
        ax.plot(Z, sigma_Z, label=f"Stereo (B={B:.2f}m)")

    sigma_tof = np.full_like(Z, tof_sigma)
    sigma_tof[Z > tof_max] = np.inf
    ax.plot(Z, np.clip(sigma_tof, 0, 5), "--", label=f"ToF (σ={tof_sigma}m, max={tof_max}m)")

    sigma_lidar = np.full_like(Z, lidar_sigma)
    sigma_lidar[Z > lidar_max] = np.inf
    ax.plot(Z, np.clip(sigma_lidar, 0, 5), ":", label=f"LiDAR (σ={lidar_sigma}m)")

    ax.set_xlabel("Depth Z [m]")
    ax.set_ylabel("σ_Z [m]")
    ax.set_title("Depth Uncertainty Comparison Across Modalities")
    ax.legend()
    ax.set_ylim(0, 2)
    ax.set_xlim(0, 30)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# V12 / V13: SfM + BA
# ---------------------------------------------------------------------------

def plot_sfm_reconstruction(result: SfmResult) -> plt.Figure:
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    pts = result.points_3d
    mask = np.all(np.isfinite(pts), axis=1) & (np.abs(pts) < 50).all(axis=1)
    pts = pts[mask]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, alpha=0.5)
    ax.scatter(0, 0, 0, c="r", s=60, marker="^", label="Camera 1")
    cam2_pos = -result.R.T @ result.t
    ax.scatter(*cam2_pos, c="g", s=60, marker="^", label="Camera 2")
    ax.set_xlabel("X [arb.]")
    ax.set_ylabel("Y [arb.]")
    ax.set_zlabel("Z [arb.]")
    ax.set_title(f"Two-View SfM at arbitrary scale ({len(pts)} points)")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_ba_sparsity(result: BundleAdjustmentResult) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].spy(np.abs(result.hessian) > 1e-10, markersize=0.5, aspect="auto")
    axes[0].set_title("Full Hessian H (poses + landmarks)")
    axes[0].set_xlabel("variable index")
    axes[0].set_ylabel("variable index")

    axes[1].spy(np.abs(result.hessian_reduced) > 1e-10, markersize=1, aspect="auto")
    axes[1].set_title("Reduced system after Schur complement (poses only)")
    axes[1].set_xlabel("pose param index")
    axes[1].set_ylabel("pose param index")

    plt.tight_layout()
    return fig


def plot_ba_convergence(result: BundleAdjustmentResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(result.errors_per_iter) <= 2:
        ax.bar(["initial", "optimized"], result.errors_per_iter[:2], color=["#C44E52", "#55A868"])
        ax.set_xlabel("Solve stage")
        ax.set_title("Bundle Adjustment Error Before/After Optimization")
    else:
        ax.plot(result.errors_per_iter, "o-", markersize=4)
        ax.set_xlabel("Outer solve")
        ax.set_title("Bundle Adjustment Error Across Outer Solves")
    ax.set_ylabel("RMS reprojection error [px]")
    ax.set_yscale("log")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# V14b: TSDF evolution
# ---------------------------------------------------------------------------

def plot_tsdf_evolution(
    tsdf: TSDFFusion, n_frames: int = 10,
    rng: np.random.Generator | None = None,
) -> plt.Figure:
    from lib.depth import TSDFFusion as _TF

    if rng is None:
        rng = np.random.default_rng()

    fig, axes = plt.subplots(1, min(n_frames, 4), figsize=(14, 3.5))
    if not hasattr(axes, "__len__"):
        axes = [axes]

    show_indices = np.linspace(0, n_frames - 1, len(axes), dtype=int)

    for i in range(n_frames):
        surface_z = tsdf.grid_size[2] * tsdf.voxel_size * 0.5
        sdf, w = _TF.generate_synthetic_frame(
            tsdf.grid_size, tsdf.voxel_size, tsdf.trunc_dist,
            surface_z=surface_z, noise_std=0.01, rng=rng,
        )
        tsdf.integrate(sdf, w)

        if i in show_indices:
            ax_idx = np.where(show_indices == i)[0][0]
            mid_slice = tsdf.tsdf[:, :, tsdf.grid_size[2] // 2]
            axes[ax_idx].imshow(mid_slice.T, cmap="RdBu", vmin=-tsdf.trunc_dist, vmax=tsdf.trunc_dist,
                                origin="lower")
            axes[ax_idx].set_title(f"Frame {i + 1}")
            axes[ax_idx].axis("off")

    fig.suptitle("TSDF Fusion — Cross-Section Evolution", fontsize=11)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# V15: Beam model
# ---------------------------------------------------------------------------

def plot_beam_model_interactive(
    beam: BeamModel, z_true: float = 5.0,
) -> plt.Figure:
    z = np.linspace(0, beam.z_max, 500)
    components = beam.pdf_components(z, z_true)
    mixture = beam.pdf(z, z_true)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    for name, p in components.items():
        axes[0].plot(z, p, label=name, alpha=0.7)
    axes[0].axvline(z_true, ls="--", color="k", lw=0.8, label=f"z* = {z_true}")
    axes[0].set_xlabel("z [m]")
    axes[0].set_ylabel("p(z)")
    axes[0].set_title("Individual Components")
    axes[0].legend(fontsize=8)

    axes[1].fill_between(z, mixture, alpha=0.3, color="#4C72B0")
    axes[1].plot(z, mixture, color="#4C72B0", lw=1.5)
    axes[1].axvline(z_true, ls="--", color="k", lw=0.8)
    axes[1].set_xlabel("z [m]")
    axes[1].set_ylabel("p(z)")
    axes[1].set_title(
        f"Mixture (α_hit={beam.alpha_hit}, α_short={beam.alpha_short}, "
        f"α_max={beam.alpha_max}, α_rand={beam.alpha_rand})"
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# V16: Occupancy grid evolution
# ---------------------------------------------------------------------------

def plot_occupancy_grid_evolution(
    grid: OccupancyGrid,
    scans: list[dict],
    n_snapshots: int = 4,
) -> plt.Figure:
    indices = np.linspace(0, len(scans) - 1, n_snapshots, dtype=int)
    fig, axes = plt.subplots(1, n_snapshots, figsize=(14, 3.5))
    fig.suptitle("Occupancy Grid — Map Build-Up", fontsize=11)

    for i, scan in enumerate(scans):
        grid.update(scan["pos"], scan["endpoints"])
        if i in indices:
            ax_idx = np.where(indices == i)[0][0]
            axes[ax_idx].imshow(
                grid.probability, cmap="gray_r", vmin=0, vmax=1,
                origin="lower",
                extent=[grid.origin[0], grid.origin[0] + grid.size[0] * grid.resolution,
                        grid.origin[1], grid.origin[1] + grid.size[1] * grid.resolution],
            )
            axes[ax_idx].plot(scan["pos"][0], scan["pos"][1], "r^", markersize=6)
            axes[ax_idx].set_title(f"After scan {i + 1}")
            axes[ax_idx].set_aspect("equal")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# V17: ICP convergence
# ---------------------------------------------------------------------------

def plot_icp_convergence(
    source: np.ndarray,
    target: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    errors: list[float],
) -> plt.Figure:
    aligned = (R @ source.T).T + t

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].scatter(target[:, 0], target[:, 1], s=5, alpha=0.5, label="target", c="#55A868")
    axes[0].scatter(source[:, 0], source[:, 1], s=5, alpha=0.3, label="source (before)", c="#C44E52")
    axes[0].scatter(aligned[:, 0], aligned[:, 1], s=5, alpha=0.5, label="source (aligned)", c="#4C72B0")
    axes[0].legend(fontsize=8)
    axes[0].set_aspect("equal")
    axes[0].set_title("ICP Alignment")

    axes[1].plot(errors, "o-", markersize=3)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("RMS error")
    axes[1].set_title("ICP Convergence")
    axes[1].set_yscale("log")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# V19: Failure mode gallery
# ---------------------------------------------------------------------------

def plot_failure_gallery() -> plt.Figure:
    scenarios = [
        ("Low texture", "Feature matching fails\n→ Camera SLAM degrades"),
        ("Glass / mirrors", "LiDAR pass-through\n→ Phantom points"),
        ("Fast rotation", "Motion blur + IMU saturation"),
        ("Forward motion", "Degenerate epipolar geometry\n→ Depth collapses"),
        ("Long tunnel", "No loop closure\n→ Drift accumulates"),
        ("Dynamic objects", "Map corruption\n→ Wrong associations"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    fig.suptitle("Failure Mode Gallery", fontsize=12)
    colors = ["#C44E52", "#DD8452", "#937860", "#8172B3", "#4C72B0", "#55A868"]

    for ax, (title, desc), color in zip(axes.ravel(), scenarios, colors):
        ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, transform=ax.transAxes,
                                   fc=color, alpha=0.15, ec=color, lw=2))
        ax.text(0.5, 0.65, title, ha="center", va="center", fontsize=11,
                fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.35, desc, ha="center", va="center", fontsize=9,
                style="italic", transform=ax.transAxes)
        ax.axis("off")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Live demo visualization helpers
# ---------------------------------------------------------------------------

class LivePlot:
    """Manages a matplotlib figure that updates in-place in a Jupyter notebook.

    Uses IPython.display.DisplayHandle.update() so the figure is replaced
    without scrolling. Works with %matplotlib inline.
    """

    def __init__(
        self,
        fig: plt.Figure,
        update_fn: Callable[[plt.Figure, Any], None],
    ):
        self.fig = fig
        self.update_fn = update_fn
        self._handle = None

    def show(self) -> None:
        try:
            from IPython.display import display
            self._handle = display(self.fig, display_id=True)
        except Exception:
            plt.show()

    def update(self, data: Any) -> None:
        self.update_fn(self.fig, data)
        if self._handle is not None:
            try:
                self._handle.update(self.fig)
            except Exception:
                pass

    def close(self) -> None:
        plt.close(self.fig)


# ---------------------------------------------------------------------------
# V-live-1: Encoder / servo live plot
# ---------------------------------------------------------------------------

def _update_encoder_live(fig: plt.Figure, samples: list) -> None:
    if not samples:
        return
    ticks = np.array([s.position_ticks for s in samples])
    rads = np.array([s.position_rad for s in samples])
    ts = np.array([s.timestamp for s in samples])
    delta = 2 * np.pi / 4096
    error = rads - (np.round(rads / delta) * delta)

    axes = fig.axes
    for ax in axes:
        ax.cla()

    axes[0].plot(ts, np.degrees(rads), "b-", lw=1.5, label="position (deg)")
    axes[0].set_ylabel("Angle (°)")
    axes[0].set_title("Live Encoder Position")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(ts, ticks, "g-", lw=1)
    axes[1].set_ylabel("Ticks (0–4095)")
    axes[1].set_title("Raw Ticks")

    axes[2].plot(ts, np.degrees(error), "r-", lw=1)
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_ylabel("Quantization error (°)")
    axes[2].set_title("Error")
    axes[2].set_xlabel("Time (s)")

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()


def create_live_encoder_plot() -> LivePlot:
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    fig.suptitle("Live Servo Encoder", fontsize=11)
    return LivePlot(fig, _update_encoder_live)


# ---------------------------------------------------------------------------
# V-live-2: IMU drift live plot
# ---------------------------------------------------------------------------

def _update_imu_drift_live(fig: plt.Figure, tracker: Any) -> None:
    if not tracker.timestamps:
        return
    ts = np.array(tracker.timestamps)
    accels = np.array(tracker.accels) if tracker.accels else np.zeros((1, 3))
    gyros = np.array(tracker.gyros) if tracker.gyros else np.zeros((1, 3))
    positions = np.array(tracker.positions) if tracker.positions else np.zeros((1, 3))

    axes = fig.axes
    for ax in axes:
        ax.cla()

    axes[0].plot(ts, accels[:, 0], label="ax", lw=1)
    axes[0].plot(ts, accels[:, 1], label="ay", lw=1)
    axes[0].plot(ts, accels[:, 2], label="az", lw=1)
    axes[0].set_ylabel("Accel (m/s²)")
    axes[0].set_title("Live IMU — Accelerometer")
    axes[0].legend(loc="upper right", fontsize=7)

    axes[1].plot(ts, gyros[:, 0], label="gx", lw=1)
    axes[1].plot(ts, gyros[:, 1], label="gy", lw=1)
    axes[1].plot(ts, gyros[:, 2], label="gz", lw=1)
    axes[1].set_ylabel("Gyro (rad/s)")
    axes[1].set_title("Live IMU — Gyroscope")
    axes[1].legend(loc="upper right", fontsize=7)

    axes[2].plot(ts, positions[:, 0], label="x", lw=1.5)
    axes[2].plot(ts, positions[:, 1], label="y", lw=1.5)
    axes[2].plot(ts, positions[:, 2], label="z", lw=1.5)
    axes[2].set_ylabel("Position (m)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Integrated Position (drift)")
    axes[2].legend(loc="upper right", fontsize=7)

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()


def create_live_imu_drift_plot() -> LivePlot:
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    fig.suptitle("Live IMU Drift", fontsize=11)
    return LivePlot(fig, _update_imu_drift_live)


# ---------------------------------------------------------------------------
# V-live-3: Camera / fisheye side-by-side live plot
# ---------------------------------------------------------------------------

def _update_camera_live(fig: plt.Figure, data: tuple) -> None:
    raw, undistorted = data
    axes = fig.axes
    for ax in axes:
        ax.cla()
        ax.axis("off")

    if raw is not None:
        axes[0].imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Raw fisheye")

    if undistorted is not None:
        axes[1].imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Undistorted")
    else:
        axes[1].text(0.5, 0.5, "Calibrating…", ha="center", va="center",
                     transform=axes[1].transAxes)

    fig.tight_layout()


def create_live_camera_plot() -> LivePlot:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Fisheye Calibration Live Preview", fontsize=11)
    return LivePlot(fig, _update_camera_live)


# ---------------------------------------------------------------------------
# V-live-4: ArUco detection live plot
# ---------------------------------------------------------------------------

def _update_aruco_live(fig: plt.Figure, frame: Any) -> None:
    ax = fig.axes[0]
    ax.cla()
    ax.axis("off")
    if frame is not None:
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.set_title("Live ArUco Detection")
    fig.tight_layout()


def create_live_aruco_plot() -> LivePlot:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.axis("off")
    fig.suptitle("Live ArUco Detection", fontsize=11)
    return LivePlot(fig, _update_aruco_live)


# ---------------------------------------------------------------------------
# V-live-5: iPhone depth point cloud live plot
# ---------------------------------------------------------------------------

def _update_depth_live(fig: plt.Figure, data: Any) -> None:
    axes = fig.axes
    for ax in axes:
        ax.cla()

    if data is None:
        return

    if hasattr(data, "depth_m"):
        depth = data.depth_m
    else:
        depth = np.asarray(data)

    axes[0].imshow(depth, cmap="plasma", vmin=0, vmax=5.0)
    axes[0].set_title("Depth map (m)")
    axes[0].axis("off")

    valid = depth[(depth > 0) & (depth < 5.0)]
    if valid.size > 0:
        axes[1].hist(valid.ravel(), bins=60, color="#4C72B0", edgecolor="none")
        axes[1].set_xlabel("Depth (m)")
        axes[1].set_ylabel("Pixel count")
        axes[1].set_title("Depth histogram")
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout()


def create_live_depth_plot() -> LivePlot:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Live iPhone LiDAR Depth", fontsize=11)
    return LivePlot(fig, _update_depth_live)
