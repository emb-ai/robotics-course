from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse, FancyArrowPatch, Arc
from typing import Optional


COLORS = {
    "dubins": "#2176AE",
    "ackermann": "#E07A5F",
    "diffdrive": "#3D9970",
    "path_L": "#D62828",
    "path_R": "#003049",
    "path_S": "#F77F00",
    "particle": "#457B9D",
    "estimate": "#E63946",
    "truth": "#2A9D8F",
    "measurement": "#264653",
    "kf_pred": "#E9C46A",
    "kf_upd": "#2176AE",
}


# ── kinematic trajectory comparison ──────────────────────────────────

def plot_kinematic_comparison(
    trajectories: dict[str, np.ndarray],
    titles: dict[str, str] | None = None,
    figsize: tuple = (14, 4),
) -> plt.Figure:
    n = len(trajectories)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    color_map = {"dubins": COLORS["dubins"], "ackermann": COLORS["ackermann"],
                 "diffdrive": COLORS["diffdrive"]}
    for ax, (key, trajs) in zip(axes, trajectories.items()):
        color = color_map.get(key, "steelblue")
        for label, traj in trajs.items():
            ax.plot(traj[:, 0], traj[:, 1], linewidth=1.8, label=label, alpha=0.85)
            _draw_heading(ax, traj[-1], color="gray", length=0.3)
        _draw_heading(ax, trajs[list(trajs.keys())[0]][0], color="black", length=0.4)
        ax.set_xlabel("$x$ [m]")
        ax.set_ylabel("$y$ [m]")
        ax.set_title(titles.get(key, key) if titles else key)
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def _draw_heading(ax, state, color="black", length=0.3):
    x, y, th = state
    ax.annotate("", xy=(x + length * np.cos(th), y + length * np.sin(th)),
                xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))


# ── Dubins curves ────────────────────────────────────────────────────

def plot_dubins_all_paths(
    solver,
    start: np.ndarray,
    goal: np.ndarray,
    figsize: tuple = (15, 8),
) -> plt.Figure:
    paths = solver.all_paths(start, goal)
    shortest = solver.shortest(start, goal)
    n = len(paths)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_2d(axes)
    for idx, (ax_flat, dp) in enumerate(zip(axes.ravel(), paths)):
        pts = solver.sample_path(start, dp, n_points=300)
        is_shortest = dp.path_type == shortest.path_type
        lw = 2.5 if is_shortest else 1.5
        ax_flat.plot(pts[:, 0], pts[:, 1], linewidth=lw,
                     color="#D62828" if is_shortest else "#457B9D")
        _draw_config(ax_flat, start, color="#2A9D8F", label="start")
        _draw_config(ax_flat, goal, color="#E63946", label="goal")
        _draw_turning_circles(ax_flat, start, goal, solver.rho, alpha=0.08)
        title = f"{dp.path_type}  L={dp.total_length:.2f}"
        if is_shortest:
            title += " ★"
        ax_flat.set_title(title, fontweight="bold" if is_shortest else "normal", fontsize=9)
        ax_flat.set_aspect("equal")
        ax_flat.grid(True, alpha=0.15)
        if idx == 0:
            ax_flat.legend(fontsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("All valid Dubins paths", fontsize=12, y=1.01)
    fig.tight_layout()
    return fig


def _draw_config(ax, q, color="black", label=None, r=0.25):
    ax.plot(q[0], q[1], "o", color=color, markersize=5, zorder=5, label=label)
    ax.annotate("", xy=(q[0] + r * np.cos(q[2]), q[1] + r * np.sin(q[2])),
                xytext=(q[0], q[1]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))


def _draw_turning_circles(ax, start, goal, rho, alpha=0.1):
    for q in [start, goal]:
        cl = (q[0] - rho * np.sin(q[2]), q[1] + rho * np.cos(q[2]))
        cr = (q[0] + rho * np.sin(q[2]), q[1] - rho * np.cos(q[2]))
        for c, ls in [(cl, "--"), (cr, ":")]:
            circle = plt.Circle(c, rho, fill=False, linestyle=ls,
                                color="gray", alpha=alpha * 3, linewidth=0.8)
            ax.add_patch(circle)


# ── motion model clouds ──────────────────────────────────────────────

def plot_motion_cloud(
    particles: np.ndarray,
    title: str = "",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()
    ax.scatter(particles[:, 0], particles[:, 1], s=1.5, alpha=0.25,
               c=COLORS["particle"], rasterized=True)
    ax.plot(0, 0, "ko", markersize=6, zorder=5)
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.set_title(title)
    ax.set_aspect("equal")
    if standalone:
        fig.tight_layout()
    return fig


def plot_banana_vs_gaussian(
    particles: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.scatter(particles[:, 0], particles[:, 1], s=1.5, alpha=0.2,
                c=COLORS["particle"], rasterized=True)
    ax1.plot(0, 0, "ko", markersize=6, zorder=5, label="start")
    ax1.set_title("True distribution (particles)")
    ax1.set_aspect("equal")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.15)

    ax2.scatter(particles[:, 0], particles[:, 1], s=1.5, alpha=0.12,
                c=COLORS["particle"], rasterized=True)
    plot_covariance_ellipse(mu, Sigma, ax2, n_sigma=1.0, color=COLORS["estimate"], alpha=0.15)
    plot_covariance_ellipse(mu, Sigma, ax2, n_sigma=2.0, color=COLORS["estimate"], alpha=0.08)
    ax2.plot(mu[0], mu[1], "x", color=COLORS["estimate"], markersize=8, markeredgewidth=2, label="Gaussian approx")
    ax2.plot(0, 0, "ko", markersize=6, zorder=5)
    ax2.set_title("Linearised Gaussian vs true")
    ax2.set_aspect("equal")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.15)

    for a in (ax1, ax2):
        a.set_xlabel("$x$ [m]")
        a.set_ylabel("$y$ [m]")
    fig.tight_layout()
    return fig


# ── 1-D Bayes filter grid ────────────────────────────────────────────

def plot_bayes_1d_steps(
    positions: np.ndarray,
    belief_history: list[tuple[str, np.ndarray]],
    true_positions: list[float] | None = None,
    figsize: tuple = (12, 2.8),
) -> plt.Figure:
    n = len(belief_history)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0], figsize[1]))
    if n == 1:
        axes = [axes]
    for i, (ax, (label, bel)) in enumerate(zip(axes, belief_history)):
        ax.bar(positions, bel, width=positions[1] - positions[0],
               color=COLORS["kf_upd"] if "update" in label.lower() else COLORS["kf_pred"],
               alpha=0.7, edgecolor="white", linewidth=0.3)
        if true_positions is not None and i // 2 < len(true_positions):
            tp = true_positions[i // 2] if "predict" in label.lower() else true_positions[min(i // 2, len(true_positions) - 1)]
            ax.axvline(tp, color=COLORS["truth"], linewidth=2, linestyle="--", label="true")
        ax.set_title(label, fontsize=9)
        ax.set_ylim(0, max(bel) * 1.3 + 1e-6)
        ax.set_xlabel("$x$")
        if i == 0:
            ax.set_ylabel("$\\mathrm{bel}(x)$")
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


# ── Kalman filter tracking ───────────────────────────────────────────

def plot_kf_1d(
    t: np.ndarray,
    x_true: np.ndarray,
    x_meas: np.ndarray,
    x_est: np.ndarray,
    sigma: np.ndarray,
    state_idx: int = 0,
    ylabel: str = "position",
    figsize: tuple = (10, 4),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t, x_true, "-", color=COLORS["truth"], linewidth=2, label="true", zorder=3)
    ax.scatter(t, x_meas, s=12, color=COLORS["measurement"], alpha=0.5,
               label="measurements", zorder=2)
    ax.plot(t, x_est, "-", color=COLORS["estimate"], linewidth=1.5, label="KF estimate", zorder=4)
    ax.fill_between(t, x_est - 2 * sigma, x_est + 2 * sigma,
                    color=COLORS["estimate"], alpha=0.15, label="$\\pm 2\\sigma$")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


# ── Covariance ellipse ───────────────────────────────────────────────

def plot_covariance_ellipse(
    mu: np.ndarray,
    Sigma: np.ndarray,
    ax: plt.Axes,
    n_sigma: float = 2.0,
    color: str = "steelblue",
    alpha: float = 0.25,
    zorder: int = 2,
) -> None:
    Sigma_xy = Sigma[:2, :2]
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma_xy)
    eigenvalues = np.maximum(eigenvalues, 0)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width = 2 * n_sigma * np.sqrt(eigenvalues[0])
    height = 2 * n_sigma * np.sqrt(eigenvalues[1])
    ell = Ellipse(xy=mu[:2], width=width, height=height, angle=angle,
                  color=color, alpha=alpha, zorder=zorder)
    ax.add_patch(ell)
    ell_edge = Ellipse(xy=mu[:2], width=width, height=height, angle=angle,
                       fill=False, edgecolor=color, linewidth=1.5, zorder=zorder + 1)
    ax.add_patch(ell_edge)


# ── EKF 2-D ──────────────────────────────────────────────────────────

def plot_ekf_2d(
    true_traj: np.ndarray,
    est_traj: np.ndarray,
    Sigmas: list[np.ndarray],
    landmarks: np.ndarray,
    every_n: int = 5,
    figsize: tuple = (8, 7),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(true_traj[:, 0], true_traj[:, 1], "-", color=COLORS["truth"],
            linewidth=2, label="true path", zorder=3)
    ax.plot(est_traj[:, 0], est_traj[:, 1], "--", color=COLORS["estimate"],
            linewidth=1.5, label="EKF estimate", zorder=4)
    for i in range(0, len(est_traj), every_n):
        plot_covariance_ellipse(est_traj[i], Sigmas[i], ax,
                                n_sigma=2.0, color=COLORS["estimate"], alpha=0.1)
    ax.scatter(landmarks[:, 0], landmarks[:, 1], marker="*", s=120,
               color=COLORS["measurement"], zorder=5, label="landmarks")
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_title("EKF Localisation with Range-Bearing Landmarks")
    fig.tight_layout()
    return fig


# ── Particle filter ──────────────────────────────────────────────────

def plot_particles(
    particles: np.ndarray,
    weights: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    color: str = COLORS["particle"],
    max_display: int = 1000,
) -> plt.Figure:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()
    idx = np.arange(len(particles))
    if len(particles) > max_display:
        idx = np.random.choice(idx, max_display, replace=False)
    pts = particles[idx]
    s = 8 if weights is None else 5 + 200 * weights[idx] / (weights[idx].max() + 1e-12)
    ax.scatter(pts[:, 0], pts[:, 1], s=s, alpha=0.4, c=color)
    ax.quiver(pts[:, 0], pts[:, 1],
              np.cos(pts[:, 2]) * 0.1, np.sin(pts[:, 2]) * 0.1,
              alpha=0.3, scale=10, color=color)
    if standalone:
        fig.tight_layout()
    return fig


def plot_mcl_result(
    true_traj: np.ndarray,
    est_traj: np.ndarray,
    landmarks: np.ndarray,
    final_particles: np.ndarray,
    ess_history: list[float],
    figsize: tuple = (14, 5),
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.plot(true_traj[:, 0], true_traj[:, 1], "-", color=COLORS["truth"],
             linewidth=2, label="true", zorder=3)
    ax1.plot(est_traj[:, 0], est_traj[:, 1], "--", color=COLORS["estimate"],
             linewidth=1.5, label="MCL mean", zorder=4)
    plot_particles(final_particles, ax=ax1, max_display=500)
    ax1.scatter(landmarks[:, 0], landmarks[:, 1], marker="*", s=120,
                color=COLORS["measurement"], zorder=5, label="landmarks")
    ax1.set_xlabel("$x$ [m]")
    ax1.set_ylabel("$y$ [m]")
    ax1.set_aspect("equal")
    ax1.legend(fontsize=8)
    ax1.set_title("MCL Localisation")
    ax1.grid(True, alpha=0.2)

    ax2.plot(ess_history, color=COLORS["kf_upd"], linewidth=1.5)
    ax2.axhline(len(final_particles) * 0.5, color="gray", linestyle="--",
                linewidth=1, label="50% threshold")
    ax2.set_xlabel("step")
    ax2.set_ylabel("ESS")
    ax2.set_title("Effective Sample Size")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


# ── PID tuning ────────────────────────────────────────────────────────

def plot_pid_tuning(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    figsize: tuple = (14, 4),
) -> plt.Figure:
    """results: {label: (trajectory (N,3), error_over_time (N,))}"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    for label, (traj, err) in results.items():
        ax1.plot(traj[:, 0], traj[:, 1], linewidth=1.5, label=label, alpha=0.8)
        ax2.plot(err, linewidth=1.5, label=label, alpha=0.8)
    ax1.set_xlabel("$x$ [m]")
    ax1.set_ylabel("$y$ [m]")
    ax1.set_aspect("equal")
    ax1.legend(fontsize=7)
    ax1.set_title("Trajectories")
    ax1.grid(True, alpha=0.2)
    ax2.set_xlabel("step")
    ax2.set_ylabel("cross-track error [m]")
    ax2.legend(fontsize=7)
    ax2.set_title("Tracking error")
    ax2.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


# ── path planning exploration ─────────────────────────────────────────

def plot_planning_comparison(
    occupancy: np.ndarray,
    resolution: float,
    origin: np.ndarray,
    results: dict[str, tuple[list, list]],
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """results: {label: (path_xy_list, explored_cells_list)}"""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    H, W = occupancy.shape
    extent = [origin[0], origin[0] + W * resolution,
              origin[1], origin[1] + H * resolution]
    for ax, (label, (path, explored)) in zip(axes, results.items()):
        ax.imshow(occupancy.astype(float), origin="lower", extent=extent,
                  cmap="gray_r", vmin=0, vmax=1, alpha=0.3)
        if explored:
            ex = np.array(explored)
            ey = origin[1] + (ex[:, 0] + 0.5) * resolution
            exx = origin[0] + (ex[:, 1] + 0.5) * resolution
            colors = np.linspace(0, 1, len(explored))
            ax.scatter(exx, ey, c=colors, cmap="YlOrRd", s=1, alpha=0.4, rasterized=True)
        if path:
            path_arr = np.array(path)
            ax.plot(path_arr[:, 0], path_arr[:, 1], "-", color=COLORS["estimate"],
                    linewidth=2, zorder=5)
            ax.plot(path_arr[0, 0], path_arr[0, 1], "go", markersize=8, zorder=6)
            ax.plot(path_arr[-1, 0], path_arr[-1, 1], "rs", markersize=8, zorder=6)
        ax.set_title(f"{label}  ({len(explored)} cells explored)")
        ax.set_xlabel("$x$ [m]")
        ax.set_ylabel("$y$ [m]")
        ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_rrt_tree(
    tree: list[tuple[np.ndarray, int | None]],
    path: list[np.ndarray] | None,
    occupancy: np.ndarray,
    resolution: float,
    origin: np.ndarray,
    figsize: tuple = (7, 6),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    H, W = occupancy.shape
    extent = [origin[0], origin[0] + W * resolution,
              origin[1], origin[1] + H * resolution]
    ax.imshow(occupancy.astype(float), origin="lower", extent=extent,
              cmap="gray_r", vmin=0, vmax=1, alpha=0.3)
    for node, parent_idx in tree:
        if parent_idx is not None:
            parent = tree[parent_idx][0]
            ax.plot([parent[0], node[0]], [parent[1], node[1]],
                    "-", color="#BDBDBD", linewidth=0.4, alpha=0.5)
    if path:
        path_arr = np.array(path)
        ax.plot(path_arr[:, 0], path_arr[:, 1], "-", color=COLORS["estimate"],
                linewidth=2.5, zorder=5)
    ax.plot(tree[0][0][0], tree[0][0][1], "go", markersize=10, zorder=6, label="start")
    if path:
        ax.plot(path[-1][0], path[-1][1], "rs", markersize=10, zorder=6, label="goal")
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.set_title(f"RRT  ({len(tree)} nodes)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ── Map + path (kept from original) ──────────────────────────────────

def plot_map_and_path(
    occupancy: np.ndarray,
    resolution: float,
    origin: np.ndarray,
    path: Optional[list] = None,
    robot_pose: Optional[np.ndarray] = None,
    landmarks: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Map",
) -> plt.Figure:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()
    H, W = occupancy.shape
    extent = [origin[0], origin[0] + W * resolution,
              origin[1], origin[1] + H * resolution]
    ax.imshow(~occupancy, origin="lower", extent=extent, cmap="gray", vmin=0, vmax=1)
    if landmarks is not None:
        ax.plot(landmarks[:, 0], landmarks[:, 1], "b*", markersize=10,
                label="landmarks", zorder=8)
    if path is not None and len(path) > 1:
        path_arr = np.array([[p[0], p[1]] for p in path])
        ax.plot(path_arr[:, 0], path_arr[:, 1], "r-", linewidth=2, label="path", zorder=7)
        ax.plot(path_arr[0, 0], path_arr[0, 1], "go", markersize=8, zorder=9)
        ax.plot(path_arr[-1, 0], path_arr[-1, 1], "rs", markersize=8, zorder=9)
    if robot_pose is not None:
        ax.plot(robot_pose[0], robot_pose[1], "r^", markersize=10, zorder=10)
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    if standalone:
        fig.tight_layout()
    return fig
