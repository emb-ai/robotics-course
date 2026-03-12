from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from IPython.display import HTML, display


@dataclass
class Configuration:
    x: float
    y: float
    z: float
    theta: float
    phi: float

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def direction(self) -> np.ndarray:
        ct, st = np.cos(self.theta), np.sin(self.theta)
        cp, sp = np.cos(self.phi), np.sin(self.phi)
        return np.array([ct * cp, st * cp, sp])


@dataclass
class XYZConfiguration:
    x: float
    y: float
    z: float

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


KAPPA_MAX = 1.0
PHI_MIN = -np.pi / 4
PHI_MAX = np.pi / 4
V = 1.0


def _sample_curve(
    curve_fn: Callable[[np.ndarray], Configuration],
    n_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s = np.linspace(0.0, 1.0, n_points)
    xs, ys, zs, thetas, phis = [], [], [], [], []
    for si in s:
        c = curve_fn(np.atleast_1d(si))
        xs.append(c.x)
        ys.append(c.y)
        zs.append(c.z)
        thetas.append(c.theta)
        phis.append(c.phi)
    return (
        np.array(xs),
        np.array(ys),
        np.array(zs),
        np.array(thetas),
        np.array(phis),
    )


def check_all(
    curve_fn: Callable[[np.ndarray], Configuration],
    start: Configuration,
    goal: Configuration | None = None,
    goal_xyz: XYZConfiguration | None = None,
    n_points: int = 500,
    eom_tol: float = 0.05,
    constraint_tol: float = 1e-3,
    pos_tol: float = 0.02,
    angle_tol: float = 0.05,
) -> tuple[bool, list[str]]:
    all_errors: list[str] = []
    xs, ys, zs, thetas, phis = _sample_curve(curve_fn, n_points)
    L = curve_length(curve_fn, n_points)
    ds = L / (n_points - 1)

    if ds >= 1e-12:
        dx = np.diff(xs) / ds
        dy = np.diff(ys) / ds
        dz = np.diff(zs) / ds
        ct = np.cos(thetas[:-1])
        st = np.sin(thetas[:-1])
        cp = np.cos(phis[:-1])
        sp = np.sin(phis[:-1])
        res_x = np.abs(dx - V * ct * cp)
        res_y = np.abs(dy - V * st * cp)
        res_z = np.abs(dz - V * sp)
        if np.max(res_x) > eom_tol:
            all_errors.append(f"EOM x residual max={np.max(res_x):.4f} > {eom_tol}")
        if np.max(res_y) > eom_tol:
            all_errors.append(f"EOM y residual max={np.max(res_y):.4f} > {eom_tol}")
        if np.max(res_z) > eom_tol:
            all_errors.append(f"EOM z residual max={np.max(res_z):.4f} > {eom_tol}")

        phi_lo, phi_hi = np.min(phis), np.max(phis)
        if phi_lo < PHI_MIN - constraint_tol:
            all_errors.append(f"Pitch below limit: min(phi)={phi_lo:.4f} < {PHI_MIN:.4f}")
        if phi_hi > PHI_MAX + constraint_tol:
            all_errors.append(f"Pitch above limit: max(phi)={phi_hi:.4f} > {PHI_MAX:.4f}")
        dtheta = np.diff(thetas) / ds
        dphi = np.diff(phis) / ds
        u1 = dtheta * cp
        u2 = dphi
        kappa = np.sqrt(u1**2 + u2**2)
        if np.max(kappa) > KAPPA_MAX + constraint_tol:
            all_errors.append(f"Curvature exceeded: max(kappa)={np.max(kappa):.4f} > {KAPPA_MAX}")
        speed = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2 + np.diff(zs)**2) / ds
        if np.max(np.abs(speed - V)) > 0.1:
            all_errors.append(f"Speed deviation from v={V}: max|v-1|={np.max(np.abs(speed - V)):.4f}")

    c0 = curve_fn(np.atleast_1d(0.0))
    c1 = curve_fn(np.atleast_1d(1.0))
    d_start = np.linalg.norm(np.array([c0.x - start.x, c0.y - start.y, c0.z - start.z]))
    if d_start > pos_tol:
        all_errors.append(f"Start position error: {d_start:.4f}")
    for attr in ("theta", "phi"):
        diff = abs(getattr(c0, attr) - getattr(start, attr))
        if diff > angle_tol:
            all_errors.append(f"Start {attr} error: {diff:.4f}")
    if goal is not None:
        d_goal = np.linalg.norm(np.array([c1.x - goal.x, c1.y - goal.y, c1.z - goal.z]))
        if d_goal > pos_tol:
            all_errors.append(f"Goal position error: {d_goal:.4f}")
        for attr in ("theta", "phi"):
            diff = abs(getattr(c1, attr) - getattr(goal, attr))
            if diff > angle_tol:
                all_errors.append(f"Goal {attr} error: {diff:.4f}")
    elif goal_xyz is not None:
        d_goal = np.linalg.norm(np.array([c1.x - goal_xyz.x, c1.y - goal_xyz.y, c1.z - goal_xyz.z]))
        if d_goal > pos_tol:
            all_errors.append(f"Goal XYZ error: {d_goal:.4f}")

    return len(all_errors) == 0, all_errors


def curve_length(
    curve_fn: Callable[[np.ndarray], Configuration],
    n_points: int = 1000,
) -> float:
    xs, ys, zs, _, _ = _sample_curve(curve_fn, n_points)
    return float(np.sum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2 + np.diff(zs)**2)))


ARROW_SCALE_FRAC = 0.2
ARROW_LENGTH_RATIO = 0.25


def show_broom_path(
    curve_fn: Callable[[np.ndarray], Configuration],
    start: Configuration,
    goal: Configuration | XYZConfiguration | None = None,
    n_points: int = 200,
    title: str = "",
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    s = np.linspace(0.0, 1.0, n_points)
    xs, ys, zs, phis = [], [], [], []
    for si in s:
        c = curve_fn(np.atleast_1d(si))
        xs.append(c.x)
        ys.append(c.y)
        zs.append(c.z)
        phis.append(c.phi)
    xs, ys, zs, phis = np.array(xs), np.array(ys), np.array(zs), np.array(phis)

    pitch_ok = (phis >= PHI_MIN - 1e-3) & (phis <= PHI_MAX + 1e-3)

    created_fig = ax is None
    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    for i in range(len(xs) - 1):
        color = "tab:green" if pitch_ok[i] else "tab:red"
        ax.plot(xs[i : i + 2], ys[i : i + 2], zs[i : i + 2], color=color, linewidth=1.5)

    ax.scatter(*start.position(), color="blue", s=60, marker="o", label="start")
    scale = 0.3
    ax.quiver(*start.position(), *start.direction() * scale, color="blue", arrow_length_ratio=0.2)

    if goal is not None:
        gp = goal.position()
        ax.scatter(*gp, color="red", s=60, marker="x", label="goal")
        if isinstance(goal, Configuration):
            ax.quiver(*gp, *goal.direction() * scale, color="red", arrow_length_ratio=0.2)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    all_x = np.concatenate([[start.x], xs, [goal.position()[0]] if goal is not None else []])
    all_y = np.concatenate([[start.y], ys, [goal.position()[1]] if goal is not None else []])
    all_z = np.concatenate([[start.z], zs, [goal.position()[2]] if goal is not None else []])
    r = max(np.ptp(all_x), np.ptp(all_y), np.ptp(all_z), 1e-6)
    mid_x, mid_y, mid_z = np.mean(all_x), np.mean(all_y), np.mean(all_z)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(mid_x - r / 2, mid_x + r / 2)
    ax.set_ylim(mid_y - r / 2, mid_y + r / 2)
    ax.set_zlim(mid_z - r / 2, mid_z + r / 2)
    if created_fig:
        plt.tight_layout()
    return fig if created_fig else None


def show_broom_path_four_panels(
    curve_fn: Callable[[np.ndarray], Configuration],
    start: Configuration,
    goal: Configuration | XYZConfiguration | None = None,
    n_points: int = 200,
    title: str = "",
    intermediate_xyz: np.ndarray | None = None,
) -> plt.Figure:
    s = np.linspace(0.0, 1.0, n_points)
    xs, ys, zs, phis = [], [], [], []
    for si in s:
        c = curve_fn(np.atleast_1d(si))
        xs.append(c.x)
        ys.append(c.y)
        zs.append(c.z)
        phis.append(c.phi)
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    phis = np.array(phis)
    pitch_ok = (phis >= PHI_MIN - 1e-3) & (phis <= PHI_MAX + 1e-3)

    all_x = np.concatenate([[start.x], xs, [goal.position()[0]] if goal is not None else []])
    all_y = np.concatenate([[start.y], ys, [goal.position()[1]] if goal is not None else []])
    all_z = np.concatenate([[start.z], zs, [goal.position()[2]] if goal is not None else []])
    if intermediate_xyz is not None:
        all_x = np.concatenate([all_x, [intermediate_xyz[0]]])
        all_y = np.concatenate([all_y, [intermediate_xyz[1]]])
        all_z = np.concatenate([all_z, [intermediate_xyz[2]]])
    r = max(float(np.ptp(all_x)), float(np.ptp(all_y)), float(np.ptp(all_z)), 1e-6)
    mid_x = float(np.mean(all_x))
    mid_y = float(np.mean(all_y))
    mid_z = float(np.mean(all_z))
    scale = ARROW_SCALE_FRAC * r
    xlo, xhi = mid_x - r / 2, mid_x + r / 2
    ylo, yhi = mid_y - r / 2, mid_y + r / 2
    zlo, zhi = mid_z - r / 2, mid_z + r / 2

    sd = start.direction()
    gd = goal.direction() if isinstance(goal, Configuration) else None
    gp = goal.position() if goal is not None else None

    fig = plt.figure(figsize=(10, 10))
    ax_xy = fig.add_subplot(2, 2, 1)
    ax_yz = fig.add_subplot(2, 2, 2)
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_3d = fig.add_subplot(2, 2, 4, projection="3d")

    quiver_scale = 1.0 / (scale + 1e-12)
    quiver_kw = dict(scale=quiver_scale, scale_units="xy", angles="xy", width=0.008 * r, headwidth=4, headlength=5)

    for i in range(len(xs) - 1):
        color = "tab:green" if pitch_ok[i] else "tab:red"
        ax_xy.plot(xs[i : i + 2], ys[i : i + 2], color=color, linewidth=1.5)
        ax_yz.plot(ys[i : i + 2], zs[i : i + 2], color=color, linewidth=1.5)
        ax_xz.plot(xs[i : i + 2], zs[i : i + 2], color=color, linewidth=1.5)
    ax_xy.scatter(start.x, start.y, color="blue", s=80, marker="o", label="start", zorder=5)
    ax_xy.quiver(start.x, start.y, sd[0], sd[1], color="blue", **quiver_kw)
    ax_yz.scatter(start.y, start.z, color="blue", s=80, marker="o", zorder=5)
    ax_yz.quiver(start.y, start.z, sd[1], sd[2], color="blue", **quiver_kw)
    ax_xz.scatter(start.x, start.z, color="blue", s=80, marker="o", zorder=5)
    ax_xz.quiver(start.x, start.z, sd[0], sd[2], color="blue", **quiver_kw)

    if goal is not None and gp is not None:
        ax_xy.scatter(gp[0], gp[1], color="red", s=80, marker="x", label="goal", zorder=5)
        ax_yz.scatter(gp[1], gp[2], color="red", s=80, marker="x", zorder=5)
        ax_xz.scatter(gp[0], gp[2], color="red", s=80, marker="x", zorder=5)
        if gd is not None:
            ax_xy.quiver(gp[0], gp[1], gd[0], gd[1], color="red", **quiver_kw)
            ax_yz.quiver(gp[1], gp[2], gd[1], gd[2], color="red", **quiver_kw)
            ax_xz.quiver(gp[0], gp[2], gd[0], gd[2], color="red", **quiver_kw)

    if intermediate_xyz is not None:
        ax_xy.scatter(intermediate_xyz[0], intermediate_xyz[1], color="orange", s=100, marker="o", label="ball", zorder=5)
        ax_yz.scatter(intermediate_xyz[1], intermediate_xyz[2], color="orange", s=100, marker="o", zorder=5)
        ax_xz.scatter(intermediate_xyz[0], intermediate_xyz[2], color="orange", s=100, marker="o", zorder=5)

    ax_xy.set_aspect("equal")
    ax_xy.set_xlim(xlo, xhi)
    ax_xy.set_ylim(ylo, yhi)
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xy.set_title("xy")

    ax_yz.set_aspect("equal")
    ax_yz.set_xlim(ylo, yhi)
    ax_yz.set_ylim(zlo, zhi)
    ax_yz.set_xlabel("y")
    ax_yz.set_ylabel("z")
    ax_yz.set_title("yz")

    ax_xz.set_aspect("equal")
    ax_xz.set_xlim(xlo, xhi)
    ax_xz.set_ylim(zlo, zhi)
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")
    ax_xz.set_title("xz")

    for i in range(len(xs) - 1):
        color = "tab:green" if pitch_ok[i] else "tab:red"
        ax_3d.plot(xs[i : i + 2], ys[i : i + 2], zs[i : i + 2], color=color, linewidth=1.5)
    ax_3d.scatter(*start.position(), color="blue", s=80, marker="o", label="start")
    ax_3d.quiver(*start.position(), *start.direction() * scale, color="blue", arrow_length_ratio=ARROW_LENGTH_RATIO)
    if goal is not None:
        ax_3d.scatter(*gp, color="red", s=80, marker="x", label="goal")
        if gd is not None:
            ax_3d.quiver(*gp, *gd * scale, color="red", arrow_length_ratio=ARROW_LENGTH_RATIO)
    if intermediate_xyz is not None:
        ax_3d.scatter(*intermediate_xyz, color="orange", s=100, marker="o", label="ball")
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("z")
    ax_3d.set_title("3D")
    ax_3d.set_box_aspect([1, 1, 1])
    ax_3d.set_xlim(xlo, xhi)
    ax_3d.set_ylim(ylo, yhi)
    ax_3d.set_zlim(zlo, zhi)
    ax_3d.legend(fontsize=8)

    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    return fig


def show_quiddich_viewer(nb_dir: Path | None = None) -> None:
    nb_dir = nb_dir or Path.cwd()
    lib_dir = nb_dir / "lib"
    assets_dir = nb_dir / "assets"
    if not assets_dir.exists():
        assets_dir = nb_dir.parent / "homework" / "assets"
    js_path = lib_dir / "quiddich.js"
    glb_path = assets_dir / "quiddich.glb"
    js_code = js_path.read_text()
    js_code = js_code.replace("</script>", "<\\/script>")
    glb_b64 = base64.b64encode(glb_path.read_bytes()).decode("ascii")
    glb_data_url = f"data:model/gltf-binary;base64,{glb_b64}"
    container_id = "quiddich-viz-container"
    config = {"containerId": container_id, "glbUrl": glb_data_url}
    config_json = json.dumps(config).replace("</", "<\\/")
    html = f"""<div id="{container_id}" style="width:100%; min-height:400px; background:#1a1a2e;"></div>
<script>window.QUIDDICH_VISUALIZER = {config_json};</script>
<script type="module">
{js_code}
</script>"""
    display(HTML(html))
