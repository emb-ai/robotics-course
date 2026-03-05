from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _fk_joint_positions(urdf_path: Path, q: np.ndarray) -> np.ndarray:
    import torch
    import pytorch_kinematics as pk

    chain = pk.build_chain_from_urdf(open(urdf_path, "rb").read())
    serial = pk.SerialChain(chain, "gripper_frame_link", "base_link")
    names = serial.get_joint_parameter_names()
    full_names = chain.get_joint_parameter_names()

    th = {n: torch.tensor(0.0, dtype=torch.float64) for n in full_names}
    q_arr = np.asarray(q, dtype=np.float64).ravel()
    for i, n in enumerate(names):
        if i < len(q_arr):
            th[n] = torch.tensor(q_arr[i], dtype=torch.float64)

    ret = chain.forward_kinematics(th)
    positions = []
    for link_name, tg in ret.items():
        T = tg.get_matrix()[0].detach().numpy()
        positions.append(T[:3, 3])
    return np.array(positions)


def show_pre_grasp_grid(
    urdf_path: Path,
    poses: list[tuple[np.ndarray, float]],
    *,
    cube_size: float = 0.025,
    ik_solver,
) -> None:
    solved = []
    for position_xyz, yaw in poses:
        q = ik_solver(position_xyz, yaw, urdf_path)
        if q is None:
            continue
        solved.append((position_xyz, yaw, q))

    if not solved:
        print("No IK solutions found for the given poses.")
        return

    n = len(solved)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 4 * nrows),
        subplot_kw={"projection": "3d"},
    )
    axes = np.atleast_2d(axes)

    for idx, (position_xyz, yaw, q) in enumerate(solved):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        pts = _fk_joint_positions(urdf_path, q)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "o-", color="steelblue", markersize=3, linewidth=1.5)

        x, y, _ = np.asarray(position_xyz).ravel()[:3]
        half = cube_size / 2
        cx, cy, cz = float(x), float(y), cube_size / 2
        r = [[-half, half], [-half, half], [0, cube_size]]
        for s, e in [([0, 1], [0, 0]), ([1, 1], [0, 1]), ([0, 1], [1, 1]), ([0, 0], [0, 1])]:
            for zi in range(2):
                ax.plot3D(
                    [cx + r[0][s[0]], cx + r[0][s[1]]],
                    [cy + r[1][e[0]], cy + r[1][e[1]]],
                    [r[2][zi], r[2][zi]],
                    color="slateblue", linewidth=0.8, alpha=0.7,
                )
            ax.plot3D(
                [cx + r[0][s[0]], cx + r[0][s[0]]],
                [cy + r[1][e[0]], cy + r[1][e[0]]],
                [r[2][0], r[2][1]],
                color="slateblue", linewidth=0.8, alpha=0.7,
            )

        ax.set_title(f"yaw={yaw:.1f}", fontsize=9)
        ax.set_xlabel("x", fontsize=7)
        ax.set_ylabel("y", fontsize=7)
        ax.set_zlabel("z", fontsize=7)
        ax.tick_params(labelsize=6)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.show()
