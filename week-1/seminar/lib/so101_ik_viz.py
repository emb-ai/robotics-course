from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import trimesh
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from scipy.spatial.transform import Rotation

import pytorch_kinematics as pk
from pytorch_kinematics.transforms import rotation_conversions

from .urdf_visuals import parse_urdf_visuals, origin_to_matrix

POSITION_WEIGHT = 100.0
ROTATION_WEIGHT = 1.0

try:
    from trimesh.viewer.notebook import scene_to_notebook
except (AttributeError, ImportError):
    def scene_to_notebook(scene: trimesh.Scene, height: int = 500):
        return HTML(
            '<p style="padding:1em">Trimesh notebook viewer not available. '
            "Install it with: <code>pip install trimesh[easy]</code></p>"
        )


def _random_target_pose(
    pos_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    x_lo, x_hi = pos_bounds[0]
    y_lo, y_hi = pos_bounds[1]
    z_lo, z_hi = pos_bounds[2]
    pos = np.array([
        np.random.uniform(x_lo, x_hi),
        np.random.uniform(y_lo, y_hi),
        np.random.uniform(z_lo, z_hi),
    ])
    euler = np.random.uniform(-np.pi, np.pi, size=3)
    R = Rotation.from_euler("xyz", euler).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def show_so101_ik_demo(urdf_dir: Path, seed: int | None = None) -> None:
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    so101_urdf = urdf_dir / "so101" / "robot.urdf"
    chain = pk.build_chain_from_urdf(open(so101_urdf, mode="rb").read())
    serial_chain = pk.SerialChain(chain, "gripper_frame_link", "base_link")
    link_visuals = parse_urdf_visuals(so101_urdf)

    full_joint_names = chain.get_joint_parameter_names()
    serial_joint_names = serial_chain.get_joint_parameter_names()
    low, high = serial_chain.get_joint_limits()
    lim = torch.tensor([low, high], dtype=torch.float32)

    mesh_cache: dict[Path, trimesh.Trimesh] = {}

    def load_mesh(path: Path) -> trimesh.Trimesh | None:
        if path not in mesh_cache:
            if not path.exists():
                return None
            try:
                mesh_cache[path] = trimesh.load(path, force="mesh")
            except Exception:
                return None
        return mesh_cache[path].copy()

    pos_bounds = ((0.2, 0.32), (-0.12, 0.12), (0.10, 0.20))
    serial_chain_f32 = serial_chain.to(dtype=torch.float32)
    dof = len(serial_joint_names)
    num_retries = 5
    max_iterations = 100
    lr = 0.2
    reg = 1e-6

    def weighted_ik_solve(goal_tf: pk.Transform3d) -> torch.Tensor:
        target = goal_tf.get_matrix()
        target_pos = target[0, :3, 3]
        target_wxyz = rotation_conversions.matrix_to_quaternion(target[:, :3, :3])
        q = (
            torch.rand(num_retries, dof, dtype=torch.float32, device=serial_chain_f32.device)
            * (lim[1] - lim[0])
            + lim[0]
        )
        for _ in range(max_iterations):
            J, m = serial_chain_f32.jacobian(q, ret_eef_pose=True)
            pos_diff = target_pos.unsqueeze(0) - m[:, :3, 3]
            cur_wxyz = rotation_conversions.matrix_to_quaternion(m[:, :3, :3])
            diff_wxyz = rotation_conversions.quaternion_multiply(
                target_wxyz.expand(q.shape[0], 4),
                rotation_conversions.quaternion_invert(cur_wxyz),
            )
            rot_diff = rotation_conversions.quaternion_to_axis_angle(diff_wxyz)
            dx = torch.cat(
                [
                    pos_diff * POSITION_WEIGHT,
                    rot_diff * ROTATION_WEIGHT,
                ],
                dim=1,
            ).unsqueeze(2)
            reg_mat = reg * torch.eye(6, device=q.device, dtype=q.dtype)
            tmp_a = J @ J.transpose(1, 2) + reg_mat.unsqueeze(0)
            dq = torch.linalg.solve(tmp_a, dx)
            dq = (J.transpose(1, 2) @ dq).squeeze(2)
            q = torch.clamp(q + lr * dq, min=lim[0], max=lim[1])
        J_final, m_final = serial_chain_f32.jacobian(q, ret_eef_pose=True)
        pos_err = (target_pos.unsqueeze(0) - m_final[:, :3, 3]).norm(dim=1)
        best_idx = int(torch.argmin(pos_err).item())
        return q[best_idx].clone()

    out = widgets.Output()

    def render() -> None:
        T_target = _random_target_pose(pos_bounds)
        goal_tf = pk.Transform3d(
            matrix=torch.tensor(T_target, dtype=torch.float32).unsqueeze(0),
        )
        q_ik = weighted_ik_solve(goal_tf)
        q_ik = torch.clamp(q_ik, min=lim[0], max=lim[1])

        th_full = {
            name: torch.tensor(0.0, dtype=torch.float64)
            for name in full_joint_names
        }
        for i, name in enumerate(serial_joint_names):
            th_full[name] = q_ik[i].clone().to(torch.float64)

        ret = chain.forward_kinematics(th_full)
        scene = trimesh.Scene()

        for link_name, tg in ret.items():
            if link_name not in link_visuals:
                continue
            T_link = tg.get_matrix()[0].detach().numpy()
            for entry in link_visuals[link_name]:
                mesh = load_mesh(entry.mesh_path)
                if mesh is None:
                    continue
                T_origin = origin_to_matrix(entry.origin_xyz, entry.origin_rpy)
                T_world = T_link @ T_origin
                mesh.apply_transform(T_world)
                r, g, b, a = entry.rgba
                mesh.visual.face_colors = np.tile(
                    np.array([r * 255, g * 255, b * 255, a * 255], dtype=np.uint8),
                    (len(mesh.faces), 1),
                )
                scene.add_geometry(mesh, geom_name=f"{link_name}_{entry.mesh_path.name}")

        axis_target = trimesh.creation.axis(
            transform=T_target,
            origin_size=0.02,
            axis_length=0.08,
            axis_radius=0.008,
        )
        scene.add_geometry(axis_target, geom_name="target_frame")

        with out:
            clear_output(wait=True)
            display(scene_to_notebook(scene, height=500))

    button = widgets.Button(description="Randomize target")
    button.on_click(lambda _: render())

    display(button)
    display(out)
    render()
