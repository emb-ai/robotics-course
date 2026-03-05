from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
import pytorch_kinematics as pk

_topic = Path(__file__).resolve().parent.parent.parent
if str(_topic) not in sys.path:
    sys.path.insert(0, str(_topic))
from lib.urdf_visuals import parse_urdf_visuals, origin_to_matrix


def build_so101_scene(
    so101_urdf: Path,
    q: np.ndarray,
    cube_center: tuple[float, float, float] | None = None,
    cube_size: float = 0.02,
) -> trimesh.Scene:
    chain = pk.build_chain_from_urdf(open(so101_urdf, mode="rb").read())
    serial_chain = pk.SerialChain(chain, "gripper_frame_link", "base_link")
    link_visuals = parse_urdf_visuals(so101_urdf)
    full_joint_names = chain.get_joint_parameter_names()
    serial_joint_names = serial_chain.get_joint_parameter_names()

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

    th_full = {
        name: torch.tensor(0.0, dtype=torch.float64)
        for name in full_joint_names
    }
    q_arr = np.asarray(q, dtype=np.float64).ravel()
    for i, name in enumerate(serial_joint_names):
        if i < len(q_arr):
            th_full[name] = torch.tensor(q_arr[i], dtype=torch.float64)
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
    if cube_center is not None:
        box = trimesh.creation.box(extents=(cube_size, cube_size, cube_size))
        box.apply_translation(cube_center)
        box.visual.face_colors = (100, 100, 200, 255)
        scene.add_geometry(box, geom_name="cube")
    return scene
