from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import trimesh
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

from .urdf_visuals import parse_urdf_visuals, origin_to_matrix

try:
    from trimesh.viewer.notebook import scene_to_notebook
except (AttributeError, ImportError):
    def scene_to_notebook(scene: trimesh.Scene, height: int = 500):
        return HTML(
            '<p style="padding:1em">Trimesh notebook viewer not available. '
            "Install it with: <code>pip install trimesh[easy]</code></p>"
        )


def show_so101_interactive(urdf_dir: Path) -> None:
    import pytorch_kinematics as pk

    so101_urdf = urdf_dir / "so101" / "robot.urdf"
    chain = pk.build_chain_from_urdf(open(so101_urdf, mode="rb").read())
    link_visuals = parse_urdf_visuals(so101_urdf)
    joint_names = chain.get_joint_parameter_names()
    low, high = chain.get_joint_limits()
    joint_limits = [
        (float(lo), float(hi)) if np.isfinite(lo) and np.isfinite(hi) else (-np.pi, np.pi)
        for lo, hi in zip(low, high)
    ]

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

    out = widgets.Output()

    def render_robot(**kwargs):
        q_vals = [kwargs[f"q{i}"] for i in range(len(joint_names))]
        th = {
            name: torch.tensor(q_vals[i], dtype=torch.float64)
            for i, name in enumerate(joint_names)
        }
        ret = chain.forward_kinematics(th)
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
        with out:
            clear_output(wait=True)
            display(scene_to_notebook(scene, height=500))

    display(out)
    with out:
        render_robot(**{f"q{i}": 0.0 for i in range(len(joint_names))})

    slider_kw = {
        f"q{i}": widgets.FloatSlider(
            min=joint_limits[i][0],
            max=joint_limits[i][1],
            step=0.05,
            value=0.0,
            description=joint_names[i],
        )
        for i in range(len(joint_names))
    }
    widgets.interact(render_robot, **slider_kw)
