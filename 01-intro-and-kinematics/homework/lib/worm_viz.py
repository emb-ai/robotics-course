from __future__ import annotations

import numpy as np
import trimesh

try:
    from trimesh.viewer.notebook import scene_to_notebook
except (AttributeError, ImportError):
    def scene_to_notebook(scene: trimesh.Scene, height: int = 500):
        from IPython.display import HTML
        return HTML(
            '<p style="padding:1em">Trimesh notebook viewer not available. '
            "Install it with: <code>pip install trimesh[easy]</code></p>"
        )

R_MAX = np.pi / 2


def _urdf_worm_chain(link_lengths: np.ndarray, r_max: float) -> str:
    N = len(link_lengths)
    r_max_str = f"{r_max:.6f}"
    parts = ['<?xml version="1.0"?><robot name="worm">']
    parts.append('<link name="base_link"><inertial><mass value="0.001"/><inertia ixx="1e-6" iyy="1e-6" izz="1e-6" ixy="0" ixz="0" iyz="0"/></inertial></link>')
    L0 = float(link_lengths[0])
    parts.append(f'<joint name="base_to_link0" type="fixed"><parent link="base_link"/><child link="link0"/>')
    parts.append('<origin xyz="0 0 0" rpy="0 0 0"/></joint>')
    parts.append(f'<link name="link0"><inertial><origin xyz="0 0 {L0/2:.6f}" rpy="0 0 0"/><mass value="0.1"/><inertia ixx="1e-5" iyy="1e-5" izz="1e-5" ixy="0" ixz="0" iyz="0"/></inertial></link>')
    for i in range(N - 1):
        L = float(link_lengths[i + 1])
        L_prev = float(link_lengths[i])
        parts.append(f'<joint name="j{i}a" type="revolute"><parent link="link{i}"/><child link="link{i}_mid"/>')
        parts.append(f'<origin xyz="0 0 {L_prev:.6f}" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-{r_max_str}" upper="{r_max_str}" effort="1" velocity="1"/></joint>')
        parts.append(f'<link name="link{i}_mid"><inertial><mass value="0.001"/><inertia ixx="1e-6" iyy="1e-6" izz="1e-6" ixy="0" ixz="0" iyz="0"/></inertial></link>')
        parts.append(f'<joint name="j{i}b" type="revolute"><parent link="link{i}_mid"/><child link="link{i+1}"/>')
        parts.append(f'<origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-{r_max_str}" upper="{r_max_str}" effort="1" velocity="1"/></joint>')
        parts.append(f'<link name="link{i+1}"><inertial><origin xyz="0 0 {L/2:.6f}" rpy="0 0 0"/><mass value="0.1"/><inertia ixx="1e-5" iyy="1e-5" izz="1e-5" ixy="0" ixz="0" iyz="0"/></inertial></link>')
    parts.append('</robot>')
    return "".join(parts)


def segment_endpoints(link_lengths: np.ndarray, angles: np.ndarray) -> np.ndarray:
    import torch
    import pytorch_kinematics as pk
    N = len(link_lengths)
    urdf = _urdf_worm_chain(link_lengths, R_MAX)
    chain = pk.build_chain_from_urdf(urdf.encode("utf-8"))
    th = {f"j{i}a": torch.tensor(float(angles[2*i]), dtype=torch.float32) for i in range(N - 1)}
    th.update({f"j{i}b": torch.tensor(float(angles[2*i + 1]), dtype=torch.float32) for i in range(N - 1)})
    ret = chain.forward_kinematics(th)
    endpoints = np.zeros((N + 1, 3))
    endpoints[0] = 0.0, 0.0, 0.0
    for i in range(N):
        T = ret[f"link{i}"].get_matrix()[0].numpy()
        endpoints[i + 1] = T[:3, 3] + link_lengths[i] * T[:3, 2]
    return endpoints


def _cylinder_between(p0: np.ndarray, p1: np.ndarray, radius: float = 1.0) -> trimesh.Trimesh:
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length < 1e-10:
        axis = np.array([0, 0, 1], dtype=np.float64)
        length = 1e-10
    axis = axis / length
    z = np.array([0, 0, 1], dtype=np.float64)
    R = np.eye(3)
    if np.abs(np.dot(axis, z)) < 1 - 1e-10:
        y = np.cross(axis, z)
        y /= np.linalg.norm(y)
        x = np.cross(y, axis)
        R = np.column_stack([x, y, axis])
    else:
        if np.dot(axis, z) < 0:
            R = np.diag([1, -1, -1])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p0 + 0.5 * length * (R @ z)
    cyl = trimesh.creation.cylinder(radius=radius, height=length)
    cyl.apply_transform(T)
    return cyl


def worm_scene(
    link_lengths: np.ndarray,
    angles: np.ndarray,
    endpoints: np.ndarray | None = None,
    show_aabb: bool = True,
    cylinder_radius: float = 1.0,
) -> trimesh.Scene:
    if endpoints is None:
        endpoints = segment_endpoints(link_lengths, angles)
    scene = trimesh.Scene()
    N = len(link_lengths)
    for i in range(N):
        cyl = _cylinder_between(endpoints[i], endpoints[i + 1], radius=cylinder_radius)
        cyl.visual.face_colors = np.tile([180, 200, 220, 255], (len(cyl.faces), 1))
        scene.add_geometry(cyl, geom_name=f"link{i}")
    if show_aabb:
        lo = endpoints.min(axis=0)
        hi = endpoints.max(axis=0)
        center = (lo + hi) / 2
        extent = hi - lo
        box = trimesh.creation.box(extents=extent)
        box.apply_translation(center)
        box.visual.face_colors = np.tile([80, 80, 80, 60], (len(box.faces), 1))
        scene.add_geometry(box, geom_name="aabb")
    return scene


def show_worm(
    link_lengths: np.ndarray,
    angles: np.ndarray,
    show_aabb: bool = True,
    height: int = 500,
):
    scene = worm_scene(link_lengths, angles, show_aabb=show_aabb)
    return scene_to_notebook(scene, height=height)
