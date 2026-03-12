from __future__ import annotations

import base64
import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import torch
import trimesh
import pytorch_kinematics as pk
from IPython.display import HTML, display


def show_so101_viewer() -> None:
    nb_dir = Path.cwd()
    lib_dir = nb_dir / "lib"
    so101_dir = nb_dir / "assets" / "so101"
    if not so101_dir.exists():
        so101_dir = nb_dir.parent / "homework" / "assets" / "so101"
    js_path = lib_dir / "so101_ik.js"
    urdf_path = so101_dir / "robot.urdf"
    assets_dir = so101_dir / "assets"

    js_code = js_path.read_text()
    js_code = js_code.replace("</script>", "<\\/script>")

    urdf_text = urdf_path.read_text()
    mesh_data_urls = {}
    if assets_dir.exists():
        for stl in assets_dir.glob("*.stl"):
            key = f"assets/{stl.name}"
            b64 = base64.b64encode(stl.read_bytes()).decode("ascii")
            mesh_data_urls[key] = f"data:application/octet-stream;base64,{b64}"

    obj_path = so101_dir / "so101_solvable_region_approx.obj"
    solvable_region_obj_url = None
    if obj_path.exists():
        b64 = base64.b64encode(obj_path.read_bytes()).decode("ascii")
        solvable_region_obj_url = f"data:application/octet-stream;base64,{b64}"

    container_id = "so101-viz-container"
    config = {
        "containerId": container_id,
        "urdfText": urdf_text,
        "meshDataUrls": mesh_data_urls,
        "solvableRegionObjUrl": solvable_region_obj_url,
    }
    config_json = json.dumps(config).replace("</", "<\\/")

    importmap = """<script type="importmap">
{"imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
  "three/examples/jsm/loaders/STLLoader.js": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/STLLoader.js",
  "three/examples/jsm/loaders/OBJLoader.js": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/OBJLoader.js",
  "three/examples/jsm/loaders/ColladaLoader.js": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/ColladaLoader.js"
}}</script>"""

    html = f"""{importmap}
<div id="{container_id}" style="width:100%; min-height:400px; background:#1a1a1a;"></div>
<script>window.SO101_VISUALIZER = {config_json};</script>
<script type="module">
{js_code}
</script>"""
    display(HTML(html))


class VisualEntry(NamedTuple):
    mesh_path: Path
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    rgba: tuple[float, float, float, float]


def _parse_xyz(elem: ET.Element | None) -> tuple[float, float, float]:
    if elem is None:
        return (0.0, 0.0, 0.0)
    s = (elem.get("xyz") or "0 0 0").strip().split()
    return (float(s[0]), float(s[1]), float(s[2])) if len(s) >= 3 else (0.0, 0.0, 0.0)


def _parse_rpy(elem: ET.Element | None) -> tuple[float, float, float]:
    if elem is None:
        return (0.0, 0.0, 0.0)
    s = (elem.get("rpy") or "0 0 0").strip().split()
    return (float(s[0]), float(s[1]), float(s[2])) if len(s) >= 3 else (0.0, 0.0, 0.0)


def _parse_rgba(elem: ET.Element | None) -> tuple[float, float, float, float]:
    if elem is None:
        return (0.5, 0.5, 0.5, 1.0)
    s = (elem.get("rgba") or "0.5 0.5 0.5 1.0").strip().split()
    return (float(s[0]), float(s[1]), float(s[2]), float(s[3])) if len(s) >= 4 else (0.5, 0.5, 0.5, 1.0)


def parse_urdf_visuals(urdf_path: Path) -> dict[str, list[VisualEntry]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    path_prefix = urdf_path.parent

    materials: dict[str, tuple[float, float, float, float]] = {}
    for mat in root.findall(".//material"):
        name = mat.get("name")
        if name is None:
            continue
        color = mat.find("color")
        materials[name] = _parse_rgba(color)

    result: dict[str, list[VisualEntry]] = {}
    for link in root.findall("link"):
        name = link.get("name")
        if name is None:
            continue
        entries: list[VisualEntry] = []
        for visual in link.findall("visual"):
            origin = visual.find("origin")
            geom = visual.find("geometry")
            mesh = geom.find("mesh") if geom is not None else None
            if mesh is None:
                continue
            filename = mesh.get("filename")
            if not filename:
                continue
            mesh_path = path_prefix / filename.strip()
            mat_ref = visual.find("material")
            mat_name = mat_ref.get("name") if mat_ref is not None else None
            color_elem = mat_ref.find("color") if mat_ref is not None else None
            if color_elem is not None:
                rgba = _parse_rgba(color_elem)
            elif mat_name and mat_name in materials:
                rgba = materials[mat_name]
            else:
                rgba = (0.5, 0.5, 0.5, 1.0)
            entries.append(
                VisualEntry(
                    mesh_path=mesh_path,
                    origin_xyz=_parse_xyz(origin),
                    origin_rpy=_parse_rpy(origin),
                    rgba=rgba,
                )
            )
        if entries:
            result[name] = entries
    return result


def origin_to_matrix(
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float],
) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("xyz", rpy).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T


CUBE_SIZE = 0.025
GRIPPER_OPEN = 0.0
VIEWS = ("top", "side", "front")
VIEW_LABELS = ("Top", "Side", "Front")
CELL_SIZE = (320, 240)
CELL_MARGIN = 10
MAX_CASES = 6


def _setup_chain(urdf_path: Path):
    chain = pk.build_chain_from_urdf(open(urdf_path, mode="rb").read())
    serial_chain = pk.SerialChain(chain, "gripper_frame_link", "base_link")
    link_visuals = parse_urdf_visuals(urdf_path)
    full_joint_names = chain.get_joint_parameter_names()
    serial_joint_names = serial_chain.get_joint_parameter_names()
    return chain, serial_chain, link_visuals, full_joint_names, serial_joint_names


def _load_mesh(path: Path, cache: dict) -> trimesh.Trimesh | None:
    if path not in cache:
        if not path.exists():
            return None
        try:
            cache[path] = trimesh.load(path, force="mesh")
        except Exception:
            return None
    return cache[path].copy()


def build_scene(
    chain: pk.Chain,
    th_full: dict[str, torch.Tensor],
    link_visuals: dict,
    mesh_cache: dict,
    urdf_dir: Path,
    cube_pose: np.ndarray,
    cube_size: float = CUBE_SIZE,
) -> trimesh.Scene:
    ret = chain.forward_kinematics(th_full)
    scene = trimesh.Scene()
    for link_name, tg in ret.items():
        if link_name not in link_visuals:
            continue
        T_link = tg.get_matrix()[0].detach().numpy()
        for entry in link_visuals[link_name]:
            mesh = _load_mesh(entry.mesh_path, mesh_cache)
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
    box = trimesh.creation.box(extents=(cube_size, cube_size, cube_size))
    box.apply_transform(cube_pose)
    box.visual.face_colors = np.tile(np.array([200, 100, 50, 255], dtype=np.uint8), (len(box.faces), 1))
    scene.add_geometry(box, geom_name="cube")
    T_base = ret["base_link"].get_matrix()[0].detach().numpy()
    axis_base = trimesh.creation.axis(
        transform=T_base,
        origin_size=0.02,
        axis_length=0.08,
        axis_radius=0.008,
    )
    scene.add_geometry(axis_base, geom_name="base_axes")
    return scene


def _view_matrix(center: np.ndarray, cam_pos: np.ndarray, world_up: np.ndarray) -> np.ndarray:
    forward = center - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-9)
    right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-9)
    up = np.cross(right, forward)
    R = np.column_stack([right, up, -forward])
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ cam_pos
    return T


def _camera_transform_side(center: np.ndarray, distance: float) -> np.ndarray:
    cam_pos = center + distance * np.array([0.0, 1.0, 0.0])
    return _view_matrix(center, cam_pos, np.array([0.0, 0.0, 1.0]))


def _camera_transform_top(center: np.ndarray, distance: float) -> np.ndarray:
    cam_pos = center + distance * np.array([0.0, 0.0, 1.0])
    return _view_matrix(center, cam_pos, np.array([0.0, 1.0, 0.0]))


def _camera_transform_front(center: np.ndarray, distance: float) -> np.ndarray:
    cam_pos = center + distance * np.array([1.0, 0.0, 0.0])
    return _view_matrix(center, cam_pos, np.array([0.0, 0.0, 1.0]))


def _fixed_camera_center_and_distance(
    center_xyz: tuple[float, float, float],
    margin: float = 1.5,
) -> tuple[np.ndarray, float]:
    cx, cy, cz = center_xyz
    min_b = np.array([-0.02, -0.15, 0.0])
    max_b = np.array([max(0.35, cx + 0.08), 0.15, cz + 0.15])
    center = (min_b + max_b) / 2
    extent = max_b - min_b
    r = np.linalg.norm(extent) / 2 + 0.05
    distance = margin * r / np.tan(np.pi / 8)
    return center.astype(np.float64), float(distance)


def _trimesh_scene_to_pyrender(trimesh_scene: trimesh.Scene):
    import pyrender
    pr_scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4])
    for geom in trimesh_scene.geometry.values():
        if not isinstance(geom, trimesh.Trimesh):
            continue
        mesh = pyrender.Mesh.from_trimesh(geom, smooth=False)
        pr_scene.add(mesh)
    return pr_scene


def _render_pyrender(
    trimesh_scene: trimesh.Scene,
    view: str,
    center: np.ndarray,
    distance: float,
    resolution: tuple[int, int],
) -> np.ndarray:
    import pyrender
    pr_scene = _trimesh_scene_to_pyrender(trimesh_scene)
    if view == "side":
        view_matrix = _camera_transform_side(center, distance)
    elif view == "front":
        view_matrix = _camera_transform_front(center, distance)
    else:
        view_matrix = _camera_transform_top(center, distance)
    cam_pose = np.linalg.inv(view_matrix)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=resolution[0] / resolution[1])
    pr_scene.add(camera, pose=cam_pose)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    pr_scene.add(light, pose=cam_pose)
    r = pyrender.OffscreenRenderer(resolution[0], resolution[1])
    color, _ = r.render(pr_scene)
    r.delete()
    return color


def render_view(
    scene: trimesh.Scene,
    view: str,
    center: np.ndarray,
    distance: float,
    resolution: tuple[int, int],
) -> np.ndarray:
    return _render_pyrender(scene, view, center, distance, resolution)


def floor_cube_pose(x: float, y: float, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = (x, y, CUBE_SIZE / 2)
    return T


def label_cell(rgb: np.ndarray, view_label: str, case_label: str) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
    draw.rectangle([0, 0, rgb.shape[1], 44], fill=(40, 40, 40))
    draw.text((8, 6), view_label, fill=(255, 255, 255), font=font)
    draw.text((8, 24), case_label, fill=(200, 200, 200), font=font)
    return np.array(pil)


def show_pre_grasp_grid(
    urdf_path: Path,
    poses: list[tuple[np.ndarray, float]],
    *,
    cube_size: float = CUBE_SIZE,
    ik_solver: Callable[..., np.ndarray | dict | None],
    save_path: Path | None = None,
) -> None:
    urdf_path = Path(urdf_path)
    urdf_dir = urdf_path.parent
    chain, serial_chain, link_visuals, full_joint_names, serial_joint_names = _setup_chain(urdf_path)
    joint_names = list(serial_joint_names)
    mesh_cache: dict = {}

    solved: list[tuple[np.ndarray, float, np.ndarray]] = []
    for position_xyz, yaw in poses:
        pos = np.asarray(position_xyz).ravel()
        try:
            q = ik_solver(position_xyz, yaw, urdf_path)
        except TypeError:
            q = ik_solver(float(pos[0]), float(pos[1]), float(pos[2]), yaw)
        if q is None:
            continue
        if isinstance(q, dict):
            q = np.array([q[n] for n in joint_names], dtype=np.float64)
        solved.append((position_xyz, yaw, np.asarray(q, dtype=np.float64)))

    if not solved:
        print("No IK solutions found for the given poses.")
        return

    n_cases = min(len(solved), MAX_CASES)
    solved = solved[:n_cases]

    def th_from_arm_gripper(q_arm: np.ndarray, gripper_val: float) -> dict:
        q = np.asarray(q_arm, dtype=np.float64).ravel()
        th = {name: torch.tensor(0.0, dtype=torch.float64) for name in full_joint_names}
        for i, name in enumerate(serial_joint_names):
            th[name] = torch.tensor(q[i], dtype=torch.float64)
        th["gripper"] = torch.tensor(gripper_val, dtype=torch.float64)
        return th

    cell_images: list[list[np.ndarray]] = []
    for case_idx, (position_xyz, yaw, q_pre) in enumerate(solved):
        x, y, z = np.asarray(position_xyz).ravel()[:3]
        cube_pose = floor_cube_pose(float(x), float(y), float(yaw))
        th = th_from_arm_gripper(q_pre, GRIPPER_OPEN)
        scene = build_scene(chain, th, link_visuals, mesh_cache, urdf_dir, cube_pose, cube_size=cube_size)
        center, distance = _fixed_camera_center_and_distance((float(x), float(y), float(z)))
        distance = distance / 1.75
        row = []
        for view in VIEWS:
            rgb = render_view(scene, view, center, distance, CELL_SIZE)
            row.append(rgb)
        cell_images.append(row)

    _, ch = CELL_SIZE
    margin_color = np.array([72, 72, 72], dtype=np.uint8)
    margin_v = np.tile(margin_color, (ch, CELL_MARGIN, 1))
    grid_rows = []
    for view_idx in range(len(VIEWS)):
        row_cells = []
        for case_idx in range(n_cases):
            cell = cell_images[case_idx][view_idx]
            labeled = label_cell(
                cell,
                VIEW_LABELS[view_idx],
                f"Pose {case_idx + 1}",
            )
            row_cells.append(labeled)
        row_parts = [row_cells[0]]
        for i in range(1, n_cases):
            row_parts.append(margin_v)
            row_parts.append(row_cells[i])
        grid_rows.append(np.concatenate(row_parts, axis=1))
    margin_h = np.tile(
        margin_color,
        (CELL_MARGIN, grid_rows[0].shape[1], 1),
    )
    grid_parts = [grid_rows[0]]
    for i in range(1, len(grid_rows)):
        grid_parts.append(margin_h)
        grid_parts.append(grid_rows[i])
    grid = np.concatenate(grid_parts, axis=0)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        Image.fromarray(grid).save(save_path)
        return

    from PIL import Image as PILImage
    buf = io.BytesIO()
    PILImage.fromarray(grid).save(buf, format="png")
    buf.seek(0)
    from IPython.display import Image as IPImage
    display(IPImage(data=buf.getvalue()))
