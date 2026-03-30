#!/usr/bin/env python3
"""
Differentiable FK to `wrist_cam_mount_link` + live digital twin (trimesh).

Mesh placement follows lerubick `digital_twin/robot_visualizer.py`: URDF visuals,
servo calibration -> radians, recursive link transforms.
"""
from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_kinematics as pk
import torch
import trimesh
from scipy.spatial.transform import Rotation

TARGET_FRAME = "wrist_cam_mount_link"
ROOT_FRAME = "base_link"
DEFAULT_HZ = 30.0

SERVO_RESOLUTION = 4096
RADIANS_PER_STEP = 2 * np.pi / SERVO_RESOLUTION

LINK_COLORS_RGBA = {
    "base": [180, 180, 190, 255],
    "shoulder": [200, 200, 210, 255],
    "upper_arm": [210, 210, 220, 255],
    "lower_arm": [220, 220, 230, 255],
    "wrist": [230, 230, 240, 255],
    "gripper": [220, 120, 70, 255],
    "jaw": [220, 120, 70, 255],
    "default": [200, 200, 210, 255],
}

try:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    LEROBOT_AVAILABLE = True
except ImportError:
    SO101Follower = None
    SO101FollowerConfig = None
    LEROBOT_AVAILABLE = False


def default_urdf_path() -> Path:
    return Path(__file__).resolve().parent / "so101" / "robot.urdf"


def resolve_seminar_path(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def load_lerobot_calibration(robot_id_or_path: str) -> dict | None:
    """Load lerobot servo calibration JSON (path, or `~/.cache/.../so101_follower/<id>.json`)."""
    path = Path(robot_id_or_path)
    if path.exists() and path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    cache_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "lerobot"
        / "calibration"
        / "robots"
        / "so101_follower"
    )
    calibration_path = cache_dir / f"{robot_id_or_path}.json"
    if calibration_path.exists():
        with open(calibration_path, encoding="utf-8") as f:
            return json.load(f)
    return None


def normalized_to_radians(
    normalized_angles: np.ndarray,
    joint_names: list[str],
    calibration: dict | None,
) -> np.ndarray:
    """Same convention as lerubick `robot_utils.kinematics.normalized_to_radians`."""
    angles_rad = np.zeros(len(normalized_angles))
    for i, (norm_val, joint_name) in enumerate(zip(normalized_angles, joint_names)):
        if calibration and joint_name in calibration:
            cal = calibration[joint_name]
            range_min = cal["range_min"]
            range_max = cal["range_max"]
        else:
            range_min = 0
            range_max = SERVO_RESOLUTION - 1
        servo_pos = ((norm_val + 100.0) / 200.0) * (range_max - range_min) + range_min
        center_pos = (range_min + range_max) / 2.0
        angles_rad[i] = (servo_pos - center_pos) * RADIANS_PER_STEP
    return angles_rad


def parse_urdf_joints(urdf_path: str | Path) -> dict:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints: dict[str, Any] = {}
    for joint in root.findall(".//joint"):
        joint_name = joint.get("name")
        joint_type = joint.get("type")
        parent_elem = joint.find("parent")
        child_elem = joint.find("child")
        if parent_elem is None or child_elem is None:
            continue
        origin = joint.find("origin")
        t_joint = np.eye(4)
        if origin is not None:
            xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()]
            rpy = [float(x) for x in origin.get("rpy", "0 0 0").split()]
            r_mat = Rotation.from_euler("xyz", rpy).as_matrix()
            t_joint[:3, :3] = r_mat
            t_joint[:3, 3] = xyz
        axis_elem = joint.find("axis")
        axis = np.array([0.0, 0.0, 1.0])
        if axis_elem is not None:
            axis = np.array([float(x) for x in axis_elem.get("xyz", "0 0 1").split()])
        joints[joint_name] = {
            "parent": parent_elem.get("link"),
            "child": child_elem.get("link"),
            "type": joint_type,
            "origin": t_joint,
            "axis": axis,
        }
    return joints


def compute_link_transforms(
    joints: dict,
    joint_angles_dict: dict[str, float],
) -> dict[str, np.ndarray]:
    children: dict[str, list[tuple[str, str]]] = {}
    for joint_name, joint_data in joints.items():
        parent = joint_data["parent"]
        if parent not in children:
            children[parent] = []
        children[parent].append((joint_name, joint_data["child"]))

    link_transforms: dict[str, np.ndarray] = {"base_link": np.eye(4)}

    def recurse(parent_link: str, t_parent: np.ndarray) -> None:
        if parent_link not in children:
            return
        for joint_name, child_link in children[parent_link]:
            joint_data = joints[joint_name]
            t_joint_origin = joint_data["origin"]
            t_joint_motion = np.eye(4)
            if joint_data["type"] == "revolute":
                angle = joint_angles_dict.get(joint_name, 0.0)
                axis = joint_data["axis"]
                r_joint = Rotation.from_rotvec(angle * axis).as_matrix()
                t_joint_motion[:3, :3] = r_joint
            t_child = t_parent @ t_joint_origin @ t_joint_motion
            link_transforms[child_link] = t_child
            recurse(child_link, t_child)

    recurse("base_link", np.eye(4))
    return link_transforms


def parse_urdf_joint_limits(urdf_path: str | Path) -> dict[str, tuple[float, float]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joint_limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall(".//joint"):
        joint_name = joint.get("name")
        joint_type = joint.get("type")
        parent_elem = joint.find("parent")
        if parent_elem is None:
            continue
        if joint_type == "revolute":
            limit_elem = joint.find("limit")
            if limit_elem is not None:
                lower = float(limit_elem.get("lower", "-3.14159"))
                upper = float(limit_elem.get("upper", "3.14159"))
                joint_limits[joint_name] = (lower, upper)
    return joint_limits


def mesh_scale_xyz_from_urdf(scale_attr: str | None) -> tuple[float, float, float]:
    """
    URDF mesh `scale` is three space-separated factors for x, y, z.
    If a single value is given, apply uniformly (common in the wild).
    """
    if not scale_attr or not scale_attr.strip():
        return (1.0, 1.0, 1.0)
    vals = [float(x) for x in scale_attr.split()]
    if len(vals) == 1:
        s = vals[0]
        return (s, s, s)
    if len(vals) == 2:
        return (vals[0], vals[1], vals[1])
    return (vals[0], vals[1], vals[2])


def parse_urdf_link_visuals(
    urdf_path: str | Path,
) -> dict[str, list[tuple[str, np.ndarray, tuple[float, float, float]]]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = Path(urdf_path).parent
    link_visuals: dict[str, list[tuple[str, np.ndarray, tuple[float, float, float]]]] = {}

    for link in root.findall(".//link"):
        link_name = link.get("name")
        visuals: list[tuple[str, np.ndarray, tuple[float, float, float]]] = []
        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            if geometry is None:
                continue
            mesh_elem = geometry.find("mesh")
            if mesh_elem is None:
                continue
            mesh_filename = mesh_elem.get("filename", "")
            if not mesh_filename:
                continue
            if mesh_filename.startswith("package://"):
                mesh_filename = mesh_filename.replace("package://", "")
            mesh_path = urdf_dir / mesh_filename
            if not mesh_path.exists():
                mesh_path = urdf_dir / Path(mesh_filename).name
            if not mesh_path.exists():
                continue
            origin = visual.find("origin")
            t_visual = np.eye(4)
            if origin is not None:
                xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()]
                rpy = [float(x) for x in origin.get("rpy", "0 0 0").split()]
                r_mat = Rotation.from_euler("xyz", rpy).as_matrix()
                t_visual[:3, :3] = r_mat
                t_visual[:3, 3] = xyz
            scale_xyz = mesh_scale_xyz_from_urdf(mesh_elem.get("scale"))
            visuals.append((str(mesh_path), t_visual, scale_xyz))
        if visuals:
            link_visuals[link_name] = visuals
    return link_visuals


def _link_color_rgba(link_name: str) -> list[int]:
    lower = link_name.lower()
    for key, rgba in LINK_COLORS_RGBA.items():
        if key in lower:
            return list(rgba)
    return list(LINK_COLORS_RGBA["default"])


def load_mesh_trimesh(mesh_path: str) -> trimesh.Trimesh:
    """Load mesh file; URDF scale is applied separately via `apply_urdf_mesh_scale`."""
    loaded = trimesh.load(mesh_path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        geom = [g.copy() for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geom:
            raise RuntimeError(f"No mesh geometry in scene: {mesh_path}")
        mesh = trimesh.util.concatenate(geom)
    else:
        mesh = loaded
    mesh.fix_normals()
    return mesh


def apply_urdf_mesh_scale(mesh: trimesh.Trimesh, scale_xyz: tuple[float, float, float]) -> None:
    """Apply URDF visual mesh scale (per-axis or uniform)."""
    sx, sy, sz = scale_xyz
    if sx == 1.0 and sy == 1.0 and sz == 1.0:
        return
    if sx == sy == sz:
        mesh.apply_scale(float(sx))
    else:
        mesh.apply_scale(np.array([sx, sy, sz], dtype=np.float64))


def joint_angles_dict_from_robot_observation(
    joint_state: dict[str, float],
    gripper_angle: float | None,
    joint_names_ordered: list[str],
    joint_limits: dict[str, tuple[float, float]],
    calibration: dict | None,
) -> dict[str, float]:
    arr = np.array([joint_state[n] for n in joint_names_ordered], dtype=np.float64)
    rad = normalized_to_radians(arr, joint_names_ordered, calibration)
    out = {n: float(r) for n, r in zip(joint_names_ordered, rad)}
    if gripper_angle is not None and "gripper" in joint_limits:
        lo, hi = joint_limits["gripper"]
        out["gripper"] = lo + (gripper_angle / 100.0) * (hi - lo)
    return out


def build_chain(
    urdf_path: str | Path | None = None,
    root_frame: str = ROOT_FRAME,
    target_frame: str = TARGET_FRAME,
) -> pk.chain.SerialChain:
    path = Path(urdf_path) if urdf_path is not None else default_urdf_path()
    if not path.exists():
        raise FileNotFoundError(f"URDF not found: {path}")

    urdf_bytes = path.read_bytes()
    return pk.build_serial_chain_from_urdf(
        data=urdf_bytes,
        end_link_name=target_frame,
        root_link_name=root_frame,
    )


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor, got shape={tuple(value.shape)}")
        return float(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"Expected scalar ndarray, got shape={value.shape}")
        return float(value.reshape(-1)[0])
    return float(value)


def _action_joint_names(robot: Any) -> list[str]:
    action_features = getattr(robot, "action_features", {})
    names = [
        key[:-4]
        for key in action_features.keys()
        if key.endswith(".pos") and key != "gripper.pos"
    ]
    return sorted(names)


def read_lerobot_joint_state(robot: Any) -> tuple[dict[str, float], float | None]:
    observation = robot.get_observation()
    if observation is None:
        raise RuntimeError("Robot observation is empty")

    joint_state: dict[str, float] = {}
    for joint_name in _action_joint_names(robot):
        key = f"{joint_name}.pos"
        if key in observation:
            joint_state[joint_name] = _to_float(observation[key])

    if not joint_state:
        raise RuntimeError("No joint positions found in robot observation")

    gripper = _to_float(observation["gripper.pos"]) if "gripper.pos" in observation else None
    return joint_state, gripper


def joint_vector_for_chain(
    chain: pk.chain.SerialChain,
    joint_state: dict[str, float],
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    chain_joint_names = list(chain.get_joint_parameter_names())
    missing = [name for name in chain_joint_names if name not in joint_state]
    if missing:
        raise KeyError(f"Missing joints in state: {missing}")

    q_values = [joint_state[name] for name in chain_joint_names]
    return torch.tensor(q_values, dtype=dtype, device=device)


def forward_camera_mount_transform(chain: pk.chain.SerialChain, q: torch.Tensor) -> torch.Tensor:
    q = q.to(dtype=torch.float32)
    if q.ndim == 1:
        q = q.unsqueeze(0)
    if q.ndim != 2:
        raise ValueError(f"Expected q shape [N, DoF] or [DoF], got {tuple(q.shape)}")

    ee_tf = chain.forward_kinematics(q)
    matrix = ee_tf.get_matrix()
    return matrix[0] if matrix.shape[0] == 1 else matrix


def make_robot(robot_id: str, port: str | None, calibration_dir: str | Path) -> Any:
    if not LEROBOT_AVAILABLE:
        raise RuntimeError(
            "lerobot is not installed. Activate env `ai_in_robotics` and install lerobot."
        )
    cal_dir = Path(calibration_dir)
    if not cal_dir.is_absolute():
        cal_dir = resolve_seminar_path(cal_dir)
    config = SO101FollowerConfig(
        id=robot_id,
        port=port,
        calibration_dir=cal_dir,
        use_degrees=False,
    )
    return SO101Follower(config)


def release_motors_for_viewer(robot: Any) -> None:
    """
    Disable servo torque on the lerobot follower bus (same idea as lerubick
    `disable_torque`): arm can move by hand while the viewer tracks pose.
    """
    bus = getattr(robot, "bus", None)
    if bus is None:
        print("Warning: robot has no `bus`; cannot disable torque.")
        return
    disable = getattr(bus, "disable_torque", None)
    if not callable(disable):
        print("Warning: motor bus has no `disable_torque()`; motors left as-is.")
        return
    try:
        disable()
        print("Motors: torque released (free motion).")
    except Exception as exc:
        print(f"Warning: could not disable torque: {exc}")


def build_robot_mesh_scene(
    urdf_path: str | Path,
    *,
    base_transform: np.ndarray | None = None,
) -> tuple[trimesh.Scene, dict[str, str], dict[str, Any]]:
    """
    Preload URDF meshes as trimesh nodes (lerubick-style).

    Returns scene, node_name -> link_name map, and metadata dict with keys:
    joints, joint_limits, link_visuals, mesh_cache_keys
    """
    urdf_path = Path(urdf_path)
    link_visuals = parse_urdf_link_visuals(urdf_path)
    joints = parse_urdf_joints(urdf_path)
    joint_limits = parse_urdf_joint_limits(urdf_path)

    scene = trimesh.Scene()
    base = np.eye(4) if base_transform is None else np.asarray(base_transform, dtype=np.float64)
    if not np.allclose(base, np.eye(4)):
        scene.graph.update(frame_to="world", matrix=base)

    node_to_link: dict[str, str] = {}
    mesh_cache: dict[str, trimesh.Trimesh] = {}

    for link_name, visuals in link_visuals.items():
        for idx, (mesh_path, _t_vis, scale_xyz) in enumerate(visuals):
            if mesh_path not in mesh_cache:
                mesh_cache[mesh_path] = load_mesh_trimesh(mesh_path)
            template = mesh_cache[mesh_path]
            node_name = f"mesh__{link_name}__{idx}"
            geom = template.copy()
            apply_urdf_mesh_scale(geom, scale_xyz)
            rgba = _link_color_rgba(link_name)
            if geom.visual.kind is None or geom.visual.vertex_colors is None:
                geom.visual.vertex_colors = np.tile(
                    np.array(rgba, dtype=np.uint8), (len(geom.vertices), 1)
                )
            scene.add_geometry(geom, node_name=node_name)
            node_to_link[node_name] = link_name

    meta = {
        "joints": joints,
        "joint_limits": joint_limits,
        "link_visuals": link_visuals,
        "mesh_cache": mesh_cache,
    }
    return scene, node_to_link, meta


def update_robot_mesh_scene(
    scene: trimesh.Scene,
    link_visuals: dict[str, list[tuple[str, np.ndarray, tuple[float, float, float]]]],
    link_transforms: dict[str, np.ndarray],
) -> None:
    """Apply `base @ T_link @ T_visual` to each mesh node."""
    for link_name, visuals in link_visuals.items():
        if link_name not in link_transforms:
            continue
        t_link = link_transforms[link_name]
        for idx, (_mesh_path, t_visual, _scale_xyz) in enumerate(visuals):
            node_name = f"mesh__{link_name}__{idx}"
            if node_name not in scene.graph.nodes:
                continue
            t_world = t_link @ t_visual
            scene.graph.update(frame_to=node_name, matrix=t_world)


def make_camera_mount_frustum(
    scale: float = 0.04,
    *,
    color_rgba: tuple[int, int, int, int] = (255, 128, 60, 255),
) -> trimesh.path.Path3D:
    """
    Wireframe pyramid for the camera-mount frame: apex at origin, opening toward +Z.

    Same convention as `camera_aruco_demo.make_camera_frustum` (seminar): optical axis ~ +Z,
    far plane corners at z = `scale` with half-extent scaled by 0.5 in x/y.
    """
    apex = np.zeros(3, dtype=np.float64)
    corners = np.array(
        [
            [-0.5, -0.5, 1.0],
            [0.5, -0.5, 1.0],
            [0.5, 0.5, 1.0],
            [-0.5, 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    corners *= float(scale)
    segments = np.array(
        [
            [apex, corners[0]],
            [apex, corners[1]],
            [apex, corners[2]],
            [apex, corners[3]],
            [corners[0], corners[1]],
            [corners[1], corners[2]],
            [corners[2], corners[3]],
            [corners[3], corners[0]],
        ],
        dtype=np.float64,
    )
    path = trimesh.load_path(segments)
    color = np.array(color_rgba, dtype=np.uint8)
    if len(path.entities) > 0:
        path.colors = np.tile(color, (len(path.entities), 1))
    return path


def run_live_viewer(
    robot: Any,
    chain: pk.chain.SerialChain,
    urdf_path: str | Path,
    calibration: dict | None,
    hz: float,
    print_every_s: float = 0.5,
    *,
    frustum_scale: float = 0.04,
) -> None:
    urdf_path = Path(urdf_path)
    joint_limits = parse_urdf_joint_limits(urdf_path)
    link_visuals = parse_urdf_link_visuals(urdf_path)
    joints_urdf = parse_urdf_joints(urdf_path)

    scene, _node_map, _meta = build_robot_mesh_scene(urdf_path)
    scene.add_geometry(
        trimesh.creation.axis(origin_size=0.006, axis_length=0.12),
        node_name="world_axis",
    )

    cam_frustum = make_camera_mount_frustum(scale=frustum_scale)
    scene.add_geometry(cam_frustum, node_name="camera_mount_marker")
    scene.graph.update(frame_to="camera_mount_marker", matrix=np.eye(4))

    n_meshes = sum(len(v) for v in link_visuals.values())
    print(f"Digital twin: loaded {n_meshes} mesh parts from URDF ({len(link_visuals)} links).")

    update_period = 1.0 / max(hz, 1e-6)
    last_print = 0.0

    def _callback(_scene: trimesh.Scene) -> None:
        nonlocal last_print
        start = time.time()
        try:
            joint_state, gripper_angle = read_lerobot_joint_state(robot)
            joint_names = _action_joint_names(robot)
            joint_angles_dict = joint_angles_dict_from_robot_observation(
                joint_state,
                gripper_angle,
                joint_names,
                joint_limits,
                calibration,
            )

            link_transforms = compute_link_transforms(joints_urdf, joint_angles_dict)
            update_robot_mesh_scene(_scene, link_visuals, link_transforms)

            q = joint_vector_for_chain(chain, joint_state)
            tf = forward_camera_mount_transform(chain, q)
            tf_np = tf.detach().cpu().numpy()
            _scene.graph.update(frame_to="camera_mount_marker", matrix=tf_np)

            now = time.time()
            if now - last_print >= print_every_s:
                pos = tf_np[:3, 3]
                print(
                    "camera_mount xyz(m): "
                    f"[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]"
                )
                last_print = now
        except Exception as exc:
            print(f"viewer update warning: {exc}")

        elapsed = time.time() - start
        remaining = update_period - elapsed
        if remaining > 0:
            time.sleep(remaining)

    scene.show(callback=_callback, smooth=False)


def _autograd_sanity(chain: pk.chain.SerialChain) -> bool:
    dof = len(chain.get_joint_parameter_names())
    q = torch.zeros(dof, dtype=torch.float32, requires_grad=True)
    tf = forward_camera_mount_transform(chain, q)
    loss = tf[:3, 3].norm()
    loss.backward()
    return q.grad is not None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Differentiable camera-mount FK + live trimesh digital twin"
    )
    parser.add_argument("--urdf-path", type=str, default=None, help="Path to robot URDF")
    parser.add_argument("--robot-id", type=str, default="robot_sasha", help="Robot id for lerobot")
    parser.add_argument("--robot-port", type=str, default=None, help="Robot serial port")
    parser.add_argument(
        "--calibration-dir",
        type=str,
        default="calibration_data",
        help="Directory with lerobot calibration files (relative to this seminar folder)",
    )
    parser.add_argument(
        "--calibration-json",
        type=str,
        default=None,
        help="Explicit path to servo calibration JSON (overrides default lookup)",
    )
    parser.add_argument("--hz", type=float, default=DEFAULT_HZ, help="Viewer update rate")
    parser.add_argument(
        "--autograd-check",
        action="store_true",
        help="Run a quick gradient sanity check and exit",
    )
    parser.add_argument(
        "--keep-torque",
        action="store_true",
        help="Keep servo torque enabled (default: release motors for viewer / free motion)",
    )
    parser.add_argument(
        "--frustum-scale",
        type=float,
        default=0.04,
        metavar="M",
        help="Camera-mount frustum size (meters); apex at link origin, opening along +Z",
    )
    args = parser.parse_args()

    urdf = Path(args.urdf_path) if args.urdf_path else default_urdf_path()
    if not urdf.exists():
        urdf = default_urdf_path()
    urdf = urdf.resolve()

    cal: dict | None = None
    if args.calibration_json:
        p = Path(args.calibration_json)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                cal = json.load(f)
            print(f"Loaded calibration JSON: {p}")
        else:
            print(f"Warning: calibration file not found: {args.calibration_json}")
    if cal is None:
        seminar_cal = resolve_seminar_path("calibration_data") / f"{args.robot_id}.json"
        if seminar_cal.exists():
            with open(seminar_cal, encoding="utf-8") as f:
                cal = json.load(f)
            print(f"Loaded calibration JSON: {seminar_cal}")
    if cal is None:
        cal = load_lerobot_calibration(args.robot_id)
        if cal:
            print(f"Loaded calibration from lerobot cache for id={args.robot_id!r}")
    if cal is None:
        print(
            "Warning: no servo calibration found; mesh pose uses default servo range mapping."
        )

    chain = build_chain(urdf)
    joint_names = list(chain.get_joint_parameter_names())
    print(f"Loaded chain {ROOT_FRAME} -> {TARGET_FRAME} with joints: {joint_names}")

    if args.autograd_check:
        ok = _autograd_sanity(chain)
        print(f"Autograd sanity check: {'OK' if ok else 'FAILED'}")
        return

    robot = make_robot(
        robot_id=args.robot_id,
        port=args.robot_port,
        calibration_dir=args.calibration_dir,
    )

    try:
        if not robot.is_connected:
            print("Connecting to robot...")
            robot.connect(calibrate=False)
        print("Robot connected.")
        if args.keep_torque:
            print("Motors: torque kept ON (--keep-torque).")
        else:
            release_motors_for_viewer(robot)
        print("Starting live viewer...")
        run_live_viewer(
            robot,
            chain,
            urdf,
            cal,
            hz=args.hz,
            frustum_scale=args.frustum_scale,
        )
    finally:
        if getattr(robot, "is_connected", False):
            robot.disconnect()
            print("Robot disconnected.")


if __name__ == "__main__":
    main()
