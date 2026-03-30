from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from IPython.display import HTML, display

JOINT_LIMIT_RADIUS = np.pi / 3
BEAD_SPHERE_RADIUS = 1.0
# Non-adjacent bead centers must be at least this far apart (touching spheres of radius BEAD_SPHERE_RADIUS).
BEAD_MIN_NONADJACENT_CENTER_DIST = 2.0 * BEAD_SPHERE_RADIUS - 1e-4


def bead_configuration_violations(
    link_lengths: np.ndarray,
    angles: np.ndarray,
    *,
    tol: float = 1e-5,
    min_center_dist: float = BEAD_MIN_NONADJACENT_CENTER_DIST,
) -> list[str]:
    """Return human-readable violation messages; empty list means constraints satisfied."""
    link_lengths = np.asarray(link_lengths, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    n_joints = len(link_lengths) - 1
    if angles.shape != (n_joints, 2):
        return [
            f"angles shape {angles.shape} != expected ({n_joints}, 2) for {len(link_lengths)} links"
        ]

    cos_limit = np.cos(JOINT_LIMIT_RADIUS + tol)
    for i in range(n_joints):
        ax, ay = float(angles[i, 0]), float(angles[i, 1])
        cos_angle = np.cos(ax) * np.cos(ay)
        if cos_angle < cos_limit:
            return [
                f"Joint {i} violates spherical cap: cos(θ)={cos_angle:.6f} < cos(limit)={cos_limit:.6f}"
            ]

    vertices = build_necklace(link_lengths, angles)
    n_points = len(vertices)
    for i in range(n_points):
        for j in range(i + 2, n_points):
            d = float(np.linalg.norm(vertices[i] - vertices[j]))
            if d < min_center_dist:
                return [
                    f"Sphere pair ({i},{j}) collides: distance {d:.6f} < {min_center_dist:.6f}"
                ]
    return []


def build_necklace(
    link_lengths: np.ndarray, 
    angles: np.ndarray,
) -> np.ndarray:
    individual_rotations = [
        R.from_euler("xyz", np.pad(a.astype(np.float64), (0, 1))) for a in angles
    ]
    cumulative_rotations = np.cumprod(individual_rotations)
    vectors = np.array([[0, 0, 1]], dtype=np.float64) * link_lengths[:, None]
    first_link = vectors[0:1]
    rest_rotated = R.concatenate(cumulative_rotations).apply(vectors[1:])
    rotated_vectors = np.vstack([first_link, rest_rotated])
    cumulative_vectors = np.cumsum(rotated_vectors, axis=0)
    vertices = np.vstack([np.zeros((1, 3)), cumulative_vectors])
    return vertices


def _beads_viewer_data(
    link_lengths: np.ndarray,
    angles: np.ndarray,
) -> dict:
    vertices = build_necklace(link_lengths, angles)
    return {
        "endpoints": vertices.tolist(),
        "bead_radius": BEAD_SPHERE_RADIUS,
    }


def show_beads_viewer(
    link_lengths: np.ndarray,
    angles: np.ndarray,
    nb_dir: Path | None = None,
) -> None:
    nb_dir = nb_dir or Path.cwd()
    lib_dir = nb_dir / "lib"
    js_path = lib_dir / "beads.js"
    js_code = js_path.read_text()
    js_code = js_code.replace("</script>", "<\\/script>")

    data = _beads_viewer_data(link_lengths, angles)
    container_id = "beads-viz-container"
    config = {
        "containerId": container_id,
        "endpoints": data["endpoints"],
        "bead_radius": data["bead_radius"],
    }
    config_json = json.dumps(config).replace("</", "<\\/")

    html = f"""<div id="{container_id}" style="width:100%; min-height:400px; background:#1a1a1a;"></div>
<script>window.BEADS_VISUALIZER = {config_json};</script>
<script type="module">
{js_code}
</script>"""
    display(HTML(html))
