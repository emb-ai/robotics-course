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


def bounding_sphere_radius(
    link_lengths: np.ndarray,
    angles: np.ndarray,
) -> float:
    vertices = build_necklace(link_lengths, angles).astype(np.float64, copy=False)
    if len(vertices) == 0:
        return 0.0

    def _sphere_from_boundary(boundary: np.ndarray) -> tuple[float, np.ndarray]:
        """Return exact sphere passing through up to 4 boundary points."""
        if len(boundary) == 0:
            return 0.0, np.zeros(3, dtype=np.float64)
        if len(boundary) == 1:
            return 0.0, boundary[0].copy()

        p0 = boundary[0]
        if len(boundary) == 2:
            c = 0.5 * (boundary[0] + boundary[1])
            r = float(np.linalg.norm(boundary[0] - c))
            return r, c

        # Solve 2 * (pi - p0) dot c = ||pi||^2 - ||p0||^2 for i = 1..k-1.
        a = 2.0 * (boundary[1:] - p0)
        b = np.sum(boundary[1:] ** 2, axis=1) - np.dot(p0, p0)
        c, *_ = np.linalg.lstsq(a, b, rcond=None)
        c = c.astype(np.float64, copy=False)
        r = float(np.linalg.norm(boundary[0] - c))
        return r, c

    def _welzl(points: np.ndarray, boundary: list[np.ndarray]) -> tuple[float, np.ndarray]:
        eps = 1e-10
        if len(points) == 0 or len(boundary) == 4:
            return _sphere_from_boundary(np.asarray(boundary, dtype=np.float64))

        p = points[-1]
        r, c = _welzl(points[:-1], boundary)
        if np.linalg.norm(p - c) <= r + eps:
            return r, c

        boundary.append(p)
        r2, c2 = _welzl(points[:-1], boundary)
        boundary.pop()
        return r2, c2

    rng = np.random.default_rng(0xBAD5EED)
    perm = rng.permutation(len(vertices))
    shuffled = vertices[perm]
    radius, _ = _welzl(shuffled, [])
    return float(radius)


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
