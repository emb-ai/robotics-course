from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from IPython.display import HTML, display

JOINT_LIMIT_RADIUS = np.pi / 3


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
        "bead_radius": 1.0,
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
