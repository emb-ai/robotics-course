from __future__ import annotations

import json
import uuid
from pathlib import Path

from IPython.display import HTML, display


JUPYTER_FILE_BASE_JS = """
function homeworkFileBaseUrl() {
  const origin = window.location.origin;
  const path = window.location.pathname;
  for (const route of ["/notebooks/", "/lab/tree/"]) {
    const idx = path.indexOf(route);
    if (idx !== -1) {
      const base = path.slice(0, idx + 1);
      const rel = path.slice(idx + route.length);
      const dir = rel.split("/").slice(0, -1).join("/");
      return new URL(`files/${dir ? `${dir}/` : ""}`, `${origin}${base}`).href;
    }
  }
  const baseUrl = document.body?.dataset?.baseUrl || "/";
  return new URL("files/", new URL(baseUrl, origin)).href;
}
"""


def show_rocket_viewer(
    *,
    initial_scenario: str = "ascent",
    wind_enabled: bool = True,
    height: int = 520,
    nb_dir: Path | None = None,
) -> None:
    nb_dir = nb_dir or Path.cwd()
    js_path = nb_dir / "lib" / "rocket" / "viewer" / "main.js"
    js_path.read_text().replace("</script>", "<\\/script>")

    canvas_id = f"rocket-canvas-{uuid.uuid4().hex}"
    config = {
        "canvasId": canvas_id,
        "initialScenario": initial_scenario,
        "windEnabled": wind_enabled,
    }
    config_json = json.dumps(config).replace("</", "<\\/")
    viewer_version = uuid.uuid4().hex

    html = f"""<canvas id="{canvas_id}" style="display:block;width:100%;min-height:{int(height)}px;background:#0a0a2e;"></canvas>
<script>
window.ROCKET_VISUALIZER = window.ROCKET_VISUALIZER || {{}};
window.ROCKET_VISUALIZER[{json.dumps(canvas_id)}] = {config_json};
</script>
<script type="module">
{JUPYTER_FILE_BASE_JS}
const moduleUrl = new URL("lib/rocket/viewer/main.js", homeworkFileBaseUrl());
moduleUrl.searchParams.set("v", {json.dumps(viewer_version)});
const {{ startRocketViewer }} = await import(moduleUrl.href);
startRocketViewer(window.ROCKET_VISUALIZER[{json.dumps(canvas_id)}]);
</script>"""
    display(HTML(html))
