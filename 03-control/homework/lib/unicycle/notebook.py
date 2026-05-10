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


def show_unicycle_viewer(*, height: int = 560, nb_dir: Path | None = None) -> None:
    nb_dir = nb_dir or Path.cwd()
    js_path = nb_dir / "lib" / "unicycle" / "viewer" / "main.js"
    js_path.read_text().replace("</script>", "<\\/script>")

    container_id = f"unicycle-container-{uuid.uuid4().hex}"
    config = {"containerId": container_id}
    config_json = json.dumps(config).replace("</", "<\\/")
    viewer_version = uuid.uuid4().hex

    importmap = """<script type="importmap">
{"imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@0.184.0/build/three.module.js",
  "@mujoco/mujoco": "https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.7.0/mujoco.js"
}}</script>"""

    html = f"""{importmap}
<div id="{container_id}" style="position:relative;width:100%;min-height:{int(height)}px;background:#000;overflow:hidden;"></div>
<script>
window.UNICYCLE_VISUALIZER = window.UNICYCLE_VISUALIZER || {{}};
window.UNICYCLE_VISUALIZER[{json.dumps(container_id)}] = {config_json};
</script>
<script type="module">
{JUPYTER_FILE_BASE_JS}
const moduleUrl = new URL("lib/unicycle/viewer/main.js", homeworkFileBaseUrl());
moduleUrl.searchParams.set("v", {json.dumps(viewer_version)});
const {{ startUnicycleViewer }} = await import(moduleUrl.href);
startUnicycleViewer(window.UNICYCLE_VISUALIZER[{json.dumps(container_id)}]);
</script>"""
    display(HTML(html))
