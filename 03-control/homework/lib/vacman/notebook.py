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


def show_vacman_viewer(
    *,
    case_index: int = 0,
    height: int = 560,
    cat_debug: bool = False,
    nb_dir: Path | None = None,
) -> None:
    nb_dir = nb_dir or Path.cwd()
    js_path = nb_dir / "lib" / "vacman" / "viewer" / "main.js"
    js_path.read_text().replace("</script>", "<\\/script>")

    container_id = f"vacman-container-{uuid.uuid4().hex}"
    config = {
        "containerId": container_id,
        "caseIndex": case_index,
        "catDebug": cat_debug,
    }
    config_json = json.dumps(config).replace("</", "<\\/")
    viewer_version = uuid.uuid4().hex

    html = f"""<div id="{container_id}" style="position:relative;width:100%;min-height:{int(height)}px;background:#0d1117;overflow:hidden;"></div>
<script>
window.VACMAN_VISUALIZER = window.VACMAN_VISUALIZER || {{}};
window.VACMAN_VISUALIZER[{json.dumps(container_id)}] = {config_json};
</script>
<script type="module">
{JUPYTER_FILE_BASE_JS}
const moduleUrl = new URL("lib/vacman/viewer/main.js", homeworkFileBaseUrl());
moduleUrl.searchParams.set("v", {json.dumps(viewer_version)});
const {{ startVacmanViewer }} = await import(moduleUrl.href);
startVacmanViewer(window.VACMAN_VISUALIZER[{json.dumps(container_id)}]);
</script>"""
    display(HTML(html))
