"""Autograder admin dashboard: queue, logs, rate limits, grades table, CSV export, manual submit."""

import csv
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, render_template_string, send_file, request, redirect, url_for, abort

from autograder import config as autograder_config
from autograder.grading_logs import get_grading_logs_root
from autograder.grades.store import get_leaderboard_pivoted, get_metric_leaderboard
from shared.autograder_telemetry import LAST_EVENT_KEY, PROCESSING_KEY
from shared.log_utils import tail
from shared.week_config import list_weeks

app = Flask(__name__)

LOG_TAIL_LINES = 150
LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "autograder.log"


def _redis_client():
    from shared.redis_pool import get_redis
    return get_redis()


HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Autograder Admin</title>
<meta http-equiv="refresh" content="10">
<style>
  body { font-family: sans-serif; margin: 20px; max-width: 1200px; }
  pre { background: #f5f5f5; padding: 10px; max-height: 500px; overflow: auto; font-size: 0.85em; white-space: pre-wrap; word-break: break-all; }
  .stat { font-size: 1.2em; margin: 10px 0; }
  table { border-collapse: collapse; width: 100%; }
  th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; }
  th { background: #eee; }
  a { margin-right: 15px; }
  .badge-ok { color: green; font-weight: bold; }
  .badge-err { color: red; font-weight: bold; }
  .section { border: 1px solid #ddd; border-radius: 4px; padding: 12px 16px; margin-bottom: 20px; }
  .submit-form input[type=number], .submit-form select { padding: 4px 8px; }
  .submit-form input[type=submit] { padding: 6px 16px; background: #0055cc; color: white; border: none; border-radius: 3px; cursor: pointer; }
  .flash { background: #d4edda; border: 1px solid #c3e6cb; padding: 10px 14px; border-radius: 4px; margin-bottom: 14px; }
  .flash-err { background: #f8d7da; border-color: #f5c6cb; }
</style>
</head>
<body>
  <h1>Autograder Admin</h1>
  <p><small>Page auto-refreshes every 10 seconds.</small></p>

  {% if flash %}
  <div class="flash {{ 'flash-err' if flash_err else '' }}">{{ flash }}</div>
  {% endif %}

  <div class="section">
    <h2>Queue</h2>
    <p class="stat">Pending in Redis list: <strong>{{ queue_len }}</strong></p>
    <p><small>Jobs leave the queue as soon as a worker picks them up. "0" with a stuck student usually means the job already ran, failed, or timed out — see <strong>Last event</strong> below.</small></p>
  </div>

  <div class="section">
    <h2>Current / last job</h2>
    <p><strong>Now processing</strong> (if any):</p>
    <pre>{{ processing }}</pre>
    <p><strong>Last finished event</strong>:</p>
    <pre>{{ last_event }}</pre>
  </div>

  <div class="section">
    <h2>Manual submission</h2>
    <p>Inject a grading job directly into the queue (bypasses Telegram). Useful for testing and instructor regrading.</p>
    <form method="post" action="/submit" enctype="multipart/form-data" class="submit-form">
      <table style="border:none; width:auto;">
        <tr>
          <td style="border:none; padding:4px 8px 4px 0">Week:</td>
          <td style="border:none; padding:4px 0">
            <select name="week_id">
              {% for w in weeks %}<option value="{{ w }}">{{ w }}</option>{% endfor %}
            </select>
          </td>
        </tr>
        <tr>
          <td style="border:none; padding:4px 8px 4px 0">Chat ID:</td>
          <td style="border:none; padding:4px 0"><input type="number" name="chat_id" value="0" required></td>
        </tr>
        <tr>
          <td style="border:none; padding:4px 8px 4px 0">User ID:</td>
          <td style="border:none; padding:4px 0"><input type="number" name="user_id" value="0" required></td>
        </tr>
        <tr>
          <td style="border:none; padding:4px 8px 4px 0">First name:</td>
          <td style="border:none; padding:4px 0"><input type="text" name="first_name" value="instructor"></td>
        </tr>
        <tr>
          <td style="border:none; padding:4px 8px 4px 0">Username:</td>
          <td style="border:none; padding:4px 0"><input type="text" name="username" value="instructor"></td>
        </tr>
        <tr>
          <td style="border:none; padding:4px 8px 4px 0">.py files:</td>
          <td style="border:none; padding:4px 0"><input type="file" name="files" multiple accept=".py"></td>
        </tr>
        <tr>
          <td colspan="2" style="border:none; padding:8px 0 0 0">
            <input type="submit" value="Submit for grading">
          </td>
        </tr>
      </table>
    </form>
  </div>

  <div class="section">
    <h2>Limits &amp; config</h2>
    <ul>
      <li>Redis: <code>{{ redis_url }}</code></li>
      <li>Queue key: <code>{{ queue_key }}</code></li>
      <li>Per-homework <code>limits.timeout_sec</code>: {{ timeout }}s base</li>
      <li><code>AUTOGRADER_DOCKER_OVERHEAD_SEC</code>: {{ overhead }}s (added for Docker pull/build)</li>
      <li>Default container limits: {{ memory }} MB, {{ cpus }} CPU</li>
    </ul>
  </div>

  <div class="section">
    <h2>Grades</h2>
    <p>
      <a href="/grades">View grades table</a>
      <a href="/grades/export">Export CSV</a>
      <a href="/metrics">Metric leaderboards</a>
      <a href="/grading_logs">Per-submission grading logs</a>
    </p>
    <p><small>On disk: <code>{{ grading_logs_root }}</code> (full stdout/stderr per run)</small></p>
  </div>

  <div class="section">
    <h2>Logs (last {{ n }} lines)</h2>
    <pre>{{ log_tail }}</pre>
  </div>
</body>
</html>
"""

GRADES_HTML = """
<!DOCTYPE html>
<html>
<head><title>Grades Admin</title>
<style>
  body { font-family: sans-serif; margin: 20px; }
  table { border-collapse: collapse; width: 100%; }
  th, td { border: 1px solid #ccc; padding: 6px 10px; }
  th { background: #eee; }
  a { margin-right: 15px; }
</style>
</head>
<body>
  <h1>Grades Admin</h1>
  <p>
    <a href="/">← Back to Autograder</a>
    <a href="/grades/export">Export CSV</a>
    <a href="/metrics">Metric leaderboards</a>
  </p>
  <table>
    <thead><tr>
      {% for c in cols %}<th>{{ c }}</th>{% endfor %}
    </tr></thead>
    <tbody>
    {% for row in rows %}
    <tr>
      {% for c in cols %}<td>{{ row.get(c, '') }}</td>{% endfor %}
    </tr>
    {% endfor %}
    </tbody>
  </table>
</body>
</html>
"""


@app.route("/")
def index():
    flash = request.args.get("flash", "")
    flash_err = request.args.get("flash_err", "0") == "1"
    try:
        r = _redis_client()
        queue_len = r.llen(autograder_config.QUEUE_KEY)
        raw_p = r.get(PROCESSING_KEY)
        processing = (
            raw_p.decode("utf-8", errors="replace")
            if isinstance(raw_p, bytes)
            else (raw_p or "(none)")
        )
        raw_l = r.get(LAST_EVENT_KEY)
        last_event = (
            raw_l.decode("utf-8", errors="replace")
            if isinstance(raw_l, bytes)
            else (raw_l or "(none yet)")
        )
    except Exception as e:
        queue_len = "?"
        processing = f"(redis error: {e})"
        last_event = processing
    log_tail = tail(LOG_FILE, LOG_TAIL_LINES)
    return render_template_string(
        HTML,
        queue_len=queue_len,
        queue_key=autograder_config.QUEUE_KEY,
        redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        timeout=os.environ.get("AUTOGRADER_TIMEOUT_SEC", "120"),
        overhead=os.environ.get("AUTOGRADER_DOCKER_OVERHEAD_SEC", "600"),
        memory=os.environ.get("AUTOGRADER_MEMORY_MB", "512"),
        cpus=os.environ.get("AUTOGRADER_CPUS", "1"),
        processing=processing,
        last_event=last_event,
        log_tail=log_tail,
        n=LOG_TAIL_LINES,
        weeks=list_weeks(),
        flash=flash,
        flash_err=flash_err,
        grading_logs_root=str(get_grading_logs_root()),
    )


def _resolve_grading_log(rel: str) -> Path | None:
    root = get_grading_logs_root().resolve()
    try:
        path = (root / rel).resolve()
    except OSError:
        return None
    if not str(path).startswith(str(root) + os.sep) and path != root:
        return None
    if not path.is_file():
        return None
    return path


@app.route("/grading_logs")
def grading_logs_index():
    root = get_grading_logs_root()
    rows: list[dict] = []
    if root.exists():
        for p in sorted(root.rglob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)[:80]:
            try:
                rel = p.relative_to(root)
            except ValueError:
                continue
            st = p.stat()
            rel_s = str(rel).replace(os.sep, "/")
            rows.append(
                {
                    "rel": rel_s,
                    "rel_url": quote(rel_s, safe=""),
                    "size": st.st_size,
                    "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
    return render_template_string(
        """
        <!DOCTYPE html><html><head><title>Grading logs</title>
        <style>body{font-family:sans-serif;margin:20px} table{border-collapse:collapse;width:100%}
        th,td{border:1px solid #ccc;padding:6px 10px;text-align:left} th{background:#eee}
        a{margin-right:12px}</style></head><body>
        <h1>Per-submission grading logs</h1>
        <p><a href="/">← Autograder</a> · Root: <code>{{ root }}</code></p>
        <p><small>Each file: header (week, user_id, status, exit_code) + full pytest/Docker stdout and stderr.</small></p>
        <table><thead><tr><th>Modified</th><th>File</th><th>Size</th></tr></thead><tbody>
        {% for r in rows %}
        <tr><td>{{ r.mtime }}</td><td><a href="/grading_logs/raw/{{ r.rel_url }}">{{ r.rel }}</a></td><td>{{ r.size }}</td></tr>
        {% endfor %}
        </tbody></table>
        {% if not rows %}<p>No log files yet.</p>{% endif %}
        </body></html>
        """,
        root=str(root),
        rows=rows,
    )


@app.route("/grading_logs/raw/<path:rel>")
def grading_logs_raw(rel: str):
    path = _resolve_grading_log(rel)
    if path is None:
        abort(404)
    return send_file(path, mimetype="text/plain; charset=utf-8", as_attachment=False)


@app.route("/submit", methods=["POST"])
def manual_submit():
    """Inject a grading job into the Redis queue from an uploaded .py file set."""
    try:
        week_id = request.form.get("week_id", "").strip()
        chat_id = int(request.form.get("chat_id", "0") or "0")
        user_id = int(request.form.get("user_id", "0") or "0")
        first_name = request.form.get("first_name", "instructor").strip() or "instructor"
        username = request.form.get("username", "instructor").strip() or "instructor"

        uploaded = request.files.getlist("files")
        files: dict[str, str] = {}
        for f in uploaded:
            if f and f.filename and f.filename.endswith(".py"):
                content = f.read().decode("utf-8", errors="replace")
                files[Path(f.filename).name] = content

        if not files:
            return redirect(url_for("index", flash="No .py files uploaded.", flash_err=1))
        if not week_id:
            return redirect(url_for("index", flash="No week selected.", flash_err=1))

        from shared.schemas import Job
        from shared.redis_pool import get_redis

        job = Job(
            chat_id=chat_id,
            week_id=week_id,
            files=files,
            user_id=user_id,
            first_name=first_name,
            username=username,
        )
        get_redis().rpush(autograder_config.QUEUE_KEY, json.dumps(job.to_dict()))
        return redirect(url_for("index", flash=f"Job queued: week={week_id} files={list(files)}"))
    except Exception as e:
        return redirect(url_for("index", flash=f"Submit error: {e}", flash_err=1))


def _grades_cols(rows):
    cols = ["display", "total_pts"]
    for r in rows:
        for k in r:
            if k not in ("telegram_id", "first_name", "username", "display", "total_pts") and k not in cols:
                cols.append(k)
    return ["display", "total_pts"] + sorted([c for c in cols if c not in ("display", "total_pts")])


@app.route("/grades")
def grades_index():
    rows = get_leaderboard_pivoted()
    cols = _grades_cols(rows) if rows else ["display"]
    return render_template_string(GRADES_HTML, rows=rows, cols=cols)


@app.route("/metrics")
def metrics_index():
    from shared.week_config import get_metrics_config

    boards = []
    for week_id in list_weeks():
        try:
            metrics_cfg = get_metrics_config(week_id)
        except ValueError:
            continue
        for problem_id, cfg in metrics_cfg.items():
            name = cfg.get("name", problem_id)
            direction = cfg.get("direction", "minimize")
            rows = get_metric_leaderboard(week_id, problem_id, direction, limit=50)
            boards.append({"week_id": week_id, "problem_id": problem_id, "name": name, "rows": rows})
    return render_template_string(
        """
        <!DOCTYPE html>
        <html>
        <head><title>Metric Leaderboards</title>
        <style>body{font-family:sans-serif;margin:20px} table{border-collapse:collapse} th,td{border:1px solid #ccc;padding:6px 10px} th{background:#eee} a{margin-right:15px}</style>
        </head>
        <body>
        <h1>Metric Leaderboards</h1>
        <p><a href="/">← Back to Autograder</a></p>
        {% for b in boards %}
        <h2>Week {{ b.week_id }} — {{ b.name }}</h2>
        <table><thead><tr><th>Rank</th><th>Student</th><th>Value</th></tr></thead>
        <tbody>
        {% for r in b.rows %}
        <tr><td>{{ loop.index }}</td><td>{{ r.display }}</td><td>{{ r.metric_value }}</td></tr>
        {% endfor %}
        </tbody></table>
        {% endfor %}
        {% if not boards %}<p>No metric leaderboards configured.</p>{% endif %}
        </body></html>
        """,
        boards=boards,
    )


@app.route("/grades/export")
def grades_export():
    rows = get_leaderboard_pivoted()
    cols = _grades_cols(rows) if rows else ["display"]
    bio = io.StringIO()
    w = csv.DictWriter(bio, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    bio.seek(0)
    return send_file(
        io.BytesIO(bio.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="grades.csv",
    )


def run(port: int = 5002):
    host = os.environ.get("DASHBOARD_BIND_HOST", "127.0.0.1")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    run()
