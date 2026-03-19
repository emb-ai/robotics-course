"""Autograder admin dashboard: queue, logs, rate limits, grades table, CSV export."""

import csv
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, render_template_string, send_file

from autograder.grades.store import get_leaderboard_pivoted, get_metric_leaderboard
from shared.autograder_telemetry import LAST_EVENT_KEY, PROCESSING_KEY
from shared.log_utils import tail

app = Flask(__name__)

LOG_TAIL_LINES = 100
LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "autograder.log"


def _redis_client():
    from shared.redis_pool import get_redis
    return get_redis()


HTML = """
<!DOCTYPE html>
<html>
<head><title>Autograder Admin</title>
<style>
  body { font-family: sans-serif; margin: 20px; }
  pre { background: #f5f5f5; padding: 10px; max-height: 400px; overflow: auto; }
  .stat { font-size: 1.2em; margin: 10px 0; }
  table { border-collapse: collapse; }
  th, td { border: 1px solid #ccc; padding: 6px 10px; }
  a { margin-right: 15px; }
</style>
</head>
<body>
  <h1>Autograder Admin</h1>
  <h2>Queue</h2>
  <p class="stat">Pending in Redis list: <strong>{{ queue_len }}</strong></p>
  <p><small>Jobs leave this queue as soon as a worker picks them up. “0” with a stuck student usually means the job already ran, failed, or timed out — see <strong>Last event</strong> below.</small></p>
  <h2>Current / last job</h2>
  <p><strong>Now processing</strong> (if any):</p>
  <pre>{{ processing }}</pre>
  <p><strong>Last finished event</strong> (autograder writes here after each run):</p>
  <pre>{{ last_event }}</pre>
  <h2>Autograder limits (reference)</h2>
  <ul>
    <li><strong>This page</strong> uses Redis at: {{ redis_url }} (compose service name <code>redis</code>).</li>
    <li><strong>Autograder worker</strong> (separate container, host network): <code>redis://127.0.0.1:6379/0</code> — same server, different URL.</li>
    <li>Per-homework <code>limits.timeout_sec</code> in <code>autograder.yaml</code> (base for docker subprocess): typically {{ timeout }} (see week configs).</li>
    <li><code>AUTOGRADER_DOCKER_OVERHEAD_SEC</code>: {{ overhead }} — added to that base for Docker pull/build before tests run.</li>
    <li>Container limits (compose / override): memory {{ memory }} MB, cpus {{ cpus }}</li>
  </ul>
  <p><small>Edit <code>tools/deploy/.env</code> and homework <code>autograder.yaml</code>; restart autograder.</small></p>
  <h2>Grades</h2>
  <p><a href="/grades">View grades table</a> | <a href="/grades/export">Export CSV</a> | <a href="/metrics">Metric leaderboards</a></p>
  <h2>Logs (last {{ n }} lines)</h2>
  <pre>{{ log_tail }}</pre>
</body>
</html>
"""

GRADES_HTML = """
<!DOCTYPE html>
<html>
<head><title>Grades Admin</title>
<style>
  body { font-family: sans-serif; margin: 20px; }
  table { border-collapse: collapse; }
  th, td { border: 1px solid #ccc; padding: 6px 10px; }
  a { margin-right: 15px; }
</style>
</head>
<body>
  <h1>Grades Admin</h1>
  <p><a href="/">← Back to Autograder</a> | <a href="/grades/export">Export CSV</a> | <a href="/metrics">Metric leaderboards</a></p>
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
    try:
        r = _redis_client()
        queue_len = r.llen("autograder:jobs")
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
        redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        timeout=os.environ.get("AUTOGRADER_TIMEOUT_SEC", "120"),
        overhead=os.environ.get("AUTOGRADER_DOCKER_OVERHEAD_SEC", "600"),
        memory=os.environ.get("AUTOGRADER_MEMORY_MB", "512"),
        cpus=os.environ.get("AUTOGRADER_CPUS", "1"),
        processing=processing,
        last_event=last_event,
        log_tail=log_tail,
        n=LOG_TAIL_LINES,
    )


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
    from shared.week_config import get_metrics_config, list_weeks

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
        <style>body{font-family:sans-serif;margin:20px} table{border-collapse:collapse} th,td{border:1px solid #ccc;padding:6px 10px} a{margin-right:15px}</style>
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
