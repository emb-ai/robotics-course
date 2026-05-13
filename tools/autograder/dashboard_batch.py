"""File-backed batch grading dashboard routes."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Blueprint, abort, redirect, render_template_string, request, send_file, url_for

from autograder.batch.discovery import discover_homeworks
from autograder.batch.leaderboard import aggregate_leaderboards, list_runs, per_run_leaderboards


_SAFE_RUN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


bp = Blueprint("dashboard_batch", __name__)


@bp.route("/batches")
def batches_index():
    runs = list_runs(_batch_output_root())
    return render_template_string(
        BATCHES_INDEX_HTML,
        runs=runs,
        output_root=str(_batch_output_root()),
    )


@bp.route("/batches/new", methods=["GET"])
def batch_new():
    discovery = discover_homeworks(_repo_root())
    return render_template_string(
        BATCH_NEW_HTML,
        homeworks=sorted(discovery.homeworks.values(), key=lambda spec: spec.id),
        default_run_id=_default_run_id(),
        default_output_root=str(_batch_output_root()),
        default_download_root=str(_dataschool_download_root()),
        llm_default=bool(os.environ.get("ORACLE_LLM_BASE_URL") and os.environ.get("ORACLE_LLM_MODEL")),
        warnings=discovery.warnings,
    )


@bp.route("/batches/new", methods=["POST"])
def batch_create():
    output_root = _batch_output_root()
    run_id = _require_safe_run_id(request.form.get("run_id", ""))
    run_dir = output_root / run_id
    if run_dir.exists():
        abort(400, f"Batch run already exists: {run_id}")
    homework_id = str(request.form.get("homework_id") or "").strip()
    discovery = discover_homeworks(_repo_root())
    if homework_id not in discovery.homeworks:
        abort(400, f"Unknown homework: {homework_id}")
    max_workers = _positive_int(request.form.get("max_workers"), "max workers")
    source_mode = str(request.form.get("source_mode") or "local")
    if source_mode not in {"local", "dataschool"}:
        abort(400, f"Unknown source mode: {source_mode}")

    job: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "homework_id": homework_id,
        "output_root": str(output_root),
        "max_workers": max_workers,
        "enable_diagnostics": request.form.get("enable_diagnostics") == "on",
        "enable_feedback": request.form.get("enable_feedback") == "on",
        "source_mode": source_mode,
        "created_at": _now(),
    }

    if source_mode == "local":
        submissions_root = Path(str(request.form.get("submissions_root") or "")).expanduser()
        if not submissions_root.is_dir():
            abort(400, f"Submissions root does not exist: {submissions_root}")
        job["submissions_root"] = str(submissions_root)
    else:
        _validate_dataschool_cookie(request.form.get("cookie_file", ""))
        job.update(
            {
                "queue_url": str(request.form.get("queue_url") or "").strip(),
                "course": str(request.form.get("course") or "").strip(),
                "assignments": str(request.form.get("assignments") or "").strip(),
                "statuses": str(request.form.get("statuses") or "on_checking").strip(),
                "sort": str(request.form.get("sort") or "solution_asc").strip(),
                "cookie_file": str(request.form.get("cookie_file") or "").strip(),
                "download_root": str(
                    Path(str(request.form.get("download_root") or _dataschool_download_root())).expanduser()
                ),
            }
        )
        if not job["queue_url"] and not job["course"]:
            abort(400, "DataSchool mode requires a queue URL or course filter.")
        if not job["queue_url"] and not job["assignments"]:
            abort(400, "DataSchool course mode requires an assignments filter.")

    run_dir.mkdir(parents=True)
    job_path = run_dir / "job.json"
    _atomic_write_json(job_path, job)
    cmd = [sys.executable, "-m", "autograder.batch.job_runner", "--job-config", str(job_path)]
    stdout = (run_dir / "dashboard_job_stdout.log").open("ab")
    stderr = (run_dir / "dashboard_job_stderr.log").open("ab")
    try:
        proc = subprocess.Popen(cmd, cwd=str(_repo_root()), env=_subprocess_env(), stdout=stdout, stderr=stderr)
    finally:
        stdout.close()
        stderr.close()
    _atomic_write_json(
        run_dir / "state.json",
        {
            "status": "queued",
            "counts": {"total": 0, "queued": 0, "running": 0, "completed": 0, "failed": 0, "scan_error": 0},
            "students": {},
            "pid": proc.pid,
            "started_at": _now(),
            "job_config": "job.json",
            "command": cmd[:3] + ["--job-config", "job.json"],
        },
    )
    return redirect(url_for("dashboard_batch.batch_run", run_id=run_id))


@bp.route("/batches/leaderboard")
def batch_aggregate_leaderboard():
    homework_id = request.args.get("homework_id") or None
    boards = aggregate_leaderboards(_batch_output_root(), _repo_root(), homework_id=homework_id)
    return render_template_string(AGGREGATE_LEADERBOARD_HTML, boards=boards, homework_id=homework_id or "")


@bp.route("/batches/<run_id>")
def batch_run(run_id: str):
    run_dir = _run_dir(run_id)
    state = _read_json(run_dir / "state.json", {})
    results = _read_json(run_dir / "results.json", {})
    boards = per_run_leaderboards(results, _repo_root()) if results else {"points": [], "metrics": []}
    return render_template_string(
        BATCH_RUN_HTML,
        run_id=run_id,
        state=state,
        results=results,
        students=results.get("students", []) if isinstance(results, dict) else [],
        boards=boards,
    )


@bp.route("/batches/<run_id>/students/<student_path_id>")
def batch_student(run_id: str, student_path_id: str):
    run_dir = _run_dir(run_id)
    student_dir = _safe_child(run_dir / "students", student_path_id)
    if not student_dir.is_dir():
        abort(404)
    student = _read_json(student_dir / "student.json", {})
    result = _read_json(student_dir / "result.json", {})
    artifacts = _read_json(student_dir / "artifacts.json", {}).get("artifacts", [])
    feedback = []
    feedback_dir = student_dir / "feedback"
    if feedback_dir.is_dir():
        for path in sorted(feedback_dir.glob("*.md")):
            feedback.append({"name": path.name, "text": path.read_text(encoding="utf-8", errors="replace")})
    submitted = sorted(path.name for path in (student_dir / "submitted").glob("*") if path.is_file())
    logs = {
        "stdout": _read_text(student_dir / "stdout.log"),
        "stderr": _read_text(student_dir / "stderr.log"),
    }
    return render_template_string(
        BATCH_STUDENT_HTML,
        run_id=run_id,
        student=student,
        result=result,
        artifacts=artifacts,
        feedback=feedback,
        submitted=submitted,
        logs=logs,
    )


@bp.route("/batches/<run_id>/summary.csv")
def batch_summary_csv(run_id: str):
    return _send_run_file(run_id, "summary.csv", "text/csv")


@bp.route("/batches/<run_id>/results.json")
def batch_results_json(run_id: str):
    return _send_run_file(run_id, "results.json", "application/json")


@bp.route("/batches/<run_id>/index.html")
def batch_static_index(run_id: str):
    return _send_run_file(run_id, "index.html", "text/html")


@bp.route("/batches/<run_id>/artifacts/<path:rel>")
def batch_artifact(run_id: str, rel: str):
    run_dir = _run_dir(run_id)
    path = _safe_child(run_dir, rel)
    if not path.is_file():
        abort(404)
    return send_file(path)


def _send_run_file(run_id: str, filename: str, mimetype: str):
    path = _run_dir(run_id) / filename
    if not path.is_file():
        abort(404)
    return send_file(path, mimetype=mimetype)


def _validate_dataschool_cookie(cookie_file: str | None) -> None:
    raw = str(cookie_file or "").strip()
    if raw:
        if not Path(raw).expanduser().is_file():
            abort(400, f"Cookie file does not exist: {raw}")
        return
    if not os.environ.get("DATASCHOOL_COOKIE"):
        abort(400, "DataSchool mode requires DATASCHOOL_COOKIE or a cookie file.")


def _positive_int(raw: str | None, label: str) -> int:
    try:
        value = int(raw or "0")
    except ValueError:
        abort(400, f"{label} must be a positive integer.")
    if value <= 0:
        abort(400, f"{label} must be a positive integer.")
    return value


def _require_safe_run_id(value: str) -> str:
    run_id = str(value or "").strip()
    if not _SAFE_RUN_RE.fullmatch(run_id) or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        abort(400, f"Unsafe run id: {value}")
    return run_id


def _run_dir(run_id: str) -> Path:
    safe = _require_safe_run_id(run_id)
    path = _batch_output_root() / safe
    if not path.is_dir():
        abort(404)
    return path


def _safe_child(root: Path, rel: str | Path) -> Path:
    base = root.resolve()
    candidate = (root / rel).resolve(strict=False)
    try:
        candidate.relative_to(base)
    except ValueError:
        abort(400)
    return candidate


def _batch_output_root() -> Path:
    return Path(os.environ.get("AUTOGRADER_BATCH_OUTPUT_ROOT") or (_repo_root() / "dev" / "grading_batches")).expanduser()


def _dataschool_download_root() -> Path:
    return Path(os.environ.get("DATASCHOOL_DOWNLOAD_ROOT") or (_repo_root() / "dev" / "downloads" / "dataschool_submissions")).expanduser()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    tools = str(_repo_root() / "tools")
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{tools}{os.pathsep}{current}" if current else tools
    return env


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            json.dump(data, tmp, ensure_ascii=False, indent=2, sort_keys=True)
            tmp.write("\n")
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _default_run_id() -> str:
    return "batch-" + datetime.now().strftime("%Y%m%d-%H%M%S")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


BATCHES_INDEX_HTML = """
<!doctype html>
<html><head><title>Batch grading</title>
<style>body{font-family:sans-serif;margin:20px;max-width:1200px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:6px 10px;text-align:left}th{background:#eee}a{margin-right:12px}.section{border:1px solid #ddd;border-radius:4px;padding:12px 16px;margin-bottom:20px}</style>
</head><body>
<h1>Batch grading</h1>
<p><a href="/">Back</a><a href="/batches/new">Start batch</a><a href="/batches/leaderboard">Aggregate leaderboard</a></p>
<p><small>Output root: <code>{{ output_root }}</code></small></p>
{% if runs %}
<table><thead><tr><th>Run</th><th>Homework</th><th>Status</th><th>Counts</th><th>Started</th><th>Finished</th><th>Reports</th></tr></thead><tbody>
{% for run in runs %}
<tr>
<td><a href="/batches/{{ run.run_id }}">{{ run.run_id }}</a></td>
<td>{{ run.homework_id }}</td>
<td>{{ run.status }}</td>
<td>{{ run.counts }}</td>
<td>{{ run.started_at or "" }}</td>
<td>{{ run.finished_at or "" }}</td>
<td><a href="/batches/{{ run.run_id }}/summary.csv">CSV</a><a href="/batches/{{ run.run_id }}/results.json">JSON</a><a href="/batches/{{ run.run_id }}/index.html">HTML</a></td>
</tr>
{% endfor %}
</tbody></table>
{% else %}
<p>No batch runs yet.</p>
{% endif %}
</body></html>
"""


BATCH_NEW_HTML = """
<!doctype html>
<html><head><title>Start batch</title>
<style>body{font-family:sans-serif;margin:20px;max-width:900px}label{display:block;margin:10px 0 4px}input,select{padding:5px 8px;min-width:360px}.inline{display:inline-block;min-width:auto}.section{border:1px solid #ddd;border-radius:4px;padding:12px 16px;margin-bottom:18px}button{padding:7px 16px;background:#0055cc;color:white;border:0;border-radius:3px}</style>
</head><body>
<h1>Start batch</h1>
<p><a href="/batches">Back to batches</a></p>
<form method="post" action="/batches/new">
<div class="section">
<label>Homework</label>
<select name="homework_id">{% for hw in homeworks %}<option value="{{ hw.id }}">{{ hw.id }} — {{ hw.topic_slug }}</option>{% endfor %}</select>
<label>Run id</label><input name="run_id" value="{{ default_run_id }}" required>
<label>Max workers</label><input name="max_workers" value="2" type="number" min="1" required>
<label><input class="inline" type="checkbox" name="enable_diagnostics" checked> Run diagnostics</label>
<label><input class="inline" type="checkbox" name="enable_feedback" {% if llm_default %}checked{% endif %}> Generate LLM drafts</label>
</div>
<div class="section">
<h2>Local submissions root</h2>
<label><input class="inline" type="radio" name="source_mode" value="local" checked> Use local folder</label>
<input name="submissions_root" placeholder="/path/to/one-folder-per-student">
</div>
<div class="section">
<h2>DataSchool</h2>
<label><input class="inline" type="radio" name="source_mode" value="dataschool"> Download then grade</label>
<label>Queue URL</label><input name="queue_url" placeholder="https://lk.dataschool.yandex.ru/teaching/assignments/?...">
<label>Course</label><input name="course" value="1704">
<label>Assignments</label><input name="assignments" placeholder="5898,5692,5635">
<label>Statuses</label><input name="statuses" value="on_checking">
<label>Sort</label><input name="sort" value="solution_asc">
<label>Cookie file path</label><input name="cookie_file" placeholder="optional if DATASCHOOL_COOKIE is set">
<label>Download root</label><input name="download_root" value="{{ default_download_root }}">
</div>
<button type="submit">Start</button>
</form>
{% if warnings %}<h2>Discovery warnings</h2><pre>{{ warnings }}</pre>{% endif %}
</body></html>
"""


BATCH_RUN_HTML = """
<!doctype html>
<html><head><title>Batch {{ run_id }}</title>
<meta http-equiv="refresh" content="10">
<style>body{font-family:sans-serif;margin:20px;max-width:1200px}table{border-collapse:collapse;width:100%;margin-bottom:20px}th,td{border:1px solid #ccc;padding:6px 10px;text-align:left}th{background:#eee}pre{background:#f5f5f5;padding:8px;white-space:pre-wrap;max-height:280px;overflow:auto}a{margin-right:12px}</style>
</head><body>
<h1>Batch {{ run_id }}</h1>
<p><a href="/batches">Batches</a><a href="/batches/{{ run_id }}/summary.csv">summary.csv</a><a href="/batches/{{ run_id }}/results.json">results.json</a><a href="/batches/{{ run_id }}/index.html">static report</a></p>
<h2>Status</h2>
<p><strong>{{ state.status or "unknown" }}</strong> {{ state.counts or {} }}</p>
<p>Active: {{ state.active_jobs or [] }}</p>
{% if state.error %}<pre>{{ state.error }}</pre>{% endif %}
<h2>Students</h2>
{% if students %}
<table><thead><tr><th>Student</th><th>Status</th><th>Points</th><th>Failed problems</th><th>Diagnostics</th><th>Feedback</th></tr></thead><tbody>
{% for student in students %}
{% set total = namespace(points=0, max=0, failed=[]) %}
{% for pid, problem in (student.problems or {}).items() %}
{% set total.points = total.points + (problem.points or 0) %}
{% set total.max = total.max + (problem.max_points or 0) %}
{% if problem.status != "passed" %}{% set _ = total.failed.append(pid ~ ": " ~ problem.message) %}{% endif %}
{% endfor %}
<tr>
<td><a href="/batches/{{ run_id }}/students/{{ student.student_path_id or student.student_id }}">{{ student.student_id }}</a></td>
<td>{{ student.status }}</td>
<td>{{ total.points }}/{{ total.max }}</td>
<td>{{ total.failed|join("; ") }}</td>
<td>{% for d in student.diagnostics or [] %}{{ d.summary }} {% endfor %}</td>
<td>{{ (student.feedback or [])|length }}</td>
</tr>
{% endfor %}
</tbody></table>
{% else %}<p>No results yet.</p>{% endif %}
<h2>Points leaderboard</h2>
<table><thead><tr><th>Rank</th><th>Student</th><th>Total</th></tr></thead><tbody>{% for row in boards.points %}<tr><td>{{ loop.index }}</td><td>{{ row.student_id }}</td><td>{{ row.total_points }}/{{ row.max_points }}</td></tr>{% endfor %}</tbody></table>
{% for board in boards.metrics %}
<h2>Metric: {{ board.name }}</h2>
<table><thead><tr><th>Rank</th><th>Student</th><th>Value</th></tr></thead><tbody>{% for row in board.rows %}<tr><td>{{ loop.index }}</td><td>{{ row.student_id }}</td><td>{{ row.value }}</td></tr>{% endfor %}</tbody></table>
{% endfor %}
</body></html>
"""


BATCH_STUDENT_HTML = """
<!doctype html>
<html><head><title>{{ student.student_id or result.student_id }}</title>
<style>body{font-family:sans-serif;margin:20px;max-width:1100px}table{border-collapse:collapse;width:100%;margin-bottom:20px}th,td{border:1px solid #ccc;padding:6px 10px;text-align:left}th{background:#eee}pre{background:#f5f5f5;padding:8px;white-space:pre-wrap;max-height:320px;overflow:auto}a{margin-right:12px}</style>
</head><body>
<h1>{{ student.student_id or result.student_id }}</h1>
<p><a href="/batches/{{ run_id }}">Back to run</a></p>
<h2>Submitted files</h2><p>{{ submitted|join(", ") }}</p>
<h2>Problems</h2>
<table><thead><tr><th>Problem</th><th>Status</th><th>Points</th><th>Message</th></tr></thead><tbody>
{% for pid, problem in (result.problems or {}).items() %}
<tr><td>{{ pid }}</td><td>{{ problem.status }}</td><td>{{ problem.points }}/{{ problem.max_points }}</td><td>{{ problem.message }}</td></tr>
{% endfor %}
</tbody></table>
<h2>Logs</h2>
<h3>stdout.log</h3><pre>{{ logs.stdout }}</pre>
<h3>stderr.log</h3><pre>{{ logs.stderr }}</pre>
<h2>Artifacts</h2>
<ul>{% for ref in artifacts %}<li>{{ ref.kind }} {{ ref.problem_id }}: <a href="/batches/{{ run_id }}/artifacts/{{ ref.path }}">{{ ref.label }}</a></li>{% endfor %}</ul>
<h2>Feedback</h2>
{% for item in feedback %}<h3>{{ item.name }}</h3><pre>{{ item.text }}</pre>{% endfor %}
</body></html>
"""


AGGREGATE_LEADERBOARD_HTML = """
<!doctype html>
<html><head><title>Batch aggregate leaderboard</title>
<style>body{font-family:sans-serif;margin:20px;max-width:1200px}table{border-collapse:collapse;width:100%;margin-bottom:22px}th,td{border:1px solid #ccc;padding:6px 10px;text-align:left}th{background:#eee}a{margin-right:12px}</style>
</head><body>
<h1>Batch aggregate leaderboard</h1>
<p><a href="/batches">Batches</a></p>
<h2>Best points{% if homework_id %} for homework {{ homework_id }}{% endif %}</h2>
<table><thead><tr><th>Rank</th><th>Homework</th><th>Student</th><th>Total</th><th>Run</th></tr></thead><tbody>
{% for row in boards.points %}<tr><td>{{ loop.index }}</td><td>{{ row.homework_id }}</td><td>{{ row.student_id }}</td><td>{{ row.total_points }}/{{ row.max_points }}</td><td><a href="/batches/{{ row.run_id }}">{{ row.run_id }}</a></td></tr>{% endfor %}
</tbody></table>
{% for board in boards.metrics %}
<h2>{{ board.homework_id }} {{ board.name }} ({{ board.direction }})</h2>
<table><thead><tr><th>Rank</th><th>Student</th><th>Value</th><th>Run</th></tr></thead><tbody>
{% for row in board.rows %}<tr><td>{{ loop.index }}</td><td>{{ row.student_id }}</td><td>{{ row.value }}</td><td><a href="/batches/{{ row.run_id }}">{{ row.run_id }}</a></td></tr>{% endfor %}
</tbody></table>
{% endfor %}
</body></html>
"""
