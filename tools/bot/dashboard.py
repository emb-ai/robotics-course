"""Bot admin dashboard: allowlist, logs."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, render_template_string, request, redirect

from allowlist_db.store import list_users, add_user, remove_user
from shared.log_utils import tail

app = Flask(__name__)

LOG_TAIL_LINES = 100
LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "bot.log"

HTML = """
<!DOCTYPE html>
<html>
<head><title>Bot Admin</title>
<style>
  body { font-family: sans-serif; margin: 20px; }
  table { border-collapse: collapse; }
  th, td { border: 1px solid #ccc; padding: 6px 10px; }
  pre { background: #f5f5f5; padding: 10px; overflow-x: auto; max-height: 400px; overflow-y: auto; }
  form { display: inline; }
  .btn { padding: 4px 10px; cursor: pointer; }
  .btn-del { background: #faa; }
</style>
</head>
<body>
  <h1>Bot Admin</h1>
  <h2>Allowlist</h2>
  <form method="post" action="/add">
    <input type="number" name="telegram_id" placeholder="Telegram ID" required />
    <button type="submit">Add</button>
  </form>
  <table>
    <tr><th>Telegram ID</th><th>Action</th></tr>
    {% for uid in users %}
    <tr>
      <td>{{ uid }}</td>
      <td>
        <form method="post" action="/remove" style="display:inline">
          <input type="hidden" name="telegram_id" value="{{ uid }}" />
          <button type="submit" class="btn btn-del">Remove</button>
        </form>
      </td>
    </tr>
    {% endfor %}
  </table>
  <h2>Logs (last {{ n }} lines)</h2>
  <pre>{{ log_tail }}</pre>
</body>
</html>
"""


@app.route("/")
def index():
    users = list_users()
    log_tail = tail(LOG_FILE, LOG_TAIL_LINES)
    return render_template_string(HTML, users=users, log_tail=log_tail, n=LOG_TAIL_LINES)


@app.route("/add", methods=["POST"])
def do_add():
    uid = request.form.get("telegram_id")
    if uid:
        try:
            add_user(int(uid))
        except ValueError:
            pass
    return redirect("/")


@app.route("/remove", methods=["POST"])
def do_remove():
    uid = request.form.get("telegram_id")
    if uid:
        try:
            remove_user(int(uid))
        except ValueError:
            pass
    return redirect("/")


def run(port: int = 5001):
    # 0.0.0.0 required inside Docker so published ports reach Flask (127.0.0.1 is container-only)
    host = os.environ.get("DASHBOARD_BIND_HOST", "127.0.0.1")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    run()
