#!/usr/bin/env python3
"""
Manual job submission CLI — inject a grading job into the Redis queue.

Usage examples:
  # Submit specific files for week 01 (results sent to chat_id 0, no Telegram reply)
  python -m submit_job --week 01 --files dev/01-intro-and-kinematics/homework/student_solutions/beads.py

  # Submit all .py files in a directory
  python -m submit_job --week 01 --dir dev/01-intro-and-kinematics/homework/student_solutions/

  # Override user identity (useful to test specific student re-grading)
  python -m submit_job --week 01 --files beads.py --user-id 123456 --chat-id 123456 --first-name Alice

  # Dry-run: show what would be submitted without pushing to queue
  python -m submit_job --week 01 --files beads.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure tools/ on path when run as script from repo root
_tools = Path(__file__).resolve().parent
if str(_tools) not in sys.path:
    sys.path.insert(0, str(_tools))


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        env_path = _tools.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass


def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser(
        description="Manually inject a grading job into the autograder Redis queue.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--week", required=True, help="Week ID, e.g. 01 or 02")
    parser.add_argument(
        "--files", nargs="*", metavar="FILE",
        help="One or more .py solution files to submit",
    )
    parser.add_argument(
        "--dir", metavar="DIR",
        help="Submit all .py files from this directory",
    )
    parser.add_argument("--chat-id", type=int, default=0, help="Telegram chat_id for result reply (0 = no reply)")
    parser.add_argument("--user-id", type=int, default=0, help="Telegram user_id for grade storage")
    parser.add_argument("--first-name", default="instructor", help="First name for grade records")
    parser.add_argument("--username", default="instructor", help="Username for grade records")
    parser.add_argument("--dry-run", action="store_true", help="Print job JSON without pushing to queue")
    parser.add_argument(
        "--redis-url",
        help="Redis URL (overrides REDIS_URL env var)",
    )
    args = parser.parse_args()

    # Collect files
    file_paths: list[Path] = []
    if args.files:
        for f in args.files:
            p = Path(f)
            if not p.exists():
                print(f"ERROR: file not found: {f}", file=sys.stderr)
                sys.exit(1)
            if not p.suffix == ".py":
                print(f"WARNING: skipping non-.py file: {f}", file=sys.stderr)
                continue
            file_paths.append(p)
    if args.dir:
        d = Path(args.dir)
        if not d.is_dir():
            print(f"ERROR: not a directory: {args.dir}", file=sys.stderr)
            sys.exit(1)
        file_paths.extend(sorted(d.glob("*.py")))

    if not file_paths:
        print("ERROR: no .py files specified (use --files or --dir)", file=sys.stderr)
        sys.exit(1)

    files: dict[str, str] = {}
    for p in file_paths:
        content = p.read_text(encoding="utf-8", errors="replace")
        files[p.name] = content
        print(f"  + {p.name}  ({len(content)} bytes)")

    # Validate week
    from shared.week_config import list_weeks
    available = list_weeks()
    if args.week not in available:
        print(f"ERROR: unknown week '{args.week}'. Available: {', '.join(available)}", file=sys.stderr)
        sys.exit(1)

    # Check files against expected solution_files
    from shared.week_config import get_solution_files
    expected = set(get_solution_files(args.week))
    submitted = set(files)
    unknown = submitted - expected
    missing = expected - submitted
    if unknown:
        print(f"WARNING: unknown files for week {args.week} (will be ignored by autograder): {sorted(unknown)}")
    if missing:
        print(f"WARNING: missing expected files for week {args.week}: {sorted(missing)}")
    if not (submitted & expected):
        print("ERROR: none of the submitted files match expected solution files. Aborting.", file=sys.stderr)
        sys.exit(1)

    from shared.schemas import Job
    job = Job(
        chat_id=args.chat_id,
        week_id=args.week,
        files=files,
        user_id=args.user_id,
        first_name=args.first_name,
        username=args.username,
    )
    payload = json.dumps(job.to_dict(), ensure_ascii=False, indent=2)

    if args.dry_run:
        print("\n--- DRY RUN: job payload (not submitted) ---")
        print(payload)
        return

    import os
    if args.redis_url:
        os.environ["REDIS_URL"] = args.redis_url

    from shared.redis_pool import get_redis
    from autograder.config import QUEUE_KEY

    try:
        r = get_redis()
        r.ping()
    except Exception as e:
        print(f"ERROR: cannot connect to Redis: {e}", file=sys.stderr)
        sys.exit(1)

    r.rpush(QUEUE_KEY, json.dumps(job.to_dict()))
    queue_len = r.llen(QUEUE_KEY)
    print(f"\nJob pushed to queue '{QUEUE_KEY}'. Queue length now: {queue_len}")
    print(f"Week: {args.week}  user_id: {args.user_id}  chat_id: {args.chat_id}  files: {sorted(files)}")
    if args.chat_id == 0:
        print("NOTE: chat_id=0 — autograder will attempt to send Telegram reply to chat 0 (likely fails; grades still stored).")


if __name__ == "__main__":
    main()
