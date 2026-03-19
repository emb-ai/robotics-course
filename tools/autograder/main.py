#!/usr/bin/env python3
"""Autograder daemon: consume Redis queue, run jobs."""

import json
import logging
import signal
import sys
from pathlib import Path

import redis

# Ensure tools/ is on path for shared, grades_db
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.schemas import Job

from . import config as autograder_config
from .worker import process_job

logger = logging.getLogger(__name__)
shutdown = False


def main() -> None:
    import os

    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent.parent / ".env")
    except ImportError:
        pass

    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "autograder.log", encoding="utf-8"),
        ],
    )

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not set")
        sys.exit(1)

    try:
        from shared.redis_pool import get_redis
    except ImportError:
        logger.error("redis package not installed. pip install redis")
        sys.exit(1)

    r = get_redis()
    queue_key = autograder_config.QUEUE_KEY

    def on_sig(signum, frame):
        global shutdown
        shutdown = True

    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    logger.info("Autograder daemon started, consuming from %s", queue_key)

    while not shutdown:
        try:
            # BLPOP blocks until a job is available (timeout 5s for responsive shutdown)
            result = r.blpop(queue_key, timeout=5)
            if result is None:
                continue
            _, payload = result
            job = Job.from_dict(json.loads(payload))
            logger.info("Processing job: week=%s chat=%s", job.week_id, job.chat_id)
            try:
                process_job(job)
            except Exception:
                logger.exception("Job failed")
                try:
                    from .worker import _send_telegram
                    _send_telegram(job.chat_id, "An error occurred during grading. Please try again later.")
                except Exception:
                    pass
        except redis.ConnectionError:
            logger.warning("Redis connection lost, retrying...")
        except Exception:
            logger.exception("Unexpected error")
            if shutdown:
                break

    logger.info("Autograder daemon stopped")


if __name__ == "__main__":
    main()
