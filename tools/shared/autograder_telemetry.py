"""Redis keys so the admin dashboard can show job state (autograder updates, dashboard reads)."""

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

PROCESSING_KEY = "autograder:processing"
LAST_EVENT_KEY = "autograder:last_event"


def _payload(job: Any, status: str, detail: str = "") -> str:
    return json.dumps(
        {
            "status": status,
            "week_id": getattr(job, "week_id", ""),
            "chat_id": getattr(job, "chat_id", 0),
            "user_id": getattr(job, "user_id", 0),
            "detail": (detail or "")[:800],
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        },
        ensure_ascii=False,
    )


def mark_job_started(job: Any) -> None:
    try:
        from shared.redis_pool import get_redis

        r = get_redis()
        r.setex(PROCESSING_KEY, 7200, _payload(job, "started", "grading in progress"))
    except Exception as e:
        logger.debug("telemetry mark_job_started: %s", e)


def mark_job_finished(job: Any, status: str, detail: str = "") -> None:
    try:
        from shared.redis_pool import get_redis

        r = get_redis()
        r.delete(PROCESSING_KEY)
        r.set(LAST_EVENT_KEY, _payload(job, status, detail))
    except Exception as e:
        logger.debug("telemetry mark_job_finished: %s", e)
