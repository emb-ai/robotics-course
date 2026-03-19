"""Push grading jobs to Redis queue."""

import json

from shared.redis_pool import get_redis
from shared.schemas import Job

from . import config as bot_config


def push_job(job: Job) -> None:
    get_redis().rpush(bot_config.QUEUE_KEY, json.dumps(job.to_dict()))
