"""Shared Redis connection pool for bot, autograder, dashboards."""

import os
from typing import Optional

_redis_client: Optional["redis.Redis"] = None
_pool: Optional["redis.ConnectionPool"] = None


def get_redis():
    """Return a Redis client using a connection pool. Pool is shared per process."""
    global _redis_client, _pool
    if _redis_client is not None:
        return _redis_client
    try:
        import redis
    except ImportError:
        raise ImportError("redis package not installed. pip install redis")
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    _pool = redis.ConnectionPool.from_url(url)
    _redis_client = redis.Redis(connection_pool=_pool)
    return _redis_client
