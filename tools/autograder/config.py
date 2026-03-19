"""Autograder config from env."""

from shared.env_helpers import env_bool, env_int, env_str

REDIS_URL = env_str("REDIS_URL", "redis://localhost:6379/0")
TELEGRAM_BOT_TOKEN = env_str("TELEGRAM_BOT_TOKEN")
TIMEOUT_SEC = env_int("AUTOGRADER_TIMEOUT_SEC", 120)
MEMORY_MB = env_int("AUTOGRADER_MEMORY_MB", 512)
CPUS = env_int("AUTOGRADER_CPUS", 1)
QUEUE_KEY = "autograder:jobs"
# Concurrency: single-worker BLPOP processes one job at a time
ORACLE_BASE_URL = env_str("ORACLE_BASE_URL", "http://localhost:9000")
ORACLE_ENABLED = env_bool("ORACLE_ENABLED", False)
ORACLE_TIMEOUT_SEC = env_int("ORACLE_TIMEOUT_SEC", 60)
