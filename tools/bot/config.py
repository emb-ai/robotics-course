"""Bot config from env."""

from shared.env_helpers import env_bool, env_int, env_str

TELEGRAM_BOT_TOKEN = env_str("TELEGRAM_BOT_TOKEN")
REDIS_URL = env_str("REDIS_URL", "redis://localhost:6379/0")
QUEUE_KEY = "autograder:jobs"
RATE_LIMIT_SEC = env_int("RATE_LIMIT_SEC", 300)
RATE_LIMIT_KEY_PREFIX = "ratelimit:"
ORACLE_BASE_URL = env_str("ORACLE_BASE_URL", "http://localhost:9000")
ORACLE_ENABLED = env_bool("ORACLE_ENABLED", False)
ORACLE_TIMEOUT_SEC = env_int("ORACLE_TIMEOUT_SEC", 60)
