"""Oracle config from env."""

from shared.env_helpers import env_bool, env_int, env_str

ORACLE_LLM_BASE_URL = env_str("ORACLE_LLM_BASE_URL", "http://localhost:8000/v1")
ORACLE_LLM_MODEL = env_str("ORACLE_LLM_MODEL", "glm-5-fp8")
ORACLE_ENABLED = env_bool("ORACLE_ENABLED", False)
ORACLE_TIMEOUT_SEC = env_int("ORACLE_TIMEOUT_SEC", 60)
