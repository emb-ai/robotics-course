# Oracle + SGLang LLM in one container.
# Requires: NVIDIA GPU, nvidia-docker. For production, override SGLANG_MODEL (e.g. meta-llama/Llama-3.2-1B).
FROM lmsysorg/sglang:latest

WORKDIR /app

# System deps for search and env
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ripgrep \
    && rm -rf /var/lib/apt/lists/*

# Install rtk for token-optimized file operations
RUN curl -fsSL https://raw.githubusercontent.com/rtk-ai/rtk/refs/heads/master/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Oracle Python deps
RUN pip install --no-cache-dir \
    httpx \
    nbformat \
    pydantic \
    python-dotenv \
    pyyaml

# Copy oracle tools (oracle code must run; repo is cloned at runtime)
COPY tools/ /app/tools/

# Clone course repo from GitHub (no dev/, reflects published state)
ARG REPO_GIT_URL=https://github.com/emb-ai/robotics-course.git
ARG REPO_GIT_BRANCH=main
RUN git clone --depth 1 -b "${REPO_GIT_BRANCH}" "${REPO_GIT_URL}" /app/repo

ENV PYTHONPATH=/app/tools
ENV AI_ROBOTICS_REPO_ROOT=/app/repo

COPY tools/oracle/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# SGLang on 8000, Oracle on 9000
EXPOSE 8000 9000

ENTRYPOINT ["/entrypoint.sh"]
