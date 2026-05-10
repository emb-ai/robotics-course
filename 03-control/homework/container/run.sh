#!/usr/bin/env bash
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
export REPO_ROOT="$(pwd -P)"
docker compose -f 03-control/homework/container/docker_compose.yaml run --rm homework-tests "$@"
