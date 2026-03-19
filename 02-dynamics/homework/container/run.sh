#!/usr/bin/env bash
set -e
repo_root="$(cd "$(dirname "$0")/../../.." && pwd -P)"
export REPO_ROOT="$repo_root"
cd "$repo_root"
exec docker compose -f 02-dynamics/homework/container/docker_compose.yaml run --rm homework-tests "$@"
