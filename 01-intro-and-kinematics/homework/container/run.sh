#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):$PYTHONPATH"
pytest tests -v "$@"
