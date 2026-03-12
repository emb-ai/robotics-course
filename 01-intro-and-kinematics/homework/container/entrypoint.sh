#!/usr/bin/env bash
set -e
cd /app/01-intro-and-kinematics/homework
if [ -d "reference_solution" ] && [ -n "$(ls -A reference_solution 2>/dev/null)" ]; then
  exec pytest tests/ hidden_tests/ -v "$@"
else
  exec pytest tests/ -v "$@"
fi
