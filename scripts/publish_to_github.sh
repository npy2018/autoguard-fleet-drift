#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 https://github.com/<account>/<repository>.git" >&2
  exit 2
fi
remote="$1"
if [[ ! -d .git ]]; then
  git init
  git add .
  git commit -m "Initial research-grade MVP"
fi
git branch -M main
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$remote"
else
  git remote add origin "$remote"
fi
git push -u origin main
