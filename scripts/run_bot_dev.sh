#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${DB_PATH:=db_prod_snapshot.sqlite}"
: "${DEV_MODE:=1}"
: "${PYTHONUNBUFFERED:=1}"

export DB_PATH DEV_MODE PYTHONUNBUFFERED

exec python main.py

