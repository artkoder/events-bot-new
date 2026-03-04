#!/usr/bin/env bash
# Скачивает snapshot prod БД только если текущий файл отсутствует или "старше" заданного порога.
# Использование:
#   ./scripts/sync_prod_db_if_stale.sh [--max-age-hours 6] [--db ./db_prod_snapshot.sqlite] [--app APP_NAME]
#
# Примечание: это thin-wrapper над ./scripts/sync_prod_db.sh.

set -euo pipefail

MAX_AGE_HOURS="${MAX_AGE_HOURS:-6}"
DB_PATH_LOCAL="${DB_PATH:-./db_prod_snapshot.sqlite}"
APP_NAME=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --max-age-hours|--max_age_hours)
      MAX_AGE_HOURS="$2"
      shift 2
      ;;
    --db|--path|--output)
      DB_PATH_LOCAL="$2"
      shift 2
      ;;
    --app)
      APP_NAME="$2"
      shift 2
      ;;
    --help)
      echo "Использование: $0 [--max-age-hours 6] [--db ./db_prod_snapshot.sqlite] [--app APP_NAME]"
      exit 0
      ;;
    *)
      echo "Неизвестная опция: $1"
      exit 1
      ;;
  esac
done

if ! [[ "$MAX_AGE_HOURS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "❌ Ошибка: MAX_AGE_HOURS должен быть числом, получено: $MAX_AGE_HOURS"
  exit 1
fi

needs_sync=0
MARKER_PATH="${DB_PATH_LOCAL}.downloaded_at"
if [ ! -f "$DB_PATH_LOCAL" ]; then
  needs_sync=1
else
  # If we don't have a download marker yet, treat the snapshot as stale.
  # Otherwise the DB file mtime can be refreshed by local writes (E2E),
  # which makes the "age" check unreliable.
  if [ ! -f "$MARKER_PATH" ]; then
    needs_sync=1
  fi
  now_epoch="$(date +%s)"
  # IMPORTANT: do NOT use DB file mtime as a freshness signal. Local E2E runs
  # mutate the sqlite file, which makes the snapshot look "fresh" even if the
  # last download happened long ago. We track the actual download time via
  # a sidecar marker written by sync_prod_db.sh.
  if [ -f "$MARKER_PATH" ]; then
    file_epoch="$(stat -c %Y "$MARKER_PATH" 2>/dev/null || echo 0)"
  else
    file_epoch="$(stat -c %Y "$DB_PATH_LOCAL" 2>/dev/null || echo 0)"
  fi
  age_sec=$(( now_epoch - file_epoch ))
  max_age_sec="$(python3 - "$MAX_AGE_HOURS" <<'PY'
import sys
v=float(sys.argv[1])
print(int(v*3600))
PY
)"
  if [ "$age_sec" -ge "$max_age_sec" ]; then
    needs_sync=1
  fi
fi

if [ "$needs_sync" -eq 0 ]; then
  echo "✅ Snapshot свежий: $DB_PATH_LOCAL (<= ${MAX_AGE_HOURS}h)"
  exit 0
fi

echo "🔄 Snapshot отсутствует/устарел (> ${MAX_AGE_HOURS}h). Скачиваю свежий…"
args=(--output "$DB_PATH_LOCAL")
if [ -n "$APP_NAME" ]; then
  args=(--app "$APP_NAME" "${args[@]}")
fi
./scripts/sync_prod_db.sh "${args[@]}"
