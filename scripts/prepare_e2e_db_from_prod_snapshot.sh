#!/usr/bin/env bash
# Prepare an isolated writable DB copy for live E2E runs.
#
# Why: using db_prod_snapshot.sqlite directly will mutate it during E2E (events created/updated),
# which breaks test determinism and can fool "stale snapshot" checks based on mtime.
#
# Usage:
#   ./scripts/prepare_e2e_db_from_prod_snapshot.sh --max-age-hours 6
#   eval "$(./scripts/prepare_e2e_db_from_prod_snapshot.sh --max-age-hours 6)"
#
# Prints a single line: export DB_PATH=...

set -euo pipefail

MAX_AGE_HOURS="6"
SNAPSHOT_PATH="./db_prod_snapshot.sqlite"
OUT_DIR="./artifacts/test-results"

while [[ $# -gt 0 ]]; do
  case $1 in
    --max-age-hours|--max_age_hours)
      MAX_AGE_HOURS="$2"
      shift 2
      ;;
    --snapshot|--db|--path)
      SNAPSHOT_PATH="$2"
      shift 2
      ;;
    --out-dir|--out_dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [--max-age-hours 6] [--snapshot ./db_prod_snapshot.sqlite] [--out-dir ./artifacts/test-results]"
      echo "Prints: export DB_PATH=..."
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

./scripts/sync_prod_db_if_stale.sh --max-age-hours "$MAX_AGE_HOURS" --db "$SNAPSHOT_PATH" >/dev/null

mkdir -p "$OUT_DIR"
ts="$(date +%Y%m%d_%H%M%S)"
out_path="${OUT_DIR%/}/db_e2e_prod_snapshot_${ts}.sqlite"

python3 - "$SNAPSHOT_PATH" "$out_path" <<'PY'
import sqlite3
import sys
from pathlib import Path

src_path = Path(sys.argv[1])
dst_path = Path(sys.argv[2])

if not src_path.exists():
    raise SystemExit(f"Snapshot not found: {src_path}")

dst_path.parent.mkdir(parents=True, exist_ok=True)
if dst_path.exists():
    dst_path.unlink()

src = sqlite3.connect(str(src_path))
dst = sqlite3.connect(str(dst_path))
try:
    src.backup(dst)
finally:
    dst.close()
    src.close()
print(f"export DB_PATH={dst_path}")
PY

