#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 \"<prompt>\""
  echo "Writes output to artifacts/codex/reports/ (gitignored)."
  exit 2
fi

out_dir="${CODEX_REPORT_DIR:-artifacts/codex/reports}"
mkdir -p "$out_dir"

ts="$(date -u +%Y%m%dT%H%M%SZ)"
out_file="${out_dir}/${ts}.md"

codex exec --sandbox workspace-write -o "$out_file" "$*"
echo "Saved: $out_file"
