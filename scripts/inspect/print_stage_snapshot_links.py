#!/usr/bin/env python3
"""Print Telegraph URLs from E2E stage snapshots.

Usage:
  python scripts/inspect/print_stage_snapshot_links.py
  python scripts/inspect/print_stage_snapshot_links.py --dir artifacts/e2e/stage_snapshots/<run_dir>
  python scripts/inspect/print_stage_snapshot_links.py --filter figaro
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _pick_latest_snapshot_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", dest="dir_path", default="", help="Snapshot run dir (under artifacts/e2e/stage_snapshots)")
    ap.add_argument("--filter", dest="filter", default="", help="Only show labels containing this substring")
    args = ap.parse_args()

    root = Path("artifacts/e2e/stage_snapshots")
    if args.dir_path:
        snap_dir = Path(args.dir_path)
    else:
        snap_dir = _pick_latest_snapshot_dir(root)
    if not snap_dir or not snap_dir.exists():
        raise SystemExit(f"snapshot dir not found (root={root})")

    filt = (args.filter or "").strip().lower()
    items = sorted([p for p in snap_dir.glob("*.json") if p.is_file()])
    if not items:
        raise SystemExit(f"no json snapshots in {snap_dir}")

    print(f"snapshot_dir={snap_dir}")
    for path in items:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        label = str(data.get("label") or path.stem)
        if filt and filt not in label.lower():
            continue
        ev = data.get("event") or {}
        tel = data.get("telegraph") or {}
        event_id = ev.get("id")
        telegraph_url = tel.get("url") or ev.get("telegraph_url")
        snapshot_url = tel.get("snapshot_url")
        print("")
        print(f"label={label}")
        print(f"event_id={event_id}")
        print(f"telegraph_url={telegraph_url}")
        print(f"snapshot_url={snapshot_url}")
        artifact_json = data.get("artifact_json") or str(path)
        print(f"artifact_json={artifact_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

