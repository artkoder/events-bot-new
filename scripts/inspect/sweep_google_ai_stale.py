#!/usr/bin/env python3
"""Sweep stale Google AI reservations in Supabase.

Usage:
  python scripts/inspect/sweep_google_ai_stale.py --older-than-minutes 30 --limit 500
  python scripts/inspect/sweep_google_ai_stale.py --use-service --schema public
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests


def _load_dotenv() -> dict[str, str]:
    path = Path(".env")
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _get_env(key: str, env_file: dict[str, str]) -> str:
    return (os.getenv(key) or env_file.get(key) or "").strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default=None, help="PostgREST schema (default: env or public)")
    ap.add_argument(
        "--use-service",
        action="store_true",
        help="Use SUPABASE_SERVICE_KEY if available",
    )
    ap.add_argument("--older-than-minutes", type=int, default=30)
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()

    env_file = _load_dotenv()
    url = _get_env("SUPABASE_URL", env_file)
    if not url:
        raise SystemExit("SUPABASE_URL missing")

    key = _get_env("SUPABASE_KEY", env_file)
    service = _get_env("SUPABASE_SERVICE_KEY", env_file)
    if args.use_service and service:
        key = service
    if not key:
        raise SystemExit("SUPABASE_KEY missing (and SUPABASE_SERVICE_KEY not used/available)")

    schema = (args.schema or _get_env("SUPABASE_SCHEMA", env_file) or "public").strip() or "public"
    endpoint = url.rstrip("/") + "/rest/v1/rpc/google_ai_sweep_stale"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Accept-Profile": schema,
        "Content-Profile": schema,
    }
    payload = {
        "p_older_than_minutes": max(1, int(args.older_than_minutes)),
        "p_limit": max(1, int(args.limit)),
    }
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=30)
    print("status=", resp.status_code)
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text[:2000]}
    print(json.dumps(body, ensure_ascii=False, indent=2))
    return 0 if 200 <= resp.status_code < 300 else 1


if __name__ == "__main__":
    raise SystemExit(main())
