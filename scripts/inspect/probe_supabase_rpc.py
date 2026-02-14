#!/usr/bin/env python3
"""Probe Supabase PostgREST RPC routes from local env.

This script helps diagnose PGRST202/404 issues for RPC functions like google_ai_reserve.
It intentionally prints only status codes and short response heads (no secrets).

Usage:
  python scripts/inspect/probe_supabase_rpc.py google_ai_reserve --schema public
  python scripts/inspect/probe_supabase_rpc.py google_ai_reserve --schema private --use-service
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
    ap.add_argument("fn", help="RPC function name, e.g. google_ai_reserve")
    ap.add_argument("--schema", default=None, help="PostgREST schema (Content-Profile/Accept-Profile)")
    ap.add_argument(
        "--use-service",
        action="store_true",
        help="Use SUPABASE_SERVICE_KEY if available (recommended for privileged RPCs)",
    )
    ap.add_argument(
        "--payload-json",
        default=None,
        help="Raw JSON payload (string). If omitted, uses a safe default for known RPCs.",
    )
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

    if args.payload_json:
        payload = json.loads(args.payload_json)
    else:
        # Minimal defaults: enough to validate route+signature.
        if args.fn == "google_ai_reserve":
            payload = {
                "p_request_uid": "00000000-0000-0000-0000-000000000000",
                "p_attempt_no": 1,
                "p_consumer": "probe",
                "p_account_name": "probe",
                "p_model": "probe",
                "p_reserved_tpm": 1,
                "p_candidate_key_ids": None,
            }
        elif args.fn == "google_ai_finalize":
            payload = {
                "p_request_uid": "00000000-0000-0000-0000-000000000000",
                "p_attempt_no": 1,
                "p_usage_input_tokens": 1,
                "p_usage_output_tokens": 1,
                "p_usage_total_tokens": 2,
                "p_duration_ms": 1,
                "p_provider_status": "succeeded",
                "p_error_type": None,
                "p_error_code": None,
                "p_error_message": None,
            }
        else:
            payload = {}

    endpoint = url.rstrip("/") + f"/rest/v1/rpc/{args.fn}"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Accept-Profile": schema,
        "Content-Profile": schema,
    }

    resp = requests.post(endpoint, headers=headers, json=payload, timeout=20)
    body = (resp.text or "").replace("\n", " ")
    print("fn=", args.fn)
    print("schema=", schema)
    print("use_service=", bool(args.use_service and service))
    print("status=", resp.status_code)
    print("body_head=", body[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
