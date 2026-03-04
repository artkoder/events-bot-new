#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = (raw or "").strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = (k or "").strip()
        if not key or key in os.environ:
            continue
        val = (v or "").strip()
        if (len(val) >= 2) and (val[0] == val[-1]) and val[0] in {"'", '"'}:
            val = val[1:-1]
        if val:
            os.environ[key] = val


def _has_any(*keys: str) -> bool:
    return any((os.getenv(k) or "").strip() for k in keys)


def _need_any(label: str, keys: list[str]) -> list[str]:
    return [] if _has_any(*keys) else [f"{label}: one of {', '.join(keys)}"]


def _need_all(keys: list[str]) -> list[str]:
    missing = []
    for k in keys:
        if not (os.getenv(k) or "").strip():
            missing.append(k)
    return missing


def _probe_url_head(url: str, *, timeout: int = 3) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(getattr(resp, "status", 0) or 0)
        ok = 200 <= status < 500
        return ok, f"status={status}"
    except Exception as exc:
        return False, f"error={type(exc).__name__}:{exc}"


def _telegram_getme(token: str) -> tuple[bool, str]:
    url = f"https://api.telegram.org/bot{token}/getMe"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return False, f"getMe_error={type(exc).__name__}:{exc}"
    if not isinstance(payload, dict) or not payload.get("ok"):
        return False, f"getMe_failed={payload!r}"
    username = ((payload.get("result") or {}).get("username") or "").strip()
    return True, f"bot_username=@{username}" if username else "bot_username=(empty)"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    _load_dotenv(root / ".env")

    issues: list[str] = []

    issues.extend(_need_all(["TELEGRAM_BOT_TOKEN"]))
    issues.extend(_need_any("E2E_SESSION", ["TELEGRAM_AUTH_BUNDLE_E2E", "TELEGRAM_SESSION"]))
    issues.extend(_need_any("E2E_API", ["TELEGRAM_API_ID", "TG_API_ID"]))
    issues.extend(_need_any("E2E_HASH", ["TELEGRAM_API_HASH", "TG_API_HASH"]))

    # Telegram Monitoring / Kaggle / Gemma
    issues.extend(_need_all(["GOOGLE_API_KEY", "KAGGLE_USERNAME", "KAGGLE_KEY"]))

    print("preflight_release_smoke")
    if issues:
        print("MISSING:")
        for item in issues:
            print(f"- {item}")
    else:
        print("OK: base env keys present")

    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    if token:
        ok, info = _telegram_getme(token)
        print(f"telegram_bot_api_getMe: {'OK' if ok else 'FAIL'} {info}")

    ok, info = _probe_url_head("https://catbox.moe")
    print(f"catbox.moe: {'OK' if ok else 'FAIL'} {info}")
    ok, info = _probe_url_head("https://files.catbox.moe")
    print(f"files.catbox.moe: {'OK' if ok else 'FAIL'} {info}")

    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())

