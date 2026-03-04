#!/usr/bin/env python3
"""
Kaggle kernel: probe Telegram web preview (cached_page + photo) for Telegraph URLs.

Inputs (via attached Kaggle datasets, searched under /kaggle/input):
- config.json: {"run_id": "...", "targets": [{"url": "...", "kind": "...", "ref_id": 123, "ref_key": "..."}], ...}
- secrets.enc + fernet.key: encrypted JSON with at least TG_API_ID/TG_API_HASH and either
  TELEGRAM_AUTH_BUNDLE_S22 (base64 JSON) or TG_SESSION (Telethon StringSession).

Output:
- telegraph_cache_report.json with per-URL probe results.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _install_if_missing() -> None:
    import subprocess

    def _pip(*args: str) -> None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

    try:
        import telethon  # noqa: F401
    except Exception:
        _pip("telethon>=1.34.0")

    try:
        import cryptography  # noqa: F401
    except Exception:
        _pip("cryptography>=41.0.0")


_install_if_missing()

from cryptography.fernet import Fernet  # noqa: E402
from telethon import TelegramClient  # noqa: E402
from telethon.errors import FloodWaitError  # noqa: E402
from telethon import functions  # noqa: E402
from telethon.sessions import StringSession  # noqa: E402
from telethon.tl.types import MessageMediaWebPage  # noqa: E402


@dataclass(frozen=True)
class Target:
    url: str
    kind: str
    ref_id: int | None = None
    ref_key: str | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_first(path: Path, name: str) -> Path | None:
    try:
        for p in path.rglob(name):
            if p.is_file():
                return p
    except Exception:
        return None
    return None


def _load_inputs() -> tuple[dict[str, Any], dict[str, str]]:
    root = Path("/kaggle/input")
    cfg_path = _find_first(root, "config.json")
    enc_path = _find_first(root, "secrets.enc")
    key_path = _find_first(root, "fernet.key")
    if not cfg_path:
        raise RuntimeError("config.json not found under /kaggle/input")
    if not enc_path or not key_path:
        raise RuntimeError("secrets.enc / fernet.key not found under /kaggle/input")

    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    fernet_key = key_path.read_bytes().strip()
    encrypted = enc_path.read_bytes()
    payload = Fernet(fernet_key).decrypt(encrypted).decode("utf-8")
    secrets = json.loads(payload)

    # Normalize secrets into a flat str dict (like env).
    env: dict[str, str] = {}
    for k, v in (secrets or {}).items():
        if v is None:
            continue
        env[str(k)] = str(v)
    return config, env


def _parse_auth_bundle(b64: str) -> dict[str, str]:
    raw = base64.urlsafe_b64decode(b64.encode("ascii")).decode("utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict) or not data.get("session"):
        raise RuntimeError("Invalid TELEGRAM_AUTH_BUNDLE payload")
    return {str(k): str(v) for k, v in data.items() if v is not None}


def _read_int(env: dict[str, str], key: str, *, fallback: int | None = None) -> int:
    raw = (env.get(key) or "").strip()
    if not raw:
        if fallback is None:
            raise RuntimeError(f"{key} is missing")
        return int(fallback)
    try:
        return int(raw)
    except Exception as exc:
        raise RuntimeError(f"Invalid {key}: expected int") from exc


def _read_str(env: dict[str, str], key: str, *, fallback: str | None = None) -> str:
    raw = (env.get(key) or "").strip()
    if raw:
        return raw
    if fallback is None:
        raise RuntimeError(f"{key} is missing")
    return str(fallback)


async def _sleep_jitter(min_sec: float, max_sec: float) -> None:
    await asyncio.sleep(random.uniform(max(0.0, min_sec), max(0.0, max_sec)))


async def _probe_one(
    client: TelegramClient,
    *,
    url: str,
    attach_wait_sec: int,
    delete_messages: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    sent = await client.send_message("me", url)
    webpage = None
    error = ""
    # Telegram can attach a basic preview first and fill `cached_page` a bit later.
    # Wait for `cached_page` (Instant View) within the same budget when possible.
    for _ in range(max(1, int(attach_wait_sec))):
        msg = await client.get_messages("me", ids=sent.id)
        media = getattr(msg, "media", None)
        if isinstance(media, MessageMediaWebPage):
            webpage = getattr(media, "webpage", None)
            if webpage and getattr(webpage, "cached_page", None):
                break
        await asyncio.sleep(1.0)
    if not webpage:
        error = "no_web_preview_attached"
    has_cached_page = bool(getattr(webpage, "cached_page", None)) if webpage else False
    has_photo = bool(getattr(webpage, "photo", None)) if webpage else False
    # Best-effort: force-refresh the full WebPage object when cached_page is missing
    # in the message preview (reduces false negatives).
    if webpage and not has_cached_page:
        try:
            refreshed = await client(
                functions.messages.GetWebPageRequest(url=url, hash=0)
            )
            if getattr(refreshed, "cached_page", None):
                webpage = refreshed
                has_cached_page = True
            if getattr(refreshed, "photo", None):
                has_photo = True
        except Exception:
            pass
    # Primary signal for Telegram “Instant View / cached preview” availability is `cached_page`.
    # A preview photo is nice-to-have, but it must not mark the page as a hard failure.
    ok = bool(webpage and has_cached_page)
    if webpage and not has_cached_page:
        error = error or "no_cached_page"
    if webpage and has_cached_page and not has_photo:
        error = error or "no_photo"
    title = getattr(webpage, "title", None) if webpage else None
    site_name = getattr(webpage, "site_name", None) if webpage else None
    if delete_messages:
        try:
            await client.delete_messages("me", sent.id)
        except Exception:
            pass
    took_ms = int((time.monotonic() - started) * 1000)
    return {
        "ok": ok,
        "has_cached_page": has_cached_page,
        "has_photo": has_photo,
        "title": title,
        "site_name": site_name,
        "error": error,
        "took_ms": took_ms,
    }


async def _probe_target(
    client: TelegramClient,
    target: Target,
    *,
    repeats: int,
    attach_wait_sec: int,
    per_url_timeout_sec: float,
    delete_messages: bool,
    pause_between_repeats_sec: tuple[float, float],
) -> dict[str, Any]:
    url = target.url
    attempts: list[dict[str, Any]] = []
    last_error = ""
    ok_any = False
    has_cached_any = False
    has_photo_any = False
    title = None
    site_name = None

    for attempt in range(1, max(1, int(repeats)) + 1):
        try:
            res = await asyncio.wait_for(
                _probe_one(
                    client,
                    url=url,
                    attach_wait_sec=attach_wait_sec,
                    delete_messages=delete_messages,
                ),
                timeout=max(1.0, float(per_url_timeout_sec)),
            )
        except FloodWaitError as e:
            wait = int(getattr(e, "seconds", 0) or 0)
            # Best-effort: respect small flood waits; abort on huge ones.
            if wait <= 0:
                last_error = "flood_wait"
                attempts.append({"ok": False, "error": last_error, "took_ms": 0})
                break
            if wait > 1800:
                last_error = f"flood_wait_too_long:{wait}"
                attempts.append({"ok": False, "error": last_error, "took_ms": 0})
                break
            await asyncio.sleep(wait + random.randint(6, 18))
            last_error = f"flood_wait:{wait}"
            attempts.append({"ok": False, "error": last_error, "took_ms": 0})
        except asyncio.TimeoutError:
            last_error = "timeout"
            attempts.append({"ok": False, "error": last_error, "took_ms": 0})
        except Exception as exc:
            last_error = f"error:{exc.__class__.__name__}"
            attempts.append({"ok": False, "error": last_error, "took_ms": 0})
        else:
            attempts.append(res)
            ok_any = ok_any or bool(res.get("ok"))
            has_cached_any = has_cached_any or bool(res.get("has_cached_page"))
            has_photo_any = has_photo_any or bool(res.get("has_photo"))
            title = res.get("title") or title
            site_name = res.get("site_name") or site_name
            last_error = str(res.get("error") or "").strip() or last_error

        if attempt < repeats:
            await _sleep_jitter(*pause_between_repeats_sec)

    return {
        "url": url,
        "kind": target.kind,
        "ref_id": target.ref_id,
        "ref_key": target.ref_key,
        "ok": bool(ok_any),
        "has_cached_page": bool(has_cached_any),
        "has_photo": bool(has_photo_any),
        "title": title,
        "site_name": site_name,
        "error": last_error,
        "attempts": attempts,
    }


async def main() -> None:
    random.seed(int(time.time()))
    config, env = _load_inputs()
    run_id = str(config.get("run_id") or "").strip() or "run"
    schema_version = int(config.get("schema_version") or 1)

    api_id = _read_int(env, "TG_API_ID")
    api_hash = _read_str(env, "TG_API_HASH")

    bundle_b64 = (env.get("TELEGRAM_AUTH_BUNDLE_S22") or env.get("TELEGRAM_AUTH_BUNDLE") or "").strip()
    session_string = (env.get("TG_SESSION") or env.get("TELEGRAM_SESSION") or "").strip()

    device_kwargs: dict[str, Any] = {}
    if bundle_b64:
        bundle = _parse_auth_bundle(bundle_b64)
        session_string = bundle.get("session", session_string)
        device_kwargs = {
            "device_model": bundle.get("device_model"),
            "system_version": bundle.get("system_version"),
            "app_version": bundle.get("app_version"),
            "lang_code": bundle.get("lang_code"),
            "system_lang_code": bundle.get("system_lang_code"),
        }

    if not session_string:
        raise RuntimeError("Missing Telegram session (TELEGRAM_AUTH_BUNDLE_* or TG_SESSION)")

    settings = dict(config.get("settings") or {})
    repeats = int(settings.get("repeats") or 2)
    attach_wait_sec = int(settings.get("attach_wait_sec") or 20)
    per_url_timeout_sec = float(settings.get("per_url_timeout_sec") or 35)
    delete_messages = bool(settings.get("delete_messages", True))
    delay_min = float(settings.get("delay_min_sec") or 1.0)
    delay_max = float(settings.get("delay_max_sec") or 2.2)
    repeat_pause_min = float(settings.get("repeat_pause_min_sec") or 1.5)
    repeat_pause_max = float(settings.get("repeat_pause_max_sec") or 3.5)

    raw_targets = list(config.get("targets") or [])
    targets: list[Target] = []
    for item in raw_targets:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            continue
        kind = str(item.get("kind") or "").strip() or "unknown"
        ref_id = item.get("ref_id")
        try:
            ref_id_i = int(ref_id) if ref_id is not None else None
        except Exception:
            ref_id_i = None
        ref_key = str(item.get("ref_key") or "").strip() or None
        targets.append(Target(url=url, kind=kind, ref_id=ref_id_i, ref_key=ref_key))

    started = time.monotonic()
    results: list[dict[str, Any]] = []
    async with TelegramClient(
        StringSession(session_string),
        api_id,
        api_hash,
        **{k: v for k, v in device_kwargs.items() if v},
    ) as client:
        me = await client.get_me()
        me_id = int(getattr(me, "id", 0) or 0)
        for idx, target in enumerate(targets, 1):
            res = await _probe_target(
                client,
                target,
                repeats=repeats,
                attach_wait_sec=attach_wait_sec,
                per_url_timeout_sec=per_url_timeout_sec,
                delete_messages=delete_messages,
                pause_between_repeats_sec=(repeat_pause_min, repeat_pause_max),
            )
            res["idx"] = idx
            results.append(res)
            # Human-like jitter between URLs.
            if idx < len(targets):
                await _sleep_jitter(delay_min, delay_max)

    took_sec = float(time.monotonic() - started)
    ok = sum(1 for r in results if r.get("ok"))
    fail = len(results) - ok
    payload = {
        "schema_version": schema_version,
        "generated_at": _utc_now_iso(),
        "run_id": run_id,
        "counts": {"total": len(results), "ok": ok, "fail": fail},
        "results": results,
        "took_sec": took_sec,
        "meta": {"me_id": me_id},
    }
    Path("telegraph_cache_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"OK: wrote telegraph_cache_report.json total={len(results)} ok={ok} fail={fail} took_sec={took_sec:.1f}")


if __name__ == "__main__":
    asyncio.run(main())
