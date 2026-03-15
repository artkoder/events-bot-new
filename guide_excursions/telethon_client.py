from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TelethonRuntimeAuth:
    api_id: int
    api_hash: str
    session: str
    device_model: str | None = None
    system_version: str | None = None
    app_version: str | None = None
    lang_code: str | None = None
    system_lang_code: str | None = None


def _decode_auth_bundle(bundle_b64: str) -> dict:
    raw = base64.urlsafe_b64decode(bundle_b64.encode("ascii")).decode("utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise EnvironmentError("TELEGRAM_AUTH_BUNDLE_E2E must decode to JSON object")
    return data


def resolve_telethon_runtime_auth() -> TelethonRuntimeAuth:
    api_id = os.environ.get("TELEGRAM_API_ID") or os.environ.get("TG_API_ID")
    api_hash = os.environ.get("TELEGRAM_API_HASH") or os.environ.get("TG_API_HASH")
    if not api_id or not api_hash:
        raise EnvironmentError(
            "Missing TELEGRAM_API_ID/TELEGRAM_API_HASH or TG_API_ID/TG_API_HASH for guide scan"
        )

    session = (os.environ.get("TELEGRAM_SESSION") or "").strip()
    bundle_b64 = (os.environ.get("TELEGRAM_AUTH_BUNDLE_E2E") or "").strip()
    bundle: dict | None = None
    if not session and bundle_b64:
        bundle = _decode_auth_bundle(bundle_b64)
        session = str(bundle.get("session") or "").strip()
    if not session:
        raise EnvironmentError(
            "Missing TELEGRAM_SESSION or TELEGRAM_AUTH_BUNDLE_E2E for guide scan"
        )

    return TelethonRuntimeAuth(
        api_id=int(api_id),
        api_hash=str(api_hash),
        session=session,
        device_model=(bundle or {}).get("device_model"),
        system_version=(bundle or {}).get("system_version"),
        app_version=(bundle or {}).get("app_version"),
        lang_code=(bundle or {}).get("lang_code"),
        system_lang_code=(bundle or {}).get("system_lang_code"),
    )


async def create_telethon_runtime_client():
    try:
        from telethon import TelegramClient
        from telethon.sessions import StringSession
    except Exception as exc:
        raise RuntimeError("telethon is required for guide excursions monitoring") from exc

    auth = resolve_telethon_runtime_auth()
    client = TelegramClient(
        StringSession(auth.session),
        auth.api_id,
        auth.api_hash,
        device_model=auth.device_model or "iPhone 14 Pro",
        system_version=auth.system_version or "iOS 17.2",
        app_version=auth.app_version or "10.5.1",
        lang_code=auth.lang_code or "ru",
        system_lang_code=auth.system_lang_code or auth.lang_code or "ru",
    )
    await client.connect()
    if not await client.is_user_authorized():
        await client.disconnect()
        raise RuntimeError("Telethon runtime client is not authorized")
    return client
