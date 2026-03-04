"""
E2E Test Configuration - pytest fixtures for human-like Telegram testing.

Provides a session-scoped HumanUserClient fixture for E2E bot testing.
"""

import pytest
import pytest_asyncio
import asyncio
import os
import logging
import json
import urllib.request

from tests.e2e.human_client import HumanUserClient, create_human_client

logger = logging.getLogger(__name__)


def _load_dotenv_local() -> None:
    """Best-effort `.env` loader for local dev/E2E.

    Pytest does not auto-export `.env` into os.environ; many IDE runs rely on it.
    We set only missing variables to avoid overriding explicit env configuration.
    """
    try:
        from pathlib import Path

        env_path = Path(__file__).resolve().parents[2] / ".env"
    except Exception:
        return
    if not env_path.exists():
        return
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return
    for line in lines:
        s = (line or "").strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("export "):
            s = s[len("export ") :].strip()
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        key = (k or "").strip()
        if not key or key in os.environ:
            continue
        val = (v or "").strip()
        if (len(val) >= 2) and ((val[0] == val[-1]) and val[0] in ("'", '"')):
            val = val[1:-1]
        if val:
            os.environ[key] = val


def pytest_configure(config):
    """Register custom markers."""
    _load_dotenv_local()
    config.addinivalue_line(
        "markers",
        "e2e: marks tests as E2E tests requiring real Telegram credentials"
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def human_client() -> HumanUserClient:
    """
    Session-scoped fixture providing HumanUserClient.
    
    Requires environment variables:
    - TELEGRAM_API_ID
    - TELEGRAM_API_HASH
    - TELEGRAM_SESSION or TELEGRAM_AUTH_BUNDLE_E2E
    
    Skips tests if credentials are not available.
    """
    # TELEGRAM_API_* are preferred names for E2E, but TG_API_* are used across the repo.
    missing = []
    if not (os.environ.get("TELEGRAM_API_ID") or os.environ.get("TG_API_ID")):
        missing.append("TELEGRAM_API_ID or TG_API_ID")
    if not (os.environ.get("TELEGRAM_API_HASH") or os.environ.get("TG_API_HASH")):
        missing.append("TELEGRAM_API_HASH or TG_API_HASH")
    has_session = bool(os.environ.get("TELEGRAM_SESSION"))
    has_bundle = bool(os.environ.get("TELEGRAM_AUTH_BUNDLE_E2E"))

    if not has_session and not has_bundle:
        missing.append("TELEGRAM_SESSION or TELEGRAM_AUTH_BUNDLE_E2E")

    if missing:
        pytest.skip(
            f"E2E tests require Telegram credentials. "
            f"Missing: {', '.join(missing)}"
        )
    
    # Create and connect client
    client = create_human_client()
    
    try:
        await client.connect()
        logger.info("HumanUserClient connected for E2E tests")
        yield client
    finally:
        await client.disconnect()
        logger.info("HumanUserClient disconnected")


@pytest.fixture
def bot_username() -> str:
    """Target bot username for testing."""
    token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        pytest.skip("Missing TELEGRAM_BOT_TOKEN (required to resolve bot username for E2E tests)")
    url = f"https://api.telegram.org/bot{token}/getMe"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        pytest.skip(f"Failed to resolve bot username via getMe: {exc}")
    if not isinstance(payload, dict) or not payload.get("ok"):
        pytest.skip(f"Telegram getMe failed: {payload}")
    username = ((payload.get('result') or {}).get("username") or "").strip()
    if not username:
        pytest.skip("Telegram getMe returned empty username")
    return username
