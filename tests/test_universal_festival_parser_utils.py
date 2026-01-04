from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SRC_PATH = Path(__file__).resolve().parents[1] / "kaggle" / "UniversalFestivalParser" / "src"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


llm_logger_module = _load_module("festival_parser_llm_logger", SRC_PATH / "llm_logger.py")
rate_limit_module = _load_module("festival_parser_rate_limit", SRC_PATH / "rate_limit.py")
secrets_module = _load_module("festival_parser_secrets", SRC_PATH / "secrets.py")

LLMLogger = llm_logger_module.LLMLogger
GemmaRateLimiter = rate_limit_module.GemmaRateLimiter
get_api_key = secrets_module.get_api_key


def test_llm_logger_records_response_tokens_override() -> None:
    logger = LLMLogger("run-1")
    with logger.track("reason", model="test-model", prompt="hi") as tracker:
        tracker.set_response("hello", tokens=42)

    interaction = logger.interactions[0]
    assert interaction.response_tokens == 42


def test_llm_logger_records_exception() -> None:
    logger = LLMLogger("run-2")
    with pytest.raises(RuntimeError):
        with logger.track("reason", model="test-model", prompt="hi"):
            raise RuntimeError("boom")

    interaction = logger.interactions[0]
    assert interaction.success is False
    assert "boom" in (interaction.error or "")


@pytest.mark.asyncio
async def test_rate_limiter_acquire_context_manager() -> None:
    limiter = GemmaRateLimiter()
    async with limiter.acquire(estimated_tokens=1) as context:
        assert context.wait_time >= 0.0


def test_get_api_key_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    assert get_api_key() == "test-key"
