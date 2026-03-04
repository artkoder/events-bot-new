import pytest


@pytest.mark.asyncio
async def test_parse_event_via_gemma_retries_on_rate_limit(monkeypatch):
    from types import SimpleNamespace

    import main
    from google_ai.exceptions import RateLimitError

    class StubClient:
        def __init__(self):
            self.calls = 0

        async def generate_content_async(self, *args, **kwargs):  # noqa: ARG002
            self.calls += 1
            if self.calls == 1:
                raise RateLimitError(blocked_reason="tpm", retry_after_ms=10)
            usage = SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2)
            raw = '{"events":[{"title":"T","date":"2026-03-19","time":"19:00","location_name":"X"}]}'
            return raw, usage

    client = StubClient()

    async def noop_sleep(_sec):  # noqa: ARG001
        return None

    async def noop_log(*_args, **_kwargs):
        return None

    monkeypatch.setattr(main, "_get_event_parse_gemma_client", lambda: client)
    monkeypatch.setattr(main.asyncio, "sleep", noop_sleep)
    monkeypatch.setattr(main, "log_token_usage", noop_log)

    parsed = await main._parse_event_via_gemma("text", source_channel="ch")  # type: ignore[attr-defined]
    assert client.calls >= 2
    assert len(parsed) == 1
    assert parsed[0]["title"] == "T"
