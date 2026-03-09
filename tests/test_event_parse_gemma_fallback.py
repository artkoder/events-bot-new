from types import SimpleNamespace

import pytest

import main


class _StubGemmaClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[str] = []

    async def generate_content_async(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.calls.append(kwargs.get("prompt", ""))
        raw = self._responses.pop(0)
        usage = SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2)
        return raw, usage


@pytest.mark.asyncio
async def test_parse_event_via_gemma_falls_back_to_4o_after_invalid_json(monkeypatch):
    client = _StubGemmaClient(["{bad json", "{still bad json"])
    fallback_seen = {}

    async def fake_log(*_args, **_kwargs):
        return None

    async def fake_notify(kind, payload):  # noqa: ANN001
        fallback_seen["incident"] = (kind, payload)

    async def fake_parse_via_4o(text, source_channel=None, **kwargs):  # noqa: ANN001
        fallback_seen["call"] = {
            "text": text,
            "source_channel": source_channel,
            "poster_texts": kwargs.get("poster_texts"),
        }
        return [{"title": "Fallback event", "date": "2026-03-09", "time": "19:00"}]

    monkeypatch.setattr(main, "_get_event_parse_gemma_client", lambda: client)
    monkeypatch.setattr(main, "log_token_usage", fake_log)
    monkeypatch.setattr(main, "notify_llm_incident", fake_notify)
    monkeypatch.setattr(main, "_parse_event_via_4o", fake_parse_via_4o)
    monkeypatch.setenv("FOUR_O_TOKEN", "test-token")

    parsed = await main._parse_event_via_gemma(  # type: ignore[attr-defined]
        "Листайте афиши и выбирайте события.",
        source_channel="Теплосеть",
        poster_texts=["Афиша 1"],
    )

    assert len(parsed) == 1
    assert parsed[0]["title"] == "Fallback event"
    assert fallback_seen["call"] == {
        "text": "Листайте афиши и выбирайте события.",
        "source_channel": "Теплосеть",
        "poster_texts": ["Афиша 1"],
    }
    kind, payload = fallback_seen["incident"]
    assert kind == "event_parse_gemma_fallback_4o"
    assert payload["next_model"] == "gpt-4o"
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_parse_event_via_gemma_prompt_allows_empty_array_for_image_only_intro(monkeypatch):
    client = _StubGemmaClient(["[]"])

    async def fake_log(*_args, **_kwargs):
        return None

    monkeypatch.setattr(main, "_get_event_parse_gemma_client", lambda: client)
    monkeypatch.setattr(main, "log_token_usage", fake_log)

    parsed = await main._parse_event_via_gemma(  # type: ignore[attr-defined]
        "Листайте афиши и выбирайте события, на которые хотите попасть.",
        source_channel="Теплосеть",
    )

    assert list(parsed) == []
    prompt = client.calls[0]
    assert "Return ONLY JSON: either a JSON array of events or a JSON object with an `events` array." in prompt
    assert "return [] as a valid empty JSON array" in prompt
