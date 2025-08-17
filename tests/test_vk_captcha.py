import pytest
import main

class DummyResp:
    def __init__(self, data):
        self._data = data
    def json(self):
        return self._data


@pytest.mark.asyncio
async def test_vk_api_captcha_cached(monkeypatch):
    calls = 0
    async def fake_http_call(*args, **kwargs):
        nonlocal calls
        calls += 1
        return DummyResp({"error": {"error_code": 14, "error_msg": "Captcha needed"}})
    monkeypatch.setattr(main, "http_call", fake_http_call)
    monkeypatch.setattr(main, "_vk_captcha_needed", False)
    with pytest.raises(main.VKAPIError) as e:
        await main._vk_api("wall.get", {}, token="t")
    assert e.value.code == 14
    assert calls == 1
    with pytest.raises(main.VKAPIError):
        await main._vk_api("wall.get", {}, token="t")
    assert calls == 1
