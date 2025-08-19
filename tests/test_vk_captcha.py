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
        return DummyResp({
            "error": {
                "error_code": 14,
                "error_msg": "Captcha needed",
                "captcha_sid": "sid",
                "captcha_img": "img",
            }
        })
    monkeypatch.setattr(main, "http_call", fake_http_call)
    monkeypatch.setattr(main, "_vk_captcha_needed", False)
    monkeypatch.setattr(main, "_vk_captcha_sid", None)
    monkeypatch.setattr(main, "_vk_captcha_img", None)
    with pytest.raises(main.VKAPIError) as e1:
        await main._vk_api("wall.get", {}, token="t")
    assert e1.value.code == 14
    assert e1.value.captcha_sid == "sid"
    assert e1.value.captcha_img == "img"
    assert calls == 1
    with pytest.raises(main.VKAPIError) as e2:
        await main._vk_api("wall.get", {}, token="t")
    assert e2.value.captcha_sid == "sid"
    assert e2.value.captcha_img == "img"
    assert calls == 1
