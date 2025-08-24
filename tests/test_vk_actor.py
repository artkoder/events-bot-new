import pytest
from collections import defaultdict
import main

class DummyResp:
    def __init__(self, data):
        self._data = data
    def json(self):
        return self._data

@pytest.mark.asyncio
async def test_vk_actor_auto_fallback(monkeypatch):
    monkeypatch.setattr(main, "_vk_captcha_needed", False)
    monkeypatch.setattr(main, "VK_ACTOR_MODE", "auto")
    monkeypatch.setattr(main, "VK_TOKEN", "g")
    monkeypatch.setenv("VK_USER_TOKEN", "u")
    monkeypatch.setattr(main, "_vk_user_token_bad", None)
    monkeypatch.setattr(main, "BACKOFF_DELAYS", [0])
    main.vk_fallback_group_to_user_total = defaultdict(int)
    main.vk_group_blocked.clear()

    calls = []
    async def fake_http_call(name, method, url, timeout, data, **kwargs):
        calls.append(data["access_token"])
        if len(calls) == 1:
            return DummyResp({"error": {"error_code": 200, "error_msg": "no"}})
        return DummyResp({"response": "ok"})

    monkeypatch.setattr(main, "http_call", fake_http_call)
    data = await main._vk_api("wall.post", {}, token=None, db=None, bot=None)
    assert data["response"] == "ok"
    assert calls == ["g", "u"]
    assert main.vk_fallback_group_to_user_total["wall.post"] == 1

@pytest.mark.asyncio
async def test_vk_actor_auto_no_fallback(monkeypatch):
    monkeypatch.setattr(main, "_vk_captcha_needed", False)
    monkeypatch.setattr(main, "VK_ACTOR_MODE", "auto")
    monkeypatch.setattr(main, "VK_TOKEN", "g")
    monkeypatch.setenv("VK_USER_TOKEN", "u")
    monkeypatch.setattr(main, "_vk_user_token_bad", None)
    monkeypatch.setattr(main, "BACKOFF_DELAYS", [0])
    main.vk_fallback_group_to_user_total = defaultdict(int)
    main.vk_group_blocked.clear()

    calls = []
    async def fake_http_call(name, method, url, timeout, data, **kwargs):
        calls.append(data["access_token"])
        return DummyResp({"error": {"error_code": 3, "error_msg": "bad"}})

    monkeypatch.setattr(main, "http_call", fake_http_call)
    with pytest.raises(main.VKAPIError) as e:
        await main._vk_api("wall.post", {}, db=None, bot=None)
    assert e.value.code == 3
    assert calls == ["g"]
    assert main.vk_fallback_group_to_user_total["wall.post"] == 0


@pytest.mark.asyncio
async def test_vk_actor_circuit_breaker(monkeypatch):
    """Permission error caches group failure"""
    monkeypatch.setattr(main, "_vk_captcha_needed", False)
    monkeypatch.setattr(main, "VK_ACTOR_MODE", "auto")
    monkeypatch.setattr(main, "VK_TOKEN", "g")
    monkeypatch.setenv("VK_USER_TOKEN", "u")
    monkeypatch.setattr(main, "_vk_user_token_bad", None)
    monkeypatch.setattr(main, "BACKOFF_DELAYS", [0])
    main.vk_fallback_group_to_user_total = defaultdict(int)
    main.vk_group_blocked.clear()

    calls = []
    async def fake_http_call(name, method, url, timeout, data, **kwargs):
        calls.append(data["access_token"])
        return DummyResp({"error": {"error_code": 200, "error_msg": "no"}})

    monkeypatch.setattr(main, "http_call", fake_http_call)
    # first attempt: group then user
    with pytest.raises(main.VKAPIError):
        await main._vk_api("wall.post", {}, db=None, bot=None)
    assert calls == ["g", "u"]
    assert main.vk_group_blocked.get("wall.post")

    calls.clear()
    # second attempt should skip group token
    with pytest.raises(main.VKAPIError):
        await main._vk_api("wall.post", {}, db=None, bot=None)
    assert calls == ["u"]
