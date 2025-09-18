import logging
import pytest
from collections import defaultdict
import main


class DummyResp:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


@pytest.mark.asyncio
async def test_vk_actor_auto_fallback_and_circuit_breaker(monkeypatch):
    monkeypatch.setattr(main, "_vk_captcha_needed", False)
    monkeypatch.setattr(main, "VK_ACTOR_MODE", "auto")
    monkeypatch.setattr(main, "VK_TOKEN", "g")
    monkeypatch.setenv("VK_USER_TOKEN", "u")
    monkeypatch.setattr(main, "_vk_user_token_bad", None)
    monkeypatch.setattr(main, "BACKOFF_DELAYS", [0])
    main.vk_fallback_group_to_user_total = defaultdict(int)
    main.vk_group_blocked.clear()

    now = 0
    monkeypatch.setattr(main._time, "time", lambda: now)

    calls: list[str] = []
    attempt = 0

    async def fake_http_call(name, method, url, timeout, data, **kwargs):
        nonlocal attempt
        attempt += 1
        calls.append(data["access_token"])
        if attempt == 1:
            return DummyResp({"error": {"error_code": 15, "error_msg": "access denied"}})
        return DummyResp({"response": "ok"})

    monkeypatch.setattr(main, "http_call", fake_http_call)

    data = await main._vk_api("wall.post", {}, db=None, bot=None)
    assert data["response"] == "ok"
    assert calls == ["g", "u"]

    calls.clear()
    data = await main._vk_api("wall.post", {}, db=None, bot=None)
    assert data["response"] == "ok"
    assert calls == ["u"]

    now += main.VK_CB_TTL + 1
    calls.clear()
    data = await main._vk_api("wall.post", {}, db=None, bot=None)
    assert data["response"] == "ok"
    assert calls == ["g"]


@pytest.mark.asyncio
async def test_vk_actor_auto_no_fallback(monkeypatch, caplog):
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

    caplog.set_level(logging.ERROR)
    with pytest.raises(main.VKAPIError) as e:
        await main._vk_api("wall.post", {}, db=None, bot=None)
    assert e.value.code == 3
    assert e.value.actor == "group"
    assert e.value.token == "<redacted>"
    assert calls == ["g"]
    assert main.vk_fallback_group_to_user_total["wall.post"] == 0
    assert any(
        "actor=group" in rec.getMessage() and "token=<redacted>" in rec.getMessage()
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_vk_actor_no_retry_edit_time_expired(monkeypatch):
    monkeypatch.setattr(main, "_vk_captcha_needed", False)
    monkeypatch.setattr(main, "VK_ACTOR_MODE", "group")
    monkeypatch.setattr(main, "VK_TOKEN", "g")
    monkeypatch.setattr(main, "_vk_user_token", lambda: None)
    monkeypatch.setattr(main, "BACKOFF_DELAYS", [0, 0, 0])

    attempts = 0

    async def fake_http_call(name, method, url, timeout, data, **kwargs):
        nonlocal attempts
        attempts += 1
        return DummyResp(
            {"error": {"error_code": 15, "error_msg": "Edit time expired"}}
        )

    monkeypatch.setattr(main, "http_call", fake_http_call)

    with pytest.raises(main.VKAPIError) as exc:
        await main._vk_api("wall.edit", {}, db=None, bot=None)

    assert exc.value.code == 15
    assert attempts == 1


def test_choose_vk_actor(monkeypatch):
    monkeypatch.setattr(main, "VK_MAIN_GROUP_ID", "1")
    monkeypatch.setattr(main, "VK_AFISHA_GROUP_ID", "2")
    monkeypatch.setattr(main, "VK_TOKEN", "gm")
    monkeypatch.setattr(main, "VK_TOKEN_AFISHA", "ga")
    monkeypatch.setattr(main, "VK_USER_TOKEN", "u")

    actors_main = main.choose_vk_actor(-1, "wall.post")
    assert [a.label for a in actors_main] == ["group:main", "user"]
    assert actors_main[0].token == "gm"

    actors_afisha = main.choose_vk_actor(-2, "wall.post")
    assert [a.label for a in actors_afisha] == ["group:afisha", "user"]
    assert actors_afisha[0].token == "ga"

