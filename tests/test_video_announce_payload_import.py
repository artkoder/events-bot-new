import json

import pytest

from video_announce.scenario import VideoAnnounceScenario


def _scenario():
    return VideoAnnounceScenario(db=None, bot=None, chat_id=0, user_id=0)


def test_parse_import_payload_ok():
    raw = json.dumps(
        {
            "intro": {"text": "AFISHA"},
            "scenes": [
                {
                    "about": "Text",
                    "description": "",
                    "date": "2024-01-01",
                    "location": "City",
                    "images": ["https://example.com/1.jpg"],
                    "is_free": True,
                }
            ],
        }
    )
    json_text, scene_count = _scenario()._parse_import_payload(raw)

    assert scene_count == 1
    parsed = json.loads(json_text)
    assert parsed["intro"]["text"] == "AFISHA"
    assert len(parsed["scenes"]) == 1


def test_parse_import_payload_rejects_invalid_json():
    with pytest.raises(ValueError):
        _scenario()._parse_import_payload("not-json")


def test_parse_import_payload_requires_intro_and_scenes():
    raw = json.dumps({"intro": {"text": "AFISHA"}})
    with pytest.raises(ValueError):
        _scenario()._parse_import_payload(raw)


def test_parse_import_payload_requires_scenes():
    raw = json.dumps({"intro": {"text": "AFISHA"}, "scenes": []})
    with pytest.raises(ValueError):
        _scenario()._parse_import_payload(raw)


def test_pick_default_kernel_prefers_local(monkeypatch):
    scenario = _scenario()
    monkeypatch.setattr(
        "video_announce.scenario.list_local_kernels",
        lambda: [{"ref": "local:VideoAfisha", "title": "Video"}],
    )
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)

    assert scenario._pick_default_kernel_ref() == "local:VideoAfisha"


def test_pick_default_kernel_falls_back_to_env(monkeypatch):
    scenario = _scenario()
    monkeypatch.setattr("video_announce.scenario.list_local_kernels", lambda: [])
    monkeypatch.setenv("KAGGLE_USERNAME", "codex")

    assert scenario._pick_default_kernel_ref() == "codex/video-announce-renderer"
