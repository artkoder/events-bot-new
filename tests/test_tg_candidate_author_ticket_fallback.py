from __future__ import annotations

from types import SimpleNamespace

from source_parsing.telegram.handlers import _build_candidate


def _source() -> SimpleNamespace:
    return SimpleNamespace(
        default_location=None,
        default_ticket_link=None,
        trust_level="medium",
    )


def _message(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "source_username": "kraftmarket39",
        "source_type": "supergroup",
        "message_id": 101,
        "source_link": "https://t.me/kraftmarket39/101",
        "text": (
            "14.03 с 15.00 до 19.00\n\n"
            "Разговорный квартирник\n"
            "Стоимость 3300\n"
            "Запись по предоплате."
        ),
        "post_author": {
            "is_user": True,
            "is_channel": False,
            "is_chat": False,
            "user_id": 424242,
            "username": "elena_hod",
            "display_name": "Елена Ходаковская",
        },
        "events": [],
        "posters": [],
    }
    payload.update(overrides)
    return payload


def _event(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "title": "Разговорный квартирник",
        "date": "2026-03-14",
        "time": "15:00",
        "location_name": "Квартира в центре города",
        "city": "Калининград",
    }
    payload.update(overrides)
    return payload


def test_tg_build_candidate_uses_group_post_author_as_ticket_link_when_missing() -> None:
    cand = _build_candidate(_source(), _message(), _event())
    assert cand.ticket_link == "https://t.me/elena_hod"
    assert cand.metrics is not None
    assert cand.metrics.get("tg_ticket_link_from_post_author") is True


def test_tg_build_candidate_uses_tg_user_deep_link_when_author_has_no_username() -> None:
    message = _message(
        post_author={
            "is_user": True,
            "is_channel": False,
            "is_chat": False,
            "user_id": 424242,
            "username": None,
            "display_name": "Елена Ходаковская",
        }
    )
    cand = _build_candidate(_source(), message, _event())
    assert cand.ticket_link == "tg://user?id=424242"


def test_tg_build_candidate_does_not_override_phone_contact_with_post_author() -> None:
    message = _message(
        text=(
            "14.03 с 15.00 до 19.00\n\n"
            "Разговорный квартирник\n"
            "Стоимость 3300\n"
            "Запись по телефону +7 (900) 123-45-67."
        )
    )
    cand = _build_candidate(_source(), message, _event())
    assert cand.ticket_link is None


def test_tg_build_candidate_does_not_use_post_author_for_channel_posts() -> None:
    cand = _build_candidate(_source(), _message(source_type="channel"), _event())
    assert cand.ticket_link is None
