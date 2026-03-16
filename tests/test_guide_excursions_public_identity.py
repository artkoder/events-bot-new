from __future__ import annotations

import pytest

from guide_excursions import public_identity


class _FakeEntity:
    def __init__(self, first_name: str, last_name: str):
        self.first_name = first_name
        self.last_name = last_name
        self.title = None


class _FakeClient:
    async def get_entity(self, username: str):
        if username == "ann_tuz":
            return _FakeEntity("Анастасия", "Туз")
        if username == "reiseleiterin_tanja":
            return _FakeEntity("Татьяна", "Удовенко")
        raise ValueError(username)

    async def disconnect(self):
        return None


@pytest.mark.asyncio
async def test_resolve_public_guide_names_replaces_partial_name_from_username_profile(monkeypatch):
    async def _fake_client():
        return _FakeClient()

    monkeypatch.setattr(public_identity, "create_telethon_runtime_client", _fake_client)

    rows = [
        {
            "id": 1,
            "source_username": "tanja_from_koenigsberg",
            "guide_profile_display_name": "Татьяна Удовенко",
            "guide_profile_marketing_name": "Татьяна Удовенко",
            "guide_names": ["Татьяна Удовенко", "Анна Туз"],
            "dedup_source_text": "И 22 марта случится премьера - мы с @ann_tuz готовим новую прогулку. Запись в л/с @reiseleiterin_tanja",
        }
    ]

    out = await public_identity.resolve_public_guide_names(rows)
    assert out[0]["guide_names"] == ["Татьяна Удовенко", "Анастасия Туз"]
    assert out[0]["resolved_guide_profiles"][0]["display_name"] == "Анастасия Туз"


def test_extract_public_guide_usernames_ignores_booking_contact_username():
    text = "И 22 марта случится премьера - мы с @ann_tuz готовим новую прогулку. Запись в л/с @reiseleiterin_tanja"
    assert public_identity.extract_public_guide_usernames(text) == ["ann_tuz"]


@pytest.mark.asyncio
async def test_resolve_public_guide_names_collapses_marketing_alias_to_profile_name(monkeypatch):
    async def _fake_client():
        return _FakeClient()

    monkeypatch.setattr(public_identity, "create_telethon_runtime_client", _fake_client)

    rows = [
        {
            "id": 2,
            "source_username": "amber_fringilla",
            "guide_profile_display_name": "Юлия Гришанова",
            "guide_profile_marketing_name": "Amber Fringilla",
            "guide_names": ["Amber Fringilla"],
            "dedup_source_text": "Экопрогулка по Южному парку. Запись @Yulia_Grishanova",
        }
    ]

    out = await public_identity.resolve_public_guide_names(rows)
    assert out[0]["guide_names"] == ["Юлия Гришанова"]
