from __future__ import annotations

from guide_excursions.llm_support import resolve_candidate_key_ids


class _FakeQuery:
    def __init__(self, rows):
        self.rows = rows
        self.filters = {}

    def select(self, _fields):
        return self

    def eq(self, key, value):
        self.filters[key] = value
        return self

    def in_(self, key, values):
        self.filters[key] = list(values)
        return self

    def order(self, _field):
        return self

    def execute(self):
        env_names = set(self.filters.get("env_var_name") or [])
        data = [row for row in self.rows if row.get("env_var_name") in env_names and row.get("is_active") is True]
        return type("Result", (), {"data": data})()


class _FakeSupabase:
    def __init__(self, rows):
        self.rows = rows

    def table(self, name):
        assert name == "google_ai_api_keys"
        return _FakeQuery(self.rows)


def test_resolve_candidate_key_ids_prefers_primary_key2_metadata():
    supabase = _FakeSupabase(
        [
            {"id": "fallback", "env_var_name": "GOOGLE_API_KEY", "is_active": True, "priority": 10},
            {"id": "primary", "env_var_name": "GOOGLE_API_KEY2", "is_active": True, "priority": 5},
        ]
    )

    out = resolve_candidate_key_ids(
        supabase=supabase,
        primary_key_env="GOOGLE_API_KEY2",
        fallback_key_env="GOOGLE_API_KEY",
        consumer="guide_excursions_digest_batch:test-primary",
    )

    assert out == ["primary"]


def test_resolve_candidate_key_ids_falls_back_when_primary_missing():
    supabase = _FakeSupabase(
        [
            {"id": "fallback", "env_var_name": "GOOGLE_API_KEY", "is_active": True, "priority": 10},
        ]
    )

    out = resolve_candidate_key_ids(
        supabase=supabase,
        primary_key_env="GOOGLE_API_KEY2",
        fallback_key_env="GOOGLE_API_KEY",
        consumer="guide_occurrence_enrich:test-fallback",
    )

    assert out == ["fallback"]
