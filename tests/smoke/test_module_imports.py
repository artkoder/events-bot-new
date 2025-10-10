"""Smoke tests ensuring critical modules can be imported without side effects."""

from __future__ import annotations

import importlib
from typing import List

import pytest


CRITICAL_MODULES: List[str] = [
    "main",
    "db",
    "imagekit_poster",
    "vk_intake",
    "vk_review",
    "markup",
    "scheduling",
    "digests",
    "sections",
    "shortlinks",
    "supabase_export",
    "safe_bot",
    "span",
    "net",
    "models",
]


@pytest.mark.parametrize("module_name", CRITICAL_MODULES)
def test_module_can_be_imported(module_name: str) -> None:
    """Import each critical module, failing if the import raises an exception."""

    module = importlib.import_module(module_name)
    assert module is not None
