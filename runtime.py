from __future__ import annotations

"""Helpers for accessing state from the running ``main`` module."""

import sys
from types import ModuleType
from typing import Any


def get_running_main() -> ModuleType | None:
    """Return the loaded ``main`` module without importing it."""

    module = sys.modules.get("main")
    if module is not None:
        return module
    return sys.modules.get("__main__")


def require_main_attr(name: str) -> Any:
    """Return an attribute from the running ``main`` module.

    Raises ``RuntimeError`` when the module is not loaded or the attribute is
    missing.  The function never imports the module.
    """

    module = get_running_main()
    if module is None:
        raise RuntimeError("main module is not loaded; cannot access %s" % name)
    try:
        return getattr(module, name)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"running main module does not define {name!r}"
        ) from exc

