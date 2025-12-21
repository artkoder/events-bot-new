from __future__ import annotations

from pathlib import Path

_ASSETS_DIR = Path(__file__).parent / "assets"
_DEFAULT_PROMPT = (
    "Собери сценарий видеоролика по списку событий. Используй живой тон,"
    " привяжи хронометраж, упомяни площадку и время."
)


def load_prompt(name: str = "script") -> str:
    """Return prompt text for the requested name, falling back to default."""

    asset = _ASSETS_DIR / f"{name}.txt"
    if asset.exists():
        return asset.read_text(encoding="utf-8").strip()
    return _DEFAULT_PROMPT


def available_prompts() -> list[str]:
    prompts = []
    for path in _ASSETS_DIR.glob("*.txt"):
        prompts.append(path.stem)
    return sorted(set(prompts + ["script"]))
