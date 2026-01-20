from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock

# Mock 'main' to avoid circular import chain via video_announce.__init__
sys.modules["main"] = MagicMock()

from video_announce import pattern_preview


def _write_preview(out_path: Path, pattern_name: str, intro_text: str) -> None:
    data = pattern_preview.generate_intro_preview(
        pattern_name=pattern_name,
        intro_text=intro_text,
    )
    out_path.write_bytes(data)


def main() -> None:
    out_dir = Path("artifacts/codex/tasks/intro_visuals")
    out_dir.mkdir(parents=True, exist_ok=True)

    intro_text = "WEEKEND\nPLANS"
    _write_preview(
        out_dir / "test_sticker_dark.png",
        pattern_preview.PATTERN_STICKER,
        intro_text,
    )
    _write_preview(
        out_dir / "test_sticker_yellow.png",
        f"{pattern_preview.PATTERN_STICKER}_YELLOW",
        intro_text,
    )

    print(f"Wrote previews to {out_dir}")


if __name__ == "__main__":
    main()
