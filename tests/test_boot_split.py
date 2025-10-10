import builtins
from pathlib import Path

import pytest


START_MARKER = "# --- split-loader"
END_MARKER = "# --- end split-loader ---"


def _extract_loader_block(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    lines: list[str] = []
    recording = False
    for line in text.splitlines():
        if line.startswith(START_MARKER):
            recording = True
            continue
        if line.startswith(END_MARKER):
            break
        if recording:
            lines.append(line)
    loader_src = "\n".join(lines)
    if not loader_src.strip():
        raise AssertionError("split-loader block not found in main.py")
    return loader_src


def test_split_loader_propagates_syntax_error(monkeypatch):
    main_path = Path(__file__).resolve().parents[1] / "main.py"
    loader_block = _extract_loader_block(main_path)

    original_compile = builtins.compile

    def failing_compile(source, filename, mode, *args, **kwargs):
        if filename == "main_part2.py":
            raise SyntaxError("boom")
        return original_compile(source, filename, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "compile", failing_compile)

    module_globals = {
        "__file__": str(main_path),
        "__builtins__": builtins,
    }

    with pytest.raises(SyntaxError):
        exec(loader_block, module_globals, module_globals)
