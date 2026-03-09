from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "crumple_video.ipynb"
MODULE_PATH = ROOT / "poster_overlay.py"


def _escape_single_quoted_python(source: str) -> str:
    return source.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")


def main() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    module_source = MODULE_PATH.read_text(encoding="utf-8").rstrip("\n")
    escaped_module = _escape_single_quoted_python(module_source)

    overlay_call_old = (
        "        overlay = scene.get(\"poster_overlay\")\n"
        "        if isinstance(overlay, dict):\n"
        "            overlay_text = overlay.get(\"text\")\n"
        "        else:\n"
        "            overlay_text = None\n"
        "        if found and isinstance(overlay_text, str) and overlay_text.strip():\n"
        "            try:\n"
        "                found = apply_poster_overlay(found, text=overlay_text, out_dir=posters_dir, search_roots=OVERLAY_FONT_ROOTS)\n"
        "            except Exception as e:\n"
        "                log(f\"⚠️ Overlay failed for scene {i}: {e}\")\n"
    )
    overlay_call_new = (
        "        overlay = scene.get(\"poster_overlay\")\n"
        "        overlay_text = None\n"
        "        highlight_title = None\n"
        "        if isinstance(overlay, dict):\n"
        "            overlay_text = overlay.get(\"text\")\n"
        "            missing = overlay.get(\"missing\")\n"
        "            if isinstance(missing, list):\n"
        "                highlight_title = \"title\" in {str(part).strip().casefold() for part in missing if isinstance(part, str)}\n"
        "        if found and isinstance(overlay_text, str) and overlay_text.strip():\n"
        "            try:\n"
        "                found = apply_poster_overlay(\n"
        "                    found,\n"
        "                    text=overlay_text,\n"
        "                    out_dir=posters_dir,\n"
        "                    search_roots=OVERLAY_FONT_ROOTS,\n"
        "                    highlight_title=highlight_title,\n"
        "                )\n"
        "            except Exception as e:\n"
        "                log(f\"⚠️ Overlay failed for scene {i}: {e}\")\n"
    )

    replaced = False
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if "def _ensure_poster_overlay_module():" not in source:
            continue
        start = source.find("        code = '")
        end_marker = "'\n        target.write_text(code, encoding='utf-8')"
        end = source.find(end_marker, start)
        if start < 0 or end < 0:
            raise RuntimeError("Could not locate embedded poster_overlay.py block in notebook")
        replacement = f"        code = '{escaped_module}'"
        source = source[:start] + replacement + source[end:]
        if overlay_call_old in source:
            source = source.replace(overlay_call_old, overlay_call_new, 1)
        elif overlay_call_new not in source:
            raise RuntimeError("Could not locate overlay call block in notebook")
        cell["source"] = source
        replaced = True
        break

    if not replaced:
        raise RuntimeError("Could not find CrumpleVideo pipeline cell in notebook")

    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
