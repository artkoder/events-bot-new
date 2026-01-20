import json
from pathlib import Path

NOTEBOOK_PATH = Path("kaggle/CrumpleVideo/crumple_video.ipynb")

NEW_SOURCE = [
    "# ==========================================\n",
    "# 3. ГЕНЕРАТОР ОБЛОЖКИ (Pixel Perfect v2)\n",
    "# ==========================================\n",
    "\n",
    "import os\n",
    "from pathlib import Path\n",
    "from PIL import Image, ImageDraw, ImageFont\n",
    "from dataclasses import dataclass\n",
    "from typing import Optional\n",
    "\n",
    "def find_font(name: str) -> str:\n",
    "    \"\"\"Ищет шрифт в SOURCE_FOLDER и стандартных путях.\"\"\"\n",
    "    candidates = [\n",
    "        SOURCE_FOLDER / name,\n",
    "        SOURCE_FOLDER / \"fonts\" / name,\n",
    "        Path(f\"/usr/share/fonts/truetype/{name}\"),\n",
    "        Path(\"assets\") / name,\n",
    "    ]\n",
    "    # Также ищем глобально\n",
    "    try: candidates.extend(Path(KAGGLE_INPUT_ROOT).rglob(name))\n",
    "    except: pass\n",
    "    for p in candidates:\n",
    "        if p.exists():\n",
    "            return str(p)\n",
    "    return None\n",
    "\n",
    "# --- Layout Configuration ---\n",
    "CANVAS_W, CANVAS_H = 1080, 1920\n",
    "DESIGN_H = 1572\n",
    "OFFSET_Y = (CANVAS_H - DESIGN_H) // 2\n",
    "\n",
    "@dataclass\n",
    "class TextStyle:\n",
    "    font_name: str\n",
    "    size: int\n",
    "    x: int\n",
    "    y: int\n",
    "    width: int = 0\n",
    "    height: int = 0\n",
    "    align: str = \"left\"\n",
    "    rotate: int = 0\n",
    "    line_height: Optional[int] = None\n",
    "    uppercase: bool = False\n",
    "\n",
    "@dataclass\n",
    "class IntroConfig:\n",
    "    date: TextStyle\n",
    "    month: TextStyle\n",
    "    title: TextStyle\n",
    "    cities: TextStyle\n",
    "\n",
    "# --- Style Definitions (Pixel-Perfect from CSS) ---\n",
    "STYLE_DAY = IntroConfig(\n",
    "    date=TextStyle(\"Benzin-Bold.ttf\", 224, 676, 270, width=324, height=308, align=\"right\"),\n",
    "    month=TextStyle(\"BebasNeue-Bold.ttf\", 200, 850, 541, width=476, height=200, rotate=-90, uppercase=True),\n",
    "    title=TextStyle(\"DrukCyr-Bold.ttf\", 180, 73, 827, width=724, height=228, align=\"right\", uppercase=True),\n",
    "    cities=TextStyle(\"Oswald-VariableFont_wght.ttf\", 60, 435, 1058, width=357, height=267, align=\"right\", line_height=89, uppercase=True)\n",
    ")\n",
    "\n",
    "STYLE_WEEKEND = IntroConfig(\n",
    "    date=TextStyle(\"Benzin-Bold.ttf\", 224, 55, 270, width=945, height=308, align=\"right\"),\n",
    "    month=TextStyle(\"BebasNeue-Bold.ttf\", 200, 850, 541, width=476, height=200, rotate=-90, uppercase=True),\n",
    "    title=TextStyle(\"DrukCyr-Bold.ttf\", 220, 82, 779, width=710, height=279, align=\"left\", uppercase=True),\n",
    "    cities=TextStyle(\"Oswald-VariableFont_wght.ttf\", 60, 435, 1058, width=357, height=267, align=\"right\", line_height=89, uppercase=True)\n",
    ")\n",
    "\n",
    "def generate_cover_image(date_range: str, month: str, output_path: Path, title: str = \"ВЫХОДНЫЕ\", cities_list: list[str] = None) -> Path:\n",
    "    \"\"\"V2 Generator supporting rigid layouts and Cities list.\"\"\"\n",
    "    is_weekend = \"ВЫХОДНЫЕ\" in (title or \"\").upper()\n",
    "    cfg = STYLE_WEEKEND if is_weekend else STYLE_DAY\n",
    "    \n",
    "    img = Image.new(\"RGBA\", (CANVAS_W, CANVAS_H), (241, 228, 75, 255))\n",
    "    draw = ImageDraw.Draw(img)\n",
    "    \n",
    "    def get_font(name: str, size: int):\n",
    "        path = find_font(name)\n",
    "        if path: return ImageFont.truetype(str(path), size)\n",
    "        # Fallback names mapping\n",
    "        if \"Oswald\" in name: alt = \"Oswald-VariableFont_wght.ttf\"\n",
    "        elif \"Druk\" in name: alt = \"DrukCyr-Bold.ttf\"\n",
    "        elif \"Bebas\" in name: alt = \"BebasNeue-Bold.ttf\"\n",
    "        elif \"Benzin\" in name: alt = \"Benzin-Bold.ttf\"\n",
    "        else: alt = name\n",
    "        \n",
    "        path = find_font(alt)\n",
    "        if path: return ImageFont.truetype(str(path), size)\n",
    "        log(f\"Font {name} not found, using default\")\n",
    "        return ImageFont.load_default()\n",
    "\n",
    "    def render_element(text, style: TextStyle):\n",
    "        if not text: return\n",
    "        content = text.upper() if style.uppercase else text\n",
    "        font = get_font(style.font_name, style.size)\n",
    "        \n",
    "        # Adjust Y for 1920 canvas centering\n",
    "        x, y = style.x, style.y + OFFSET_Y\n",
    "        \n",
    "        spacing = (style.line_height - style.size) if style.line_height else 4\n",
    "        fill = (16, 14, 14, 255)\n",
    "        \n",
    "        if style.rotate != 0:\n",
    "             # Helper for rotated text (Month)\n",
    "             # Create temp buffer\n",
    "             dummy = Image.new(\"RGBA\", (1,1))\n",
    "             d = ImageDraw.Draw(dummy)\n",
    "             bbox = d.textbbox((0,0), content, font=font, spacing=spacing)\n",
    "             w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]\n",
    "             \n",
    "             txt = Image.new(\"RGBA\", (w+20, h+20), (0,0,0,0))\n",
    "             d2 = ImageDraw.Draw(txt)\n",
    "             d2.text((0,0), content, font=font, fill=fill, spacing=spacing)\n",
    "             \n",
    "             cropped = txt.crop(txt.getbbox())\n",
    "             # User CSS says rotate(-90deg) -> we want vertical reading UP. \n",
    "             # Standard PIL rotate(90) -> CCW 90 -> UP.\n",
    "             rotated = cropped.rotate(90, expand=True, resample=Image.BICUBIC)\n",
    "             img.alpha_composite(rotated, (x, y))\n",
    "        else:\n",
    "             # Alignment logic\n",
    "            if style.align == \"right\" and style.width > 0:\n",
    "                dummy = Image.new(\"RGBA\", (1,1))\n",
    "                d = ImageDraw.Draw(dummy)\n",
    "                bbox = d.multiline_textbbox((0,0), content, font=font, spacing=spacing, align=\"right\")\n",
    "                w = bbox[2] - bbox[0]\n",
    "                draw_x = (x + style.width) - w\n",
    "                draw.multiline_text((draw_x, y), content, font=font, fill=fill, spacing=spacing, align=\"right\")\n",
    "            elif style.align == \"center\" and style.width > 0:\n",
    "                dummy = Image.new(\"RGBA\", (1,1))\n",
    "                d = ImageDraw.Draw(dummy)\n",
    "                bbox = d.multiline_textbbox((0,0), content, font=font, spacing=spacing, align=\"center\")\n",
    "                w = bbox[2] - bbox[0]\n",
    "                draw_x = x + (style.width - w) // 2\n",
    "                draw.multiline_text((draw_x, y), content, font=font, fill=fill, spacing=spacing, align=\"center\")\n",
    "            else:\n",
    "                draw.multiline_text((x, y), content, font=font, fill=fill, spacing=spacing, align=style.align)\n",
    "\n",
    "    # Render items\n",
    "    render_element(date_range, cfg.date)\n",
    "    render_element(month, cfg.month)\n",
    "    render_element(title, cfg.title)\n",
    "    \n",
    "    # Cities\n",
    "    c_text = \"\\n\".join(cities_list[:3]) if cities_list else \"\"\n",
    "    if is_weekend and not c_text:\n",
    "         c_text = \"КАЛИНИНГРАД\\nСВЕТЛОГОРСК\\nЗЕЛЕНОГРАДСК\"\n",
    "    render_element(c_text, cfg.cities)\n",
    "    \n",
    "    out_p = Path(output_path)\n",
    "    img.convert(\"RGB\").save(out_p)\n",
    "    log(f\"Saved intro cover: {out_p}\")\n",
    "    return out_p\n"
]

def main():
    if not NOTEBOOK_PATH.exists():
        print(f"Error: {NOTEBOOK_PATH} not found")
        return

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Find cell
    updated = False
    for cell in nb["cells"]:
        if cell.get("id") == "cover-generator":
            cell["source"] = NEW_SOURCE
            updated = True
            print("Updated 'cover-generator' cell.")
            break
            
    if not updated:
        print("Warning: Cell with id 'cover-generator' not found. Appending new cell.")
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "id": "cover-generator",
            "metadata": {},
            "source": NEW_SOURCE
        })

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook saved.")

if __name__ == "__main__":
    main()
