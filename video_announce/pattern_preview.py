"""
Pattern Preview Generator for Video Announce Bot.

Generates intro pattern preview images server-side.
Patterns: RISING, STICKER, COMPACT
"""
from __future__ import annotations

import io
import os
from PIL import Image, ImageDraw, ImageFont

# Constants
W, H = 1080, 1920
SCALE = 2

# Asset paths (relative to video_announce/assets/)
_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
FONT_PATH = os.path.join(_ASSETS_DIR, "BebasNeue-Bold.ttf")
CITIES_FONT_PATH = os.path.join(_ASSETS_DIR, "Bebas-Neue-Pro-SemiExpanded-Bold-BF66cf3d78c1f4e.ttf")

# Colors
BG_CANVAS = (30, 30, 30)
BG_ACCENT = (241, 156, 28, 255)
BG_YELLOW = (241, 228, 75)
BG_ACCENT_YELLOW = (255, 255, 255, 255)
TEXT_BLACK = (0, 0, 0, 255)

# Pattern names
PATTERN_RISING = "RISING"
PATTERN_STICKER = "STICKER"
PATTERN_COMPACT = "COMPACT"

ALL_PATTERNS = [PATTERN_RISING, PATTERN_STICKER, PATTERN_COMPACT]


def _resolve_pattern_theme(pattern_name: str) -> tuple[str, tuple, tuple]:
    if pattern_name.endswith("_YELLOW"):
        return pattern_name[:-7], BG_YELLOW, BG_ACCENT_YELLOW
    return pattern_name, BG_CANVAS, BG_ACCENT


def _get_font(size: int, path: str = FONT_PATH) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, int(size))


def _create_strip(
    text: str,
    font_size: int,
    bg_color: tuple,
    text_color: tuple,
    bevel_ratio: float = 0.25,
    font_path: str = FONT_PATH,
    pad_y_override: int | None = None,
) -> tuple[Image.Image, int]:
    """Create a single text strip with bevel."""
    font = _get_font(font_size, font_path)
    
    # Reference Height for centering
    is_digit = any(c.isdigit() for c in text)
    ref_char = "5" if is_digit else "H"
    
    ref_bbox = font.getbbox(ref_char)
    if not ref_bbox:
        ref_bbox = font.getbbox("I")
    
    ref_center_rel = (ref_bbox[1] + ref_bbox[3]) / 2
    
    # Canvas Height from full alphabet
    full_bbox = font.getbbox('ЙрАБВду0123456789')
    full_height_safe = full_bbox[3] - full_bbox[1]
    
    if pad_y_override is not None:
        pad_y = pad_y_override
    else:
        pad_y = 10 * SCALE if is_digit else 18 * SCALE
    
    content_h = int(full_height_safe + pad_y * 2)
    
    # Width
    pad_x = 12 * SCALE
    text_bbox = font.getbbox(text)
    text_w = text_bbox[2] - text_bbox[0]
    content_w = int(text_w + pad_x * 2)
    
    img = Image.new('RGBA', (content_w, content_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    
    # Polygon with TR Bevel
    b_len = content_h * bevel_ratio
    
    p1 = (0, 0)
    p2_a = (content_w - b_len, 0)
    p2_b = (content_w, b_len)
    p3 = (content_w, content_h)
    p4 = (0, content_h)
    
    d.polygon([p1, p2_a, p2_b, p3, p4], fill=bg_color)
    
    # Text Placement
    ty = (content_h / 2) - ref_center_rel
    if is_digit:
        ty += content_h * 0.1
    
    tx = (content_w - text_w) // 2
    d.text((tx, ty), text, font=font, fill=text_color)
    
    return img, content_h


def _create_skewed_strip(
    text: str,
    font_size: int,
    bg_color: tuple,
    text_color: tuple,
    strip_skew_deg: float = 4.0,
    bevel_ratio: float = 0.20,
    font_path: str = FONT_PATH,
    is_digit: bool = False,
    is_last: bool = False,
) -> tuple[Image.Image, int]:
    """Create a skewed parallelogram strip for Rising pattern.
    
    Unlike _create_strip which makes rectangles, this creates parallelograms
    with proper skew angle and optional bevel on the top-right corner.
    """
    import math
    
    font = _get_font(font_size, font_path)
    
    # Reference Height for centering
    ref_char = "5" if is_digit else "H"
    ref_bbox = font.getbbox(ref_char)
    if not ref_bbox:
        ref_bbox = font.getbbox("I")
    ref_center_rel = (ref_bbox[1] + ref_bbox[3]) / 2
    
    # Canvas Height from full alphabet
    full_bbox = font.getbbox('ЙрАБВду0123456789')
    full_height_safe = full_bbox[3] - full_bbox[1]
    
    PAD_Y_DIGIT = 10 * SCALE
    PAD_Y_TEXT = 18 * SCALE
    pad_y = PAD_Y_DIGIT if is_digit else PAD_Y_TEXT
    content_h = int(full_height_safe + pad_y * 2)
    
    # Width
    text_bbox = font.getbbox(text)
    if not text_bbox:
        text_bbox = (0, 0, 10, 10)
    text_w = text_bbox[2] - text_bbox[0]
    pad_x = 10 * SCALE
    content_w = int(text_w + pad_x * 2)
    
    # Calculate skew offset
    skew_px = int(abs(math.tan(math.radians(strip_skew_deg))) * content_h)
    
    BUFFER = 40
    canvas_w = content_w + skew_px + BUFFER
    canvas_h = content_h
    
    # Create strip canvas
    strip = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(strip)
    
    # Bevel calculation
    b_len = content_h * bevel_ratio if is_last else 0
    
    if strip_skew_deg > 0:
        # Parallelogram points (skewed right / rising)
        # P1 (TL), P2 (TR with bevel), P3 (BR), P4 (BL)
        
        if is_last and b_len > 0:
            side_len = math.hypot(skew_px, canvas_h)
            p2_a = (content_w + skew_px - b_len, 0)
            ratio = b_len / side_len if side_len > 0 else 0
            p2_b = (content_w + skew_px + ratio * (-skew_px), 0 + ratio * canvas_h)
            p2_points = [p2_a, p2_b]
        else:
            p2_points = [(content_w + skew_px, 0)]
        
        points = [(skew_px, 0)] + p2_points + [(content_w, canvas_h), (0, canvas_h)]
    else:
        # Straight rectangle fallback
        points = [(0, 0), (content_w, 0), (content_w, canvas_h), (0, canvas_h)]
    
    sd.polygon(points, fill=bg_color)
    
    # Text rendering
    txt_layer_w = int(content_w * 1.5)
    txt_layer = Image.new('RGBA', (txt_layer_w, content_h), (0, 0, 0, 0))
    td = ImageDraw.Draw(txt_layer)
    
    ty = (content_h / 2) - ref_center_rel
    if is_digit:
        ty += content_h * 0.1
    
    tx = (txt_layer_w - text_w) // 2
    td.text((tx, ty), text, font=font, fill=text_color)
    
    # Paste text centered on polygon
    poly_mid_x = (content_w / 2) + (skew_px / 2)
    paste_x = int(poly_mid_x - (txt_layer_w / 2))
    strip.paste(txt_layer, (paste_x, 0), txt_layer)
    
    # Crop to content
    bbox = strip.getbbox()
    if bbox:
        strip = strip.crop(bbox)
    
    return strip, content_h


def _render_sticker(
    lines: list[dict],
    cities: str | None = None,
    bg_canvas: tuple = BG_CANVAS,
    bg_accent: tuple = BG_ACCENT,
) -> Image.Image:
    """Pattern STICKER: Messy alternating rotation."""
    all_items = list(lines)
    if cities:
        all_items.append({
            "text": cities,
            "size": 50 * SCALE,
            "rot": 4,
            "font": CITIES_FONT_PATH,
            "pad_y": 6 * SCALE,
        })
    
    canvas = Image.new('RGB', (W * SCALE, H * SCALE), bg_canvas)
    curr_y = 600 * SCALE
    
    for item in all_items:
        f_path = item.get("font", FONT_PATH)
        py_ov = item.get("pad_y")
        img, content_h = _create_strip(
            item["text"], item["size"], bg_accent, TEXT_BLACK,
            bevel_ratio=0.25, font_path=f_path, pad_y_override=py_ov
        )
        
        if item.get("rot", 0) != 0:
            img = img.rotate(item["rot"], expand=True, resample=Image.BICUBIC)
        
        x = (W * SCALE - img.width) // 2
        canvas.paste(img, (x, curr_y), mask=img)
        
        buffer_y = int(content_h * 0.15)
        curr_y += int(content_h + buffer_y)
    
    return canvas.resize((W, H), resample=Image.LANCZOS)


def _render_compact(
    lines: list[dict],
    cities: str | None = None,
    bg_canvas: tuple = BG_CANVAS,
    bg_accent: tuple = BG_ACCENT,
) -> Image.Image:
    """Pattern COMPACT: Dynamic spacing based on rotated height."""
    all_items = list(lines)
    if cities:
        all_items.append({
            "text": cities,
            "size": 50 * SCALE,
            "rot": 4,
            "font": CITIES_FONT_PATH,
            "pad_y": 6 * SCALE,
        })
    
    canvas = Image.new('RGB', (W * SCALE, H * SCALE), bg_canvas)
    curr_y = 600 * SCALE
    
    for item in all_items:
        f_path = item.get("font", FONT_PATH)
        py_ov = item.get("pad_y")
        img, content_h = _create_strip(
            item["text"], item["size"], bg_accent, TEXT_BLACK,
            bevel_ratio=0.25, font_path=f_path, pad_y_override=py_ov
        )
        
        if item.get("rot", 0) != 0:
            img = img.rotate(item["rot"], expand=True, resample=Image.BICUBIC)
        
        x = (W * SCALE - img.width) // 2
        canvas.paste(img, (x, curr_y), mask=img)
        
        # Dynamic spacing based on rotated height
        advance = int(img.height - (18 * SCALE))
        curr_y += advance
    
    return canvas.resize((W, H), resample=Image.LANCZOS)


def _render_rising(
    lines: list[dict],
    cities: str | None = None,
    bg_canvas: tuple = BG_CANVAS,
    bg_accent: tuple = BG_ACCENT,
) -> Image.Image:
    """Pattern RISING: Composite lines with number highlighting, unidirectional 7° shear.
    
    Lines can be:
    - Simple: {"text": "...", "size": ..., "is_last": bool}
    - Composite: {"composite": True, "elements": [...], "is_last": bool}
    """
    import math
    
    all_items = list(lines)
    if cities:
        all_items.append({
            "composite": False,
            "text": cities,
            "size": 50 * SCALE,
            "type": "cities",
            "font": CITIES_FONT_PATH,
            "pad_y": 6 * SCALE,
            "is_last": True,
        })
        # Update previous last
        if len(all_items) > 1:
            all_items[-2]["is_last"] = False
    
    SHEAR_DEG = 7
    GAP_PX = -12 * SCALE  # Negative gap for overlapping elements
    VERTICAL_SPACING = 0.82
    
    canvas = Image.new('RGB', (W * SCALE, H * SCALE), bg_canvas)
    curr_y = 600 * SCALE
    
    line_images = []
    
    for item in all_items:
        if item.get("composite"):
            # Composite line: multiple elements horizontally
            line_img = _create_composite_line_rising(
                item["elements"],
                item["is_last"],
                bg_accent,
            )
        else:
            # Simple text line
            f_path = item.get("font", FONT_PATH)
            py_ov = item.get("pad_y")
            is_last = item.get("is_last", False)
            
            bevel = 0.20 if is_last else 0.0
            line_img, _ = _create_strip(
                item["text"], item["size"], bg_accent, TEXT_BLACK,
                bevel_ratio=bevel, font_path=f_path, pad_y_override=py_ov
            )
        
        # Apply unidirectional VERTICAL shear to the whole line (Rising up-right)
        # Original uses: coeffs = (1, 0, 0, factor, 1, y_offset)
        # This means: y' = factor * x + y + y_offset
        factor = math.tan(math.radians(SHEAR_DEG))
        w, h = line_img.size
        
        # Correction for Rising Shear (Factor > 0)
        # To keep Top-Right visible, we must offset Y equal to the rise
        if factor > 0:
            y_offset = -factor * w
        else:
            y_offset = 0
        
        coeffs = (1, 0, 0, factor, 1, y_offset)
        new_h = int(h + abs(factor) * w)
        
        line_img = line_img.transform((w, new_h), Image.AFFINE, coeffs, resample=Image.BICUBIC)
        
        line_images.append((line_img, new_h))
    
    # Paste lines from bottom to top (last line first) for proper overlapping
    for line_img, content_h in reversed(line_images):
        x = (W * SCALE - line_img.width) // 2
        canvas.paste(line_img, (x, curr_y), mask=line_img)
        curr_y += int(content_h * VERTICAL_SPACING)
    
    # Actually we want top-to-bottom, then reverse z-order. Let me fix:
    canvas = Image.new('RGB', (W * SCALE, H * SCALE), bg_canvas)
    curr_y = 600 * SCALE
    positions = []
    
    for line_img, content_h in line_images:
        x = (W * SCALE - line_img.width) // 2
        positions.append((x, curr_y, line_img))
        curr_y += int(content_h * VERTICAL_SPACING)
    
    # Paste in reverse order for proper z-ordering (later lines on top)
    for x, y, img in reversed(positions):
        canvas.paste(img, (x, y), mask=img)
    
    return canvas.resize((W, H), resample=Image.LANCZOS)


def _create_composite_line_rising(
    elements: list[dict],
    is_last_line: bool,
    bg_accent: tuple = BG_ACCENT,
) -> Image.Image:
    """Create a composite line with multiple elements horizontally aligned.
    
    Elements have different sizes (e.g., big number, medium text).
    """
    import math
    
    GAP_PX = -12 * SCALE
    DIGIT_OFFSET_Y = 50 * SCALE
    
    parts = []
    for i, elem in enumerate(elements):
        is_last_elem = is_last_line and (i == len(elements) - 1)
        is_digit = elem.get("type") == "big"
        
        # Use skewed strip function directly (no additional transform needed)
        strip_img, content_h = _create_skewed_strip(
            elem["text"], elem["size"], bg_accent, TEXT_BLACK,
            strip_skew_deg=4.0,
            bevel_ratio=0.20,
            font_path=FONT_PATH,
            is_digit=is_digit,
            is_last=is_last_elem,
        )
        
        parts.append({
            "img": strip_img,
            "is_digit": is_digit,
            "height": content_h,
        })
    
    # Calculate total width and max height
    total_width = sum(p["img"].width for p in parts) + GAP_PX * (len(parts) - 1)
    max_height = max(p["img"].height for p in parts)
    
    # Create canvas for composite line
    canvas_w = int(total_width * 1.2)
    canvas_h = int(max_height * 1.5)
    line_canvas = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
    
    # Calculate vertical alignment (baseline)
    target_y = canvas_h // 2
    
    # Paste elements
    curr_x = (canvas_w - total_width) // 2
    for p in parts:
        # Vertical offset for digits (higher)
        if p["is_digit"]:
            y = target_y - p["img"].height // 2 - DIGIT_OFFSET_Y
        else:
            y = target_y - p["img"].height // 2
        
        line_canvas.paste(p["img"], (curr_x, y), p["img"])
        curr_x += p["img"].width + GAP_PX
    
    # Crop to content
    bbox = line_canvas.getbbox()
    if bbox:
        line_canvas = line_canvas.crop(bbox)
    
    return line_canvas


def _parse_intro_text(intro_text: str, pattern: str = "STICKER") -> list[dict]:
    """Parse intro text into line items for rendering.
    
    Handles:
    - Multi-line input (split by newlines)
    - Single-line input with smart splitting
    - Preposition handling (НА, В, ДО etc. attach to following word)
    - Number extraction for Rising pattern composite lines
    """
    import re
    
    # Russian prepositions that must attach to the following word
    PREPOSITIONS = {'НА', 'В', 'ДО', 'ДЛЯ', 'ИЗ', 'ПО', 'С', 'К', 'О', 'У', 'ОТ', 'ЗА'}
    
    text = intro_text.strip().upper()
    
    def _group_words_with_prepositions(words: list[str]) -> list[str]:
        """Group prepositions with following words."""
        result = []
        i = 0
        while i < len(words):
            word = words[i]
            # Check if this word is a preposition and there's a next word
            if word in PREPOSITIONS and i + 1 < len(words):
                # Merge preposition with next word
                result.append(word + ' ' + words[i + 1])
                i += 2
            else:
                result.append(word)
                i += 1
        return result
    
    def _split_into_lines(words: list[str], max_lines: int = 3) -> list[str]:
        """Split words into lines, respecting preposition groups."""
        if len(words) <= 2:
            return [' '.join(words)]
        elif len(words) <= 4:
            mid = len(words) // 2
            return [' '.join(words[:mid]), ' '.join(words[mid:])]
        else:
            # Split into 3 lines
            third = len(words) // 3
            return [
                ' '.join(words[:third]),
                ' '.join(words[third:third*2]),
                ' '.join(words[third*2:])
            ]
    
    # Check if already multi-line
    if '\n' in text:
        text_lines = [l.strip() for l in text.split('\n') if l.strip()]
    else:
        # Single line - need smart splitting
        text_lines = []
        
        # Split at colons first
        if ':' in text:
            parts = text.split(':', 1)
            text_lines.append(parts[0].strip() + ':')
            remaining = parts[1].strip()
        else:
            remaining = text
        
        # For Rising pattern: Don't pre-split, let Rising parser handle structure
        if pattern == "RISING" and remaining:
            # Pass the full remaining text as a single element for Rising to parse
            return _parse_for_rising([remaining])
        
        # For other patterns: Process remaining words with preposition grouping
        if remaining:
            words = remaining.split()
            grouped = _group_words_with_prepositions(words)
            text_lines.extend(_split_into_lines(grouped))
    
    # For Rising pattern with already multi-line input
    if pattern == "RISING":
        return _parse_for_rising(text_lines)
    
    # For other patterns: Simple line-based rendering
    # Default rotation pattern: +4, -3, +4, ...
    rotations = [4, -3, 4, -3, 4]
    
    items = []
    for i, line_text in enumerate(text_lines):
        rot = rotations[i % len(rotations)]
        # Adjust size based on text length
        if len(line_text) > 25:
            size = 85 * SCALE
        elif len(line_text) > 15:
            size = 95 * SCALE
        else:
            size = 110 * SCALE
        
        items.append({"text": line_text, "size": size, "rot": rot})
    
    return items


def _parse_for_rising(text_lines: list[str]) -> list[dict]:
    """Parse text lines into Rising pattern composite elements.
    
    Rising structure (from reference):
    - Line 1: VERB + NUMBER + NOUN (e.g., "ПОДОБРАЛИ" + "5" + "СОБЫТИЙ") - composite
    - Line 2: "НА ВЫХОДНЫЕ" or similar preposition phrase - nav style
    - Line 3: Date range + month (e.g., "27-28" + "ДЕКАБРЯ") - small, NOT big
    - Line 4: Cities (optional)
    """
    import re
    
    result = []
    
    # Check if we have a single line that contains a count number pattern
    # Pattern: WORD NUMBER WORD (e.g., "ПОДОБРАЛИ 2 СПЕКТАКЛЯ")
    if len(text_lines) == 1:
        # Single line - need to split into proper Rising structure
        line = text_lines[0]
        
        # Check for count pattern: WORD NUMBER WORD(S)
        count_match = re.match(r'^(\w+)\s+(\d+)\s+(.+)$', line)
        if count_match:
            verb = count_match.group(1)
            number = count_match.group(2)
            rest = count_match.group(3)
            
            # Split rest to find preposition phrase
            words = rest.split()
            
            # Find where preposition starts (НА, В, ДО, etc.)
            prep_idx = None
            for i, w in enumerate(words):
                if w in ('НА', 'В', 'ДО', 'ДЛЯ', 'К'):
                    prep_idx = i
                    break
            
            if prep_idx is not None and prep_idx > 0:
                # Object (СПЕКТАКЛЯ, СОБЫТИЯ, etc.)
                obj = ' '.join(words[:prep_idx])
                # Everything after preposition
                after_prep = words[prep_idx:]
                
                # Find where date/month starts (detect date pattern)
                # Date indicators: numbers, month names, day ranges
                MONTHS = {'ЯНВАРЯ', 'ФЕВРАЛЯ', 'МАРТА', 'АПРЕЛЯ', 'МАЯ', 'ИЮНЯ', 
                          'ИЮЛЯ', 'АВГУСТА', 'СЕНТЯБРЯ', 'ОКТЯБРЯ', 'НОЯБРЯ', 'ДЕКАБРЯ'}
                
                date_idx = None
                for i, w in enumerate(after_prep):
                    # Check if word is a number or date range (27-28) or month
                    if re.match(r'^\d+[-–]?\d*$', w) or w in MONTHS:
                        date_idx = i
                        break
                
                if date_idx is not None and date_idx > 0:
                    # Split: prep phrase vs date line
                    prep_phrase = ' '.join(after_prep[:date_idx])
                    date_phrase = ' '.join(after_prep[date_idx:])
                    
                    # Line 1: VERB + NUMBER + OBJECT (composite)
                    result.append({
                        "composite": True,
                        "elements": [
                            {"text": verb, "size": 90 * SCALE, "type": "med"},
                            {"text": number, "size": 230 * SCALE, "type": "big"},
                            {"text": obj, "size": 90 * SCALE, "type": "med"},
                        ],
                        "is_last": False,
                    })
                    
                    # Line 2: Preposition phrase (НА ВЫХОДНЫЕ)
                    result.append({
                        "composite": False,
                        "text": prep_phrase,
                        "size": 110 * SCALE,
                        "type": "nav",
                        "is_last": False,
                    })
                    
                    # Line 3: Date (27-28 ДЕКАБРЯ) - parse into composite
                    date_words = date_phrase.split()
                    date_elements = []
                    for dw in date_words:
                        date_elements.append({"text": dw, "size": 75 * SCALE, "type": "small"})
                    
                    result.append({
                        "composite": True,
                        "elements": date_elements,
                        "is_last": True,
                    })
                else:
                    # No date found - prep_phrase is everything
                    prep_phrase = ' '.join(after_prep)
                    
                    # Line 1: VERB + NUMBER + OBJECT (composite)
                    result.append({
                        "composite": True,
                        "elements": [
                            {"text": verb, "size": 90 * SCALE, "type": "med"},
                            {"text": number, "size": 230 * SCALE, "type": "big"},
                            {"text": obj, "size": 90 * SCALE, "type": "med"},
                        ],
                        "is_last": False,
                    })
                    
                    # Line 2: Preposition phrase
                    result.append({
                        "composite": False,
                        "text": prep_phrase,
                        "size": 110 * SCALE,
                        "type": "nav",
                        "is_last": True,
                    })
            else:
                # No preposition found - keep as single composite
                obj = ' '.join(words) if words else ""
                result.append({
                    "composite": True,
                    "elements": [
                        {"text": verb, "size": 90 * SCALE, "type": "med"},
                        {"text": number, "size": 230 * SCALE, "type": "big"},
                        {"text": obj, "size": 90 * SCALE, "type": "med"} if obj else None,
                    ],
                    "is_last": True,
                })
                # Remove None elements
                result[-1]["elements"] = [e for e in result[-1]["elements"] if e]
        else:
            # No count pattern - simple line
            result.append({
                "composite": False,
                "text": line,
                "size": 90 * SCALE,
                "type": "small",
                "is_last": True,
            })
    else:
        # Multiple lines already - process each
        for line_idx, line in enumerate(text_lines):
            # Check for date range pattern (27-28, 25-26, etc.) - should be SMALL, not big
            date_range_match = re.search(r'\b(\d{1,2}[-–]\d{1,2})\b', line)
            
            # Check for single count number (not date range)
            count_match = re.search(r'\b(\d{1,2})\b', line)
            is_date_range = bool(date_range_match)
            
            if is_date_range:
                # Line with date range - all elements SMALL
                # Split into parts: before, date range, after
                before = line[:date_range_match.start()].strip()
                date_range = date_range_match.group(1)
                after = line[date_range_match.end():].strip()
                
                elements = []
                if before:
                    elements.append({"text": before, "size": 75 * SCALE, "type": "small"})
                elements.append({"text": date_range, "size": 75 * SCALE, "type": "small"})
                if after:
                    elements.append({"text": after, "size": 75 * SCALE, "type": "small"})
                
                result.append({
                    "composite": True,
                    "elements": elements,
                    "is_last": False,
                })
            elif count_match and not is_date_range:
                # Line with count number - number is BIG
                before = line[:count_match.start()].strip()
                number = count_match.group(1)
                after = line[count_match.end():].strip()
                
                elements = []
                if before:
                    elements.append({"text": before, "size": 90 * SCALE, "type": "med"})
                elements.append({"text": number, "size": 230 * SCALE, "type": "big"})
                if after:
                    elements.append({"text": after, "size": 90 * SCALE, "type": "med"})
                
                result.append({
                    "composite": True,
                    "elements": elements,
                    "is_last": False,
                })
            else:
                # Simple text line
                if any(line.startswith(prep + ' ') for prep in ['НА', 'В', 'ДО', 'К']):
                    size = 110 * SCALE
                    line_type = "nav"
                else:
                    size = 90 * SCALE
                    line_type = "small"
                
                result.append({
                    "composite": False,
                    "text": line,
                    "size": size,
                    "type": line_type,
                    "is_last": False,
                })
    
    # Mark last element
    if result:
        result[-1]["is_last"] = True
    
    return result


def generate_intro_preview(
    pattern_name: str,
    intro_text: str,
    cities: str | None = None,
) -> bytes:
    """
    Generate a pattern preview image.
    
    Args:
        pattern_name: One of RISING, STICKER, COMPACT (optionally with _YELLOW)
        intro_text: Multi-line intro text (e.g., "АФИША:\\nПЛАНЫ НА\\nВЫХОДНЫЕ")
        cities: Optional cities string (e.g., "Калининград, Светлогорск")
    
    Returns:
        PNG image bytes
    """
    base_pattern, bg_canvas, bg_accent = _resolve_pattern_theme(pattern_name)
    lines = _parse_intro_text(intro_text, pattern=base_pattern)
    
    if base_pattern == PATTERN_RISING:
        img = _render_rising(lines, cities, bg_canvas=bg_canvas, bg_accent=bg_accent)
    elif base_pattern == PATTERN_COMPACT:
        img = _render_compact(lines, cities, bg_canvas=bg_canvas, bg_accent=bg_accent)
    else:  # Default to STICKER
        img = _render_sticker(lines, cities, bg_canvas=bg_canvas, bg_accent=bg_accent)
    
    # Convert to bytes
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def get_next_pattern(current: str) -> str:
    """Get the next pattern in the cycle."""
    try:
        idx = ALL_PATTERNS.index(current)
        return ALL_PATTERNS[(idx + 1) % len(ALL_PATTERNS)]
    except ValueError:
        return PATTERN_STICKER


def get_prev_pattern(current: str) -> str:
    """Get the previous pattern in the cycle."""
    try:
        idx = ALL_PATTERNS.index(current)
        return ALL_PATTERNS[(idx - 1) % len(ALL_PATTERNS)]
    except ValueError:
        return PATTERN_STICKER
