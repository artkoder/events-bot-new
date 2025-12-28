"""
3D Pattern Preview Generator - Flat visualization of cube texts.

Generates preview images showing how text will appear on 3D cubes,
in a flat 2D format for quick iteration before 3D rendering.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

# Import font manager
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "3d_intro"))
from font_manager import get_font_str, DRUK_CYR_BOLD, BENZIN_BOLD, BEBAS_NEUE_REGULAR, BEBAS_NEUE_BOLD

# Canvas settings
CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 1920  # Vertical format to match 3D output
BG_COLOR = (30, 30, 30)  # Dark background
CUBE_BG = (50, 50, 50)  # Cube background color
TEXT_COLOR = (255, 255, 255)  # White text
ACCENT_COLOR = (241, 156, 28)  # Orange accent


def _get_font(font_id: str, size: int) -> ImageFont.FreeTypeFont:
    """Load font with given size."""
    try:
        font_path = get_font_str(font_id, allow_fallback=True)
        return ImageFont.truetype(font_path, size)
    except Exception as e:
        print(f"Warning: Failed to load font {font_id}: {e}")
        # Fallback to default
        return ImageFont.load_default()


def generate_cube_text_preview(
    texts_data: dict,
    show_cube_outlines: bool = True,
) -> bytes:
    """
    Generate flat preview of text layout on cubes.
    
    Args:
        texts_data: Dictionary with cube texts:
            {
                "main_cube": {"text": "АФИША", "font": "druk_cyr_bold"},
                "info_cubes": [
                    {"text": "НА ВЫХОДНЫЕ", "font": "benzin_bold"},
                    {"text": "27-28", "font": "bebas_neue_regular"},
                    {"text": "ДЕКАБРЯ", "font": "bebas_neue_regular"},
                ],
                "cities": "Калининград, Светлогорск"  # Optional
            }
        show_cube_outlines: If True, draw cube wireframes
    
    Returns:
        PNG image bytes
    """
    # Create canvas
    canvas = Image.new('RGB', (CANVAS_WIDTH, CANVAS_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    
    # Layout parameters
    y_offset = 100
    cube_spacing = 50
    
    # Draw title
    title_font = _get_font(BEBAS_NEUE_BOLD, 40)
    title_text = "3D INTRO PREVIEW (ПЛОСКИЙ ФОРМАТ)"
    title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text(
        ((CANVAS_WIDTH - title_width) // 2, 30),
        title_text,
        fill=ACCENT_COLOR,
        font=title_font
    )
    
    # Main cube (largest)
    if "main_cube" in texts_data:
        main = texts_data["main_cube"]
        main_font = _get_font(main.get("font", DRUK_CYR_BOLD), 120)
        
        # Draw cube outline
        if show_cube_outlines:
            cube_size = 400
            cube_x = (CANVAS_WIDTH - cube_size) // 2
            cube_y = y_offset
            draw.rectangle(
                [cube_x, cube_y, cube_x + cube_size, cube_y + cube_size],
                outline=ACCENT_COLOR,
                width=3
            )
        
        # Draw text
        text = main["text"]
        text_bbox = draw.textbbox((0, 0), text, font=main_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        draw.text(
            ((CANVAS_WIDTH - text_width) // 2, y_offset + 200 - text_height // 2),
            text,
            fill=TEXT_COLOR,
            font=main_font
        )
        
        y_offset += 450
    
    # Info cubes (stacked vertically)
    if "info_cubes" in texts_data:
        for info in texts_data["info_cubes"]:
            font_id = info.get("font", BEBAS_NEUE_REGULAR)
            
            # Determine font size based on text length
            text = info["text"]
            if text.isdigit() or "-" in text:  # Dates/numbers
                font_size = 80
            else:
                font_size = 60
            
            info_font = _get_font(font_id, font_size)
            
            # Draw cube outline
            if show_cube_outlines:
                cube_size = 300
                cube_x = (CANVAS_WIDTH - cube_size) // 2
                cube_y = y_offset
                draw.rectangle(
                    [cube_x, cube_y, cube_x + cube_size, cube_y + cube_size],
                    outline=(100, 100, 100),
                    width=2
                )
            
            # Draw text
            text_bbox = draw.textbbox((0, 0), text, font=info_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            draw.text(
                ((CANVAS_WIDTH - text_width) // 2, y_offset + 150 - text_height // 2),
                text,
                fill=TEXT_COLOR,
                font=info_font
            )
            
            y_offset += 300 + cube_spacing
    
    # Cities (if provided)
    if "cities" in texts_data and texts_data["cities"]:
        cities_font = _get_font(BEBAS_NEUE_REGULAR, 35)
        cities_text = texts_data["cities"]
        
        # Split into lines if too long
        max_width = CANVAS_WIDTH - 100
        cities_bbox = draw.textbbox((0, 0), cities_text, font=cities_font)
        
        if cities_bbox[2] - cities_bbox[0] > max_width:
            # Split by comma
            parts = cities_text.split(", ")
            cities_lines = []
            current_line = ""
            
            for part in parts:
                test_line = current_line + (", " if current_line else "") + part
                test_bbox = draw.textbbox((0, 0), test_line, font=cities_font)
                
                if test_bbox[2] - test_bbox[0] <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        cities_lines.append(current_line)
                    current_line = part
            
            if current_line:
                cities_lines.append(current_line)
        else:
            cities_lines = [cities_text]
        
        # Draw cities
        for line in cities_lines:
            line_bbox = draw.textbbox((0, 0), line, font=cities_font)
            line_width = line_bbox[2] - line_bbox[0]
            
            draw.text(
                ((CANVAS_WIDTH - line_width) // 2, y_offset),
                line,
                fill=(180, 180, 180),
                font=cities_font
            )
            y_offset += 50
    
    # Export to bytes
    output = io.BytesIO()
    canvas.save(output, format='PNG')
    return output.getvalue()


def generate_cube_unwrap(
    cube_id: str,
    textures: dict[str, str],
) -> bytes:
    """
    Generate cube unwrap/net showing all 6 faces.
    
    Args:
        cube_id: Identifier for cube (e.g., "main", "info_1")
        textures: Dict mapping face name to texture path or text
            Example: {"front": "/path/to/image.jpg", "top": "ТЕКСТ"}
    
    Returns:
        PNG image bytes with cube unwrap layout
    """
    # Cube unwrap layout (cross pattern):
    #       [top]
    # [left][front][right][back]
    #      [bottom]
    
    face_size = 300
    canvas_width = face_size * 4
    canvas_height = face_size * 3
    
    canvas = Image.new('RGB', (canvas_width, canvas_height), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    
    # Face positions in unwrap
    face_positions = {
        "top": (face_size, 0),
        "left": (0, face_size),
        "front": (face_size, face_size),
        "right": (face_size * 2, face_size),
        "back": (face_size * 3, face_size),
        "bottom": (face_size, face_size * 2),
    }
    
    # Draw each face
    for face_name, (x, y) in face_positions.items():
        # Draw face outline
        draw.rectangle(
            [x, y, x + face_size, y + face_size],
            outline=(100, 100, 100),
            fill=CUBE_BG,
            width=2
        )
        
        # Draw face label
        label_font = _get_font(BEBAS_NEUE_REGULAR, 20)
        draw.text(
            (x + 10, y + 10),
            face_name.upper(),
            fill=(150, 150, 150),
            font=label_font
        )
        
        # Draw texture/text if provided
        if face_name in textures:
            texture = textures[face_name]
            
            # Check if it's a file path or text
            if Path(texture).exists():
                # Load and paste image
                try:
                    img = Image.open(texture)
                    img = img.resize((face_size - 40, face_size - 40))
                    canvas.paste(img, (x + 20, y + 40))
                except Exception as e:
                    print(f"Warning: Failed to load texture {texture}: {e}")
            else:
                # Render as text
                text_font = _get_font(BEBAS_NEUE_BOLD, 50)
                text_bbox = draw.textbbox((0, 0), texture, font=text_font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                draw.text(
                    (x + (face_size - text_width) // 2, y + (face_size - text_height) // 2 + 20),
                    texture,
                    fill=TEXT_COLOR,
                    font=text_font
                )
    
    # Add title
    title_font = _get_font(BEBAS_NEUE_BOLD, 40)
    title = f"CUBE UNWRAP: {cube_id.upper()}"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    
    # Draw title background
    title_bg_y = canvas_height - 60
    draw.rectangle(
        [0, title_bg_y, canvas_width, canvas_height],
        fill=BG_COLOR
    )
    draw.text(
        ((canvas_width - title_width) // 2, title_bg_y + 10),
        title,
        fill=ACCENT_COLOR,
        font=title_font
    )
    
    # Export
    output = io.BytesIO()
    canvas.save(output, format='PNG')
    return output.getvalue()


if __name__ == "__main__":
    # Test preview generation
    test_data = {
        "main_cube": {"text": "АФИША", "font": DRUK_CYR_BOLD},
        "info_cubes": [
            {"text": "НА ВЫХОДНЫЕ", "font": BENZIN_BOLD},
            {"text": "27-28", "font": BEBAS_NEUE_REGULAR},
            {"text": "ДЕКАБРЯ", "font": BEBAS_NEUE_REGULAR},
        ],
        "cities": "Калининград, Светлогорск, Зеленоградск"
    }
    
    # Generate preview
    preview_bytes = generate_cube_text_preview(test_data)
    
    # Save to file
    output_path = "/tmp/3d_pattern_preview_test.png"
    with open(output_path, "wb") as f:
        f.write(preview_bytes)
    
    print(f"✓ Preview saved to: {output_path}")
    
    # Generate unwrap
    unwrap_data = {
        "front": "АФИША",
        "top": "TOP",
        "bottom": "BOTTOM",
        "left": "LEFT",
        "right": "RIGHT",
        "back": "BACK"
    }
    
    unwrap_bytes = generate_cube_unwrap("main", unwrap_data)
    unwrap_path = "/tmp/3d_cube_unwrap_test.png"
    
    with open(unwrap_path, "wb") as f:
        f.write(unwrap_bytes)
    
    print(f"✓ Unwrap saved to: {unwrap_path}")
