from __future__ import annotations

import argparse
from typing import Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

# --- Configuration ---
ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT / "video_announce"
# Font locations - matching previous logic
TEST_AFISHA_DIR = VIDEO_DIR / "test_afisha"
ASSETS_DIR = VIDEO_DIR / "assets"
OUT_DIR = VIDEO_DIR / "test_output"

# Standard Canvas
CANVAS_SIZE = (1080, 1572)
BG_COLOR = (241, 228, 75, 255)  # #F1E44B from CSS
TEXT_COLOR = (16, 14, 14, 255)  # #100E0E from CSS

@dataclass
class TextStyle:
    font_name: str
    size: int
    x: int
    y: int
    # Optional width/height if needed for alignment, strictly following CSS
    width: Optional[int] = None 
    height: Optional[int] = None
    align: str = "left"
    rotate: int = 0
    line_height: Optional[int] = None
    uppercase: bool = False

@dataclass
class IntroConfig:
    date: TextStyle
    month: TextStyle
    title: TextStyle
    cities: TextStyle

# CSS-derived configurations
PAD = 0 # No padding needed if we use exact coordinates? Maybe for image generation

# Fonts mapping
FONT_FILES = {
    "Benzin-Bold": "Benzin-Bold.ttf",
    "Bebas Neue": "BebasNeue-Bold.ttf",
    "Oswald": "Oswald-VariableFont_wght.ttf",
    "Druk Cyr": "DrukCyr-Bold.ttf" 
}

# --- Style Definitions based on USER CSS ---
STYLE_DAY = IntroConfig(
    date=TextStyle(
        font_name="Benzin-Bold",
        size=224,
        x=676, y=270,
        width=324, height=308,
        align="right"
    ),
    month=TextStyle(
        font_name="Bebas Neue",
        size=200,
        x=850, y=541,
        width=476, height=200,
        rotate=-90,
        uppercase=True
    ),
    title=TextStyle(
        font_name="Druk Cyr",
        size=180,
        x=73, y=827,
        width=724, height=228,
        align="right",
        uppercase=True
    ),
    cities=TextStyle(
        font_name="Oswald",
        size=60,
        x=435, y=1058,
        width=357, height=267,
        align="right",
        line_height=89,
        uppercase=True
    )
)

STYLE_WEEKEND = IntroConfig(
    date=TextStyle(
        font_name="Benzin-Bold",
        size=224,
        x=55, y=270,
        width=945, height=308,
        align="right"
     ),
    month=TextStyle(
        font_name="Bebas Neue",
        size=200,
        x=850, y=541,
        width=476, height=200,
        rotate=-90,
        uppercase=True
        # Uses same month style as day
    ),
    title=TextStyle(
        font_name="Druk Cyr",
        size=220,
        x=82, y=779,
        width=710, height=279,

        # Weekend ref usually looks left aligned or centered, but CSS for Day said right.
        # User CSS for Weekend Title:
        # width: 710px; left: 82px; top: 779px;
        # text-align is NOT specified in Weekend CSS provided! 
        # But Day was text-align: right.
        # Checking ref... Weekend title "ВЫХОДНЫЕ" usually spans nicely.
        # Let's assume Left or Center if not specified.
        # Day: text-align: right.
        # Weekend: "ВЫХОДНЫЕ". Left 82. Width 710.
        # I will start with Left for Weekend unless specified.
        # Wait, if I align right, x + width = right edge. 
        # Let's look at reference image alignment... "ВЫХОДНЫЕ" looks big.
        # I'll stick to 'left' default if missing, or maybe 'right' if consistent with Day.
        # Let's assume 'left' because x=82 is quite far left.
        align="left", 
        uppercase=True
    ),
    cities=TextStyle(
        font_name="Oswald",
        size=60,
        x=435, y=1058,
        width=357, height=267,
        align="right",
        line_height=89,
        uppercase=True
    )
)

def _find_font(font_name: str) -> Path:
    filename = FONT_FILES.get(font_name, font_name)
    candidates = [
        TEST_AFISHA_DIR / filename,
        ASSETS_DIR / filename,
        ASSETS_DIR / "fonts" / filename,
        VIDEO_DIR / "assets" / filename
    ]
    for p in candidates:
        if p.exists():
            return p
    # Fallback try checking if filename provided directly
    if (ASSETS_DIR / font_name).exists():
        return ASSETS_DIR / font_name
    
    raise FileNotFoundError(f"Font {font_name} ({filename}) not found in search paths.")

def _load_font(font_name: str, size: int) -> ImageFont.FreeTypeFont:
    path = _find_font(font_name)
    return ImageFont.truetype(str(path), size)

def render_text(
    draw: ImageDraw.ImageDraw, 
    img: Image.Image,
    text: str, 
    style: TextStyle
):
    text_content = text.upper() if style.uppercase else text
    font = _load_font(style.font_name, style.size)
    
    # Coordinates in CSS are usually Top-Left of the bounding box
    x, y = style.x, style.y
    
    # Handle Rotation
    if style.rotate != 0:
        # Render to temp image first
        # For Bebas Month rotated -90
        # CSS: width 476, height 200. transform rotate(-90).
        # When rotated -90 (CCW 270 or CW 90?), standard CSS rotate(-90) is Counter-Clockwise?
        # No, CSS rotate(90deg) is Clockwise. rotate(-90deg) is Counter-Clockwise.
        # Wait, standard math is CCW positive. CSS is CW positive.
        # So -90 CSS is 90 degrees CCW (pointing up?).
        # User CSS: "transform: rotate(-90deg);"
        # "ЯНВАРЯ" reads bottom-to-top.
        # PIL rotate is CCW. So rotate(90) makes it vertical reading up.
        
        # Create a temp surface large enough
        # We render HORIZONTALLY first.
        # CSS Box: W=476, H=200 (post-rotation? or pre-rotation?)
        # Usually CSS width/height applies to the element *before* transform.
        # So we render into 476x200 area horizontally.
        
        # Render text
        dummy = Image.new("RGBA", (10,10))
        d_dummy = ImageDraw.Draw(dummy)
        
        spacing = 4
        if style.line_height:
             # spacing approx line_height - size? 
             spacing = style.line_height - style.size
        
        bbox = d_dummy.textbbox((0,0), text_content, font=font, spacing=spacing)
        w_text = bbox[2] - bbox[0]
        h_text = bbox[3] - bbox[1]
        
        # Create image size of text or specified width?
        # Using specified width from CSS as max container
        box_w = style.width if style.width else w_text
        box_h = style.height if style.height else h_text
        
        txt_img = Image.new("RGBA", (w_text + 20, h_text + 20), (0,0,0,0))
        d_txt = ImageDraw.Draw(txt_img)
        d_txt.text((0,0), text_content, font=font, fill=TEXT_COLOR, spacing=spacing, align=style.align)
        
        # Crop to content
        txt_crop = txt_img.crop(txt_img.getbbox())
        
        # Rotate
        # CSS: rotate(-90deg) -> Counter Clockwise 90.
        # PIL .rotate(90) -> Counter Clockwise 90.
        rotated = txt_crop.rotate(90, expand=True, resample=Image.BICUBIC)
        
        # Placement
        # CSS absolute positioning with transforms can be tricky.
        # Usually it means the top-left corner of the element *before* rotation is at (x,y)? 
        # Or the center?
        # CSS `transform-origin` defaults to center.
        # IF transform-origin is center:
        #   Center of unrotated box at x + w/2, y + h/2.
        #   We place center of rotated box there.
        # BUT specific layout engines might vary.
        # User CSS: left 850, top 541. Width 476, Height 200.
        # Center = 850+238, 541+100 = 1088, 641.
        # If we rotate, the visual bounding box changes.
        
        # TRICK: Let's assume standard top-left anchor for the *visual* result or just try to match reference logic.
        # Previous reference checks showed correct month placement.
        # Let's stick to the visual coordinates derived from CSS 'rect'.
        # If I place it at (x,y), let's see.
        # Actually, let's use the layout engine values directly.
        # If I put it at x,y strictly, it might look wrong if rotation pivot is weird.
        # However, for 'Bebas' month 'JANUARY', standard placement is vertical.
        
        img.alpha_composite(rotated, (x, y))
        return

    # Standard Text
    # Handle Alignment within Box
    if style.align == "right" and style.width:
        # Align right edge of text to x + width
        # Draw text at right-aligned position
        
        # Measure first
        dummy = Image.new("RGBA", (10,10))
        d_dummy = ImageDraw.Draw(dummy)
        spacing = 4
        if style.line_height:
             spacing = style.line_height - style.size
             
        # Use multiline if needed
        bbox = d_dummy.multiline_textbbox((0,0), text_content, font=font, spacing=spacing, align="right")
        w_text = bbox[2] - bbox[0]
        
        # x is left edge of box. x+width is right edge.
        # Text should end at x+width.
        # So start x = (x + width) - w_text
        draw_x = (x + style.width) - w_text
        draw.multiline_text((draw_x, y), text_content, font=font, fill=TEXT_COLOR, spacing=spacing, align="right")
        
    elif style.align == "center" and style.width:
        # Center in box
        # Not used in current CSS but good to have
        dummy = Image.new("RGBA", (10,10))
        d_dummy = ImageDraw.Draw(dummy)
        spacing = 4
        if style.line_height:
             spacing = style.line_height - style.size
        bbox = d_dummy.multiline_textbbox((0,0), text_content, font=font, spacing=spacing, align="center")
        w_text = bbox[2] - bbox[0]
        
        draw_x = x + (style.width - w_text) // 2
        draw.multiline_text((draw_x, y), text_content, font=font, fill=TEXT_COLOR, spacing=spacing, align="center")

    else:
        # Left align or no width constraint
        spacing = 4
        if style.line_height:
             spacing = style.line_height - style.size
        # For Cities, user CSS says line-height 89, font size 60. Spacing = 29.
        draw.multiline_text((x, y), text_content, font=font, fill=TEXT_COLOR, spacing=spacing, align=style.align)


def generate_intro(
    style_name: str,
    date_text: str,
    month_text: str,
    title_text: str,
    cities_text: str,
    output_path: Path
):
    cfg = STYLE_DAY if style_name == "day" else STYLE_WEEKEND
    
    img = Image.new("RGBA", CANVAS_SIZE, BG_COLOR)
    draw = ImageDraw.Draw(img)
    
    render_text(draw, img, date_text, cfg.date)
    render_text(draw, img, month_text, cfg.month)
    render_text(draw, img, title_text, cfg.title)
    render_text(draw, img, cities_text, cfg.cities)
    
    img.save(output_path)
    print(f"Generated {output_path}")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate Day Version
    generate_intro(
        "day",
        "19", 
        "ЯНВАРЯ", 
        "ПОНЕДЕЛЬНИК", 
        "КАЛИНИНГРАД\nСВЕТЛОГОРСК\nЗЕЛЕНОГРАДСК",
        OUT_DIR / "codex_intro_day_aligned.png"
    )
    
    # Generate Weekend Version
    generate_intro(
        "weekend",
        "24-25", 
        "ДЕКАБРЯ", 
        "ВЫХОДНЫЕ", 
        "КАЛИНИНГРАД\nСВЕТЛОГОРСК\nЗЕЛЕНОГРАДСК",
        OUT_DIR / "codex_intro_weekend_aligned.png"
    )

if __name__ == "__main__":
    main()
