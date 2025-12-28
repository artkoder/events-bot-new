#!/usr/bin/env python3
"""
Generate cube face texture with EXACT Figma CSS positioning.
Reference: 1080x1080 canvas with text positioned according to CSS spec.
"""
from PIL import Image, ImageDraw, ImageFont
import os

OUTPUT_DIR = "/tmp/scene_textures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Canvas size from CSS
SIZE = 1080

# Fonts
FONTS_DIR_INTRO = "/workspaces/events-bot-new/3d_intro/assets/fonts"
FONTS_DIR_VIDEO = "/workspaces/events-bot-new/video_announce/assets"


def create_front_face_texture():
    """
    Create front face texture with EXACT Figma CSS positioning.
    
    CSS Reference:
    - "27-28": left=93px, top=232px, font-size=224px, text-align=right, width=907px
    - "Декабря": left=316px, top=540px, font-size=240px, text-align=right, width=684px
    - "АФИША": left=761px, top=870px, font-size=110px, text-align=right, width=239px
    """
    img = Image.new("RGB", (SIZE, SIZE), "#F1E44B")
    draw = ImageDraw.Draw(img)
    
    # Load fonts
    try:
        font_benzin = ImageFont.truetype(f"{FONTS_DIR_INTRO}/Benzin-Bold.ttf", 224)
    except Exception as e:
        print(f"Error loading Benzin-Bold: {e}")
        font_benzin = ImageFont.load_default()
    
    try:
        font_bebas = ImageFont.truetype(f"{FONTS_DIR_VIDEO}/BebasNeue-Regular.ttf", 240)
    except Exception as e:
        print(f"Error loading BebasNeue: {e}")
        font_bebas = ImageFont.load_default()
    
    try:
        font_druk = ImageFont.truetype(f"{FONTS_DIR_INTRO}/DrukCyr-Bold.ttf", 110)
    except Exception as e:
        print(f"Error loading DrukCyr: {e}")
        font_druk = ImageFont.load_default()
    
    color = "#100E0E"
    
    # 1. "27-28" - CSS: left=93, top=232, width=907, text-align=right
    # Right-aligned means: text ends at left + width = 93 + 907 = 1000
    text_27_28 = "27-28"
    bbox = draw.textbbox((0, 0), text_27_28, font=font_benzin)
    text_width = bbox[2] - bbox[0]
    x_27_28 = 93 + 907 - text_width  # Right-align within the box
    y_27_28 = 232
    draw.text((x_27_28, y_27_28), text_27_28, font=font_benzin, fill=color)
    
    # 2. "Декабря" - CSS: left=316, top=540, width=684, text-align=right
    text_dec = "Декабря"
    bbox = draw.textbbox((0, 0), text_dec, font=font_bebas)
    text_width = bbox[2] - bbox[0]
    x_dec = 316 + 684 - text_width  # Right-align within the box
    y_dec = 540
    draw.text((x_dec, y_dec), text_dec, font=font_bebas, fill=color)
    
    # 3. "АФИША" - CSS: left=761, top=870, width=239, text-align=right
    text_afisha = "АФИША"
    bbox = draw.textbbox((0, 0), text_afisha, font=font_druk)
    text_width = bbox[2] - bbox[0]
    x_afisha = 761 + 239 - text_width  # Right-align within the box
    y_afisha = 870
    draw.text((x_afisha, y_afisha), text_afisha, font=font_druk, fill=color)
    
    # Rotate 90° counter-clockwise to correct orientation on front face (-Y normal)
    img = img.rotate(90, expand=True)
    
    # Save 
    path = os.path.join(OUTPUT_DIR, "front_face_exact.png")
    img.save(path)
    print(f"Saved {path} ({SIZE}x{SIZE})")
    
    # Copy to assets for easy viewing
    img.save("/workspaces/events-bot-new/3d_intro/assets/front_face_exact.png")
    
    return path


if __name__ == "__main__":
    create_front_face_texture()
    print("Done!")
