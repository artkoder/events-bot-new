#!/usr/bin/env python3
"""Generate base textures without rotation (to be called before rotation tests)"""
from PIL import Image, ImageDraw, ImageFont
import os

OUTPUT_DIR = "/tmp/scene_textures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SIZE = 1080
FONTS_DIR_INTRO = "/workspaces/events-bot-new/3d_intro/assets/fonts"
FONTS_DIR_VIDEO = "/workspaces/events-bot-new/video_announce/assets"

# Create front texture
img_front = Image.new("RGB", (SIZE, SIZE), "#F1E44B")
draw = ImageDraw.Draw(img_front)

try:
    font_big = ImageFont.truetype(f"{FONTS_DIR_INTRO}/Benzin-Bold.ttf", 300)
    font_med = ImageFont.truetype(f"{FONTS_DIR_VIDEO}/BebasNeue-Regular.ttf", 130)
    font_small = ImageFont.truetype(f"{FONTS_DIR_INTRO}/DrukCyr-Bold.ttf", 75)
except Exception as e:
    print(f"Font error: {e}")
    font_big = font_med = font_small = ImageFont.load_default()

draw.text((65, 130), "27-28", font=font_big, fill="#100E0E")
draw.text((76, 518), "ДЕКАБРЯ", font=font_med, fill="#100E0E")
bbox = draw.textbbox((0, 0), "АФИША", font=font_small)
text_width = bbox[2] - bbox[0]
draw.text((1080 - text_width - 65, 929), "АФИША", font=font_small, fill="#100E0E")

img_front.save(f"{OUTPUT_DIR}/main_front_orig.png")
print(f"Saved {OUTPUT_DIR}/main_front_orig.png")

# Create right texture
img_right = Image.new("RGB", (SIZE, SIZE), "#F1E44B")
draw = ImageDraw.Draw(img_right)

cities = ["Калининград", "Светлогорск", "Зеленоградск", "Гурьевск", "Гвардейск"]
for i, city in enumerate(cities):
    y = 108 + (i * 173)
    draw.text((76, y), city, font=font_med, fill="#100E0E")

img_right.save(f"{OUTPUT_DIR}/main_right_orig.png")
print(f"Saved {OUTPUT_DIR}/main_right_orig.png")
