from PIL import Image, ImageDraw, ImageFont
import os

# Configuration
OUTPUT_DIR = "/tmp/scene_textures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# IMPORTANT: Generate at ORIGINAL CSS size (1080x1080)
# No pre-scaling - let Blender's UV mapping handle the fit
SIZE = 1080

# Fonts
FONTS_DIR_INTRO = "/workspaces/events-bot-new/3d_intro/assets/fonts"
FONTS_DIR_VIDEO = "/workspaces/events-bot-new/video_announce/assets"

FONT_BENZIN = os.path.join(FONTS_DIR_INTRO, "Benzin-Bold.ttf")
FONT_DRUK = os.path.join(FONTS_DIR_INTRO, "DrukCyr-Bold.ttf")
FONT_BEBAS = os.path.join(FONTS_DIR_VIDEO, "BebasNeue-Regular.ttf")


def create_main_front():
    """
    Front face: 27-28, ДЕКАБРЯ, АФИША
    
    Based on reference image (1024x1024):
    - "27-28": Large bold Benzin font, top area (~12% from top)
    - "ДЕКАБРЯ": Bebas Neue, thinner, below numbers (~48% from top)
    - "АФИША": Druk italic, bottom-right corner
    """
    img = Image.new("RGB", (SIZE, SIZE), "#F1E44B")
    draw = ImageDraw.Draw(img)

    # 1. "27-28" - Benzin-Bold, very large (~28% of height)
    try:
        font_numbers = ImageFont.truetype(FONT_BENZIN, 300)
    except:
        print(f"Warning: Could not load {FONT_BENZIN}")
        font_numbers = ImageFont.load_default()
    
    # Position: ~6% left margin, ~12% top margin (matching reference)
    x_numbers = int(SIZE * 0.06)
    y_numbers = int(SIZE * 0.12)
    draw.text((x_numbers, y_numbers), "27-28", font=font_numbers, fill="#100E0E")

    # 2. "ДЕКАБРЯ" - Bebas Neue (thinner than Benzin), below numbers
    try:
        font_month = ImageFont.truetype(FONT_BEBAS, 130)
    except:
        font_month = ImageFont.load_default()
    
    # Position below numbers, ~7% left margin, ~48% from top
    x_month = int(SIZE * 0.07)
    y_month = int(SIZE * 0.48)
    draw.text((x_month, y_month), "ДЕКАБРЯ", font=font_month, fill="#100E0E")

    # 3. "АФИША" - Druk Cyr Bold (italic style), bottom-right
    try:
        font_afisha = ImageFont.truetype(FONT_DRUK, 75)
    except:
        font_afisha = ImageFont.load_default()
    
    # Get text size for right-alignment
    bbox = draw.textbbox((0, 0), "АФИША", font=font_afisha)
    text_width = bbox[2] - bbox[0]
    
    # Position: ~6% right margin, ~86% from top
    x_afisha = SIZE - text_width - int(SIZE * 0.06)
    y_afisha = int(SIZE * 0.86)
    draw.text((x_afisha, y_afisha), "АФИША", font=font_afisha, fill="#100E0E")

    # Rotate 90° counter-clockwise for correct orientation on front face (-Y normal)
    img = img.rotate(90, expand=True)
    
    path = os.path.join(OUTPUT_DIR, "main_front.png")
    img.save(path)
    print(f"Saved {path} ({SIZE}x{SIZE})")
    return path


def create_main_right():
    """
    Right face: Cities list
    
    Based on reference image (1024x1024):
    - 5 cities in Bebas Neue (thin font)
    - ~7% left margin
    - ~10% top margin, evenly spaced vertically
    """
    img = Image.new("RGB", (SIZE, SIZE), "#F1E44B")
    draw = ImageDraw.Draw(img)

    # Bebas Neue, matching reference size (~12% of height)
    try:
        font = ImageFont.truetype(FONT_BEBAS, 130)
    except:
        print(f"Warning: Could not load {FONT_BEBAS}")
        font = ImageFont.load_default()

    color = "#100E0E"

    # Calculate positions based on reference
    left_margin = int(SIZE * 0.07)   # ~7% left margin
    top_margin = int(SIZE * 0.10)    # ~10% top margin
    vertical_step = int(SIZE * 0.16) # ~16% vertical spacing between cities
    
    cities = ["Калининград", "Светлогорск", "Зеленоградск", "Гурьевск", "Гвардейск"]
    
    for i, city in enumerate(cities):
        y = top_margin + (i * vertical_step)
        draw.text((left_margin, y), city, font=font, fill=color)

    # Rotate 270° counter-clockwise (= 90° clockwise) for correct orientation on right face (+X normal)
    img = img.rotate(270, expand=True)
    
    path = os.path.join(OUTPUT_DIR, "main_right.png")
    img.save(path)
    print(f"Saved {path} ({SIZE}x{SIZE})")
    return path


if __name__ == "__main__":
    create_main_front()
    create_main_right()
