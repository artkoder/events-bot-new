import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Local paths
SOURCE_FOLDER = Path("video_announce/test_afisha")
OUTPUT_FOLDER = Path("video_announce/test_output")
OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)

def find_font(name: str) -> str:
    """Simple local font finder."""
    candidates = [
        SOURCE_FOLDER / name,
        SOURCE_FOLDER / "fonts" / name,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None

def generate_cover_image(date_range: str, month: str, title: str, output_path: Path):
    """
    Generates cover in 'Poster' style.
    Ref: Yellow background, large date top-left, vertical month right, localized title.
    """
    width, height = 1080, 1920 # Full HD Portrait for Stories/Reels usually? Or 864x1104 per notebook? 
    # Notebook said 864x1104. User ref seems vertical. Let's stick to notebook dims for now or 1080x1920?
    # Ref check: 'intro ref (day).png' - let's assume standard Story 9:16 or the notebook's 864x1104.
    # Notebook default was 864x1104. I will use that for consistency with the pipeline.
    width, height = 864, 1104

    bg_color = (255, 242, 0) # UPDATED Yellow from rough guess, will refine.
    # User said "different shade of yellow". Let's try to pick a vibrant poster yellow.
    bg_color = (245, 230, 20) # A bit darker/gold? 
    # Reverting to a standard "Post-it" yellow common in these designs, or picking from previous code (245, 225, 14)
    bg_color = (245, 225, 14) 
    
    text_color = (0, 0, 0)

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Fonts
    font_benzin = find_font("Benzin-Bold.ttf")
    font_bebas = find_font("BebasNeue-Bold.ttf") or find_font("BebasNeue-Regular.ttf")
    font_oswald = find_font("Oswald-VariableFont_wght.ttf")
    font_druk = find_font("DrukCyr-Bold.ttf")

    def load_font(path, size):
        if path: return ImageFont.truetype(path, size)
        return ImageFont.load_default()

    # 1. DATE (Top Left)
    # Style: "25-27"
    date_font = load_font(font_benzin, 180)
    draw.text((40, 40), date_range, font=date_font, fill=text_color)

    # 2. MONTH (Vertical Right)
    # Style: "ДЕКАБРЯ" rotated 90 deg, aligned to right edge.
    month_font_size = 200
    month_font = load_font(font_bebas, month_font_size)
    
    # Create separate image for rotation
    # Measure text
    l, t, r, b = month_font.getbbox(month)
    w, h = r - l, b - t
    # Add padding
    txt_img = Image.new('RGBA', (w, h + 40), (0,0,0,0))
    d_txt = ImageDraw.Draw(txt_img)
    d_txt.text((-l, -t), month, font=month_font, fill=text_color)
    
    # Rotate
    rotated_txt = txt_img.rotate(90, expand=True)
    # Paste on right edge
    img.paste(rotated_txt, (width - rotated_txt.width - 20, 150), rotated_txt)

    # 3. TITLE (Centered Bottom/Middle)
    # Style: "ВЫХОДНЫЕ"
    # Fit width logic
    title_font_path = font_druk or font_oswald
    
    # Simple fitting loop
    t_size = 300
    while t_size > 50:
        f = load_font(title_font_path, t_size)
        l,t,r,b = f.getbbox(title)
        tw = r - l
        if tw < width - 80: # Margin 40px * 2
            break
        t_size -= 10
    
    title_font = load_font(title_font_path, t_size)
    l,t,r,b = title_font.getbbox(title)
    tw = r - l
    
    # Position: Bottom third?
    x_pos = (width - tw) // 2
    y_pos = height - 300 # Approx from bottom
    
    draw.text((x_pos, y_pos), title, font=title_font, fill=text_color)

    print(f"Generated: {output_path}")
    img.save(output_path)

if __name__ == "__main__":
    generate_cover_image("20-21", "ЯНВАРЯ", "ВЫХОДНЫЕ", OUTPUT_FOLDER / "test_cover_weekend.png")
    generate_cover_image("21", "ЯНВАРЯ", "СОБЫТИЯ ЗАВТРА", OUTPUT_FOLDER / "test_cover_tomorrow.png")
