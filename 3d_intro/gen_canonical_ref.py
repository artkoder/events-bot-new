#!/usr/bin/env python3
"""
Generate Canonical Reference Textures (PIL) for INK-to-INK Calibration.
Follows the protocol:
- Canvas: 1080x1080
- Alignment: Right-align based on INK bbox (visible pixels), not advance.
- Vertical: Baseline positioning using ascent/descent.
- Output: 
    - Full reference (Gray)
    - Separate masks (White on Transparent) for each layer.
"""
from PIL import Image, ImageDraw, ImageFont
import os

# --- Configuration ---
OUTPUT_DIR = "/workspaces/events-bot-new/3d_intro/assets/calibration"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CANVAS_SIZE = 1080

# Font Paths
FONTS_DIR_INTRO = "/workspaces/events-bot-new/3d_intro/assets/fonts"
FONTS_DIR_VIDEO = "/workspaces/events-bot-new/video_announce/assets"

FONT_BENZIN = os.path.join(FONTS_DIR_INTRO, "Benzin-Bold.ttf")
FONT_BEBAS = os.path.join(FONTS_DIR_VIDEO, "BebasNeue-Regular.ttf")
FONT_DRUK = os.path.join(FONTS_DIR_INTRO, "DrukCyr-Bold.ttf")
FONT_BEBAS_PRO = os.path.join(FONTS_DIR_VIDEO, "Bebas-Neue-Pro-SemiExpanded-Bold-BF66cf3d78c1f4e.ttf")

# --- Specs (from User Protocol) ---
LAYERS = [
    # --- Front Face ---
    {
        "name": "27_28",
        "text": "27-28",
        "font_path": FONT_BENZIN,
        "size": 224,
        "line_height": 308,
        "left_css": 93,
        "top_css": 232,
        "width_css": 907, 
    },
    {
        "name": "month",
        "text": "ДЕКАБРЯ",
        "font_path": FONT_BEBAS,
        "size": 240,
        "line_height": 240,
        "left_css": 316,
        "top_css": 540,
        "width_css": 684,
    },
    {
        "name": "afisha",
        "text": "АФИША",
        "font_path": FONT_DRUK,
        "size": 110,
        "line_height": 140,
        "left_css": 761,
        "top_css": 870,
        "width_css": 239,
    },
    # --- Right Face ---
    {
        "name": "kaliningrad",
        "text": "Калининград",
        "font_path": FONT_BEBAS_PRO,
        "size": 140,
        "line_height": 159,
        "left_css": 80,
        "top_css": 142,
        "width_css": 592,
        "align": "left"
    },
    {
        "name": "svetlogorsk",
        "text": "Светлогорск",
        "font_path": FONT_BEBAS_PRO,
        "size": 140,
        "line_height": 159,
        "left_css": 80,
        "top_css": 301,
        "width_css": 560,
        "align": "left"
    },
    {
        "name": "zelenogradsk",
        "text": "Зеленоградск",
        "font_path": FONT_BEBAS_PRO,
        "size": 140,
        "line_height": 159,
        "left_css": 80,
        "top_css": 460,
        "width_css": 630,
        "align": "left"
    },
    {
        "name": "guryevsk",
        "text": "Гурьевск",
        "font_path": FONT_BEBAS_PRO,
        "size": 140,
        "line_height": 159,
        "left_css": 80,
        "top_css": 619,
        "width_css": 404,
        "align": "left"
    },
    {
        "name": "gvardeysk",
        "text": "Гвардейск",
        "font_path": FONT_BEBAS_PRO,
        "size": 140,
        "line_height": 159,
        "left_css": 80,
        "top_css": 778,
        "width_css": 467,
        "align": "left"
    }
]

def get_font_metrics(font):
    ascent, descent = font.getmetrics()
    return ascent, descent

def render_layer(layer_spec, debug_color=(255, 255, 255)):
    """
    Render a single layer to a transparent image using strict positioning.
    """
    img = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype(layer_spec["font_path"], layer_spec["size"])
    except IOError:
        print(f"ERROR: Could not load font {layer_spec['font_path']}")
        return img
        
    text = layer_spec["text"]
    
    # 1. Alignment Logic
    align = layer_spec.get("align", "right")
    
    # Measure visible bounding box (INK)
    bbox = font.getbbox(text)
    
    if align == "right":
        # right_css = left_css + width_css
        right_css = layer_spec["left_css"] + layer_spec["width_css"]
        
        # Target: The RIGHTMOST pixel of the ink should align with right_css
        target_ink_right = right_css
        
        # x_draw = target_ink_right - bbox[2]
        x_draw = target_ink_right - bbox[2]
        
    elif align == "left":
        # Target: The LEFTMOST pixel of the ink should align with left_css
        target_ink_left = layer_spec["left_css"]
        
        # We want x_draw + bbox[0] = target_ink_left
        # => x_draw = target_ink_left - bbox[0]
        x_draw = target_ink_left - bbox[0]

    
    # 2. Vertical Alignment (Baseline-based)
    # baseline_y = top_css + leading/2 + ascent
    # content_h = ascent + descent
    ascent, descent = get_font_metrics(font)
    content_h = ascent + descent
    leading = layer_spec["line_height"] - content_h
    
    # If leading is negative, we trust line_height as the bounding box height constraint?
    # Protocol says: baseline_y = top + leading/2 + ascent
    baseline_y = layer_spec["top_css"] + (leading / 2) + ascent
    
    # draw.text using coordinates usually expects Top-Left of the Advance box,
    # but PIL's y-coordinate behavior can depend on anchor.
    # Default anchor 'la' (Left-Ascender) means (x, y) is top-left of ascender line?
    # Let's use anchor='ls' (Left-Baseline) to be precise with baseline calculation.
    
    draw.text((x_draw, baseline_y), text, font=font, fill=debug_color, anchor="ls")
    
    return img

def main():
    print("Generating Canonical Reference Textures...")
    
    # 1. Generate Full Reference (Gray for visual check)
    full_ref = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "#000000") # Black background
    draw_full = ImageDraw.Draw(full_ref)
    
    # Create a composite of all layers
    for layer in LAYERS:
        print(f"Processing {layer['name']}...")
        
        # Render Mask (White on Transparent)
        mask_img = render_layer(layer, debug_color=(255, 255, 255, 255))
        mask_path = os.path.join(OUTPUT_DIR, f"ref_mask_{layer['name']}.png")
        mask_img.save(mask_path)
        print(f"  -> Saved Mask: {mask_path}")
        
        # Add to Full Reference (Gray)
        # We'll re-render in Gray or just composite. Let's re-render to be safe.
        gray_layer = render_layer(layer, debug_color="#CCCCCC")
        full_ref.paste(gray_layer, (0, 0), gray_layer)
        
    full_path = os.path.join(OUTPUT_DIR, "ref_full_check.png")
    full_ref.save(full_path)
    print(f"  -> Saved Full Check: {full_path}")
    print("Done.")

if __name__ == "__main__":
    main()
