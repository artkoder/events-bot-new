#!/usr/bin/env python3
"""
Setup Final Scene with Custom Fonts and Textures
"""
import sys
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from font_manager import get_font_str, DRUK_CYR_BOLD, BENZIN_BOLD, BEBAS_NEUE_REGULAR, BEBAS_NEUE_BOLD
from generate_intro import generate_complete_intro

# Fix for PIL DecompressionBombError with large fonts
Image.MAX_IMAGE_PIXELS = None

# Settings
BG_YELLOW = (255, 235, 59)  # FFEB3B
TEXT_BLACK = (0, 0, 0)
TEXT_WHITE = (255, 255, 255)

def create_text_texture(output_path: str, width: int, height: int, 
                       bg_color: tuple, elements: list):
    """
    Create a texture with text elements.
    elements: list of dicts with 'text', 'font_id', 'size', 'pos', 'color', 'align'
    """
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    for el in elements:
        text = el['text']
        font_id = el['font_id']
        size = el['size']
        color = el.get('color', TEXT_BLACK)
        
        try:
            # Load font
            font_path = get_font_str(font_id, allow_fallback=True)
            font = ImageFont.truetype(font_path, size)
            
            # Calculate position
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            x, y = el['pos']
            
            # Adjust for center/right alignment if needed
            if el.get('align') == 'center':
                x = x - text_w // 2
            elif el.get('align') == 'right':
                x = x - text_w
                
            draw.text((x, y), text, font=font, fill=color)
            
        except Exception as e:
            print(f"Error drawing text '{text}': {e}")
            
    img.save(output_path)
    print(f"Generated texture: {output_path}")
    return output_path

def main():
    print("=== Generating Scene Textures ===")
    
    textures_dir = "/tmp/scene_textures"
    os.makedirs(textures_dir, exist_ok=True)
    
    texture_map = {}
    
    # 1. MAIN CUBE: "27-28" (Benzin) + "ДЕКАБРЯ" (Bebas)
    # Size 2048x2048 (for 2x2 cube)
    tex_main = os.path.join(textures_dir, "main_schedule.png")
    create_text_texture(
        tex_main, 2048, 2048, BG_YELLOW,
        [
            {
                'text': "27-28", 
                'font_id': BENZIN_BOLD, 
                'size': 150, 
                'pos': (1024, 600), 
                'align': 'center'
            },
            {
                'text': "ДЕКАБРЯ", 
                'font_id': BEBAS_NEUE_REGULAR, 
                'size': 160, 
                'pos': (1024, 1200), 
                'align': 'center'
            }
        ]
    )
    texture_map['main_schedule'] = tex_main
    
    # 2. AFISHA: "АФИША" (Druk Cyr Bold)
    # Size 2048x1024 (for 2x1 cube)
    tex_afisha = os.path.join(textures_dir, "info_afisha.png")
    create_text_texture(
        tex_afisha, 2048, 1024, BG_YELLOW,
        [
            {
                'text': "АФИША", 
                'font_id': DRUK_CYR_BOLD, 
                'size': 125, 
                'pos': (1024, 260), # Centered vertically roughly
                'align': 'center'
            }
        ]
    )
    texture_map['info_afisha'] = tex_afisha
    
    # 3. TYPE: "ОТ КЛАССИКИ" + "ДО РОКА" (Bebas)
    # Size 1024x1024 (for 1x1 cube)
    tex_type = os.path.join(textures_dir, "info_type.png")
    create_text_texture(
        tex_type, 1024, 1024, BG_YELLOW,
        [
            {
                'text': "ОТ КЛАССИКИ", 
                'font_id': BEBAS_NEUE_REGULAR, 
                'size': 45, 
                'pos': (512, 300), 
                'align': 'center'
            },
             {
                'text': "ДО РОКА", 
                'font_id': BEBAS_NEUE_REGULAR, 
                'size': 45, 
                'pos': (512, 500), 
                'align': 'center'
            }
        ]
    )
    texture_map['info_type'] = tex_type
    
    # 4. CITIES
    # Size 1024x2048 (for 1x2 cube)
    # Assuming user wants list
    tex_cities = os.path.join(textures_dir, "info_cities.png")
    cities = ["Калининград", "Светлогорск", "Зеленоградск"]
    
    city_elements = []
    y_start = 600
    for i, city in enumerate(cities):
        city_elements.append({
            'text': city,
            'font_id': BEBAS_NEUE_REGULAR,
            'size': 35,
            'pos': (512, y_start + i*180),
            'align': 'center'
        })
        
    create_text_texture(tex_cities, 1024, 2048, BG_YELLOW, city_elements)
    texture_map['info_cities'] = tex_cities
    
    # 5. POSTERS
    # Use existing examples for posters
    POSTER_DIR = "/workspaces/events-bot-new/3d_intro/assets/posters"
    existing_posters = [
        os.path.join(POSTER_DIR, f) for f in os.listdir(POSTER_DIR) 
        if f.startswith('poster_example') and (f.endswith('.jpg') or f.endswith('.png'))
    ]
    if existing_posters:
        for i in range(1, 13):
            # cycle through existing
            poster_path = existing_posters[(i-1) % len(existing_posters)]
            texture_map[f'poster_{i}'] = poster_path
            
    print(f"\nCreated texture map with {len(texture_map)} entries")
    
    # Generate Scene
    print("\n=== Regenerating Scene ===")
    output_blend = "/tmp/final_scene_v2.blend"
    
    generate_complete_intro(
        poster_textures=texture_map,
        output_path=output_blend
    )
    
    # Render Preview
    print("\n=== Rendering Preview ===")
    import bpy
    bpy.ops.wm.open_mainfile(filepath=output_blend)
    
    scene = bpy.context.scene
    scene.render.resolution_x = 540
    scene.render.resolution_y = 960
    scene.cycles.samples = 32
    scene.cycles.use_denoising = False
    
    # Render Final Preview (Frame 40)
    print("Rendering Final Preview (Frame 40)...")
    scene.frame_set(40)
    preview_path = "/home/codespace/.gemini/antigravity/brain/5787ba91-955f-49ed-be99-0e6994f8f7a0/final_scene_preview.png"
    scene.render.filepath = preview_path
    
    bpy.ops.render.render(write_still=True)
    print(f"✓ Preview saved: {preview_path}")

if __name__ == "__main__":
    main()
