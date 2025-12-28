import bpy
import os
import sys

# Add path to scripts
sys.path.insert(0, os.path.dirname(__file__))

from bento_scene import generate_bento_scene
from setup_final_scene import create_text_texture, BG_YELLOW, BENZIN_BOLD, BEBAS_NEUE_REGULAR

def main():
    # 1. Generate minimal textures
    tex_dir = "/tmp/scene_textures"
    os.makedirs(tex_dir, exist_ok=True)
    
    # Mock Font textures
    tex_main = os.path.join(tex_dir, "main_schedule.png")
    create_text_texture(tex_main, 1024, 1024, BG_YELLOW, [{'text': "27-28\nDec", 'font_id': BENZIN_BOLD, 'size': 180, 'pos': (512, 512), 'align': 'center'}])
    
    tex_map = {'main_schedule': tex_main}
    
    # 2. Generate Scene
    generate_bento_scene(poster_textures=tex_map)
    
    # 3. Render
    scene = bpy.context.scene
    scene.render.resolution_x = 1080
    scene.render.resolution_y = 1080 # Output usually square for grid check using Ortho fit
    # But user wants 1080x1920? 
    # With Ortho scale fit to Width, 1920 height will show empty space top/bottom or fit perfectly?
    # Grid is Square (Total W = Total H). 1080x1080 is best for check.
    scene.render.resolution_x = 1080
    scene.render.resolution_y = 1080
    
    scene.render.engine = 'CYCLES'
    
    scene.cycles.samples = 32
    scene.cycles.use_denoising = False
    
    # Standard Transform for Colors
    # scene.view_settings.view_transform = 'Standard'
    
    out_path = "/home/codespace/.gemini/antigravity/brain/5787ba91-955f-49ed-be99-0e6994f8f7a0/layout_check_v26.png"
    scene.render.filepath = out_path
    
    print(f"Rendering Layout Check v2 to {out_path}...")
    bpy.ops.render.render(write_still=True)
    print("Done.")

if __name__ == "__main__":
    main()
