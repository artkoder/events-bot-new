#!/usr/bin/env python3
"""
Test all rotation combinations to find correct texture orientation
"""

import bpy
import os
import sys
import math

sys.path.insert(0, "/workspaces/events-bot-new/3d_intro")

OUTPUT_DIR = "/workspaces/events-bot-new/3d_intro/assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def load_image_texture(image_path: str):
    if os.path.exists(image_path):
        return bpy.data.images.load(image_path)
    return None


def create_material_with_texture(name: str, texture):
    """Create a simple material - no UV manipulation"""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (1, 1, 1, 1)
    bsdf.inputs['Roughness'].default_value = 0.3
    bsdf.inputs['Specular IOR Level'].default_value = 0.0
    
    if texture:
        tex_image = nodes.new(type='ShaderNodeTexImage')
        tex_image.image = texture
        links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])
    
    output = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    return mat


def create_test_scene(front_rotation: int, right_rotation: int):
    """Create scene with textures at specific rotations"""
    from PIL import Image
    
    # Generate textures with specific rotations
    front_path = f"/tmp/scene_textures/front_r{front_rotation}.png"
    right_path = f"/tmp/scene_textures/right_r{right_rotation}.png"
    
    # Create front texture
    img_front = Image.new("RGB", (1080, 1080), "#F1E44B")
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img_front)
    
    try:
        font_big = ImageFont.truetype("/workspaces/events-bot-new/3d_intro/assets/fonts/Benzin-Bold.ttf", 200)
        font_med = ImageFont.truetype("/workspaces/events-bot-new/video_announce/assets/BebasNeue-Regular.ttf", 100)
        font_small = ImageFont.truetype("/workspaces/events-bot-new/3d_intro/assets/fonts/DrukCyr-Bold.ttf", 60)
    except:
        font_big = font_med = font_small = ImageFont.load_default()
    
    draw.text((65, 130), "27-28", font=font_big, fill="#100E0E")
    draw.text((76, 518), "ДЕКАБРЯ", font=font_med, fill="#100E0E")
    bbox = draw.textbbox((0, 0), "АФИША", font=font_small)
    text_width = bbox[2] - bbox[0]
    draw.text((1080 - text_width - 65, 929), "АФИША", font=font_small, fill="#100E0E")
    
    if front_rotation != 0:
        img_front = img_front.rotate(front_rotation, expand=True)
    img_front.save(front_path)
    
    # Create right texture  
    img_right = Image.new("RGB", (1080, 1080), "#F1E44B")
    draw = ImageDraw.Draw(img_right)
    
    cities = ["Калининград", "Светлогорск", "Зеленоградск", "Гурьевск", "Гвардейск"]
    for i, city in enumerate(cities):
        y = 108 + (i * 173)
        draw.text((76, y), city, font=font_med, fill="#100E0E")
    
    if right_rotation != 0:
        img_right = img_right.rotate(right_rotation, expand=True)
    img_right.save(right_path)
    
    # Now create Blender scene
    clean_scene()
    
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0), size=2)
    cube = bpy.context.object
    
    blender_front = load_image_texture(front_path)
    blender_right = load_image_texture(right_path)
    
    mat_yellow = bpy.data.materials.new(name="Yellow")
    mat_yellow.use_nodes = True
    bsdf = mat_yellow.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.945, 0.894, 0.294, 1.0)
    
    mat_front = create_material_with_texture("Front", blender_front)
    mat_right = create_material_with_texture("Right", blender_right)
    
    cube.data.materials.append(mat_yellow)
    cube.data.materials.append(mat_front)
    cube.data.materials.append(mat_right)
    
    for poly in cube.data.polygons:
        n = poly.normal
        if n[1] < -0.9:  # Front: -Y
            poly.material_index = 1
        elif n[0] > 0.9:  # Right: +X
            poly.material_index = 2
        else:
            poly.material_index = 0
    
    # Camera
    bpy.ops.object.camera_add(location=(4, -4, 0))
    camera = bpy.context.object
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 4.5
    
    constraint = camera.constraints.new(type='TRACK_TO')
    constraint.target = cube
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
    
    bpy.context.scene.camera = camera
    
    # Light
    bpy.ops.object.light_add(type='SUN', location=(5, -3, 8))
    sun = bpy.context.object
    sun.data.energy = 5.0
    
    # Background
    scene = bpy.context.scene
    if not scene.world:
        world = bpy.data.worlds.new("World")
        scene.world = world
    else:
        world = scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.7, 0.7, 0.7, 1.0)


def render(output_path: str):
    scene = bpy.context.scene
    scene.render.resolution_x = 512
    scene.render.resolution_y = 256
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 32
    scene.cycles.device = 'CPU'
    scene.cycles.use_denoising = False
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print(f"Rendered: {output_path}")


if __name__ == "__main__":
    # Test all 16 combinations (4 front rotations x 4 right rotations)
    rotations = [0, 90, 180, 270]
    
    for front_rot in rotations:
        for right_rot in rotations:
            print(f"\n=== Testing Front={front_rot}°, Right={right_rot}° ===")
            create_test_scene(front_rot, right_rot)
            output = f"{OUTPUT_DIR}/rot_f{front_rot}_r{right_rot}.png"
            render(output)
    
    print("\nDone! Check assets/rot_*.png files")
