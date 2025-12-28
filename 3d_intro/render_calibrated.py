#!/usr/bin/env python3
"""
Final 3D Cube Render with Auto-Calibrated Parameters.
"""
import bpy
import os
import math
import sys

OUTPUT_DIR = "/workspaces/events-bot-new/3d_intro/assets"
REFERENCE_IMAGE = "/home/codespace/.gemini/antigravity/brain/d43d08f1-b44b-444b-8452-4f1dbed5e7b1/uploaded_image_1766911240181.png"

# Constants
CANVAS_SIZE = 1080.0
CUBE_SIZE = 2.0
K = CUBE_SIZE / CANVAS_SIZE  # 0.001852

# Font paths
FONTS = {
    "Benzin-Bold": "/workspaces/events-bot-new/3d_intro/assets/fonts/Benzin-Bold.ttf",
    "BebasNeue": "/workspaces/events-bot-new/video_announce/assets/BebasNeue-Regular.ttf",
    "DrukCyr-Bold": "/workspaces/events-bot-new/3d_intro/assets/fonts/DrukCyr-Bold.ttf",
}

# Calibrated Parameters (Strict Geometry: Verified Input Mismatch)
CALIBRATION = {
    "27-28": {
        "scale": 1.295,
        "space_factor": 0.976,
        "offset_x": -52.9,
        "offset_y": -28.3
    },
    "ДЕКАБРЯ": {
        "scale": 1.004,
        "space_factor": 1.006,
        "offset_x": -52.0,
        "offset_y": 27.3
    },
    "АФИША": {
        "scale": 1.193,
        "space_factor": 0.996,
        "offset_x": -52.6,
        "offset_y": 44.0
    }
}

# CSS Data
LAYERS = [
    {
        "id": "27-28",
        "text": "27-28",
        "font": "Benzin-Bold",
        "font_size": 224,
        "left": 93, "top": 232, "width": 907
    },
    {
        "id": "ДЕКАБРЯ",
        "text": "ДЕКАБРЯ",
        "font": "BebasNeue",
        "font_size": 240,
        "left": 316, "top": 540, "width": 684
    },
    {
        "id": "АФИША",
        "text": "АФИША",
        "font": "DrukCyr-Bold",
        "font_size": 110,
        "left": 761, "top": 870, "width": 239
    }
]


def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def create_calibrated_text(cube, layer):
    cal = CALIBRATION[layer["id"]]
    
    # 1. Dimensions
    font_size = layer["font_size"] * K * cal["scale"]
    char_spacing = cal.get("tracking", 0) * K
    space_factor = cal.get("space_factor", 1.0)
    
    # 2. Position (applying offsets)
    # Right-aligned anchor X
    x_anchor = layer["left"] + layer["width"]
    # Apply offset (Note: offset_x in calibration moves the rect, so it adds to X)
    loc_x = ((x_anchor - CANVAS_SIZE / 2) + cal["offset_x"]) * K
    
    # Z Position (Top + offset)
    # Note: offset_y in calibration was calculated as translation in Y (screen space).
    # In Blender Cube space (Z is up), screen down is -Z.
    # But our Z calc is (540 - top). Reducing top (moving up) increases Z.
    # Calibration dy was (ref - bl). If ref > bl (ref lower), dy > 0.
    # Correction was offset_y -= dy. So offset becomes negative (move up).
    # Wait, my Z calc logic in get_blender_params was:
    # Z = (540 - top)*K + offset_y*K.
    # So I just apply it directly.
    loc_z = ((CANVAS_SIZE / 2 - layer["top"]) + cal["offset_y"]) * K
    
    # Create Text
    bpy.ops.object.text_add()
    txt = bpy.context.object
    txt.data.body = layer["text"]
    
    # Font
    font_path = FONTS.get(layer["font"])
    if font_path:
        txt.data.font = bpy.data.fonts.load(font_path)
    
    # Settings
    txt.data.size = font_size
    txt.data.align_x = 'RIGHT'
    txt.data.align_y = 'TOP' # Using TOP as standardized in render_layer.py
    
    if space_factor != 1.0:
        txt.data.space_character = space_factor
    elif char_spacing != 0:
        txt.data.space_character = 1.0 + char_spacing / font_size
        
    # Placement
    txt.location = (loc_x, -1.002, loc_z) # Slightly in front of cube (-1.0)
    txt.rotation_euler = (math.radians(90), 0, 0)
    txt.data.extrude = 0.001
    txt.parent = cube
    
    # Material
    mat = bpy.data.materials.new(name=f"Mat_{layer['id']}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    # Make text INTENSE RED for visibility against black reference
    bsdf.inputs['Base Color'].default_value = (1.0, 0.0, 0.0, 1.0)
    txt.data.materials.append(mat)


def create_scene_objects():
    # Cube
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    cube = bpy.context.object
    
    mat = bpy.data.materials.new(name="Yellow")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs['Base Color'].default_value = (0.945, 0.894, 0.294, 1.0)
    cube.data.materials.append(mat)
    
    # Overlay for verification
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, -1.01, 0))
    plane = bpy.context.object
    plane.rotation_euler = (math.radians(90), 0, 0)
    
    mat_ov = bpy.data.materials.new(name="Overlay")
    mat_ov.use_nodes = True
    mat_ov.blend_method = 'BLEND'
    bsdf = mat_ov.node_tree.nodes.get("Principled BSDF")
    tex = mat_ov.node_tree.nodes.new('ShaderNodeTexImage')
    if os.path.exists(REFERENCE_IMAGE):
        tex.image = bpy.data.images.load(REFERENCE_IMAGE)
    mat_ov.node_tree.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
    
    # Increase transparency to see red text better (0.3 alpha means 30% overlay opacity)
    bsdf.inputs['Alpha'].default_value = 0.3
    plane.data.materials.append(mat_ov)
    
    # Texts
    for layer in LAYERS:
        create_calibrated_text(cube, layer)
        
    return cube


def setup_render():
    # Camera Front View
    bpy.ops.object.camera_add(location=(0, -5, 0))
    cam = bpy.context.object
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = 2.2 # Slightly zoom in
    cam.rotation_euler = (math.radians(90), 0, 0)
    bpy.context.scene.camera = cam
    
    # Light - Frontal and strong
    bpy.ops.object.light_add(type='SUN', location=(0, -10, 5))
    light = bpy.context.object
    light.rotation_euler = (math.radians(60), 0, 0)
    light.data.energy = 5.0
    
    # Render settings
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 32
    scene.cycles.use_denoising = False
    
    scene.render.resolution_x = 1080
    scene.render.resolution_y = 1080
    scene.render.filepath = f"{OUTPUT_DIR}/calibrated_final.png"
    
    print("Starting render...")
    bpy.ops.render.render(write_still=True)
    print(f"Rendered: {scene.render.filepath}")


if __name__ == "__main__":
    try:
        print("Starting render script...")
        clean_scene()
        print("Scene cleaned.")
        create_scene_objects()
        print("Objects created.")
        setup_render()
        print("Render setup complete and executed.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
