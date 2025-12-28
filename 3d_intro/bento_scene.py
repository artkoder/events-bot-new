#!/usr/bin/env python3
"""
3D Bento Grid Scene Generator (v2 - 6x6 Layout)
Generates a 3D scene with cubes arranged in a 6x6 Bento grid layout
matching the user's specific reference.
"""

import bpy
import math
import random
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# ============================================================================
# Texture and Material System
# ============================================================================

def load_image_texture(image_path: str) -> Optional[bpy.types.Image]:
    """Load an image texture from file"""
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return None
    try:
        img = bpy.data.images.load(image_path)
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def create_material(name: str, color: Tuple[float, float, float, float], texture: bpy.types.Image = None, uv_scale: float = 1.0, uv_offset: Tuple[float, float] = (0.0, 0.0)) -> bpy.types.Material:
    """Create a material with color or texture
    
    Args:
        uv_scale: Scale factor for UV mapping (3.5 = shrinks texture 3.5x)
        uv_offset: (x, y) offset for texture position. Positive X = right, Positive Y = down.
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Roughness'].default_value = 0.4
    
    if texture:
        # Texture Coordinate -> Mapping -> Image Texture -> BSDF
        tex_coord = nodes.new(type='ShaderNodeTexCoord')
        mapping = nodes.new(type='ShaderNodeMapping')
        mapping.inputs['Scale'].default_value = (uv_scale, uv_scale, uv_scale)
        # Apply offset: user wants 40% up (negative Y) and 40% right (positive X)
        mapping.inputs['Location'].default_value = (uv_offset[0], uv_offset[1], 0.0)
        
        tex_image = nodes.new(type='ShaderNodeTexImage')
        tex_image.image = texture
        
        links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
        links.new(mapping.outputs['Vector'], tex_image.inputs['Vector'])
        links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])
    
    output = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    return mat

def apply_textures_to_cube(cube: bpy.types.Object, texture_paths: List[str]):
    """Apply textures to cube faces"""
    # Simple implementation: Apply image to Front face, other faces dark/colored
    if not texture_paths:
        return
        
    # Load first image
    img = load_image_texture(texture_paths[0])
    if not img:
        return

    mat = create_material(f"{cube.name}_img", (1,1,1,1), img)
    
    # Assign to all faces for now (box mapping usually handles this, or just front)
    # The user wants "Full bleed".
    if cube.data.materials:
        cube.data.materials[0] = mat
    else:
        cube.data.materials.append(mat)

# ============================================================================
# Scene Setup Functions
# ============================================================================

def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

# 6x6 Grid Layout Definition (0-based coordinates)
# x: col (0..5), y: row (0..5) from Top-Left? 
# Blender Y is usually "Depth" or "Up" depending on view.
# Let's map Logical (Col, Row) to Blender (X, Y).
# Logical Row 0 is Top. Logical Col 0 is Left.
# Blender X+ is Right. Blender Z+ is Up.
# So Col -> X. Row -> -Z (or Y if looking down).
# User requested "Front View".
# Standard Blender Front View: X=Right, Z=Up, Y=Depth (Back).
# So Row 0 (Top) is High Z. Row 5 (Bottom) is Low Z.
BENTO_LAYOUT_6x6 = [
    # Row 1 (Top)
    {'id': 'r1c1', 'c': 0, 'r': 0, 'w': 1, 'h': 1, 'type': 'image'},
    {'id': 'r1c2', 'c': 1, 'r': 0, 'w': 1, 'h': 1, 'type': 'yellow'},
    {'id': 'r1c3', 'c': 2, 'r': 0, 'w': 1, 'h': 1, 'type': 'image'},
    {'id': 'r1c4', 'c': 3, 'r': 0, 'w': 1, 'h': 1, 'type': 'yellow'},
    {'id': 'r1c5', 'c': 4, 'r': 0, 'w': 1, 'h': 1, 'type': 'image'},
    {'id': 'r1c6', 'c': 5, 'r': 0, 'w': 1, 'h': 1, 'type': 'yellow'},

    # Row 2
    {'id': 'r2c1', 'c': 0, 'r': 1, 'w': 1, 'h': 1, 'type': 'image'},
    {'id': 'r2c2', 'c': 1, 'r': 1, 'w': 1, 'h': 1, 'type': 'image'},
    {'id': 'r2c3', 'c': 2, 'r': 1, 'w': 1, 'h': 1, 'type': 'image'},
    {'id': 'r2c4', 'c': 3, 'r': 1, 'w': 1, 'h': 1, 'type': 'image'},
    {'id': 'info_afisha', 'c': 4, 'r': 1, 'w': 1, 'h': 1, 'type': 'yellow', 'text': 'АФИША'},
    {'id': 'r2c6', 'c': 5, 'r': 1, 'w': 1, 'h': 1, 'type': 'image'},

    # Row 3
    {'id': 'r3c1', 'c': 0, 'r': 2, 'w': 1, 'h': 1, 'type': 'image'},
    {'id': 'r3c2', 'c': 1, 'r': 2, 'w': 1, 'h': 1, 'type': 'image'},
    
    # Center Big Block (2x2) at col 2, row 2 (spanning c2,c3, r2,r3)
    {'id': 'main_schedule', 'c': 2, 'r': 2, 'w': 2, 'h': 2, 'type': 'yellow', 'text': 'DATE'},

    # Right Title (2x1) at col 4, row 2 (spanning c4,c5)
    {'id': 'info_classics', 'c': 4, 'r': 2, 'w': 2, 'h': 1, 'type': 'yellow', 'text': 'CLASSICS'},

    # Row 4
    {'id': 'r4c1', 'c': 0, 'r': 3, 'w': 1, 'h': 1, 'type': 'image'},
    {'id': 'r4c2', 'c': 1, 'r': 3, 'w': 1, 'h': 1, 'type': 'image'},
    # c2,c3 taken by main
    # c4 taken by image? No, JSON says:
    # "r4c5" (col 5 in base-1 -> col 4 in base-0) is yellow list.
    # Ah, Right Title (2x1) was at Row 3 (base-1) -> Index 2. Height 1. So Row 3 is free at Index 3?
    # Let's re-read JSON "Row 4" (Index 3).
    # "r4c5" -> x=5 (Index 4), y=4 (Index 3).
    
    {'id': 'info_cities', 'c': 4, 'r': 3, 'w': 1, 'h': 1, 'type': 'yellow', 'text': 'CITIES'},
    {'id': 'r4c6', 'c': 5, 'r': 3, 'w': 1, 'h': 1, 'type': 'image'},

    # Row 5
    {'id': 'r5c1', 'c': 0, 'r': 4, 'w': 1, 'h': 1, 'type': 'image'},
    {'id': 'r5c2', 'c': 1, 'r': 4, 'w': 1, 'h': 1, 'type': 'image'},
    {'id': 'r5c3', 'c': 2, 'r': 4, 'w': 1, 'h': 1, 'type': 'image'},
    {'id': 'r5c4', 'c': 3, 'r': 4, 'w': 1, 'h': 1, 'type': 'yellow'},
    {'id': 'r5c5', 'c': 4, 'r': 4, 'w': 1, 'h': 1, 'type': 'image'},
    {'id': 'r5c6', 'c': 5, 'r': 4, 'w': 1, 'h': 1, 'type': 'yellow'},

    # Row 6
    {'id': 'r6c1', 'c': 0, 'r': 5, 'w': 1, 'h': 1, 'type': 'image'},
    {'id': 'r6c2', 'c': 1, 'r': 5, 'w': 1, 'h': 1, 'type': 'image'},
    {'id': 'r6c3', 'c': 2, 'r': 5, 'w': 1, 'h': 1, 'type': 'image'},
    {'id': 'r6c4', 'c': 3, 'r': 5, 'w': 1, 'h': 1, 'type': 'image'},
    {'id': 'r6c5', 'c': 4, 'r': 5, 'w': 1, 'h': 1, 'type': 'image'},
    {'id': 'r6c6', 'c': 5, 'r': 5, 'w': 1, 'h': 1, 'type': 'image'},
]

def generate_bento_scene(poster_textures: Optional[Dict[str, List[str]]] = None):
    clean_scene()
    
    # 1. Dimensions
    S = 1.0          # Base unit size
    G = 0.077 * S    # Gap (~7.7%)
    D = 1.0 * S      # Depth = Width (True Cubes 520x520x520)
    
    # Yellow Color (F1E44B)
    col_yellow = (0.945, 0.894, 0.294, 1.0)
    col_dark = (0.1, 0.1, 0.1, 1.0)
    
    mat_yellow = create_material("Yellow", col_yellow)
    mat_dark = create_material("Dark", col_dark)
    
    cubes = []
    
    # 2. Build Grid
    # Origin: Top-Left or Center? 
    # Let's center the whole grid around (0,0,0).
    # Total Width = 6*S + 5*G
    # Total Height = 6*S + 5*G
    total_w = 6 * S + 5 * G
    total_h = 6 * S + 5 * G
    
    start_x = -total_w / 2
    start_z = total_h / 2 # Row 0 is at top
    
    for block in BENTO_LAYOUT_6x6:
        c = block['c']
        r = block['r']
        w_span = block['w']
        h_span = block['h']
        
        # Calculate Position (Top-Left corner of block)
        # x = start_x + c*(S+G)
        # z = start_z - r*(S+G)
        # But block center is needed for primitive_cube_add.
        
        # Block dimensions
        # Width = w*S + (w-1)*G
        # Height = h*S + (h-1)*G
        block_w = w_span * S + (w_span - 1) * G
        block_h = h_span * S + (h_span - 1) * G
        
        # Center coordinates
        # X: start_x + (c * (S+G)) + block_w / 2
        # Z: start_z - (r * (S+G)) - block_h / 2 (Moving down)
        
        bs_x = start_x + (c * (S+G)) + block_w / 2
        bs_z = start_z - (r * (S+G)) - block_h / 2
        bs_y = 0 # On the "wall"
        
        # Add random depth variation?
        # User said "don't differ by more than cube size".
        # D is shallow (0.2). Let's vary slightly negative Y?
        # User said: "standing on black base".
        # Let's keep fronts aligned for now, maybe slight push back for images?
        # "Z-depth differences should not exceed 520".
        # Z-Depth Logic (Actually Y-axis depth in this XZ-grid setup)
        # Camera is at Negative Y. So "Closer" = More Negative Y.
        # User wants Main Cube to stick out "at least 50%".
        # S = 1.0. D = 1.0.
        # Let's define offsets relative to Wall at Y=0.
        
        depth_offset = 0.0
        if block['id'] == 'main_schedule':
            # Closest. Stick out by ~70-80% relative to back?
            # Let's put Center at Y = -0.8 * S
            # Front face at -1.3. Back at -0.3.
            depth_offset = -0.8 * S 
        elif block['id'] == 'info_classics':
            # Intermediate.
            # Center at Y = -0.4 * S
            # Front at -0.9. Back at 0.1.
            depth_offset = -0.4 * S
        else:
            # Background tiles.
            # Center at Y = 0.
            # Randomize slightly "behind" (positive Y) or just around 0.
            # Range [-0.1, 0.1] * S
            depth_offset = random.uniform(-0.1 * S, 0.1 * S)

        bpy.ops.mesh.primitive_cube_add(location=(bs_x, depth_offset, bs_z), size=1.0)
        cube = bpy.context.active_object
        cube.name = block['id']
        
        # Scale
        # Default Depth is D (1.0).
        # For Main Schedule (2x2), user implies it's a "Large Cube" (1080x1080).
        # To avoid stretching the side texture (which is 1080x1080), the side face must be square.
        # So Depth must equal Width/Height.
        current_d = D
        if block['id'] == 'main_schedule':
             current_d = block_w # roughly 2.077
             
        cube.dimensions = (block_w, current_d, block_h)
        
        # Assign Material
        if block['type'] == 'yellow':
            cube.data.materials.append(mat_yellow)
            if 'text' in block:
                cube['text_content'] = block['text']
        else:
            cube.data.materials.append(mat_dark)
            if poster_textures:
                 pass
        
        if block['id'] == 'main_schedule':
            cube['type'] = 'main'
            apply_special_textures(cube) # Apply Front/Right textures
        elif block['type'] == 'image':
            cube['type'] = 'poster'
        elif block['type'] == 'yellow' and 'text' in block:
            cube['type'] = 'info'
        else:
            cube['type'] = 'filler'
            
        cubes.append(cube)


    
    # 3. Camera (Angled View for Volume)
    # User said: "Turn 1 hour left" (from previous 2 hours right / 60 deg).
    # So Target = 1 hour right = 30 degrees.
    
    # User request (v13):
    # "Move tripod to 25 degrees", keep Yaw (Target X=3.5).
    # Angle: 25 degrees.
    
    angle_deg = 25
    angle_rad = math.radians(25)
    radius = 17.5
    
    # Position:
    # X = R * sin(angle)
    # Y = -R * cos(angle) (Front is -Y)
    cam_x = radius * math.sin(angle_rad)
    cam_y = -radius * math.cos(angle_rad)
    cam_z = 0 # Vertical center
    
    bpy.ops.object.camera_add(location=(cam_x, cam_y, cam_z))
    cam = bpy.context.active_object
    cam.data.type = 'PERSP'
    cam.data.lens = 120 
    
    # Track To Constraint
    # Target X:
    # v15 -> Shift Left to X=1.0 per user request.
    target_loc = (1.0, 0, 0)
    
    bpy.ops.object.empty_add(location=target_loc)
    target_empty = bpy.context.active_object
    target_empty.name = "CameraTarget"
    
    track = cam.constraints.new(type='TRACK_TO')
    track.target = target_empty
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'
    
    bpy.context.scene.camera = cam
    
    bpy.context.scene.camera = cam
    
    # 4. Lighting
    # Main Key Light (Sun) - Coming from Front-Right-Top
    # Grid is XZ plane. Camera looks +Y. Light should point +Y.
    # Default Sun points -Z.
    # Rot X +90 -> +Y.
    # We want slightly from top-right.
    bpy.ops.object.light_add(type='SUN', location=(10, -20, 20))
    sun = bpy.context.active_object
    sun.data.energy = 4.0
    sun.data.angle = 0.5 # Soft shadows (approx 30 degrees equivalent in size?) No, radians. 0.5 rad ~ 28 deg.
    # Rotation:
    # X=90 makes it horizontal (+Y). We want slightly down: X=70.
    # Z=30 makes it from right.
    sun.rotation_euler = (math.radians(60), 0, math.radians(30))
    
    # Fill Light (Area) from opposite side
    bpy.ops.object.light_add(type='AREA', location=(-15, -15, 0))
    fill = bpy.context.active_object
    fill.data.energy = 500
    fill.data.size = 20
    fill.rotation_euler = (math.radians(90), 0, math.radians(-45))
    
    # Background
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs['Color'].default_value = (0.05, 0.05, 0.05, 1) # Black/Dark Grey Gaps
        
    return cubes, cam

def apply_special_textures(cube):
    """Apply generated CSS textures to Main Cube faces"""
    # Paths (generated by generate_css_textures.py)
    path_front = "/tmp/scene_textures/main_front.png"
    path_right = "/tmp/scene_textures/main_right.png"
    
    if not os.path.exists(path_front) or not os.path.exists(path_right):
        print("Warning: Generated textures not found")
        return

    # Create Materials with UV scale = 1.0 (textures now have correct proportions)
    # No offset needed since textures are generated with proper margins
    img_f = load_image_texture(path_front)
    mat_f = create_material("Main_Front", (1,1,1,1), img_f, uv_scale=1.0, uv_offset=(0.0, 0.0))
    
    img_r = load_image_texture(path_right)
    mat_r = create_material("Main_Right", (1,1,1,1), img_r, uv_scale=1.0, uv_offset=(0.0, 0.0))
    
    # Append materials to cube
    # Note: Cube might already have mat_yellow at index 0.
    # We want Front -> mat_f, Right -> mat_r, Others -> Keep Yellow?
    # Let's ensure slots exist.
    
    # Check existing slots
    if not cube.data.materials:
        cube.data.materials.append(create_material("Default", (1,1,0,1))) # Fallback
        
    # Add new materials
    cube.data.materials.append(mat_f) # Index 1 (if 0 existed)
    cube.data.materials.append(mat_r) # Index 2
    
    idx_f = len(cube.data.materials) - 2
    idx_r = len(cube.data.materials) - 1
    
    # Map Faces
    # Normals in Local Space (if object not rotated, matches Global).
    # Cube created aligned.
    
    for poly in cube.data.polygons:
        n = poly.normal
        # Front: -Y -> (0, -1, 0)
        if n[1] < -0.9:
            poly.material_index = idx_f
        # Right: +X -> (1, 0, 0)
        elif n[0] > 0.9:
            poly.material_index = idx_r
        # Others keep default (0)

if __name__ == "__main__":
    generate_bento_scene()
    bpy.ops.wm.save_as_mainfile(filepath="/tmp/bento_6x6.blend")

# Alias for backward compatibility
BENTO_LAYOUT = {i: item for i, item in enumerate(BENTO_LAYOUT_6x6)}  # Mock dict or just list?
# Actually downstream expects a Dict with keys?
# Old layout: 'main_schedule': (row, col...).
# New layout is List of Dicts.
# Wait, generate_bento_scene (OLD) iterated .items().
# My NEW generate_bento_scene iterates struct.
# But other files might import BENTO_LAYOUT to know keys?
# generate_intro.py imports BENTO_LAYOUT?
# Let's check generate_intro.py usage.

