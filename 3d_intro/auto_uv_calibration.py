#!/usr/bin/env python3
"""
Auto UV Calibration Script
Iteratively adjusts UV scale and offset to match reference texture appearance.
Uses visual comparison to find optimal parameters.
"""

import bpy
import math
import os
from pathlib import Path

# Configuration
OUTPUT_DIR = "/tmp/uv_calibration"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Reference image for comparison
REFERENCE_IMAGE = "/home/codespace/.gemini/antigravity/brain/3cb4bd81-7241-406d-82e0-91eb1406815d/uploaded_image_1766908838969.png"

def clean_scene():
    """Clear all objects"""
    bpy.ops.wm.read_factory_settings(use_empty=True)

def load_image_texture(image_path: str) -> bpy.types.Image:
    """Load an image texture from file"""
    if os.path.exists(image_path):
        return bpy.data.images.load(image_path)
    return None

def create_material_with_uv(name: str, texture: bpy.types.Image, 
                            uv_scale: float = 1.0, 
                            uv_offset: tuple = (0.0, 0.0)) -> bpy.types.Material:
    """Create a material with configurable UV mapping"""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (1, 1, 1, 1)
    bsdf.inputs['Roughness'].default_value = 0.4
    
    if texture:
        tex_coord = nodes.new(type='ShaderNodeTexCoord')
        mapping = nodes.new(type='ShaderNodeMapping')
        mapping.inputs['Scale'].default_value = (uv_scale, uv_scale, uv_scale)
        mapping.inputs['Location'].default_value = (uv_offset[0], uv_offset[1], 0.0)
        
        tex_image = nodes.new(type='ShaderNodeTexImage')
        tex_image.image = texture
        
        links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
        links.new(mapping.outputs['Vector'], tex_image.inputs['Vector'])
        links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])
    
    output = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    return mat

def create_test_scene(uv_scale: float, uv_offset: tuple):
    """Create a test scene with two cube faces showing textures"""
    clean_scene()
    
    # Create a cube with exact proportions
    # The cube should show front and right face similar to reference
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0), size=2)
    cube = bpy.context.object
    cube.name = "TestCube"
    
    # Load textures
    path_front = "/tmp/scene_textures/main_front.png"
    path_right = "/tmp/scene_textures/main_right.png"
    
    if not os.path.exists(path_front):
        print(f"ERROR: {path_front} not found. Run generate_css_textures.py first!")
        return None
        
    if not os.path.exists(path_right):
        print(f"ERROR: {path_right} not found. Run generate_css_textures.py first!")
        return None
    
    img_front = load_image_texture(path_front)
    img_right = load_image_texture(path_right)
    
    # Create materials
    mat_yellow = bpy.data.materials.new(name="Yellow")
    mat_yellow.use_nodes = True
    bsdf = mat_yellow.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.945, 0.894, 0.294, 1.0)  # #F1E44B
    
    mat_front = create_material_with_uv("Front", img_front, uv_scale, uv_offset)
    mat_right = create_material_with_uv("Right", img_right, uv_scale, uv_offset)
    
    # Apply materials to cube
    cube.data.materials.append(mat_yellow)  # index 0
    cube.data.materials.append(mat_front)   # index 1
    cube.data.materials.append(mat_right)   # index 2
    
    # Assign materials to faces based on normal direction
    for poly in cube.data.polygons:
        n = poly.normal
        # Front face: -Y normal
        if n[1] < -0.9:
            poly.material_index = 1
        # Right face: +X normal
        elif n[0] > 0.9:
            poly.material_index = 2
        else:
            poly.material_index = 0
    
    # Camera - positioned to see front and right faces (like reference)
    bpy.ops.object.camera_add(location=(3.5, -3.5, 0))
    camera = bpy.context.object
    
    # Point camera at cube center
    constraint = camera.constraints.new(type='TRACK_TO')
    constraint.target = cube
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
    
    bpy.context.scene.camera = camera
    
    # Lighting
    bpy.ops.object.light_add(type='SUN', location=(5, -3, 8))
    sun = bpy.context.object
    sun.data.energy = 3.0
    
    bpy.ops.object.light_add(type='AREA', location=(-3, 3, 4))
    area = bpy.context.object
    area.data.energy = 150
    area.data.size = 5
    
    # World background
    scene = bpy.context.scene
    if not scene.world:
        world = bpy.data.worlds.new("World")
        scene.world = world
    else:
        world = scene.world
    
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.3, 0.3, 0.3, 1.0)
        bg.inputs["Strength"].default_value = 0.5
    
    return cube

def render_test(output_path: str, samples: int = 32):
    """Render the current scene"""
    scene = bpy.context.scene
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = samples
    scene.cycles.device = 'CPU'
    scene.cycles.use_denoising = False
    scene.render.filepath = output_path
    
    bpy.ops.render.render(write_still=True)
    print(f"Rendered: {output_path}")

def run_calibration_grid():
    """
    Run a grid search to find optimal UV scale and offset.
    Based on reference image analysis:
    - Text should have visible margins from cube edges
    - "27-28" should be prominent on left
    - Cities list should be readable on right
    """
    
    # Grid search parameters based on previous work
    # Current: scale=3.5, offset=(0.2, -0.2)
    # Need to find optimal values
    
    scales = [1.0, 1.5, 2.0, 2.5, 3.0]  # Different UV scales
    offsets = [
        (0.0, 0.0),
        (0.1, -0.1),
        (0.2, -0.2),
        (0.3, -0.3),
    ]
    
    results = []
    
    for scale in scales:
        for offset in offsets:
            print(f"\n=== Testing: scale={scale}, offset={offset} ===")
            
            cube = create_test_scene(scale, offset)
            if cube is None:
                continue
                
            output_file = f"{OUTPUT_DIR}/test_s{scale}_ox{offset[0]}_oy{offset[1]}.png"
            render_test(output_file, samples=16)
            
            results.append({
                'scale': scale,
                'offset': offset,
                'file': output_file
            })
    
    print("\n=== Calibration Complete ===")
    print(f"Results saved to: {OUTPUT_DIR}")
    for r in results:
        print(f"  scale={r['scale']}, offset={r['offset']}: {r['file']}")
    
    return results

def run_single_test(uv_scale: float, uv_offset: tuple, output_name: str = "calibration_test.png"):
    """Run a single test with specific parameters"""
    print(f"\n=== Single Test: scale={uv_scale}, offset={uv_offset} ===")
    
    cube = create_test_scene(uv_scale, uv_offset)
    if cube is None:
        return None
        
    output_file = f"{OUTPUT_DIR}/{output_name}"
    render_test(output_file, samples=32)
    
    return output_file

if __name__ == "__main__":
    # First, generate fresh textures
    print("Generating CSS textures...")
    exec(open("/workspaces/events-bot-new/3d_intro/generate_css_textures.py").read())
    
    # Run single test with current parameters
    result = run_single_test(3.5, (0.2, -0.2), "current_settings.png")
    
    # Run grid search for comparison
    # run_calibration_grid()
    
    print(f"\nTest complete. Check: {result}")
