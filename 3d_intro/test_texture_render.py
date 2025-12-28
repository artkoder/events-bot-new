#!/usr/bin/env python3
"""
Test Texture Render Script
Creates a simple scene to verify texture placement on cube faces
"""

import bpy
import os
import sys

# Add paths for imports
sys.path.insert(0, "/workspaces/events-bot-new/3d_intro")

OUTPUT_DIR = "/workspaces/events-bot-new/3d_intro/assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_scene():
    """Clear all objects"""
    bpy.ops.wm.read_factory_settings(use_empty=True)

def load_image_texture(image_path: str):
    """Load an image texture from file"""
    if os.path.exists(image_path):
        return bpy.data.images.load(image_path)
    return None

def create_material_with_texture(name: str, texture, rotation_z: float = 0.0):
    """Create a material with texture
    
    Args:
        rotation_z: Rotation in radians around Z axis for UV mapping
    """
    import math
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
        tex_coord = nodes.new(type='ShaderNodeTexCoord')
        mapping = nodes.new(type='ShaderNodeMapping')
        # Scale 1.0 = texture fits exactly to UV
        mapping.inputs['Scale'].default_value = (1.0, 1.0, 1.0)
        mapping.inputs['Location'].default_value = (0.5, 0.5, 0.0)  # Pivot at center
        mapping.inputs['Rotation'].default_value = (0.0, 0.0, rotation_z)
        
        tex_image = nodes.new(type='ShaderNodeTexImage')
        tex_image.image = texture
        
        links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
        links.new(mapping.outputs['Vector'], tex_image.inputs['Vector'])
        links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])
    
    output = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    return mat

def create_test_scene():
    """Create a test scene with textured cube"""
    clean_scene()
    
    # Create cube
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0), size=2)
    cube = bpy.context.object
    cube.name = "TestCube"
    
    # Load textures
    path_front = "/tmp/scene_textures/main_front.png"
    path_right = "/tmp/scene_textures/main_right.png"
    
    if not os.path.exists(path_front):
        print(f"ERROR: {path_front} not found!")
        return None
        
    if not os.path.exists(path_right):
        print(f"ERROR: {path_right} not found!")
        return None
    
    img_front = load_image_texture(path_front)
    img_right = load_image_texture(path_right)
    
    # Create materials
    mat_yellow = bpy.data.materials.new(name="Yellow")
    mat_yellow.use_nodes = True
    bsdf = mat_yellow.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.945, 0.894, 0.294, 1.0)  # #F1E44B
    
    import math
    mat_front = create_material_with_texture("Front", img_front, rotation_z=math.pi/2)  # 90° rotation
    mat_right = create_material_with_texture("Right", img_right, rotation_z=math.pi/2)  # 90° rotation
    
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
    # Orthographic for cleaner comparison
    bpy.ops.object.camera_add(location=(4, -4, 0))
    camera = bpy.context.object
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 4.5
    
    # Point camera at cube center
    constraint = camera.constraints.new(type='TRACK_TO')
    constraint.target = cube
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
    
    bpy.context.scene.camera = camera
    
    # Direct lighting for flat look
    bpy.ops.object.light_add(type='SUN', location=(5, -3, 8))
    sun = bpy.context.object
    sun.data.energy = 5.0
    
    # World background (neutral gray)
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
        bg.inputs["Strength"].default_value = 1.0
    
    return cube

def render_scene(output_path: str, samples: int = 64):
    """Render the current scene"""
    scene = bpy.context.scene
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 512
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = samples
    scene.cycles.device = 'CPU'
    scene.cycles.use_denoising = False
    scene.render.filepath = output_path
    
    bpy.ops.render.render(write_still=True)
    print(f"Rendered: {output_path}")
    return output_path

if __name__ == "__main__":
    print("=== Texture Test Render ===")
    
    cube = create_test_scene()
    if cube:
        output = f"{OUTPUT_DIR}/texture_test_v2.png"
        render_scene(output)
        print(f"\nResult saved to: {output}")
    else:
        print("Failed to create scene")
