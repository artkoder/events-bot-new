#!/usr/bin/env python3
"""
Test: Apply texture to SINGLE FACE of a proper 3D CUBE.
Camera is positioned to clearly show it's a cube (not a flat rectangle).
"""
import bpy
import os
import math

OUTPUT_DIR = "/workspaces/events-bot-new/3d_intro/assets"
TEXTURE_PATH = "/tmp/scene_textures/front_face_exact.png"


def clean_scene():
    """Remove all objects"""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def create_image_material(mat_name: str, image_path: str):
    """Create material with proper UV mapping (from Gemini's recommendation)"""
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # Create Nodes
    out = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    tex = nodes.new('ShaderNodeTexImage')
    mapping = nodes.new('ShaderNodeMapping')
    coord = nodes.new('ShaderNodeTexCoord')
    
    # Set BSDF properties
    bsdf.inputs['Roughness'].default_value = 0.4
    bsdf.inputs['Specular IOR Level'].default_value = 0.0
    
    # Load Image
    if os.path.exists(image_path):
        tex.image = bpy.data.images.load(image_path)
        tex.extension = 'CLIP'  # Prevents tiling
    else:
        print(f"Error: Image not found: {image_path}")
        return mat
    
    # Link: UV -> Mapping -> Texture -> BSDF -> Output
    links.new(coord.outputs['UV'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], tex.inputs['Vector'])
    links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    
    return mat


def create_yellow_material():
    """Create plain yellow material for other faces"""
    mat = bpy.data.materials.new(name="Yellow")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.945, 0.894, 0.294, 1.0)  # #F1E44B
        bsdf.inputs['Roughness'].default_value = 0.4
    return mat


def create_cube_with_single_face_texture():
    """
    Create a proper 3D cube and apply texture to FRONT face only.
    Other faces are plain yellow.
    """
    clean_scene()
    
    # Create CUBE (not rectangle!) - size=2 means 2x2x2 units
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0), size=2)
    cube = bpy.context.object
    cube.name = "TestCube"
    
    # Verify it's actually a cube
    print(f"Cube dimensions: {cube.dimensions}")  # Should be (2, 2, 2)
    
    # Create materials
    mat_yellow = create_yellow_material()
    mat_front = create_image_material("FrontFace", TEXTURE_PATH)
    
    # Assign materials to cube
    cube.data.materials.append(mat_yellow)  # Index 0 - default for all faces
    cube.data.materials.append(mat_front)   # Index 1 - for front face
    
    # Assign front face (normal -Y) to use textured material
    for poly in cube.data.polygons:
        n = poly.normal
        if n[1] < -0.9:  # Front face: normal pointing toward -Y
            poly.material_index = 1
        else:
            poly.material_index = 0
    
    return cube


def setup_camera(cube):
    """
    Setup camera to clearly show the cube is 3D (isometric-like view).
    """
    # Camera position: front-right-above for 3D perspective
    # This angle shows front face + right face + top face = clearly a cube
    bpy.ops.object.camera_add(location=(4, -4, 3))
    camera = bpy.context.object
    camera.name = "Camera"
    
    # Use perspective to enhance 3D feel
    camera.data.type = 'PERSP'
    camera.data.lens = 50
    
    # Point camera at cube center
    constraint = camera.constraints.new(type='TRACK_TO')
    constraint.target = cube
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
    
    bpy.context.scene.camera = camera
    return camera


def setup_lighting():
    """Add sun light for clean rendering"""
    bpy.ops.object.light_add(type='SUN', location=(5, -5, 10))
    sun = bpy.context.object
    sun.data.energy = 3.0
    sun.data.angle = 0.3  # Soft shadows


def setup_world():
    """Set up world background (neutral gray)"""
    scene = bpy.context.scene
    if not scene.world:
        scene.world = bpy.data.worlds.new("World")
    world = scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.3, 0.3, 0.3, 1.0)  # Dark gray background


def render(output_path: str, samples: int = 64):
    """Render the scene"""
    scene = bpy.context.scene
    scene.render.resolution_x = 800
    scene.render.resolution_y = 800
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = samples
    scene.cycles.device = 'CPU'
    scene.cycles.use_denoising = False
    scene.render.filepath = output_path
    
    bpy.ops.render.render(write_still=True)
    print(f"Rendered: {output_path}")


if __name__ == "__main__":
    print("=== Single Cube Face Texture Test ===")
    
    cube = create_cube_with_single_face_texture()
    setup_camera(cube)
    setup_lighting()
    setup_world()
    
    output = f"{OUTPUT_DIR}/cube_single_face_test.png"
    render(output, samples=32)
    
    print(f"\nResult: {output}")
    print(f"Cube dimensions: {cube.dimensions}")
