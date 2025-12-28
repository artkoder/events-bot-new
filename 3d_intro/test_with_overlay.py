#!/usr/bin/env python3
"""
3D Text on Cube with Reference Overlay.
Adds the reference image at 50% transparency on the front face
to visualize alignment differences.
"""
import bpy
import os
import math

OUTPUT_DIR = "/workspaces/events-bot-new/3d_intro/assets"
REFERENCE_IMAGE = "/home/codespace/.gemini/antigravity/brain/990c274c-40a7-4cc4-a521-59a07d231185/uploaded_image_1766957832438.png"

# CSS → Blender Constants
CANVAS_SIZE = 1080.0
CUBE_SIZE = 2.0
PIXEL_TO_METER = CUBE_SIZE / CANVAS_SIZE

# Font paths
FONTS_DIR_INTRO = "/workspaces/events-bot-new/3d_intro/assets/fonts"
FONTS_DIR_VIDEO = "/workspaces/events-bot-new/video_announce/assets"


def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def apply_css_element(cube, text_str: str, font_path: str, font_size_px: float, 
                      top_px: float, left_px: float, width_px: float, 
                      canvas_size: float = CANVAS_SIZE):
    """Create 3D text with CSS→Blender coordinate conversion."""
    cube_size = cube.dimensions.x
    px_to_m = cube_size / canvas_size
    
    bpy.ops.object.text_add()
    txt = bpy.context.object
    txt.data.body = text_str
    txt.name = f"Text_{text_str[:6]}"
    
    if font_path and os.path.exists(font_path):
        try:
            font = bpy.data.fonts.load(font_path)
            txt.data.font = font
        except Exception as e:
            print(f"Warning: Could not load font {font_path}: {e}")
    
    txt.data.size = font_size_px * px_to_m
    txt.data.align_x = 'RIGHT'
    txt.data.align_y = 'TOP'
    
    css_right_x = left_px + width_px
    loc_x = (css_right_x - (canvas_size / 2)) * px_to_m
    loc_z = ((canvas_size / 2) - top_px) * px_to_m
    loc_y = -(cube_size / 2) - 0.01
    
    txt.location = (loc_x, loc_y, loc_z)
    txt.rotation_euler = (math.radians(90), 0, 0)
    txt.data.extrude = 0.001
    
    mat = bpy.data.materials.new(name=f"TextMat_{text_str[:6]}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.063, 0.055, 0.055, 1.0)
    txt.data.materials.append(mat)
    
    txt.parent = cube
    print(f"  Added: '{text_str}' at X={loc_x:.3f}, Z={loc_z:.3f}")
    return txt


def create_yellow_cube():
    """Create yellow cube."""
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0), size=CUBE_SIZE)
    cube = bpy.context.object
    cube.name = "YellowCube"
    
    mat = bpy.data.materials.new(name="Yellow")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.945, 0.894, 0.294, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.4
    cube.data.materials.append(mat)
    
    return cube


def add_reference_overlay(cube):
    """
    Add a plane with the reference image at 50% transparency
    positioned slightly in front of the cube's front face.
    """
    # Create a plane the same size as the cube face
    bpy.ops.mesh.primitive_plane_add(size=CUBE_SIZE, location=(0, -1.02, 0))
    plane = bpy.context.object
    plane.name = "ReferenceOverlay"
    
    # Rotate to face the camera (plane is created on XY, needs to face -Y)
    plane.rotation_euler = (math.radians(90), 0, 0)
    
    # Create material with reference image and 50% alpha
    mat = bpy.data.materials.new(name="ReferenceOverlay")
    mat.use_nodes = True
    mat.blend_method = 'BLEND'  # Enable transparency
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # Create nodes
    out = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    tex = nodes.new('ShaderNodeTexImage')
    
    # Load reference image
    if os.path.exists(REFERENCE_IMAGE):
        tex.image = bpy.data.images.load(REFERENCE_IMAGE)
    else:
        print(f"Warning: Reference image not found: {REFERENCE_IMAGE}")
    
    # Set 50% transparency
    bsdf.inputs['Alpha'].default_value = 0.5
    bsdf.inputs['Roughness'].default_value = 1.0
    
    # Link nodes
    links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    
    plane.data.materials.append(mat)
    plane.parent = cube
    
    print(f"Added reference overlay at 50% transparency")
    return plane


def setup_camera(cube):
    """Camera position directly in front (анфас)."""
    bpy.ops.object.camera_add(location=(0, -5, 0))  # Directly in front
    camera = bpy.context.object
    camera.data.type = 'ORTHO'  # Orthographic for flat view
    camera.data.ortho_scale = 2.5  # Fit cube in frame
    
    constraint = camera.constraints.new(type='TRACK_TO')
    constraint.target = cube
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
    
    bpy.context.scene.camera = camera
    return camera


def setup_lighting():
    bpy.ops.object.light_add(type='SUN', location=(5, -5, 10))
    bpy.context.object.data.energy = 3.0


def setup_world():
    scene = bpy.context.scene
    if not scene.world:
        scene.world = bpy.data.worlds.new("World")
    world = scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.3, 0.3, 0.3, 1.0)


def render(output_path: str, samples: int = 32):
    scene = bpy.context.scene
    scene.render.resolution_x = 1080
    scene.render.resolution_y = 1080
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = samples
    scene.cycles.device = 'CPU'
    scene.cycles.use_denoising = False
    scene.render.film_transparent = False
    scene.render.filepath = output_path
    
    bpy.ops.render.render(write_still=True)
    print(f"Rendered: {output_path}")


if __name__ == "__main__":
    print("=== 3D Text with Reference Overlay ===")
    
    clean_scene()
    cube = create_yellow_cube()
    
    # Add 3D text elements
    print("\nAdding text elements:")
    apply_css_element(cube, "27-28",
        font_path=f"{FONTS_DIR_INTRO}/Benzin-Bold.ttf",
        font_size_px=280, top_px=200, left_px=93, width_px=907)
    
    apply_css_element(cube, "ДЕКАБРЯ",
        font_path=f"{FONTS_DIR_VIDEO}/BebasNeue-Regular.ttf",
        font_size_px=240, top_px=540, left_px=316, width_px=684)
    
    apply_css_element(cube, "АФИША",
        font_path=f"{FONTS_DIR_INTRO}/DrukCyr-Bold.ttf",
        font_size_px=110, top_px=870, left_px=761, width_px=239)
    
    # Add reference overlay with 50% transparency
    add_reference_overlay(cube)
    
    setup_camera(cube)
    setup_lighting()
    setup_world()
    
    output = f"{OUTPUT_DIR}/3d_text_with_overlay.png"
    render(output, samples=32)
    
    print(f"\nResult: {output}")
