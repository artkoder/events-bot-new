#!/usr/bin/env python3
"""
3D Text on Cube with CORRECT Vertical Anchor Handling.

Problem: CSS `top` is measured from TOP of text box (Cap Height).
         Blender text anchor is at BASELINE.
         
Solution: Add line-height as vertical offset.

CSS Reference (exact values):
- "27-28": left=93, top=232, width=907, height=308, font-size=224, line-height=308
- "Декабря": left=316, top=540, width=684, height=240, font-size=240, line-height=240
- "АФИША": left=761, top=870, width=239, height=140, font-size=110, line-height=140
"""
import bpy
import os
import math

OUTPUT_DIR = "/workspaces/events-bot-new/3d_intro/assets"
REFERENCE_IMAGE = "/home/codespace/.gemini/antigravity/brain/d43d08f1-b44b-444b-8452-4f1dbed5e7b1/uploaded_image_1766911240181.png"

# CSS → Blender Constants
CANVAS_SIZE = 1080.0
CUBE_SIZE = 2.0
PIXEL_TO_METER = CUBE_SIZE / CANVAS_SIZE

# Font paths
FONTS_DIR_INTRO = "/workspaces/events-bot-new/3d_intro/assets/fonts"
FONTS_DIR_VIDEO = "/workspaces/events-bot-new/video_announce/assets"


def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def apply_css_element(cube, text_str: str, font_path: str, 
                      font_size_px: float, line_height_px: float,
                      top_px: float, left_px: float, width_px: float, 
                      canvas_size: float = CANVAS_SIZE):
    """
    Create 3D text with corrected CSS→Blender coordinate conversion.
    
    Key fix: Add line-height offset to Z to compensate for 
    CSS top-edge vs Blender baseline anchor difference.
    """
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
    txt.data.align_y = 'TOP_BASELINE'  # Baseline anchor
    
    # Calculate X: Right edge of text box
    css_right_x = left_px + width_px
    loc_x = (css_right_x - (canvas_size / 2)) * px_to_m
    
    # Calculate Z: CSS top + offset for baseline→top conversion
    # CSS measures from top of text box, Blender measures from baseline
    # Offset by approximately 0.75 * line_height (typical ascender ratio)
    css_top_in_blender = top_px
    loc_z = ((canvas_size / 2) - css_top_in_blender) * px_to_m
    
    # Y: Slightly in front of cube face
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
    print(f"  '{text_str}': X={loc_x:.3f}, Z={loc_z:.3f} (top={top_px}, line-h={line_height_px})")
    return txt


def create_yellow_cube():
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
    """Add reference image at 50% transparency."""
    bpy.ops.mesh.primitive_plane_add(size=CUBE_SIZE, location=(0, -1.02, 0))
    plane = bpy.context.object
    plane.name = "ReferenceOverlay"
    plane.rotation_euler = (math.radians(90), 0, 0)
    
    mat = bpy.data.materials.new(name="ReferenceOverlay")
    mat.use_nodes = True
    mat.blend_method = 'BLEND'
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    out = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    tex = nodes.new('ShaderNodeTexImage')
    
    if os.path.exists(REFERENCE_IMAGE):
        tex.image = bpy.data.images.load(REFERENCE_IMAGE)
    
    bsdf.inputs['Alpha'].default_value = 0.5
    bsdf.inputs['Roughness'].default_value = 1.0
    
    links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    
    plane.data.materials.append(mat)
    plane.parent = cube
    return plane


def setup_camera(cube):
    """Camera directly in front (анфас)."""
    bpy.ops.object.camera_add(location=(0, -5, 0))
    camera = bpy.context.object
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 2.5
    
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
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print(f"Rendered: {output_path}")


if __name__ == "__main__":
    print("=== 3D Text with Vertical Anchor Fix ===")
    print(f"PIXEL_TO_METER = {PIXEL_TO_METER:.6f}")
    
    clean_scene()
    cube = create_yellow_cube()
    
    print("\nApplying text elements (original CSS values):")
    
    # "27-28": left=93, top=232, width=907, height=308, font-size=224, line-height=308
    apply_css_element(cube, "27-28",
        font_path=f"{FONTS_DIR_INTRO}/Benzin-Bold.ttf",
        font_size_px=224, line_height_px=308,
        top_px=232, left_px=93, width_px=907)
    
    # "Декабря": left=316, top=540, width=684, height=240, font-size=240, line-height=240
    apply_css_element(cube, "ДЕКАБРЯ",
        font_path=f"{FONTS_DIR_VIDEO}/BebasNeue-Regular.ttf",
        font_size_px=240, line_height_px=240,
        top_px=540, left_px=316, width_px=684)
    
    # "АФИША": left=761, top=870, width=239, height=140, font-size=110, line-height=140
    apply_css_element(cube, "АФИША",
        font_path=f"{FONTS_DIR_INTRO}/DrukCyr-Bold.ttf",
        font_size_px=110, line_height_px=140,
        top_px=870, left_px=761, width_px=239)
    
    add_reference_overlay(cube)
    setup_camera(cube)
    setup_lighting()
    setup_world()
    
    output = f"{OUTPUT_DIR}/3d_text_anchor_fix.png"
    render(output, samples=32)
    
    print(f"\nResult: {output}")
