#!/usr/bin/env python3
"""
3D Text on Cube Face with EXACT CSS-to-Blender Coordinate Transformation.

Problem: CSS uses Top-Left origin (Y down), Blender uses Center origin (Y up).
Solution: Mathematical conversion as per the System Instruction.

CSS Data:
- Canvas: 1080x1080px
- Cube: 2x2x2 meters
- "27-28": Benzin-Bold, 224px, top=232, left=93, width=907, align=RIGHT
- "ДЕКАБРЯ": Bebas Neue, 240px, top=540, left=316, width=684, align=RIGHT  
- "АФИША": Druk Cyr, 110px, top=870, left=761, width=239, align=RIGHT
"""
import bpy
import os
import math

OUTPUT_DIR = "/workspaces/events-bot-new/3d_intro/assets"

# CSS → Blender Constants
CANVAS_SIZE = 1080.0  # pixels
CUBE_SIZE = 2.0       # meters
PIXEL_TO_METER = CUBE_SIZE / CANVAS_SIZE  # ≈ 0.00185

# Font paths
FONTS_DIR_INTRO = "/workspaces/events-bot-new/3d_intro/assets/fonts"
FONTS_DIR_VIDEO = "/workspaces/events-bot-new/video_announce/assets"


def clean_scene():
    """Remove all objects"""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def apply_css_element(cube, text_str: str, font_path: str, font_size_px: float, 
                      top_px: float, left_px: float, width_px: float, 
                      canvas_size: float = CANVAS_SIZE):
    """
    Translates CSS absolute positioning to Blender Local Space on a Cube Face.
    
    CSS Origin: (0, 0) is Top-Left, Y increases downwards
    Blender Origin: (0, 0, 0) is Center, Y is depth, Z is up
    
    For right-aligned text in CSS:
    - Text is anchored to the RIGHT edge of its containing box
    - CSS Right edge X = Left + Width
    """
    cube_size = cube.dimensions.x  # Should be 2.0
    px_to_m = cube_size / canvas_size
    
    # 1. Create Text Object
    bpy.ops.object.text_add()
    txt = bpy.context.object
    txt.data.body = text_str
    txt.name = f"Text_{text_str[:6]}"
    
    # 2. Load Font (with fallback)
    if font_path and os.path.exists(font_path):
        try:
            font = bpy.data.fonts.load(font_path)
            txt.data.font = font
        except Exception as e:
            print(f"Warning: Could not load font {font_path}: {e}")
    
    # 3. Font Size (scaled from pixels to meters)
    txt.data.size = font_size_px * px_to_m
    
    # 4. Alignment (CSS text-align: right)
    txt.data.align_x = 'RIGHT'
    txt.data.align_y = 'TOP'  # Anchor at top for CSS-like behavior
    
    # 5. Calculate Blender Coordinates
    # For right-aligned text: anchor point is at CSS Right edge = Left + Width
    css_right_x = left_px + width_px
    
    # Blender X: (CSS_RightEdge - HalfCanvas) * Scale
    # This puts the right edge of text at the correct position
    loc_x = (css_right_x - (canvas_size / 2)) * px_to_m
    
    # Blender Z (vertical): (HalfCanvas - CSS_Top) * Scale
    # Inverted because CSS Y goes down, Blender Z goes up
    loc_z = ((canvas_size / 2) - top_px) * px_to_m
    
    # Blender Y: Slightly in front of cube face to avoid z-fighting
    # Front face is at Y = -1.0 (half of cube size), so place text at Y = -1.01
    loc_y = -(cube_size / 2) - 0.01
    
    # 6. Position the text
    txt.location = (loc_x, loc_y, loc_z)
    
    # 7. Rotate to face the camera (text is created on XY plane, needs to face -Y)
    # Rotate 90° around X axis to stand upright
    txt.rotation_euler = (math.radians(90), 0, 0)
    
    # 8. Set text thickness (minimal extrude - like paint on surface)
    txt.data.extrude = 0.001  # Very thin, like painted text
    
    # 9. Create dark material for text
    mat = bpy.data.materials.new(name=f"TextMat_{text_str[:6]}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.063, 0.055, 0.055, 1.0)  # #100E0E
    txt.data.materials.append(mat)
    
    # 10. Parent to cube
    txt.parent = cube
    
    print(f"  Added: '{text_str}' at X={loc_x:.3f}, Y={loc_y:.3f}, Z={loc_z:.3f}")
    
    return txt


def create_yellow_cube():
    """Create a 2x2x2 meter yellow cube"""
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0), size=CUBE_SIZE)
    cube = bpy.context.object
    cube.name = "YellowCube"
    
    # Yellow material #F1E44B
    mat = bpy.data.materials.new(name="Yellow")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.945, 0.894, 0.294, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.4
    cube.data.materials.append(mat)
    
    print(f"Cube dimensions: {cube.dimensions}")
    return cube


def setup_camera(cube):
    """Camera position at angle to show it's a 3D cube (not flat)"""
    # Position: front-right-above to see front + right + top faces
    bpy.ops.object.camera_add(location=(3, -4, 2))
    camera = bpy.context.object
    camera.data.type = 'PERSP'
    camera.data.lens = 50
    
    # Point at cube
    constraint = camera.constraints.new(type='TRACK_TO')
    constraint.target = cube
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
    
    bpy.context.scene.camera = camera
    return camera


def setup_lighting():
    """Add lighting"""
    bpy.ops.object.light_add(type='SUN', location=(5, -5, 10))
    sun = bpy.context.object
    sun.data.energy = 3.0


def setup_world():
    """Set gray background"""
    scene = bpy.context.scene
    if not scene.world:
        scene.world = bpy.data.worlds.new("World")
    world = scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.3, 0.3, 0.3, 1.0)


def render(output_path: str, samples: int = 32):
    """Render the scene"""
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
    print("=== 3D Text on Cube Face (CSS→Blender Conversion) ===")
    print(f"PIXEL_TO_METER = {PIXEL_TO_METER:.6f}")
    
    clean_scene()
    
    # Create yellow cube
    cube = create_yellow_cube()
    
    # Apply 3D text elements with CSS coordinates
    print("\nApplying text elements:")
    
    # 1. "27-28" - Benzin-Bold, increased to 280px for better match to reference
    apply_css_element(
        cube, "27-28",
        font_path=f"{FONTS_DIR_INTRO}/Benzin-Bold.ttf",
        font_size_px=280, top_px=200, left_px=93, width_px=907  # Larger size, moved up
    )
    
    # 2. "ДЕКАБРЯ" - Bebas Neue, 240px, top=540, left=316, width=684
    apply_css_element(
        cube, "ДЕКАБРЯ",
        font_path=f"{FONTS_DIR_VIDEO}/BebasNeue-Regular.ttf",
        font_size_px=240, top_px=540, left_px=316, width_px=684
    )
    
    # 3. "АФИША" - Druk Cyr, 110px, top=870, left=761, width=239
    apply_css_element(
        cube, "АФИША",
        font_path=f"{FONTS_DIR_INTRO}/DrukCyr-Bold.ttf",
        font_size_px=110, top_px=870, left_px=761, width_px=239
    )
    
    # Setup scene
    setup_camera(cube)
    setup_lighting()
    setup_world()
    
    # Render
    output = f"{OUTPUT_DIR}/3d_text_cube_test.png"
    render(output, samples=32)
    
    print(f"\nResult: {output}")
