import bpy
import math
import os

# --- Configuration ---
OUTPUT_PATH = "/workspaces/events-bot-new/3d_intro/assets/calibration_check_right.png"
# Use the NEW overlay for cities
REFERENCE_IMAGE = "/home/codespace/.gemini/antigravity/brain/990c274c-40a7-4cc4-a521-59a07d231185/uploaded_image_1766959666027.png"

# Fonts
FONTS_DIR_INTRO = "/workspaces/events-bot-new/3d_intro/assets/fonts"
FONTS_DIR_VIDEO = "/workspaces/events-bot-new/video_announce/assets"
FONT_BENZIN = os.path.join(FONTS_DIR_INTRO, "Benzin-Bold.ttf")
FONT_BEBAS = os.path.join(FONTS_DIR_VIDEO, "BebasNeue-Regular.ttf")
FONT_DRUK = os.path.join(FONTS_DIR_INTRO, "DrukCyr-Bold.ttf")
FONT_BEBAS_PRO = os.path.join(FONTS_DIR_VIDEO, "Bebas-Neue-Pro-SemiExpanded-Bold-BF66cf3d78c1f4e.ttf")

CUBE_SIZE = 2.0

def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def setup_camera(cube):
    # Setup Camera looking at RIGHT FACE (+X)
    # Position: (+5, 0, 0) looking at (0,0,0)
    # Rotation: Needs to look along -X.
    bpy.ops.object.camera_add(location=(5, 0, 0))
    cam = bpy.context.object
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = CUBE_SIZE * 1.5 
    
    constraint = cam.constraints.new(type='TRACK_TO')
    constraint.target = cube
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y' 
    # For Side View:
    # If Cam at +X, Looking -X. 
    # Up vector (0,0,1) is Z. 
    # Track -Z means Camera -Z points to target.
    # We want Camera -Z to point to (-1, 0, 0).
    # Up is Y? No, Up is Z usually.
    # Let's trust TrackTo.
    
    bpy.context.scene.camera = cam

def create_cube():
    bpy.ops.mesh.primitive_cube_add(size=CUBE_SIZE)
    cube = bpy.context.object
    # Yellow Material
    mat = bpy.data.materials.new(name="Yellow")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.945, 0.894, 0.294, 1.0)
    cube.data.materials.append(mat)
    return cube

def add_text(text, font_path, size, spacing, loc, rot):
    bpy.ops.object.text_add()
    txt = bpy.context.object
    txt.data.body = text
    
    if os.path.exists(font_path):
        font = bpy.data.fonts.load(font_path)
        txt.data.font = font
    
    txt.data.size = size
    txt.data.space_character = spacing
    txt.data.align_x = 'LEFT'
    txt.data.align_y = 'TOP'
    
    txt.location = loc
    txt.rotation_euler = rot
    
    # Material (Black)
    mat = bpy.data.materials.new(name="TextBlack")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.063, 0.055, 0.055, 1.0)
    txt.data.materials.append(mat)

def add_overlay(cube):
    # Overlay plane on Right Face
    # Right Face is at X = +1.0
    # Plane size = 2.0
    # Rotation: Needs to match text orientation?
    # Or just face +X.
    # Text rotation was (-3.14, -1.57, 0) which is facing -X (Inside).
    # If text faces inside, and overlay faces outside, they won't match.
    # But let's assume overlay needs to be readable from camera.
    # Camera is at +X looking at -X.
    # Overlay plane should face +X.
    
    bpy.ops.mesh.primitive_plane_add(size=CUBE_SIZE, location=(1.02, 0, 0))
    plane = bpy.context.object
    plane.rotation_euler = (0, math.radians(90), 0)
    
    mat = bpy.data.materials.new(name="Overlay")
    mat.use_nodes = True
    mat.blend_method = 'BLEND'
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    out = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    tex = nodes.new('ShaderNodeTexImage')
    
    if os.path.exists(REFERENCE_IMAGE):
        img = bpy.data.images.load(REFERENCE_IMAGE)
        tex.image = img
    
    bsdf.inputs['Alpha'].default_value = 0.5
    
    links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    
    plane.data.materials.append(mat)

def render():
    scene = bpy.context.scene
    scene.render.resolution_x = 1080
    scene.render.resolution_y = 1080
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 32
    scene.cycles.use_denoising = False
    scene.render.filepath = OUTPUT_PATH
    bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    clean_scene()
    cube = create_cube()
    # Ensure UV reset to match calibration condition
    if not cube.data.uv_layers: bpy.ops.mesh.uv_texture_add()
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.uv.reset()
    bpy.ops.object.mode_set(mode='OBJECT')
    
    setup_camera(cube)
    
    # Right Face Layers (from FaceMapper calibration)
    # Rotation: (-3.14159, -1.57079, 0.0)
    rot_right = (-3.141593, -1.570796, 0.0)
    
    layers_right = [
        ("Калининград", 0.2982, 1.0045, (1.002, 0.85177, 0.7468)),
        ("Светлогорск", 0.2972, 1.0007, (1.002, 0.85312, 0.4514)),
        ("Зеленоградск", 0.2972, 1.0084, (1.002, 0.85323, 0.1569)),
        ("Гурьевск", 0.2972, 0.9942, (1.002, 0.85173, -0.1367)),
        ("Гвардейск", 0.2982, 0.9941, (1.002, 0.85177, -0.4309)),
    ]
    
    for txt, sz, sp, loc in layers_right:
        add_text(txt, FONT_BEBAS_PRO, sz, sp, loc, rot_right)
             
    add_overlay(cube)
    
    # World Light
    scene = bpy.context.scene
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (1, 1, 1, 1)
    bg.inputs["Strength"].default_value = 1.0
    
    render()
