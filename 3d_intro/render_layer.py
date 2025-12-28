
import bpy
import os
import sys
import json
import mathutils

# Constants
CUBE_SIZE = 2.0
OUTPUT_DIR = "/tmp/calibration"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FONTS = {
    "Benzin-Bold": "/workspaces/events-bot-new/3d_intro/assets/fonts/Benzin-Bold.ttf",
    "BebasNeue": "/workspaces/events-bot-new/video_announce/assets/BebasNeue-Regular.ttf",
    "DrukCyr-Bold": "/workspaces/events-bot-new/3d_intro/assets/fonts/DrukCyr-Bold.ttf",
}

def clean_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def render_layer(params):
    clean_scene()
    
    # -------------------------------------------------------------------------
    # 1. View Transform (Make it Standard for pure White/Black)
    # -------------------------------------------------------------------------
    bpy.context.scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.view_settings.look = 'None'
    bpy.context.scene.view_settings.exposure = 0.0
    bpy.context.scene.view_settings.gamma = 1.0

    # -------------------------------------------------------------------------
    # 2. Background (Pure White Emission Plane)
    # -------------------------------------------------------------------------
    bpy.ops.mesh.primitive_plane_add(size=CUBE_SIZE*2, location=(0, 0, -0.1))
    bg_plane = bpy.context.object
    bg_plane.name = "Background"
    
    mat_bg = bpy.data.materials.new(name="WhiteEmission")
    mat_bg.use_nodes = True
    tree = mat_bg.node_tree
    nodes = tree.nodes
    links = tree.links
    output = nodes.get("Material Output")
    for n in nodes: 
        if n != output: nodes.remove(n)
        
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (1, 1, 1, 1) # Pure White
    emission.inputs['Strength'].default_value = 1.0
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    bg_plane.data.materials.append(mat_bg)

    # -------------------------------------------------------------------------
    # 3. Text (Pure Black Emission)
    # -------------------------------------------------------------------------
    bpy.ops.object.text_add(location=(0, 0, 0))
    txt_obj = bpy.context.object
    txt_obj.data.body = params["text"]
    txt_obj.name = f"Text_{params['layer_id']}"
    
    font_path = FONTS.get(params["font"])
    if font_path and os.path.exists(font_path):
        try:
            txt_obj.data.font = bpy.data.fonts.load(font_path)
        except Exception as e:
            print(f"Warning: Could not load font: {e}")
            
    txt_obj.data.size = params["font_size"]
    txt_obj.data.align_x = params.get("align_x", "RIGHT")
    txt_obj.data.align_y = "TOP"
    
    # Direct tracking factor (geometry based) or legacy addition
    if "space_factor" in params:
        txt_obj.data.space_character = params["space_factor"]
    elif params.get("character_spacing", 0) != 0:
        txt_obj.data.space_character = 1.0 + params["character_spacing"] / txt_obj.data.size
    else:
        txt_obj.data.space_character = 1.0
        
    txt_obj.location = (params["location_x"], params["location_z"], 0)
    
    mat_txt = bpy.data.materials.new(name="BlackEmission")
    mat_txt.use_nodes = True
    tree = mat_txt.node_tree
    nodes = tree.nodes
    links = tree.links
    output = nodes.get("Material Output")
    for n in nodes: 
        if n != output: nodes.remove(n)
        
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (0, 0, 0, 1) # Pure Black
    emission.inputs['Strength'].default_value = 1.0
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    txt_obj.data.materials.append(mat_txt)

    # -------------------------------------------------------------------------
    # 4. Camera (Ortho Front)
    # -------------------------------------------------------------------------
    bpy.ops.object.camera_add(location=(0, 0, 5))
    cam = bpy.context.object
    cam.rotation_euler = (0, 0, 0)
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = CUBE_SIZE
    bpy.context.scene.camera = cam

    # --- GEOMETRY MEASUREMENT (User Request) ---
    # Update scene to ensure geometry is generated
    bpy.context.view_layer.update()
    
    # Let's compute from bound_box in LOCAL space (no matrix_world).
    local_corners = [mathutils.Vector(corner) for corner in txt_obj.bound_box]
    xs = [v.x for v in local_corners]
    ys = [v.y for v in local_corners]
    zs = [v.z for v in local_corners]
    
    bbox_data = {
        "x_min": min(xs),
        "x_max": max(xs),
        "y_min": min(ys),
        "y_max": max(ys),
        "z_min": min(zs),
        "z_max": max(zs),
        "dimensions": [txt_obj.dimensions.x, txt_obj.dimensions.y, txt_obj.dimensions.z],
        # Plane Dimensions
        "Sx_face": CUBE_SIZE, 
        "Sz_face": CUBE_SIZE,
        "calibration_mode": params.get("calibration_mode", False)
    }
    
    # Output measurement for calibration script
    print(f"GEOMETRY_BBOX: {json.dumps(bbox_data)}")

    # -------------------------------------------------------------------------
    # 5. Render Settings
    # -------------------------------------------------------------------------
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 1 
    scene.cycles.use_denoising = False
    
    # Check for Final Camera Request
    if params.get("camera_view") == "FINAL":
        print("DEBUG: Setting FINAL Camera (Scale 2.2)...")
        # Setup Final Camera (Match render_calibrated.py)
        # Remove existing camera
        if 'cam' in locals():
            bpy.data.objects.remove(cam, do_unlink=True)
            
        bpy.ops.object.camera_add(location=(0, -5, 0))
        cam = bpy.context.object
        cam.data.type = 'ORTHO'
        cam.data.ortho_scale = 2.2 
        cam.rotation_euler = (mathutils.math.radians(90), 0, 0)
        scene.camera = cam
        
        # Rotate Text to Face (XZ plane)
        txt_obj.rotation_euler = (mathutils.math.radians(90), 0, 0)
        # Position: X, -1.002, Z
        # Note: location_z in params was for Face Basis (Y in 2D).
        # In Final 3D: Global Z = Face Basis Y. Global X = Face Basis X.
        # Global Y = -1.002 (Face depth)
        txt_obj.location = (params["location_x"], -1.002, params["location_z"])
        
        # Reference Plane (Texture or White Face)
        ref_path = "/home/codespace/.gemini/antigravity/brain/d43d08f1-b44b-444b-8452-4f1dbed5e7b1/uploaded_image_1766911240181.png"
        ref_exists = os.path.exists(ref_path)
        
        # Always create plane geometry for FACE mode or REF mode
        if ref_exists or params.get("render_mode") == "FACE":
            bpy.ops.mesh.primitive_plane_add(size=CUBE_SIZE, location=(0, -1.01, 0))
            ref_plane = bpy.context.object
            ref_plane.rotation_euler = (mathutils.math.radians(90), 0, 0)
            
            # Setup Material
            mat_ref = bpy.data.materials.new(name="RefMat")
            mat_ref.use_nodes = True
            bsdf = mat_ref.node_tree.nodes.get("Principled BSDF")
            
            # If image exists, use it. Otherwise, material stays default (white-ish) until override.
            if ref_exists:
                tex = mat_ref.node_tree.nodes.new('ShaderNodeTexImage')
                tex.image = bpy.data.images.load(ref_path)
                mat_ref.node_tree.links.new(tex.outputs['Color'], bsdf.inputs['Emission'])
                bsdf.inputs['Emission Strength'].default_value = 1.0
            
            ref_plane.data.materials.append(mat_ref)

        # Handling Render Modes
        render_mode = params.get("render_mode")
        
        if render_mode == "TEXT":
            if 'ref_plane' in locals(): ref_plane.hide_render = True
            
        elif render_mode == "REF":
            txt_obj.hide_render = True
            if 'bg_plane' in locals(): bg_plane.hide_render = True
            
        elif render_mode == "FACE":
            txt_obj.hide_render = True
            if 'bg_plane' in locals(): bg_plane.hide_render = True
            
            if 'ref_plane' in locals():
                # Override to White Emission
                mat_white = bpy.data.materials.new(name="WhiteFace")
                mat_white.use_nodes = True
                tree = mat_white.node_tree
                nodes = tree.nodes
                for n in nodes: nodes.remove(n)
                
                output = nodes.new('ShaderNodeOutputMaterial')
                emission = nodes.new('ShaderNodeEmission')
                emission.inputs['Color'].default_value = (1, 1, 1, 1)
                emission.inputs['Strength'].default_value = 1.0
                tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
                
                ref_plane.data.materials.clear()
                ref_plane.data.materials.append(mat_white)
    
    # Calibration Mode Settings (Transparency)
    if params.get("calibration_mode", False) or params.get("camera_view") == "FINAL":
        scene.render.film_transparent = True
        scene.render.image_settings.color_mode = 'RGBA'
        txt_obj.data.extrude = 0 # Force flat for calibration
        if 'bg_plane' in locals():
            bg_plane.hide_render = True
    else:
        scene.render.film_transparent = False
        scene.render.image_settings.color_mode = 'RGB'
        txt_obj.data.extrude = 0
    
    scene.render.resolution_x = 1080
    scene.render.resolution_y = 1080
    
    output_path = f"{OUTPUT_DIR}/bl_{params['layer_id']}.png"
    scene.render.filepath = output_path
    
    bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    try:
        args = sys.argv
        if "--" in args:
            json_str = args[args.index("--") + 1]
            params = json.loads(json_str)
            print(f"DEBUG: Params received: {params.keys()}")
            render_layer(params)
        else:
            print("Usage: blender ... -- '{\"json\": \"params\"}'")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
