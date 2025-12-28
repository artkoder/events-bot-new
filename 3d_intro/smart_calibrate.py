import bpy
import bmesh
import math
import os
from mathutils import Vector, Matrix

# --- Configuration ---
ASSETS_DIR = "/workspaces/events-bot-new/3d_intro/assets/calibration"
FONTS_DIR_INTRO = "/workspaces/events-bot-new/3d_intro/assets/fonts"
FONTS_DIR_VIDEO = "/workspaces/events-bot-new/video_announce/assets"

FONT_BENZIN = os.path.join(FONTS_DIR_INTRO, "Benzin-Bold.ttf")
FONT_BEBAS = os.path.join(FONTS_DIR_VIDEO, "BebasNeue-Regular.ttf")
FONT_DRUK = os.path.join(FONTS_DIR_INTRO, "DrukCyr-Bold.ttf")
FONT_BEBAS_PRO = os.path.join(FONTS_DIR_VIDEO, "Bebas-Neue-Pro-SemiExpanded-Bold-BF66cf3d78c1f4e.ttf")

CANVAS_SIZE = 1080
CUBE_SIZE = 2.0

LAYERS = [
    # Front Face (Normal 0, -1, 0)
    {"name": "27-28", "text": "27-28", "font": FONT_BENZIN, "mask": "ref_mask_27_28.png", "size": 224, "face_n": (0,-1,0)},
    {"name": "month", "text": "ДЕКАБРЯ", "font": FONT_BEBAS, "mask": "ref_mask_month.png", "size": 240, "face_n": (0,-1,0)},
    {"name": "afisha", "text": "АФИША", "font": FONT_DRUK, "mask": "ref_mask_afisha.png", "size": 110, "face_n": (0,-1,0)},
    # Right Face (Normal 1, 0, 0)
    {"name": "kaliningrad", "text": "Калининград", "font": FONT_BEBAS_PRO, "mask": "ref_mask_kaliningrad.png", "size": 140, "face_n": (1,0,0)},
    {"name": "svetlogorsk", "text": "Светлогорск", "font": FONT_BEBAS_PRO, "mask": "ref_mask_svetlogorsk.png", "size": 140, "face_n": (1,0,0)},
    {"name": "zelenogradsk", "text": "Зеленоградск", "font": FONT_BEBAS_PRO, "mask": "ref_mask_zelenogradsk.png", "size": 140, "face_n": (1,0,0)},
    {"name": "guryevsk", "text": "Гурьевск", "font": FONT_BEBAS_PRO, "mask": "ref_mask_guryevsk.png", "size": 140, "face_n": (1,0,0)},
    {"name": "gvardeysk", "text": "Гвардейск", "font": FONT_BEBAS_PRO, "mask": "ref_mask_gvardeysk.png", "size": 140, "face_n": (1,0,0)},
]

def get_ink_bbox_from_mask(mask_filename):
    path = os.path.join(ASSETS_DIR, mask_filename)
    if not os.path.exists(path): return None
    img = bpy.data.images.load(path)
    w, h = img.size
    pixels = list(img.pixels)
    min_x, max_x = w, 0
    min_y, max_y = h, 0
    found = False
    for y in range(h):
        for x in range(w):
            if pixels[(y*w+x)*4+3] > 0.1:
                found = True
                if x < min_x: min_x = x
                if x > max_x: max_x = x
                if y < min_y: min_y = y
                if y > max_y: max_y = y
    if not found: return None
    # Convert to Top-Left origin coordinates
    # Blender reading is bottom-up (y=0 is bottom).
    # We want y=0 at top.
    # pixel (x, y_blender) -> (x, H - 1 - y_blender)
    l = min_x
    r = max_x + 1 # exclusive
    # Top is max_y in blender (highest row)
    t = h - 1 - max_y 
    # Bottom is min_y in blender (lowest row)
    b = h - 1 - min_y
    # Ensure t < b
    if t > b: t, b = b, t
    return l, t, r, b

def get_face_mapper(cube, normal_target):
    """
    Returns basis vectors (origin, t_u, t_v) for the face matching normal_target.
    t_u: 1px Right vector in World
    t_v: 1px Down vector in World
    """
    mesh = cube.data
    mw = cube.matrix_world
    
    # Find face
    target_face = None
    dots = []
    
    # Calculate face normals in world space
    # (assuming no rotation/scale for simple matching, or apply matrix)
    # Actually, mesh.polygons[i].normal is local.
    # If obj rotated, need mw to transform.
    rot = mw.to_3x3()
    
    target_vec = Vector(normal_target).normalized()
    
    best_dot = -1.0
    
    for poly in mesh.polygons:
        n_world = rot @ poly.normal
        d = n_world.dot(target_vec)
        if d > best_dot:
            best_dot = d
            target_face = poly
            
    if best_dot < 0.9:
        print(f"Warning: No face found for normal {normal_target}")
        return None
        
    # Get UVs
    uv_layer = mesh.uv_layers.active.data
    
    # We need to map UV (0,1) -> TL, (1,1) -> TR, (0,0) -> BL
    # to find World Points.
    # Linear interpolation on the quad.
    
    # Collect Loops: (UV, Co_World)
    points = []
    for loop_idx in target_face.loop_indices:
        uv = uv_layer[loop_idx].uv
        v_idx = mesh.loops[loop_idx].vertex_index
        co_local = mesh.vertices[v_idx].co
        co_world = mw @ co_local
        points.append({'uv': uv, 'co': co_world})
        
    # Simple strategy: Find 3 points to solve affine, or just barycentric from Tri 0,1,2
    # P(u,v) = A + u*U_vec + v*V_vec ??
    # Not necessarily orthogonal in UV space if distorted, but on cube it is.
    
    # Let's use Triangle composed of points 0, 1, 2.
    p0, p1, p2 = points[0], points[1], points[2]
    
    # Barycentric weights for UV=(0,1) (TopLeft)
    # Solve system?
    # Better: Blender's mathutils.geometry.barycentric_transform
    # BUT we need to know WHICH triangle covers the UV coord?
    # For a quad, 0-1-2 and 0-2-3 usually.
    
    def uv_to_world(u, v):
        target_uv = Vector((u, v))
        # Try Tri 1: 0, 1, 2
        # Use simple distance check or just projection if we assume linear map?
        # Let's try to map target_uv using barycentric coords of uv0, uv1, uv2
        # And apply to co0, co1, co2
        
        # This works if point is inside triangle.
        # However, affine mapping extrapolation works too if planar.
        
        # P = P0 + (P1-P0)*a + (P2-P0)*b
        # UV = UV0 + (UV1-UV0)*a + (UV2-UV0)*b
        # Solve for a, b from UV, apply to P.
        
        uv0, uv1, uv2 = p0['uv'], p1['uv'], p2['uv']
        den = (uv1.x - uv0.x) * (uv2.y - uv0.y) - (uv1.y - uv0.y) * (uv2.x - uv0.x)
        
        if abs(den) < 1e-6:
             # Degenerate triangle in UV? Try 0, 2, 3
             return Vector((0,0,0)) # Fallback
             
        # Cramer's rule for a, b
        # u - u0 = (u1-u0)a + (u2-u0)b
        # v - v0 = (v1-v0)a + (v2-v0)b
        
        du = target_uv.x - uv0.x
        dv = target_uv.y - uv0.y
        
        a = ((uv2.y - uv0.y)*du - (uv2.x - uv0.x)*dv) / den
        b = ((uv1.x - uv0.x)*dv - (uv1.y - uv0.y)*du) / den
        
        w_p = p0['co'] + (p1['co'] - p0['co'])*a + (p2['co'] - p0['co'])*b
        return w_p

    # Top Left UV=(0, 1)
    # Top Right UV=(1, 1)
    # Bottom Left UV=(0, 0)
    
    P_TL = uv_to_world(0.0, 1.0)
    P_TR = uv_to_world(1.0, 1.0)
    P_BL = uv_to_world(0.0, 0.0)
    
    # Vectors for 1 pixel
    t_u = (P_TR - P_TL) / CANVAS_SIZE # Right
    t_v = (P_BL - P_TL) / CANVAS_SIZE # Down (Blender V goes up, but Pixel Y goes down, so BL is V=0, TL is V=1. 
    # Wait, simple math:
    # vector_down = (P_Bottom - P_Top).
    # P_Bottom (V=0) is BL. P_Top (V=1) is TL.
    # So P_BL - P_TL is indeed the vector from Top to Bottom.
    
    return P_TL, t_u, t_v


def calibrate_layer(layer, cube):
    print(f"Calibrating {layer['name']}...")
    
    # 1. BBox Ref
    bbox_ref = get_ink_bbox_from_mask(layer['mask'])
    if not bbox_ref: 
        print("No mask bbox"); return
    l_ref, t_ref, r_ref, b_ref = bbox_ref
    
    # 2. Get Face Mapper
    mapper = get_face_mapper(cube, layer['face_n'])
    if not mapper: return
    origin_px, t_u, t_v = mapper
    
    # 3. Create Text
    # Basis Rotation
    # Text Local X should match t_u direction
    # Text Local Y should match -t_v direction (vector UP)
    # Text Local Z should match Face Normal? Cross product.
    
    vec_x = t_u.normalized()
    vec_y = -t_v.normalized() # t_v is DOWN, Text Y is UP
    vec_z = vec_x.cross(vec_y).normalized()
    
    # Construct Matrix
    rot_mat = Matrix((vec_x, vec_y, vec_z)).transposed().to_4x4()
    
    bpy.ops.object.text_add()
    txt = bpy.context.object
    txt.data.body = layer['text']
    # Load font...
    font_path = layer['font']
    if os.path.exists(font_path):
        txt.data.font = bpy.data.fonts.load(font_path)
    
    # Apply Rotation
    txt.matrix_world = rot_mat # Sets orientation
    
    # Move to Face Origin (TL)
    txt.location = origin_px
    
    # Init Size
    # t_v length is size of 1 px in world.
    # Font size 1.0 usually ~ 1 meter?
    # Start with guessed scale.
    # If font size 140px... 
    # Try size = 140 * t_v.length?
    # Let's iterate.
    
    txt.data.size = layer['size'] * t_v.length * 1.5 # guess
    
    # Loop
    for i in range(10):
        # Measure Geometry in Pixels
        # Project vertices relative to origin_px onto t_u, t_v
        
        deps = bpy.context.evaluated_depsgraph_get()
        ev = txt.evaluated_get(deps)
        mesh = ev.to_mesh()
        
        min_x, max_x = 99999, -99999
        min_y, max_y = 99999, -99999
        
        mw = txt.matrix_world
        
        # Lengths of basis vectors squared for projection
        len_u2 = t_u.length_squared
        len_v2 = t_v.length_squared
        
        for v in mesh.vertices:
            w_co = mw @ v.co
            diff = w_co - origin_px
            
            # Project: dot(diff, t_u) / |t_u|^2 ???
            # No.
            # px_x = dot(diff, u_hat) / |t_u_px| ??
            # t_u is vector for 1 pixel.
            # So if diff = 10 * t_u, then px_x should be 10.
            # diff = k * t_u
            # dot(diff, t_u) = k * |t_u|^2
            # k = dot / |t_u|^2
            
            px_x = diff.dot(t_u) / len_u2
            px_y = diff.dot(t_v) / len_v2
            
            if px_x < min_x: min_x = px_x
            if px_x > max_x: max_x = px_x
            if px_y < min_y: min_y = px_y
            if px_y > max_y: max_y = px_y
            
        ev.to_mesh_clear()
        
        w_geo = max_x - min_x
        h_geo = max_y - min_y
        
        if h_geo < 0.1: h_geo = 0.1
        
        # Targets
        w_target = r_ref - l_ref
        h_target = b_ref - t_ref
        
        # Update
        ratio_h = h_target / h_geo
        ratio_w = w_target / w_geo
        
        txt.data.size *= ratio_h
        txt.data.space_character *= (ratio_w / ratio_h)
        
        # Converge check...
        if abs(ratio_h - 1) < 0.005: break

    # Final Position
    # Align Top-Left of Ref to Top-Left of Geo? 
    # Or Align Left/Top based on visual intent.
    # Cities: Left Align.
    
    dx = l_ref - min_x
    dy = t_ref - min_y
    
    # Move in World
    shift = t_u * dx + t_v * dy
    txt.location += shift
    
    # Small offset along normal to prevent trimming (Z local)
    txt.location += vec_z * 0.002
    
    # Save results...
    print(f"  [RESULT] Layer: {layer['name']}")
    print(f"    Size: {txt.data.size:.4f}")
    print(f"    Spacing: {txt.data.space_character:.4f}")
    loc = txt.location
    rot = txt.rotation_euler
    print(f"    Loc: ({loc.x:.4f}, {loc.y:.4f}, {loc.z:.4f})")
    print(f"    Rot: ({rot.x:.4f}, {rot.y:.4f}, {rot.z:.4f})")
    
    return {
        "layer": layer['name'],
        "size": txt.data.size,
        "spacing": txt.data.space_character,
        "loc": (loc.x, loc.y, loc.z),
        "rot": (rot.x, rot.y, rot.z)
    }

def main():
    # Setup scene...
    bpy.ops.mesh.primitive_cube_add(size=CUBE_SIZE)
    cube = bpy.context.object
    if not cube.data.uv_layers:
        bpy.ops.mesh.uv_texture_add()
    
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.uv.reset() 
    bpy.ops.object.mode_set(mode='OBJECT')
    
    results = []
    for l in LAYERS:
        res = calibrate_layer(l, cube)
        if res: results.append(res)
        
    print("\n=== Final JSON Validation Data ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
