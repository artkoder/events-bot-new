#!/usr/bin/env python3
"""
Strict Geometry-Based Auto-Calibration System.
Implements the User's Checklist:
1. Verify Right-Align (Ref vs CSS).
2. Measure Face Dimensions -> Kx, Kz.
3. Anchor to Glyph Geometry (Right, Top).
4. Multiplicative Tracking.
"""
import os
import json
import subprocess
import numpy as np
import sys
from PIL import Image
from typing import Tuple, Dict, Any, Optional

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

OUTPUT_DIR = "/tmp/calibration"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CANVAS_SIZE = 1080
# K will be calculated dynamically from geometry

LAYERS = [
    {
        "id": "27-28",
        "text": "27-28",
        "font": "Benzin-Bold",
        "font_size": 224,
        "line_height": 308,
        "left": 93,
        "top": 232,
        "width": 907,
        "height": 308,
        "align": "right",
    },
    {
        "id": "ДЕКАБРЯ",
        "text": "ДЕКАБРЯ",
        "font": "BebasNeue",
        "font_size": 240,
        "line_height": 240,
        "left": 316,
        "top": 540,
        "width": 684,
        "height": 240,
        "align": "right",
    },
    {
        "id": "АФИША",
        "text": "АФИША",
        "font": "DrukCyr-Bold",
        "font_size": 110,
        "line_height": 140,
        "left": 761,
        "top": 870,
        "width": 239,
        "height": 140,
        "align": "right",
    },
]

# Reset Calibration
calibration = {
    "Benzin-Bold": {"scale": 1.0, "tracking_factor": 1.0, "offset_x": 0.0, "offset_y": 0.0},
    "BebasNeue": {"scale": 1.0, "tracking_factor": 1.0, "offset_x": 0.0, "offset_y": 0.0},
    "DrukCyr-Bold": {"scale": 1.0, "tracking_factor": 1.0, "offset_x": 0.0, "offset_y": 0.0},
}

REFERENCE_IMAGE = "/home/codespace/.gemini/antigravity/brain/d43d08f1-b44b-444b-8452-4f1dbed5e7b1/uploaded_image_1766911240181.png"

MAX_ITERATIONS = 5

def render_reference_layer(layer: Dict) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    img = Image.open(REFERENCE_IMAGE).convert("L")
    
    # 50px Margin
    margin = 50
    x = max(0, layer["left"] - margin)
    y = max(0, layer["top"] - margin)
    w = layer["width"] + 2*margin
    h = layer["height"] + 2*margin
    
    img_w, img_h = img.size
    crop_box = (x, y, min(img_w, x+w), min(img_h, y+h))
    crop_arr = np.array(img.crop(crop_box))
    
    # Threshold < 50 (Shadow Exclusion)
    mask = (crop_arr < 50).astype(np.uint8)
    
    # Smart Crop (Vertical)
    rows = np.any(mask, axis=1)
    intervals = []
    start = None
    for i, val in enumerate(rows):
        if val and start is None:
            start = i
        elif not val and start is not None:
            intervals.append((start, i - 1))
            start = None
    if start is not None:
        intervals.append((start, len(rows) - 1))
        
    if not intervals:
        return np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8), (0,0,0,0)

    expected_center_y = margin + layer["height"] / 2
    best_interval = min(intervals, key=lambda iv: abs(((iv[0]+iv[1])/2) - expected_center_y))
    y_min, y_max = best_interval
    
    clean_mask = np.zeros_like(mask)
    clean_mask[y_min:y_max+1, :] = mask[y_min:y_max+1, :]
    mask = clean_mask
    
    cols = np.any(mask, axis=0)
    if not np.any(cols):
        return np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8), (0,0,0,0)
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    global_x = x + x_min
    global_y = y + y_min
    global_w = x_max - x_min + 1
    global_h = y_max - y_min + 1
    
    full_mask = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    h_src, w_src = mask.shape
    full_mask[y:y+h_src, x:x+w_src] = mask
    
    # Debug Image
    debug_img = Image.fromarray(full_mask * 255)
    debug_img.save(f"{OUTPUT_DIR}/debug_mask_{layer['id']}.png")
    
    return full_mask, (global_x, global_y, global_w, global_h)

def get_blender_params(layer: Dict, cal: Dict, K: float) -> Dict:
    # Use geometry-based offsets
    x_anchor = layer["left"] + layer["width"] 
    
    X = (x_anchor - CANVAS_SIZE / 2) * K + cal["offset_x"] * K
    Z = (CANVAS_SIZE / 2 - layer["top"]) * K + cal["offset_y"] * K
    
    size = layer["font_size"] * K * cal["scale"]
    
    return {
        "layer_id": layer["id"],
        "text": layer["text"],
        "font": layer["font"],
        "font_size": size,
        "location_x": X,
        "location_z": Z,
        "space_factor": cal["tracking_factor"],
        "align_x": "RIGHT",
        "calibration_mode": True # Enable Transparent + Alpha
    }

def render_blender_layer_with_geometry(params: Dict) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    params_json = json.dumps(params)
    print(f"DEBUG: Calling Blender with camera_view={params.get('camera_view')}")
    result = subprocess.run(
        ["blender", "--background", "--python", 
         "/workspaces/events-bot-new/3d_intro/render_layer.py",
         "--", params_json],
        capture_output=True,
        text=True,
        cwd="/workspaces/events-bot-new/3d_intro"
    )
    
    geom_data = None
    for line in result.stdout.splitlines():
        if line.startswith("GEOMETRY_BBOX:"):
            try:
                geom_data = json.loads(line.replace("GEOMETRY_BBOX:", "").strip())
            except:
                pass
    
    output_path = f"{OUTPUT_DIR}/bl_{params['layer_id']}.png"
    mask = None
    if os.path.exists(output_path):
        img_pil = Image.open(output_path).convert("RGBA")
        # Extract Alpha Channel
        alpha = np.array(img_pil)[:, :, 3]
        mask = (alpha > 0).astype(np.uint8) # Strict Alpha Mask
        
    return mask, geom_data

def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)
    return intersection / union if union > 0 else 0.0

def calibrate_layer(layer: Dict) -> Dict[str, Any]:
    cal = calibration[layer["font"]]
    print(f"\n========================================================")
    print(f"Calibrating {layer['id']}...")
    print(f"========================================================")
    
    # 1. Reference Analysis
    ref_mask, (rx, ry, rw, rh) = render_reference_layer(layer)
    if rw == 0:
        print("Error: Reference Mask Empty")
        return {"passed": False}
        
    Ref_Right_px = rx + rw
    Ref_Top_px = ry
    
    # Verify Right Align
    CSS_Right = layer["left"] + layer["width"]
    Delta_Right = Ref_Right_px - CSS_Right
    print(f"DIAGNOSTIC: Ref_Right={Ref_Right_px}, CSS_Right={CSS_Right}, DELTA={Delta_Right} px")
    if abs(Delta_Right) > 2:
        print(f"WARNING: Reference Mismatch! Screenshot contents are shifted by {Delta_Right}px relative to CSS Box.")
    
    Kx = 2.0 / CANVAS_SIZE # Default guess
    Kz = 2.0 / CANVAS_SIZE 

    iou = 0.0

    for i in range(MAX_ITERATIONS):
        print(f"--- Iter {i+1} ---")
        
        # Use current K (initially 2.0/1080)
        params = get_blender_params(layer, cal, Kx) 
        bl_img, geom = render_blender_layer_with_geometry(params)
        
        if geom is None:
            print("Render failed")
            break

        # Upgrading variable name for clarity
        bl_mask = bl_img
            
        # Update K based on measured Face Dimensions
        Sx_face = geom.get("Sx_face", 2.0)
        Sz_face = geom.get("Sz_face", 2.0)
        Kx = Sx_face / CANVAS_SIZE
        Kz = Sz_face / CANVAS_SIZE
        
        # Geometry in BU
        h_local = geom["y_max"] - geom["y_min"]
        w_local = geom["x_max"] - geom["x_min"]
        x_right_local = geom["x_max"]
        y_max_local = geom["y_max"] # Top of glyph in local coords (Global Z up)
        
        print(f"Blender Geom: h={h_local:.4f}, w={w_local:.4f}, RightLoc={x_right_local:.4f}")
        
        # 1. Scale Update (Height Ratio)
        Ref_Height_bu = rh * Kz
        if h_local > 0:
            scale_ratio = Ref_Height_bu / h_local
            cal["scale"] *= scale_ratio
            
        # 2. Tracking Update (Width Ratio Multiplicative)
        Ref_Width_bu = rw * Kx
        if w_local > 0:
            width_ratio = Ref_Width_bu / w_local
            cal["tracking_factor"] *= width_ratio
            
        # 3. Position Update (Use X_target, Z_target)
        # Z_target_top (BU)
        Z_target_top = (CANVAS_SIZE / 2 - Ref_Top_px) * Kz
        Z_loc_new = Z_target_top - y_max_local
        
        # Map back to Offset Y for param generation
        # Z_loc = (H/2 - Top_CSS)*K + OffsetY*K
        # OffsetY = Z_loc/K - (H/2 - Top_CSS)
        cal["offset_y"] = (Z_loc_new / Kz) - (CANVAS_SIZE / 2 - layer["top"])
        
        # X_target_right (BU)
        X_target_right = (Ref_Right_px - CANVAS_SIZE / 2) * Kx
        X_loc_new = X_target_right - x_right_local
        
        # Map back to Offset X
        # X_loc = (Right_CSS - W/2)*K + OffsetX*K
        # OffsetX = X_loc/K - (Right_CSS - W/2)
        cal["offset_x"] = (X_loc_new / Kx) - (CSS_Right - CANVAS_SIZE / 2)
        
        print(f"Updated Cal: Sc={cal['scale']:.3f}, TrFactor={cal['tracking_factor']:.3f}, Off=({cal['offset_x']:.1f}, {cal['offset_y']:.1f})")
        
        # IoU Check
        # bl_mask is already binary 0/1 from render_blender_layer
        current_iou = calculate_iou(ref_mask, bl_mask)
        print(f"Precision Check: IoU={current_iou:.3f}")
        iou = current_iou
        
        if current_iou > 0.88:
            break
            
    # --- Acceptance-Final (Camera View) ---
    print("--- Acceptance-Final (Camera View) ---")
    
    # helper for bbox
    def get_bbox(mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols): return 0,0,0,0
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return cmin, rmin, cmax-cmin+1, rmax-rmin+1

    # 1. Measure Face BBox (The "Canvas" in Camera)
    params_face = get_blender_params(layer, cal, Kx)
    params_face["camera_view"] = "FINAL"
    params_face["render_mode"] = "FACE"
    params_face["calibration_mode"] = True
    mask_face_cam, _ = render_blender_layer_with_geometry(params_face)
    
    face_w, face_h = 0, 0
    if mask_face_cam is not None:
        fx, fy, fw, fh = get_bbox(mask_face_cam)
        print(f"Face BBox (Cam): x={fx}, y={fy}, w={fw}, h={fh}")
        
        Sx_cam = fw / 1080.0
        Sy_cam = fh / 1080.0
        print(f"Camera Projection Scale: Sx={Sx_cam:.4f}, Sy={Sy_cam:.4f}")
        if Sx_cam > 0:
            print(f"Inverse Scale (Zoom Factor): {1.0/Sx_cam:.4f} (Expected ~1.073)")
        
        face_w, face_h = fw, fh
    
    # 2. Comparison Ref (Texture) vs Geo (Text)
    params_ref = get_blender_params(layer, cal, Kx)
    params_ref["camera_view"] = "FINAL"
    params_ref["render_mode"] = "REF"
    params_ref["calibration_mode"] = True
    mask_ref_cam, _ = render_blender_layer_with_geometry(params_ref)
    
    params_geo = get_blender_params(layer, cal, Kx)
    params_geo["camera_view"] = "FINAL"
    params_geo["render_mode"] = "TEXT"
    params_geo["calibration_mode"] = True
    mask_geo_cam, _ = render_blender_layer_with_geometry(params_geo)
    
    if mask_ref_cam is not None and mask_geo_cam is not None:
        # Sanity Checks
        ref_pix = np.count_nonzero(mask_ref_cam)
        geo_pix = np.count_nonzero(mask_geo_cam)
        print(f"Non-Zero Pixels: Ref={ref_pix}, Geo={geo_pix}")
        
        if ref_pix < 100:
            print("ERROR: Reference Mask is EMPTY or INVALID. Acceptance Failed.")
            return {"id": layer["id"], "error": "Empty Ref Mask"}
            
        x_ref, y_ref, w_ref, h_ref = get_bbox(mask_ref_cam)
        x_geo, y_geo, w_geo, h_geo = get_bbox(mask_geo_cam)
        
        print(f"Cam Ref BBox: x={x_ref}, y={y_ref}, w={w_ref}, h={h_ref}")
        print(f"Cam Geo BBox: x={x_geo}, y={y_geo}, w={w_geo}, h={h_geo}")
        
        # Calculate G (Geo vs Texture)
        if w_geo > 0 and h_geo > 0:
            Gx = w_ref / w_geo
            Gy = h_ref / h_geo
            G = (Gx + Gy) / 2.0
            print(f"G-Factor (Geo matched to Texture?): {G:.4f}")
            
            # IoU in Camera Space
            iou_cam = calculate_iou(mask_ref_cam, mask_geo_cam)
            print(f"Camera IoU: {iou_cam:.4f}")
            
            # Update results
            cal["final_camera_iou"] = iou_cam
            cal["G_factor"] = G
                
    return {"id": layer["id"], "final_iou": iou, "calibration": cal, "delta_right": Delta_Right}
                
    return {"id": layer["id"], "final_iou": iou, "calibration": cal, "delta_right": Delta_Right}

def main():
    print("Strict Geometry Calibration Starting...")
    results = []
    for layer in LAYERS:
        res = calibrate_layer(layer)
        results.append(res)
        
    print("\nDONE.")
    for r in results:
        cal = r["calibration"]
        print(f"{r['id']}: IoU={r['final_iou']:.3f}, DeltaRight={r['delta_right']}px")
        print(f"  Params: Scale={cal['scale']:.3f}, Track={cal['tracking_factor']:.3f}, Off=({cal['offset_x']:.1f}, {cal['offset_y']:.1f})")

    # Save
    with open(f"{OUTPUT_DIR}/calibration_result_strict.json", "w") as f:
        json.dump(results, f, cls=NumpyEncoder, indent=2)

if __name__ == "__main__":
    main()
