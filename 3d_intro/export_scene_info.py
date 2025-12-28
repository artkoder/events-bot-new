#!/usr/bin/env python3
"""
Export Bento scene to OBJ and create documentation
"""

import bpy
import json

# Load the scene
bpy.ops.wm.open_mainfile(filepath="/tmp/bento_scene_v1.blend")

# Collect scene information
scene_info = {
    "cubes": [],
    "camera": {},
    "lights": []
}

# Get cube information
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        cube_data = {
            "name": obj.name,
            "location": list(obj.location),
            "scale": list(obj.scale),
            "type": obj.get('type', 'unknown'),
        }
        if 'text_content' in obj:
            cube_data['text'] = obj['text_content']
        scene_info["cubes"].append(cube_data)
    elif obj.type == 'CAMERA':
        scene_info["camera"] = {
            "location": list(obj.location),
            "rotation": list(obj.rotation_euler),
            "lens": obj.data.lens
        }
    elif obj.type == 'LIGHT':
        scene_info["lights"].append({
            "name": obj.name,
            "type": obj.data.type,
            "location": list(obj.location),
            "energy": obj.data.energy
        })

# Save scene info
info_path = "/home/codespace/.gemini/antigravity/brain/ec7f7345-db56-4a74-a32a-d8d828d0a370/bento_scene_info.json"
with open(info_path, 'w', encoding='utf-8') as f:
    json.dump(scene_info, f, indent=2, ensure_ascii=False)

print(f"Scene info saved to: {info_path}")
print(f"Total cubes: {len(scene_info['cubes'])}")
print(f"Camera location: {scene_info['camera']['location']}")
