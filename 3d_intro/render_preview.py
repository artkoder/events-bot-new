#!/usr/bin/env python3
"""
Create a simple viewport render of the Bento scene
"""

import bpy
import os

# Load the scene
bpy.ops.wm.open_mainfile(filepath="/tmp/bento_scene_v1.blend")

# Make sure we have a camera
if bpy.context.scene.camera is None:
    # Find camera in scene
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            bpy.context.scene.camera = obj
            break

# Set viewport shading to solid
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = 'SOLID'
                space.shading.color_type = 'MATERIAL'
                break

# Use workbench engine for simple rendering
scene = bpy.context.scene
scene.render.engine = 'BLENDER_WORKBENCH'

# Set up render output
output_path = "/home/codespace/.gemini/antigravity/brain/ec7f7345-db56-4a74-a32a-d8d828d0a370/bento_preview_v1.png"
scene.render.filepath = output_path
scene.render.image_settings.file_format = 'PNG'
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080

# Workbench settings
scene.display.shading.light = 'STUDIO'
scene.display.shading.color_type = 'MATERIAL'

# Render
print("Rendering preview with Workbench...")
try:
    bpy.ops.render.render(write_still=True)
    print(f"Preview saved to: {output_path}")
    
    # Verify file exists
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print(f"File size: {size} bytes")
    else:
        print("ERROR: File not created!")
except Exception as e:
    print(f"Rendering failed: {e}")
    
    # Try opengl render as fallback
    print("Trying OpenGL render...")
    try:
        bpy.ops.render.opengl(write_still=True)
        print(f"OpenGL render saved to: {output_path}")
    except Exception as e2:
        print(f"OpenGL render also failed: {e2}")
