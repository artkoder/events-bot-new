#!/usr/bin/env python3
"""
Static Scene Test - No Animation
"""
import sys
import os
import bpy
import mathutils
import math

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from generate_intro import generate_bento_scene

def main():
    print("=== STATIC CAMERA TEST ===")
    
    # 1. Generate Scene
    cubes, camera = generate_bento_scene(poster_textures=None)
    
    # TEST: Add a giant plane to be SURE we see geometry
    bpy.ops.mesh.primitive_plane_add(size=50, location=(0, 0, -8))
    plane = bpy.context.active_object
    mat_plane = bpy.data.materials.new(name="TestPlane")
    mat_plane.use_nodes = True
    bsdf = mat_plane.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.0, 1.0, 0.0, 1.0) # GREEN Plane
    plane.data.materials.append(mat_plane)
    
    # 2. Force Camera Position (Overview)
    # Position: High up and back
    camera.animation_data_clear()
    camera.location = (10, -10, 10)
    
    # Force Look At
    # Point to origin (5, 3, -4) approx center of grid (based on bento_scene.py logic)
    # Target: (5, 3, -4)
    target = mathutils.Vector((5, 3, -4))
    direction = target - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    
    print(f"Camera forced to: {camera.location}")
    print(f"Looking at: {target}")
    
    # 3. Set World to WHITE (so we see void as white, objects as colored)
    world = bpy.data.worlds.new("DebugWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0) # White
        bg.inputs["Strength"].default_value = 1.0
        
    # 4. Render
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 32
    bpy.context.scene.cycles.use_denoising = False # CRITICAL fix for headless
    bpy.context.scene.render.resolution_x = 540
    bpy.context.scene.render.resolution_y = 960
    
    out_path = "/home/codespace/.gemini/antigravity/brain/5787ba91-955f-49ed-be99-0e6994f8f7a0/static_test.png"
    bpy.context.scene.render.filepath = out_path
    
    print(f"Rendering to {out_path}...")
    bpy.ops.render.render(write_still=True)
    print("Done.")

if __name__ == "__main__":
    main()
