#!/usr/bin/env python3
"""
Complete 3D Intro Generation Script
Combines scene generation, textures, and animation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import bpy
from bento_scene import generate_bento_scene, BENTO_LAYOUT
from animation_system import CameraAnimation, PosterAnimation

def generate_complete_intro(poster_textures: dict = None, 
                            selected_poster: str = 'poster_6',
                            output_path: str = "/tmp/bento_intro_complete.blend"):
    """Generate complete 3D intro with animation
    
    Args:
        poster_textures: Dict mapping poster names to texture paths
        selected_poster: Name of poster cube to zoom into in Phase 4
        output_path: Where to save the .blend file
    """
    
    print("=" * 70)
    print("3D BENTO INTRO GENERATION")
    print("=" * 70)
    
    # Step 1: Generate scene with textures
    print("\n[1/3] Generating Bento grid scene with textures...")
    cubes, camera = generate_bento_scene(poster_textures=poster_textures)
    print(f"✓ Created {len(cubes)} cubes")
    
    # Step 2: Setup animation
    print("\n[2/3] Setting up 5-phase camera animation...")
    
    # Find main cube and selected poster cube positions
    main_cube = None
    selected_cube = None
    
    for cube in cubes:
        if cube.get('type') == 'main':
            main_cube = cube
        if cube.name == selected_poster:
            selected_cube = cube
    
    if not main_cube:
        print("ERROR: Main cube not found!")
        return
    
    if not selected_cube:
        print(f"WARNING: Selected poster '{selected_poster}' not found, using first poster")
        for cube in cubes:
            if cube.get('type') == 'poster':
                selected_cube = cube
                break
    
    main_pos = tuple(main_cube.location)
    selected_pos = tuple(selected_cube.location)
    
    print(f"  Main cube at: {main_pos}")
    print(f"  Selected cube at: {selected_pos}")
    
    # Create animation
    cam_anim = CameraAnimation(camera, fps=30)
    cam_anim.setup_5_phase_animation(main_pos, selected_pos)
    
    print("✓ Animation keyframes set")
    print("  Phase 1 (0.0-0.4s): Close-up + subtle float")
    print("  Phase 2 (0.4-0.6s): Pullback (ease-in-out-cubic)")
    print("  Phase 3 (0.6-1.2s): Rotation (ease-in-out-quart)")
    print("  Phase 4 (1.2-1.8s): Zoom to poster (ease-in-expo)")
    print("  Phase 5 (1.8-2.5s): Transition (ease-out-back)")
    
    # Step 3: Configure scene for rendering
    print("\n[3/3] Configuring render settings...")
    
    # DEBUG: Check visibility
    cam = bpy.context.scene.camera
    if cam:
        print(f"DEBUG_INFO: Camera location: {cam.location}")
        print(f"DEBUG_INFO: Camera rotation: {cam.rotation_euler}")
        
    visible_objects = 0
    for obj in bpy.context.scene.objects:
        if obj.type in ['MESH', 'FONT', 'CURVE'] and not obj.hide_render:
            visible_objects += 1
            
    print(f"DEBUG_INFO: Scene has {visible_objects} renderable objects.")
    
    if visible_objects == 0:
        print("CRITICAL WARNING: SCENE IS EMPTY! No renderable objects found.")
    
    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = int(2.5 * 30)  # 2.5 seconds at 30 fps = 75 frames
    scene.render.fps = 30
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.film_transparent = False
    
    # Use CYCLES for headless compatibility
    scene.render.engine = 'CYCLES'
    scene.eevee.taa_render_samples = 64
    scene.eevee.use_gtao = True
    scene.eevee.use_ssr = True
    scene.eevee.use_bloom = False
    
    print("✓ Render settings configured")
    print(f"  Engine: {scene.render.engine}")
    print(f"  Resolution: {scene.render.resolution_x}x{scene.render.resolution_y}")
    print(f"  FPS: {scene.render.fps}")
    print(f"  Duration: {scene.frame_end / scene.render.fps:.1f} seconds")
    
    # Save blend file
    bpy.ops.wm.save_as_mainfile(filepath=output_path)
    print(f"\n✓ Scene saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Open the .blend file in Blender to preview animation")
    print("  2. Render animation: Render > Render Animation")
    print("  3. Or render via command: blender -b <file> -a")
    
    return cubes, camera

if __name__ == "__main__":
    # Define poster textures
    POSTER_DIR = "/workspaces/events-bot-new/3d_intro/assets/posters"
    
    poster_textures = {
        'poster_1': os.path.join(POSTER_DIR, 'poster_example_1.jpg'),
        'poster_2': os.path.join(POSTER_DIR, 'poster_example_2.jpg'),
        'poster_3': os.path.join(POSTER_DIR, 'poster_example_3.jpg'),
        'poster_4': os.path.join(POSTER_DIR, 'poster_example_4.png'),
        'poster_5': os.path.join(POSTER_DIR, 'poster_example_5.png'),
        'poster_6': os.path.join(POSTER_DIR, 'poster_example_1.jpg'),
        'poster_7': os.path.join(POSTER_DIR, 'poster_example_2.jpg'),
        'poster_8': os.path.join(POSTER_DIR, 'poster_example_3.jpg'),
        'poster_9': os.path.join(POSTER_DIR, 'poster_example_4.png'),
        'poster_10': os.path.join(POSTER_DIR, 'poster_example_5.png'),
        'poster_11': os.path.join(POSTER_DIR, 'poster_example_1.jpg'),
        'poster_12': os.path.join(POSTER_DIR, 'poster_example_2.jpg'),
    }
    
    # Filter to existing files
    poster_textures = {k: v for k, v in poster_textures.items() if os.path.exists(v)}
    
    # Generate complete intro
    generate_complete_intro(
        poster_textures=poster_textures,
        selected_poster='poster_6',  # Choose which poster to zoom into
        output_path="/tmp/bento_intro_v1_animated.blend"
    )
