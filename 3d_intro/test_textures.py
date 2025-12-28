#!/usr/bin/env python3
"""
Test script for texture system
Generates a scene with real poster textures
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

import bpy
from bento_scene import generate_bento_scene

# Define poster textures (using example posters)
POSTER_DIR = "/workspaces/events-bot-new/3d_intro/assets/posters"

poster_textures = {
    'poster_1': os.path.join(POSTER_DIR, 'poster_example_1.jpg'),
    'poster_2': os.path.join(POSTER_DIR, 'poster_example_2.jpg'),
    'poster_3': os.path.join(POSTER_DIR, 'poster_example_3.jpg'),
    'poster_4': os.path.join(POSTER_DIR, 'poster_example_4.png'),
    'poster_5': os.path.join(POSTER_DIR, 'poster_example_5.png'),
    'poster_6': os.path.join(POSTER_DIR, 'poster_example_1.jpg'),  # Repeat for now
    'poster_7': os.path.join(POSTER_DIR, 'poster_example_2.jpg'),
    'poster_8': os.path.join(POSTER_DIR, 'poster_example_3.jpg'),
    'poster_9': os.path.join(POSTER_DIR, 'poster_example_4.png'),
    'poster_10': os.path.join(POSTER_DIR, 'poster_example_5.png'),
    'poster_11': os.path.join(POSTER_DIR, 'poster_example_1.jpg'),
    'poster_12': os.path.join(POSTER_DIR, 'poster_example_2.jpg'),
}

# Filter to only existing files
poster_textures = {k: v for k, v in poster_textures.items() if os.path.exists(v)}

print(f"Loading {len(poster_textures)} poster textures...")
for name, path in poster_textures.items():
    print(f"  {name}: {os.path.basename(path)}")

# Generate scene with textures
cubes, camera = generate_bento_scene(poster_textures=poster_textures)

# Save the blend file
output_path = "/tmp/bento_scene_v2_textured.blend"
bpy.ops.wm.save_as_mainfile(filepath=output_path)
print(f"\nScene saved to: {output_path}")
print(f"Total cubes: {len(cubes)}")
print(f"Textured cubes: {len(poster_textures)}")
