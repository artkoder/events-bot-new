#!/usr/bin/env python3
"""
Visualize Bento scene using matplotlib
"""

import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# Load scene info
with open("/home/codespace/.gemini/antigravity/brain/ec7f7345-db56-4a74-a32a-d8d828d0a370/bento_scene_info.json", 'r') as f:
    scene_info = json.load(f)

# Create figure
fig = plt.figure(figsize=(16, 12))

# Create 3D subplots
ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222, projection='3d')
ax3 = fig.add_subplot(223, projection='3d')
ax4 = fig.add_subplot(224, projection='3d')

axes = [ax1, ax2, ax3, ax4]
titles = [
    "View from Camera (Right Corner)",
    "Top View",
    "Side View (showing Z-depth)",
    "Front View"
]
elevations = [20, 90, 0, 0]
azimuths = [-45, -90, -90, 0]

def create_cube_vertices(center, scale):
    """Create vertices for a cube"""
    x, y, z = center
    sx, sy, sz = scale
    
    # Define cube vertices (8 corners)
    vertices = [
        [x - sx, y - sy, z - sz],
        [x + sx, y - sy, z - sz],
        [x + sx, y + sy, z - sz],
        [x - sx, y + sy, z - sz],
        [x - sx, y - sy, z + sz],
        [x + sx, y - sy, z + sz],
        [x + sx, y + sy, z + sz],
        [x - sx, y + sy, z + sz],
    ]
    
    return np.array(vertices)

def plot_cube(ax, center, scale, cube_type, name):
    """Plot a single cube"""
    vertices = create_cube_vertices(center, scale)
    
    # Define the 6 faces of the cube
    faces = [
        [vertices[i] for i in [0, 1, 5, 4]],  # -Y face
        [vertices[i] for i in [2, 3, 7, 6]],  # +Y face
        [vertices[i] for i in [0, 3, 2, 1]],  # -Z face
        [vertices[i] for i in [4, 5, 6, 7]],  # +Z face
        [vertices[i] for i in [0, 4, 7, 3]],  # -X face
        [vertices[i] for i in [1, 2, 6, 5]],  # +X face
    ]
    
    # Color based on type
    colors = {
        'main': '#F5D911',      # Yellow for main
        'info': '#F5D911',      # Yellow for info
        'poster': '#333333',    # Dark for posters
        'filler': '#F5D911',    # Yellow for fillers
        'unknown': '#CCCCCC'    # Gray for unknown
    }
    
    color = colors.get(cube_type, colors['unknown'])
    
    # Create 3D polygon collection
    cube_collection = Poly3DCollection(faces, alpha=0.7, facecolor=color, edgecolor='black', linewidth=0.5)
    ax.add_collection3d(cube_collection)
    
    # Add label for main cubes
    if cube_type == 'main' or 'info' in name:
        ax.text(center[0], center[1], center[2], name.replace('_', '\n'), 
                fontsize=6, ha='center', va='center')

# Plot scene in all views
for ax, title, elev, azim in zip(axes, titles, elevations, azimuths):
    # Plot all cubes
    for cube in scene_info['cubes']:
        plot_cube(ax, cube['location'], cube['scale'], cube['type'], cube['name'])
    
    # Plot camera position
    cam_loc = scene_info['camera']['location']
    ax.scatter(*cam_loc, c='red', s=100, marker='^', label='Camera')
    
    # Plot light positions
    for light in scene_info['lights']:
        ax.scatter(*light['location'], c='yellow', s=50, marker='*', label=light['name'])
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (depth)')
    ax.set_title(title)
    
    # Set viewing angle
    ax.view_init(elev=elev, azim=azim)
    
    # Set axis limits
    ax.set_xlim(-5, 15)
    ax.set_ylim(-5, 15)
    ax.set_zlim(-10, 10)
    
    # Add grid
    ax.grid(True, alpha=0.3)

# Add legend to first plot
ax1.legend(fontsize=8, loc='upper right')

plt.tight_layout()

# Save figure
output_path = "/home/codespace/.gemini/antigravity/brain/ec7f7345-db56-4a74-a32a-d8d828d0a370/bento_scene_visualization.png"
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"Visualization saved to: {output_path}")

# Print summary
print(f"\nScene Summary:")
print(f"- Total cubes: {len(scene_info['cubes'])}")
print(f"- Camera at: {cam_loc}")
print(f"- Z-depth range: {min(c['location'][2] for c in scene_info['cubes']):.1f} to {max(c['location'][2] for c in scene_info['cubes']):.1f}")

# Count by type
from collections import Counter
type_counts = Counter(c['type'] for c in scene_info['cubes'])
print(f"- Cube types: {dict(type_counts)}")
