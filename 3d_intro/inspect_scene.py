import bpy

# Open the file
bpy.ops.wm.open_mainfile(filepath="/tmp/bento_intro_v1_animated.blend")

print("=== SCENE DESCRIPTION ===")
print(f"Scene Name: {bpy.context.scene.name}")
print(f"Frame: {bpy.context.scene.frame_current}")

# List visible objects
print("\n--- Visible Objects ---")
for obj in bpy.context.scene.objects:
    if not obj.hide_render and obj.type == 'MESH':
        # Get location
        loc = obj.location
        # Simple check if "in front" of camera (roughly)
        print(f"- {obj.name} ({obj.type}) at {loc}")
        
        # Check materials
        if obj.material_slots:
            mats = [s.name for s in obj.material_slots if s.material]
            print(f"  Materials: {mats}")

# Camera info
cam = bpy.context.scene.camera
if cam:
    print(f"\n--- Camera ---")
    print(f"Name: {cam.name}")
    print(f"Location: {cam.location}")
    print(f"Rotation: {cam.rotation_euler}")
