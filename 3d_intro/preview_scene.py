import bpy

# Open existing scene
bpy.ops.wm.open_mainfile(filepath="/tmp/bento_intro_v1_animated.blend")

scene = bpy.context.scene

# LOW QUALITY for quick preview
scene.render.resolution_x = 540  # Half of 1080
scene.render.resolution_y = 960  # Half of 1920
scene.render.engine = 'CYCLES'
scene.cycles.samples = 16  # Very low for speed
scene.cycles.device = 'CPU'
scene.cycles.use_denoising = False

# Render frame at 1 second (frame 30 at 30fps)
scene.frame_set(30)
scene.render.filepath = '/tmp/scene_preview_1s.png'

print(f"Rendering frame 30 (1.0s) at {scene.render.resolution_x}x{scene.render.resolution_y}")
print(f"Samples: {scene.cycles.samples}")
print(f"Objects in scene: {len(bpy.data.objects)}")

bpy.ops.render.render(write_still=True)
print(f"✓ Saved preview: {scene.render.filepath}")
