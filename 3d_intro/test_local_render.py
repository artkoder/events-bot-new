import bpy
import math

print("=== Fixed Camera Tracking ===")

bpy.ops.wm.read_factory_settings(use_empty=True)

# Yellow cube at origin
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0), size=2)
cube = bpy.context.object
cube.name = "TargetCube"

mat = bpy.data.materials.new(name="Yellow")
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
if bsdf:
    bsdf.inputs["Base Color"].default_value = (1.0, 0.95, 0.3, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.5

cube.data.materials.append(mat)
print(f"Cube at: {tuple(cube.location)}")

# Camera
bpy.ops.object.camera_add(location=(4, -4, 3))
camera = bpy.context.object

# CRITICAL FIX: Use Track To constraint instead of manual rotation
constraint = camera.constraints.new(type='TRACK_TO')
constraint.target = cube
constraint.track_axis = 'TRACK_NEGATIVE_Z'
constraint.up_axis = 'UP_Y'

bpy.context.scene.camera = camera
print(f"Camera at: {tuple(camera.location)}")
print(f"Camera tracking: {cube.name}")

# Lights
bpy.ops.object.light_add(type='SUN', location=(5, -3, 8))
sun = bpy.context.object
sun.data.energy = 2.0

bpy.ops.object.light_add(type='AREA', location=(-3, 3, 4))
area = bpy.context.object
area.data.energy = 100
area.data.size = 5

# World
scene = bpy.context.scene
if not scene.world:
    world = bpy.data.worlds.new("World")
    scene.world = world
else:
    world = scene.world

world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
if bg:
    bg.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)  # Mid gray
    bg.inputs["Strength"].default_value = 0.8

# CYCLES
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.engine = 'CYCLES'
scene.cycles.samples = 64
scene.cycles.device = 'CPU'
scene.cycles.use_denoising = False
scene.render.filepath = '/tmp/fixed_camera_render.png'

print("\nRendering with FIXED camera tracking...")
bpy.ops.render.render(write_still=True)
print(f"✓ Saved: {scene.render.filepath}")
