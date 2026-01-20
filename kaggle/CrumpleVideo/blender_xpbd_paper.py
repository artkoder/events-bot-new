import math
import os
import sys
import json
import hashlib
import shutil
import subprocess
import tempfile

import bpy
from mathutils import Vector


def _parse_args():
    argv = sys.argv
    if "--" not in argv:
        return {}
    args = argv[argv.index("--") + 1 :]
    out = {}
    i = 0
    while i < len(args):
        k = args[i]
        if not k.startswith("--"):
            i += 1
            continue
        k = k[2:]
        v = True
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            v = args[i + 1]
            i += 1
        out[k] = v
        i += 1
    return out


def _as_int(d, k, default):
    try:
        return int(d.get(k, default))
    except Exception:
        return default


def _as_float(d, k, default):
    try:
        return float(d.get(k, default))
    except Exception:
        return default


def _as_str(d, k, default):
    v = d.get(k, default)
    return str(v) if v is not None else default


BLENDER_VERSION = tuple(getattr(bpy.app, "version", (0, 0, 0)))
LEGACY_BLENDER = BLENDER_VERSION < (3, 0, 0)


def _is_legacy_blender():
    return LEGACY_BLENDER


def _try_set_enum(obj, attr, value):
    try:
        prop = obj.bl_rna.properties[attr]
    except Exception:
        return False
    if value in prop.enum_items.keys():
        setattr(obj, attr, value)
        return True
    return False


def _ensure_cycles(scene):
    scene.render.engine = "CYCLES"
    if scene.render.engine != "CYCLES":
        try:
            bpy.ops.preferences.addon_enable(module="cycles")
        except Exception:
            pass
        try:
            scene.render.engine = "CYCLES"
        except Exception:
            pass


def _setup_world(scene, legacy=False):
    if not scene.world:
        scene.world = bpy.data.worlds.new("World")
    world = scene.world
    world.use_nodes = True
    nt = world.node_tree
    nodes = nt.nodes
    links = nt.links
    bg = None
    out = None
    for node in nodes:
        if node.type == "BACKGROUND":
            bg = node
        elif node.type == "OUTPUT_WORLD":
            out = node
    if bg is None:
        bg = nodes.new("ShaderNodeBackground")
    if out is None:
        out = nodes.new("ShaderNodeOutputWorld")
    if not any(link.from_node == bg and link.to_node == out for link in links):
        links.new(bg.outputs["Background"], out.inputs["Surface"])
    bg.inputs[0].default_value = (0.02, 0.02, 0.02, 1)
    try:
        bg.inputs["Strength"].default_value = 0.4 if legacy else 0.02
    except Exception:
        pass


def setup_scene(res_x=864, res_y=1104, samples=64, pct=100, legacy=False):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    _ensure_cycles(scene)
    if hasattr(scene, "cycles"):
        scene.cycles.device = "CPU"
        scene.cycles.samples = samples
        if hasattr(scene.cycles, "use_adaptive_sampling"):
            scene.cycles.use_adaptive_sampling = False
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.resolution_percentage = pct
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    # Render paper with alpha so we can composite a background and also validate 'paper-only' pixels.
    scene.render.film_transparent = True
    # Keep colors stable for poster matching (avoid Filmic tonemapping).
    _try_set_enum(scene.view_settings, "view_transform", "Standard")
    _try_set_enum(scene.view_settings, "look", "None")
    _setup_world(scene, legacy=legacy)


def _point_camera_at(cam, target):
    direction = target - cam.location
    if direction.length < 1e-9:
        return
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_camera_and_light(paper_width=2.0, paper_height=2.0, render_aspect=1.0, legacy=False):
    bpy.ops.object.camera_add(location=(0, -3.0, 0.05))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    cam.data.type = "ORTHO"
    cam.data.clip_start = 0.01
    cam.data.clip_end = 100.0
    # Scale by critical dimension (Codex-recommended algorithm):
    # Compare poster aspect ratio with render aspect ratio to determine
    # which dimension (width or height) limits fitting the poster
    margin = 1.05  # minimal safety buffer
    poster_aspect = paper_width / paper_height
    
    if poster_aspect > render_aspect:
        # Poster is wider than video frame → limited by WIDTH
        ortho_scale = (paper_width / render_aspect) * margin
    else:
        # Poster is taller than video frame → limited by HEIGHT
        ortho_scale = paper_height * margin
    
    cam.data.ortho_scale = ortho_scale

    if legacy:
        _point_camera_at(cam, Vector((0.0, 0.0, 0.0)))
    else:
        bpy.ops.object.empty_add(location=(0, 0, 0))
        target = bpy.context.object
        const = cam.constraints.new(type="DAMPED_TRACK")
        const.target = target
        const.track_axis = "TRACK_NEGATIVE_Z"
    bpy.context.view_layer.update()

    # Side-ish key light similar to reference
    light_mul = 2.2 if legacy else 1.0
    bpy.ops.object.light_add(type="AREA", location=(-2.2, -2.4, 1.8))
    key = bpy.context.object
    key.data.energy = 450 * light_mul  # Reduced to prevent overexposure on creases
    key.data.size = 1.6
    key.rotation_euler = (math.radians(65), math.radians(-10), math.radians(25))

    bpy.ops.object.light_add(type="AREA", location=(1.8, -2.8, 1.4))
    fill = bpy.context.object
    fill.data.energy = 150 * light_mul  # Reduced for balanced fill
    fill.data.size = 2.2
    fill.rotation_euler = (math.radians(70), math.radians(0), math.radians(-25))
    if legacy:
        bpy.ops.object.light_add(type="SUN", location=(0.0, -1.2, 3.2))
        sun = bpy.context.object
        sun.data.energy = 1.8
        try:
            sun.data.angle = math.radians(4.0)
        except Exception:
            pass
        sun.rotation_euler = (math.radians(55), 0.0, math.radians(10))


def build_sheet(aspect, nx, ny, size=2.0):
    width = size * aspect
    height = size
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=nx,
        y_subdivisions=ny,
        size=size,
        enter_editmode=False,
        location=(0, 0, 0),
    )
    obj = bpy.context.object
    obj.name = "PaperSheet"

    obj.scale[0] = width / 2.0
    obj.scale[1] = height / 2.0
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    obj.rotation_euler = (math.radians(90), 0, 0)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    # Triangulate for hinge-based crease constraints.
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")

    # Render smoother surface while keeping the simulation on the base mesh.
    # Too much subsurf makes the sheet read like cloth. Keep it low for sharper paper folds.
    subsurf = obj.modifiers.new(name="Subsurf", type="SUBSURF")
    subsurf.levels = 1
    subsurf.render_levels = 1
    return obj


def apply_uv_map_xz(obj):
    mesh = obj.data
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="UVMap")
    uv_layer = mesh.uv_layers.active.data

    xs = [v.co.x for v in mesh.vertices]
    zs = [v.co.z for v in mesh.vertices]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)
    dx = max(max_x - min_x, 1e-9)
    dz = max(max_z - min_z, 1e-9)

    for poly in mesh.polygons:
        for li in poly.loop_indices:
            vi = mesh.loops[li].vertex_index
            co = mesh.vertices[vi].co
            u = (co.x - min_x) / dx
            v = (co.z - min_z) / dz
            uv_layer[li].uv = (u, v)


def _resolve_image_path(image_path: str):
    if not image_path:
        return ""
    candidates = [image_path]
    if not os.path.isabs(image_path):
        candidates.append(os.path.abspath(image_path))
        candidates.append(os.path.join(os.path.dirname(__file__), image_path))
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return image_path


def _convert_image_for_legacy(image_path: str, legacy: bool):
    if not legacy or not image_path:
        return image_path
    ext = os.path.splitext(image_path)[1].lower()
    if ext != ".webp":
        return image_path
    if not os.path.exists(image_path):
        return image_path
    digest = hashlib.md5(image_path.encode("utf-8")).hexdigest()[:8]
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(tempfile.gettempdir(), f"{base}_{digest}.png")
    if os.path.exists(out_path):
        return out_path

    converters = []
    if shutil.which("ffmpeg"):
        converters.append(["ffmpeg", "-y", "-i", image_path, "-frames:v", "1", out_path])
    if shutil.which("dwebp"):
        converters.append(["dwebp", image_path, "-o", out_path])
    if shutil.which("magick"):
        converters.append(["magick", image_path, out_path])
    if shutil.which("convert"):
        converters.append(["convert", image_path, out_path])

    for cmd in converters:
        try:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(out_path):
                return out_path
        except Exception:
            continue
    return image_path


def _load_image(image_path: str):
    if not image_path:
        return None
    try:
        img = bpy.data.images.load(image_path, check_existing=True)
    except Exception:
        return None
    try:
        img.colorspace_settings.name = "sRGB"
    except Exception:
        pass
    try:
        img.alpha_mode = "STRAIGHT"
    except Exception:
        pass
    return img


def apply_two_sided_paper_material(obj, image_path: str, legacy=False):
    resolved = _resolve_image_path(image_path)
    resolved = _convert_image_for_legacy(resolved, legacy)
    img = _load_image(resolved)

    mat = bpy.data.materials.new(name="PaperMat")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    out = nt.nodes.new("ShaderNodeOutputMaterial")

    geo = nt.nodes.new("ShaderNodeNewGeometry")
    mix = nt.nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MIX"

    tex = nt.nodes.new("ShaderNodeTexImage")
    if img is not None:
        tex.image = img
        tex.interpolation = "Closest"

    # Backfacing==1 -> white, else poster
    nt.links.new(geo.outputs["Backfacing"], mix.inputs["Fac"])
    if img is not None:
        nt.links.new(tex.outputs["Color"], mix.inputs["Color1"])
    else:
        mix.inputs["Color1"].default_value = (0.8, 0.8, 0.8, 1.0)
    mix.inputs["Color2"].default_value = (1.0, 1.0, 1.0, 1.0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.95
    bsdf.inputs["Specular"].default_value = 0.02
    nt.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])

    emission = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(mix.outputs["Color"], emission.inputs["Color"])
    emission.inputs["Strength"].default_value = 1.0

    unlit = nt.nodes.new("ShaderNodeValue")
    unlit.name = "UNLIT_FACTOR"
    unlit.outputs[0].default_value = 0.3 if legacy else 0.0

    mix_shader = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(unlit.outputs[0], mix_shader.inputs["Fac"])
    nt.links.new(bsdf.outputs["BSDF"], mix_shader.inputs[1])
    nt.links.new(emission.outputs["Emission"], mix_shader.inputs[2])
    nt.links.new(mix_shader.outputs["Shader"], out.inputs["Surface"])

    obj.data.materials.append(mat)


def main():
    args = _parse_args()
    poster_path = _as_str(args, "poster", None)
    out_dir = _as_str(args, "out_dir", "attempt_blender_xpbd_001/out_single")

    frame_start = _as_int(args, "frame_start", 33)
    frame_end = _as_int(args, "frame_end", 44)
    fps = _as_float(args, "fps", 24.0)
    dt_scale = _as_float(args, "dt_scale", 0.6)

    nx = _as_int(args, "nx", 35)
    ny = _as_int(args, "ny", 45)
    samples = _as_int(args, "samples", 48)
    pct = _as_int(args, "pct", 100)

    # XPBD parameters (tunable)
    pre_roll = _as_int(args, "pre_roll", 50)
    substeps = _as_int(args, "substeps", 5)
    iters = _as_int(args, "iters", 8)
    attract0 = _as_float(args, "attract0", 12.0)
    attract_tau = _as_float(args, "attract_tau", 0.6)
    # Note: `unfold_alpha` is applied multiple times per frame (per solver iteration),
    # so the effective pull-to-rest is much stronger than the raw value. Keep it small.
    unfold_max = _as_float(args, "unfold_max", 0.65)  # Increased from 0.35 for fuller unfold
    unfold_power = _as_float(args, "unfold_power", 1.2)
    unfold_gamma = _as_float(args, "unfold_gamma", 0.9)
    crease_tau = _as_float(args, "crease_tau", 1.4)
    unfold_delay = _as_float(args, "unfold_delay", 0.15)
    seed = _as_int(args, "seed", 7)
    # Keep only the final frame fully unlit so frame_0044 ≈ poster, while 43 stays visibly different.
    unlit_final_frames = _as_int(args, "unlit_final_frames", 1)

    os.makedirs(out_dir, exist_ok=True)
    print(f"[xpbd] pre_roll={pre_roll} attract0={attract0:.3f} attract_tau={attract_tau:.3f}")

    legacy = _is_legacy_blender()
    res_x = _as_int(args, "res_x", 864)
    res_y = _as_int(args, "res_y", 1104)
    setup_scene(res_x=res_x, res_y=res_y, samples=samples, pct=pct, legacy=legacy)
    # Note: setup_camera_and_light() will be called after building sheet
    bpy.context.scene.render.fps = int(round(fps))

    # Read poster aspect ratio from source image
    poster_aspect = 1.0  # default square
    if poster_path:
        poster_path = _resolve_image_path(poster_path)
        poster_path = _convert_image_for_legacy(poster_path, legacy)
    if poster_path and os.path.exists(poster_path):
        try:
            img = _load_image(poster_path)
            poster_aspect = float(img.size[0]) / max(1.0, float(img.size[1]))
        except Exception:
            poster_aspect = 1.0

    # Create paper with POSTER aspect (not fixed render aspect)
    # This ensures each poster has correct proportions: square/landscape/portrait
    sheet = build_sheet(aspect=poster_aspect, nx=nx, ny=ny, size=2.0)
    apply_uv_map_xz(sheet)
    
    # Calculate actual paper dimensions for camera framing
    # IMPORTANT: Must match build_sheet logic exactly!
    # build_sheet always uses: width = size * aspect, height = size
    paper_width = 2.0 * poster_aspect
    paper_height = 2.0
    
    # Calculate render aspect ratio for camera framing
    render_aspect = float(res_x) / float(res_y)  # e.g., 864/1104 = 0.783
    
    # Setup camera with adaptive framing - critical for all poster orientations!
    setup_camera_and_light(
        paper_width=paper_width,
        paper_height=paper_height,
        render_aspect=render_aspect,
        legacy=legacy,
    )
    
    apply_two_sided_paper_material(sheet, image_path=poster_path if poster_path else "", legacy=legacy)
    bpy.ops.object.shade_smooth()

    # Reuse the existing XPBD integrator from simulate_paper_xpbd.py by embedding the minimal parts here:
    import random
    from mathutils import Matrix

    mesh = sheet.data
    rest = [v.co.copy() for v in mesh.vertices]

    # edges
    edge_map = {}
    for e in mesh.edges:
        a, b = e.vertices[0], e.vertices[1]
        if a == b:
            continue
        k = (a, b) if a < b else (b, a)
        if k in edge_map:
            continue
        l0 = (rest[k[1]] - rest[k[0]]).length
        edge_map[k] = l0
    edges = [(a, b, l0) for (a, b), l0 in edge_map.items()]

    # hinges
    adjacent = {}
    for poly in mesh.polygons:
        if poly.loop_total != 3:
            continue
        a, b, c = poly.vertices[:]
        tri = (a, b, c)
        for (i, j, opp) in ((tri[0], tri[1], tri[2]), (tri[1], tri[2], tri[0]), (tri[2], tri[0], tri[1])):
            k = (i, j) if i < j else (j, i)
            adjacent.setdefault(k, []).append(opp)
    hinges = []
    for (i, j), opps in adjacent.items():
        if len(opps) != 2:
            continue
        k, l = opps[0], opps[1]
        if k == l or k in (i, j) or l in (i, j):
            continue
        hinges.append((i, j, k, l))

    rnd = random.Random(seed)
    v = [Vector((0.0, 0.0, 0.0)) for _ in rest]
    x = [p.copy() for p in rest]

    # Small initial perturbation breaks symmetry (helps create center folds).
    for i in range(len(x)):
        x[i].y += rnd.uniform(-0.01, 0.01)
        x[i].z += rnd.uniform(-0.01, 0.01)

    # Create crease targets; bias some through the center to match reference.
    crease_angle_rad = math.radians(135.0)
    crease_targets = {}
    if hinges:
        xs = [p.x for p in rest]
        zs = [p.z for p in rest]
        w = max(xs) - min(xs)
        h = max(zs) - min(zs)
        cx, cz = 0.5 * (max(xs) + min(xs)), 0.5 * (max(zs) + min(zs))

        def hinge_mid(hg):
            i, j, k, l = hg
            p = (rest[i] + rest[j] + rest[k] + rest[l]) * 0.25
            return p.x, p.z

        center_hinges = []
        other_hinges = []
        for hg in hinges:
            mx, mz = hinge_mid(hg)
            if abs(mx - cx) < 0.16 * w and abs(mz - cz) < 0.16 * h:
                center_hinges.append(hg)
            else:
                other_hinges.append(hg)

        for hg in rnd.sample(center_hinges, min(len(center_hinges), 220)):
            crease_targets[hg] = rnd.choice([-1.0, 1.0]) * crease_angle_rad
        for hg in rnd.sample(other_hinges, min(len(other_hinges), 360)):
            if hg not in crease_targets:
                crease_targets[hg] = rnd.choice([-1.0, 1.0]) * crease_angle_rad

    # Stiffer settings -> reads as paper/cardboard rather than cloth.
    stretch_stiffness = 0.985
    hinge_stiffness = 0.95
    hinge_max_correction_rad = math.radians(10.0)
    damping = 0.08
    gravity = -0.12

    def dihedral(xi, xj, xk, xl):
        e = xj - xi
        el = e.length
        if el < 1e-12:
            return 0.0
        e = e / el
        n1 = (xj - xi).cross(xk - xi)
        n2 = (xi - xj).cross(xl - xj)
        if n1.length < 1e-12 or n2.length < 1e-12:
            return 0.0
        n1.normalize()
        n2.normalize()
        sinv = e.dot(n1.cross(n2))
        cosv = max(-1.0, min(1.0, n1.dot(n2)))
        return math.atan2(sinv, cosv)

    def rotate_around_axis(p, axis_point, axis_dir, angle):
        if abs(angle) < 1e-12:
            return p
        rot = Matrix.Rotation(angle, 4, axis_dir)
        return axis_point + (rot @ (p - axis_point))

    def project_edges():
        for (a, b, l0) in edges:
            d = x[b] - x[a]
            ln = d.length
            if ln < 1e-9:
                continue
            err = (ln - l0) / ln
            corr = stretch_stiffness * 0.5 * err * d
            x[a] = x[a] + corr
            x[b] = x[b] - corr

    def project_unfold(alpha):
        if alpha <= 0:
            return
        for i in range(len(x)):
            x[i] = (1.0 - alpha) * x[i] + alpha * rest[i]

    def project_creases(crease_scale):
        if crease_scale <= 0 or not crease_targets:
            return
        for (i, j, k, l), theta0 in crease_targets.items():
            xi, xj, xk, xl = x[i], x[j], x[k], x[l]
            e = xj - xi
            el = e.length
            if el < 1e-12:
                continue
            axis_dir = e / el
            theta = dihedral(xi, xj, xk, xl)
            theta_target = theta0 * crease_scale
            err = theta - theta_target
            corr = -hinge_stiffness * err
            corr = max(-hinge_max_correction_rad, min(hinge_max_correction_rad, corr))
            x[k] = rotate_around_axis(xk, xi, axis_dir, 0.5 * corr)
            x[l] = rotate_around_axis(xl, xi, axis_dir, -0.5 * corr)

    def step(dt, n_iters, crease_scale, unfold_alpha, attract_strength, center=Vector((0.0, 0.0, 0.0))):
        x_prev = [p.copy() for p in x]
        for i in range(len(x)):
            f = Vector((0.0, 0.0, 0.0))
            f.z += gravity
            if attract_strength > 0:
                f += -attract_strength * (x[i] - center)
            v[i] = (1.0 - damping) * (v[i] + dt * f)
            x[i] = x[i] + dt * v[i]
        for _ in range(n_iters):
            project_creases(crease_scale)
            project_edges()
        # Important: apply unfold only once per substep (not once per solver-iteration),
        # otherwise the sheet converges to rest too fast and many late frames become near-identical.
        project_unfold(unfold_alpha)
        for i in range(len(x)):
            v[i] = (x[i] - x_prev[i]) / max(1e-9, dt)

    dt = (1.0 / fps / max(1, substeps)) * max(0.1, min(2.0, dt_scale))

    def _max_delta():
        if not x:
            return 0.0
        return max((x[i] - rest[i]).length for i in range(len(x)))

    def _log_max_delta(tag, frame=None):
        max_delta = _max_delta()
        if frame is None:
            print(f"[xpbd] max_delta[{tag}]={max_delta:.6f}")
        else:
            print(f"[xpbd] frame={frame} max_delta[{tag}]={max_delta:.6f}")
        return max_delta

    # Calibrate "full open" coverage in camera space so that target_area values normalized
    # to the reference end-frame can be matched regardless of absolute framing.
    def _apply_state():
        for i, vert in enumerate(mesh.vertices):
            vert.co = x[i]
        mesh.update()
        bpy.context.view_layer.update()

    # Pre-roll to reach a compact crumpled state by frame_start
    # (paper is still at rest orientation before this loop).
    area_flat = None
    try:
        area_flat = None
        # At this moment x==rest (with tiny perturbations), so this is a close proxy for flat.
        _apply_state()
        # _area_fraction defined below; we set area_flat right after it exists.
    except Exception:
        area_flat = None

    for _ in range(max(0, pre_roll)):
        step(dt=dt, n_iters=iters, crease_scale=1.0, unfold_alpha=0.0, attract_strength=attract0)
    pre_roll_delta = _log_max_delta("pre_roll")
    min_crumple_delta = 0.02
    if pre_roll_delta < min_crumple_delta:
        boost_steps = max(12, int(pre_roll * 0.6))
        boost_attract = max(attract0 * 1.8, attract0 + 6.0)
        print(
            f"[xpbd] pre_roll too flat (max_delta={pre_roll_delta:.6f}); "
            f"boosting {boost_steps} steps at attract={boost_attract:.3f}"
        )
        for _ in range(boost_steps):
            step(dt=dt, n_iters=iters, crease_scale=1.0, unfold_alpha=0.0, attract_strength=boost_attract)
        _log_max_delta("pre_roll_boost")

    def _smoothstep(x):
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)

    def schedule(frame):
        # u=0 at frame_start (crumpled) -> u=1 at frame_end (flat)
        u = 0.0
        if frame_end > frame_start:
            u = (frame - frame_start) / (frame_end - frame_start)
        u = max(0.0, min(1.0, u))
        # Optional delay (kept for experimentation; set unfold_delay=0 for smooth motion).
        d = max(0.0, min(0.95, unfold_delay))
        u2 = 0.0 if u <= d else (u - d) / max(1e-9, (1.0 - d))
        # Use a power curve to shape unfolding timing across frames.
        g = max(0.5, float(unfold_gamma))
        s = max(0.0, min(1.0, u2)) ** g

        # Drive the sheet back to rest continuously (avoid plateaus).
        unfold_alpha = unfold_max * (s**unfold_power)
        unfold_alpha = max(0.0, min(0.95, unfold_alpha))

        # Relax creases gradually across all frames (not only at the end).
        crease_scale = math.exp(-s / max(1e-6, crease_tau))
        crease_scale = max(0.0, min(1.0, crease_scale))

        # Center attraction fades smoothly (keeps motion across the whole interval).
        attract = attract0 * math.exp(-s / max(1e-6, attract_tau))
        if u > 0.95:
            attract = 0.0
        print(
            f"[xpbd] schedule frame={frame} u={u:.3f} s={s:.3f} "
            f"attract={attract:.3f} unfold_alpha={unfold_alpha:.4f}"
        )
        return crease_scale, unfold_alpha, attract

    scene = bpy.context.scene
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    mat = sheet.active_material
    unlit_node = mat.node_tree.nodes.get("UNLIT_FACTOR") if mat else None
    legacy_unlit = 0.3 if legacy else 0.0

    target_area_path = _as_str(args, "target_area_path", "")
    target_areas = None
    if target_area_path and os.path.exists(target_area_path):
        try:
            raw = json.loads(open(target_area_path, "r", encoding="utf-8").read())
            target_areas = {int(k): float(v) for k, v in raw.items()}
        except Exception:
            target_areas = None

    def _area_fraction():
        # Approximate visible area by rasterizing the union of projected triangles.
        # This tracks the render alpha silhouette much better than a convex hull, and helps
        # keep "paper coverage" monotonic across frames.
        from bpy_extras.object_utils import world_to_camera_view

        cam = scene.camera
        mw = sheet.matrix_world
        w = max(48, int(round(160 * (res_x / max(1.0, res_y)))))
        h = 160

        # Project vertices once.
        proj = []
        for vert in mesh.vertices:
            uvw = world_to_camera_view(scene, cam, mw @ vert.co)
            proj.append((float(uvw.x), float(uvw.y)))

        mask = bytearray(w * h)

        def clamp_int(v, lo, hi):
            return lo if v < lo else hi if v > hi else v

        for poly in mesh.polygons:
            if poly.loop_total != 3:
                continue
            i0, i1, i2 = poly.vertices[:]
            x0, y0 = proj[i0]
            x1, y1 = proj[i1]
            x2, y2 = proj[i2]

            # Convert to pixel space (y-down).
            px0, py0 = x0 * (w - 1), (1.0 - y0) * (h - 1)
            px1, py1 = x1 * (w - 1), (1.0 - y1) * (h - 1)
            px2, py2 = x2 * (w - 1), (1.0 - y2) * (h - 1)

            minx = int(math.floor(min(px0, px1, px2)))
            maxx = int(math.ceil(max(px0, px1, px2)))
            miny = int(math.floor(min(py0, py1, py2)))
            maxy = int(math.ceil(max(py0, py1, py2)))
            if maxx < 0 or maxy < 0 or minx >= w or miny >= h:
                continue
            minx = clamp_int(minx, 0, w - 1)
            maxx = clamp_int(maxx, 0, w - 1)
            miny = clamp_int(miny, 0, h - 1)
            maxy = clamp_int(maxy, 0, h - 1)

            den = (py1 - py2) * (px0 - px2) + (px2 - px1) * (py0 - py2)
            if abs(den) < 1e-9:
                continue

            for yy in range(miny, maxy + 1):
                y = yy + 0.5
                for xx in range(minx, maxx + 1):
                    x = xx + 0.5
                    w0 = ((py1 - py2) * (x - px2) + (px2 - px1) * (y - py2)) / den
                    if w0 < 0.0:
                        continue
                    w1 = ((py2 - py0) * (x - px2) + (px0 - px2) * (y - py2)) / den
                    if w1 < 0.0:
                        continue
                    w2 = 1.0 - w0 - w1
                    if w2 < 0.0:
                        continue
                    mask[yy * w + xx] = 1

        covered = sum(mask)
        return float(covered / max(1, (w * h)))

    # Now that _area_fraction exists, finalize area_flat calibration.
    if area_flat is None:
        try:
            # Temporarily force a rest pose for calibration.
            saved = [p.copy() for p in x]
            for i in range(len(x)):
                x[i] = rest[i].copy()
            _apply_state()
            area_flat = max(1e-9, float(_area_fraction()))
            for i in range(len(x)):
                x[i] = saved[i].copy()
            _apply_state()
        except Exception:
            area_flat = 1.0
    else:
        # area_flat placeholder created above; compute actual value.
        try:
            saved = [p.copy() for p in x]
            for i in range(len(x)):
                x[i] = rest[i].copy()
            _apply_state()
            area_flat = max(1e-9, float(_area_fraction()))
            for i in range(len(x)):
                x[i] = saved[i].copy()
            _apply_state()
        except Exception:
            area_flat = 1.0

    def _ctrl_from_s(s):
        s = max(0.0, min(1.0, float(s)))
        unfold_alpha = unfold_max * (s**unfold_power)
        unfold_alpha = max(0.0, min(0.95, unfold_alpha))
        crease_scale = math.exp(-s / max(1e-6, crease_tau))
        crease_scale = max(0.0, min(1.0, crease_scale))
        attract = attract0 * math.exp(-s / max(1e-6, attract_tau))
        return crease_scale, unfold_alpha, attract

    if target_areas is not None:
        # Area-matched unfolding: advance the simulation sequentially and adjust a smooth
        # control variable `s` until the coverage proxy matches the target curve.
        # This prevents "4 keyframes + duplicates" by forcing perceptible change each frame.
        _apply_state()
        a_start = float(_area_fraction()) / max(1e-9, float(area_flat))

        targets = []
        last = a_start
        raw0 = float(target_areas.get(frame_start, 0.0))
        raw0 = max(0.0, min(1.0, raw0))
        denom = max(1e-9, (1.0 - raw0))
        min_step = 0.012  # minimum visible coverage change per frame (helps avoid accidental duplicates)
        for f in range(frame_start, frame_end + 1):
            raw = float(target_areas.get(f, 0.0))
            raw = max(0.0, min(1.0, raw))
            prog = max(0.0, min(1.0, (raw - raw0) / denom))
            t = a_start + (1.0 - a_start) * prog
            # Ensure monotonic and enforce a minimum step to keep every frame perceptibly different.
            t = max(t, last + (0.0 if f == frame_start else min_step))
            t = min(1.0, t)
            last = t
            targets.append((f, float(t)))

        s_cur = 0.0
        tol = 0.004
        # Estimate current area ratio.
        a_cur = float(_area_fraction()) / max(1e-9, float(area_flat))
        a_prev = a_cur

        for f in range(frame_start, frame_end + 1):
            if f == frame_end:
                # Let the simulation reach rest naturally. Frame-end should already be very close.
                _apply_state()
            else:
                target = dict(targets).get(f, 0.0)
                target = max(0.0, min(1.0, float(target)))
                # Enforce a strictly increasing "coverage" so the sequence reads as continuous unfolding.
                target = max(target, a_prev + 0.001)

                best_err = 1e9
                best_state = None
                best_area = None
                best_under_err = None
                best_under_state = None
                best_under_area = None
                best_over_err = None
                best_over_state = None
                best_over_area = None

                max_iters = 220
                for it in range(max_iters):
                    # Stop when close enough and not shrinking vs previous frame.
                    if a_cur >= (target - tol) and a_cur >= (a_prev - 1e-4):
                        break
                    err = float(target - a_cur)
                    if err <= 0.0:
                        # Already at/above target; do not force additional unfolding.
                        break
                    # Smoothly increase s based on remaining error (never decrease).
                    ds = 0.001 + 0.06 * max(0.0, err)
                    s_cur = min(1.0, s_cur + ds)

                    crease_scale, unfold_alpha, attract = _ctrl_from_s(s_cur)
                    # Prevent large jumps/overshoot near the target by scaling the "unfold pull"
                    # with the remaining error.
                    gain = max(0.15, min(1.0, err / 0.12))
                    unfold_alpha = unfold_alpha * gain
                    # Use *one* substep per control iteration to avoid large "snaps" that
                    # skip over intermediate coverage states (causes duplicated frames).
                    for _ in range(1):
                        step(
                            dt=dt,
                            n_iters=iters,
                            crease_scale=crease_scale,
                            unfold_alpha=unfold_alpha,
                            attract_strength=attract,
                        )
                    _apply_state()
                    a_cur = float(_area_fraction()) / max(1e-9, float(area_flat))

                    if a_cur >= (a_prev - 1e-4):
                        if a_cur <= target:
                            e = float(target - a_cur)
                            if best_under_err is None or e < best_under_err:
                                best_under_err = e
                                best_under_area = a_cur
                                best_under_state = [p.copy() for p in x]
                        else:
                            e = float(a_cur - target)
                            if best_over_err is None or e < best_over_err:
                                best_over_err = e
                                best_over_area = a_cur
                                best_over_state = [p.copy() for p in x]
                        eabs = abs(a_cur - target)
                        if eabs < best_err:
                            best_err = eabs
                            best_area = a_cur
                            best_state = [p.copy() for p in x]
                    # If we overshoot by a lot, stop early and use the best-so-far state.
                    if a_cur >= target + 0.08:
                        break

                chosen_state = None
                chosen_area = None
                if best_under_state is not None:
                    chosen_state = best_under_state
                    chosen_area = best_under_area
                elif best_over_state is not None:
                    chosen_state = best_over_state
                    chosen_area = best_over_area
                else:
                    chosen_state = best_state
                    chosen_area = best_area

                if chosen_state is not None:
                    for i in range(len(x)):
                        x[i] = chosen_state[i].copy()
                    _apply_state()
                    if chosen_area is not None:
                        a_cur = float(chosen_area)
                a_prev = a_cur

            # Force final frame to be completely flat (rest state)
            if f == frame_end:
                for i in range(len(x)):
                    x[i] = rest[i].copy()
                _apply_state()
            
            if unlit_node is not None:
                # Keep consistent shaded mode for all frames
                unlit_node.outputs[0].default_value = legacy_unlit
            _log_max_delta("render", frame=f)
            scene.frame_current = f
            scene.render.filepath = os.path.join(out_dir, f"frame_{f:04d}.png")
            bpy.ops.render.render(write_still=True)
    else:
        for f in range(frame_start, frame_end + 1):
            crease_scale, unfold_alpha, attract = schedule(f)
            for _ in range(max(1, substeps)):
                step(dt=dt, n_iters=iters, crease_scale=crease_scale, unfold_alpha=unfold_alpha, attract_strength=attract)
            if f == frame_end:
                # Force final frame to be completely flat (rest state)
                for i in range(len(x)):
                    x[i] = rest[i].copy()
                _apply_state()
            if unlit_node is not None:
                # Keep consistent shaded mode for all frames
                unlit_node.outputs[0].default_value = legacy_unlit
            _apply_state()
            _log_max_delta("render", frame=f)
            scene.frame_current = f
            scene.render.filepath = os.path.join(out_dir, f"frame_{f:04d}.png")
            bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
