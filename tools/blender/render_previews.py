"""Render one preview PNG per asset category with EEVEE Next.

Run headless:

    /Applications/Blender.app/Contents/MacOS/Blender --background \
        --python tools/blender/render_previews.py

Assets are REBUILT from the vcw builders (never read back from the exported
JSON) and their AUTHORING-space vertices go straight into Blender: authoring is
already Blender-style Z-up (+X right, -Y forward, +Z up), so no axis swap is
applied here.  That is deliberate -- the (x, z, -y) rotation in
``vcw/export.py`` exists for the Y-up game runtime and applying it here would
tip every asset onto its side.

Two failure modes dominate this script and both are checked explicitly:

1. **Framing.** Camera distance comes from the asset's bounding SPHERE, so a
   0.03 m pistol and a 30 m hotel are framed identically without a per-asset
   table.  ``dist = radius / sin(half_fov) * 1.25`` keeps the whole sphere in
   frame; sphere radius, not per-axis extent, is what the FOV math needs.
2. **Blank frames.** Every PNG is read back and its per-pixel colour standard
   deviation computed.  A render that framed nothing is a uniform field and
   scores ~0.  Under 0.02 the script shouts and exits 1.

Uncaught exceptions exit 0 under ``blender --background --python``, so failures
are caught explicitly and re-raised as exit code 1.
"""

import math
import os
import sys
import traceback

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from vcw.builders import (                                      # noqa: E402
    buildings, markers, misc, palms, pedestrians, props, signs, vehicles,
    weapons,
)

BUILDER_MODULES = (
    ("buildings", buildings),
    ("vehicles", vehicles),
    ("pedestrians", pedestrians),
    ("props", props),
    ("signs", signs),
    ("palms", palms),
    ("misc", misc),
    ("weapons", weapons),
    ("markers", markers),
)

# --- render settings --------------------------------------------------------
RES = 900
SAMPLES = 32
LENS = 50.0
SENSOR = 36.0
FRAME_MARGIN = 1.25
# Azimuth is measured from +X toward +Y about the Z-up axis.  Assets are
# authored Z-up with their FRONT facing -Y (a sign's neon face, a pedestrian's
# face, a building's street front, a car's nose).  A camera at POSITIVE azimuth
# therefore sits behind them and renders their backs -- the sign preview came
# out as a blank unlit panel for exactly this reason.  Negative azimuth puts the
# camera in the -Y half-space, in front of everything, looking back toward +Y.
AZIMUTH = math.radians(-145.0)
ELEVATION = math.radians(22.0)
HALF_FOV = math.atan(SENSOR / (2.0 * LENS))     # ~19.8 deg
BLANK_STD = 0.02
WORLD_COLOR = (0.80, 0.80, 0.82, 1.0)
WORLD_STRENGTH = 0.55

# Preferred representative per category; falls back to the first asset.
PREFERRED = {
    "building": ("bldg_deco_pink", "bldg_deco_teal"),
    "vehicle": ("car_sedan",),
    "pedestrian": ("ped_suit",),
    "prop": ("prop_streetlight",),
    "sign": ("sign_hotel",),
    "palm": ("palm_tall",),
    "beach": ("beach_umbrella",),
    "misc": ("misc_helicopter",),
    "weapon": ("wep_pistol",),
    "pickup": ("pickup_health_pack",),
    # Both markers are surfaces of revolution about their own vertical axis,
    # so unlike the signs they cannot be caught from behind by the azimuth
    # above -- any angle frames the same silhouette.
    "marker": ("marker_flame",),
}


# --------------------------------------------------------------------------
# asset gathering
# --------------------------------------------------------------------------

def collect_assets():
    assets = []
    owner = {}
    for name, module in BUILDER_MODULES:
        if not hasattr(module, "build_all"):
            raise RuntimeError("builder %s exposes no build_all()" % name)
        for asset in module.build_all():
            if asset.id in owner:
                raise RuntimeError("duplicate asset id %r (%r and %r)"
                                   % (asset.id, owner[asset.id], name))
            owner[asset.id] = name
            assets.append(asset)
    if not assets:
        raise RuntimeError("no assets built")
    return assets


def pick_representatives(assets):
    """category -> asset, preferring a named representative where we have one."""
    by_cat = {}
    for a in assets:
        by_cat.setdefault(a.category, []).append(a)
    chosen = {}
    for cat in sorted(by_cat):
        group = by_cat[cat]
        pick = group[0]
        for want in PREFERRED.get(cat, ()):  # explicit first, else first-of-cat
            match = [a for a in group if a.id == want]
            if match:
                pick = match[0]
                break
        chosen[cat] = pick
    return chosen


# --------------------------------------------------------------------------
# geometry -> Blender
# --------------------------------------------------------------------------

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.objects,
                 bpy.data.lights, bpy.data.cameras, bpy.data.images):
        for item in list(coll):
            try:
                coll.remove(item)
            except (RuntimeError, ReferenceError):
                pass


def make_material(base_color, emissive, roughness, metallic):
    """One Principled material; emission rides on the BSDF's own inputs."""
    mat = bpy.data.materials.new("mat")
    mat.use_nodes = True
    bsdf = next((n for n in mat.node_tree.nodes
                 if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        raise RuntimeError("Principled BSDF node missing from new material")
    bsdf.inputs["Base Color"].default_value = (base_color[0], base_color[1],
                                              base_color[2], 1.0)
    bsdf.inputs["Roughness"].default_value = max(0.0, min(1.0, roughness))
    bsdf.inputs["Metallic"].default_value = max(0.0, min(1.0, metallic))
    if max(emissive) > 1e-6:
        # Emission Color + Emission Strength is the 4.5-safe route; the node
        # names "Emission" / "Clearcoat" from older docs no longer exist.
        bsdf.inputs["Emission Color"].default_value = (emissive[0],
                                                      emissive[1],
                                                      emissive[2], 1.0)
        strength = max(emissive)
        bsdf.inputs["Emission Strength"].default_value = max(1.0, strength)
    return mat


def build_object(asset, collection):
    """One Blender object per Part, with a material per distinct face colour."""
    objs = []
    for part in asset.parts:
        mesh = part.mesh
        if not mesh.faces:
            continue
        me = bpy.data.meshes.new("%s_%s" % (asset.id, part.name))
        # Authoring vertices ARE Blender coordinates (Z-up).  No axis change.
        me.from_pydata([tuple(p) for p in mesh.pos], [], [tuple(f) for f in
                                                          mesh.faces])
        me.update()

        obj = bpy.data.objects.new("%s_%s" % (asset.id, part.name), me)
        collection.objects.link(obj)

        # A colour -> material index map: face colours vary within a part, so
        # each distinct one needs its own material slot.
        slots = []
        index_of = {}
        for col in mesh.fcol:
            key = (round(col[0], 4), round(col[1], 4), round(col[2], 4))
            if key not in index_of:
                index_of[key] = len(slots)
                slots.append(make_material(key, part.emissive, part.roughness,
                                           part.metallic))
        for m in slots:
            me.materials.append(m)
        if len(me.polygons) != len(mesh.faces):
            raise RuntimeError("%s/%s: Blender built %d polys from %d faces"
                               % (asset.id, part.name, len(me.polygons),
                                  len(mesh.faces)))
        me.polygons.foreach_set("material_index",
                                [index_of[(round(c[0], 4), round(c[1], 4),
                                            round(c[2], 4))]
                                 for c in mesh.fcol])
        me.polygons.foreach_set("use_smooth",
                                [False] * len(me.polygons) if part.flat
                                else [True] * len(me.polygons))
        me.update()
        objs.append(obj)
    if not objs:
        raise RuntimeError("%s produced no renderable objects" % asset.id)
    return objs


# --------------------------------------------------------------------------
# scene
# --------------------------------------------------------------------------

def add_rig(scene, collection, target):
    """Key SUN + fill AREA + rim AREA, all aimed at the asset centre."""
    cx, cy, cz = target
    span = max(1.0, max(abs(v) for v in target) * 2.0)

    # Energies are tuned, not guessed: at 3.2/900/1400 the cream-white car body
    # clipped to 9.6% pure-white pixels and lost all surface detail.  These
    # values hold clipping near 1.7% while keeping mean luminance and the
    # key/fill/rim ratio intact.
    key = bpy.data.lights.new("key", type="SUN")
    key.energy = 1.8
    key.angle = math.radians(8.0)
    ko = bpy.data.objects.new("key", key)
    collection.objects.link(ko)
    ko.location = (cx + span * 0.9, cy - span * 0.9, cz + span * 1.4)
    ko.rotation_euler = (Vector(target) - ko.location).to_track_quat(
        "-Z", "Y").to_euler()

    fill = bpy.data.lights.new("fill", type="AREA")
    fill.energy = 120.0
    fill.size = span * 2.0
    fo = bpy.data.objects.new("fill", fill)
    collection.objects.link(fo)
    fo.location = (cx - span * 1.5, cy - span * 0.7, cz + span * 0.5)
    fo.rotation_euler = (Vector(target) - fo.location).to_track_quat(
        "-Z", "Y").to_euler()

    rim = bpy.data.lights.new("rim", type="AREA")
    rim.energy = 170.0
    rim.size = span * 2.0
    ro = bpy.data.objects.new("rim", rim)
    collection.objects.link(ro)
    ro.location = (cx - span * 0.5, cy + span * 1.6, cz + span * 0.9)
    ro.rotation_euler = (Vector(target) - ro.location).to_track_quat(
        "-Z", "Y").to_euler()


def setup_world(scene):
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = next((n for n in world.node_tree.nodes
               if n.type == "BACKGROUND"), None)
    if bg is None:
        bg = world.node_tree.nodes.new("ShaderNodeBackground")
        out = next((n for n in world.node_tree.nodes
                    if n.type == "OUTPUT_WORLD"), None)
        world.node_tree.links.new(bg.outputs[0], out.inputs[0])
    bg.inputs[0].default_value = WORLD_COLOR
    bg.inputs[1].default_value = WORLD_STRENGTH


def frame_camera(asset):
    """Bounding-sphere framing -> (centre, radius, distance)."""
    verts = [p for part in asset.parts for p in part.mesh.pos]
    if not verts:
        raise RuntimeError("%s has no vertices to frame" % asset.id)
    lo = [min(p[k] for p in verts) for k in range(3)]
    hi = [max(p[k] for p in verts) for k in range(3)]
    centre = Vector(((lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5,
                     (lo[2] + hi[2]) * 0.5))
    radius = max((Vector(p) - centre).length for p in verts)
    if radius < 1e-6:
        raise RuntimeError("%s has a degenerate bounding sphere" % asset.id)
    # The sphere, not the box: a sphere of `radius` is inside the frustum at
    # distance radius/sin(half_fov) from any direction.
    dist = radius / math.sin(HALF_FOV) * FRAME_MARGIN
    return centre, radius, dist


def setup_camera(scene, collection, asset):
    centre, radius, dist = frame_camera(asset)
    offset = Vector((math.cos(ELEVATION) * math.cos(AZIMUTH),
                     math.cos(ELEVATION) * math.sin(AZIMUTH),
                     math.sin(ELEVATION))) * dist
    cam_data = bpy.data.cameras.new("camera")
    cam_data.lens = LENS
    cam_data.sensor_width = SENSOR
    cam_data.clip_start = max(0.001, dist - radius * 4.0)
    cam_data.clip_end = dist + radius * 8.0 + 10.0
    cam = bpy.data.objects.new("camera", cam_data)
    collection.objects.link(cam)
    cam.location = centre + offset
    cam.rotation_euler = (centre - cam.location).to_track_quat(
        "-Z", "Y").to_euler()
    scene.camera = cam
    return centre, radius, dist


def setup_render(scene):
    # 4.5 has no "BLENDER_EEVEE"; the Next engine owns that name slot.
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = SAMPLES
    scene.render.resolution_x = RES
    scene.render.resolution_y = RES
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    # Same world on the viewport fallback path.
    setup_world(scene)


def render_one(asset, out_path):
    clear_scene()
    scene = bpy.context.scene
    collection = scene.collection
    setup_render(scene)

    objs = build_object(asset, collection)
    centre, radius, dist = setup_camera(scene, collection, asset)
    add_rig(scene, collection, centre)

    # A ground-less asset on a flat world still needs one shadow anchor, and a
    # shadow catcher would flatten the silhouette.  Keep it invisible.
    bpy.context.view_layer.update()

    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)

    if not os.path.isfile(out_path):
        raise RuntimeError("%s: Blender wrote no file at %s"
                           % (asset.id, out_path))
    return {
        "objects": len(objs),
        "radius": radius,
        "distance": dist,
        "bytes": os.path.getsize(out_path),
    }


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------

def image_std(path):
    """Per-pixel standard deviation of the RGB channels (real pixels, not a
    proxy).  Blender stores images bottom-up as RGBA floats."""
    img = bpy.data.images.load(path)
    try:
        px = img.pixels[:]          # flat list, 4 floats per pixel
    finally:
        bpy.data.images.remove(img)
    if len(px) < 4:
        raise RuntimeError("%s: image has %d floats, too few to measure"
                           % (path, len(px)))
    n_px = len(px) // 4
    # Channel means and E[x^2] in one pass -- O(n) and no float blow-up.
    s = [0.0, 0.0, 0.0]
    s2 = [0.0, 0.0, 0.0]
    for i in range(0, n_px * 4, 4):
        for c in range(3):
            v = px[i + c]
            s[c] += v
            s2[c] += v * v
    variances = []
    for c in range(3):
        mean = s[c] / n_px
        var = s2[c] / n_px - mean * mean
        variances.append(max(0.0, var))
    per_channel = [math.sqrt(v) for v in variances]
    pooled = math.sqrt(sum(per_channel) ** 2 / 3.0)
    return pooled, per_channel


def main():
    root = os.path.dirname(os.path.dirname(HERE))
    if not os.path.isdir(root):
        raise RuntimeError("repo root %s does not exist" % root)
    preview_dir = os.path.join(root, "assets", "preview")
    os.makedirs(preview_dir, exist_ok=True)

    assets = collect_assets()
    reps = pick_representatives(assets)
    out = sys.stdout.write
    out("\n=== NEON BAY preview render ===\n")
    out("engine BLENDER_EEVEE_NEXT  %dx%d  %d samples  "
        "categories=%d\n\n" % (RES, RES, SAMPLES, len(reps)))

    rows = []
    failures = []
    for cat in sorted(reps):
        asset = reps[cat]
        path = os.path.join(preview_dir, "%s.png" % cat)
        info = render_one(asset, path)
        std, per_channel = image_std(path)
        blank = std < BLANK_STD
        rows.append((cat, asset.id, path, std, per_channel, info["bytes"],
                     blank))
        if blank:
            failures.append(cat)
            out("!!! BLANK RENDER: %s (%s) -- pixel std %.5f < %.2f.  "
                "Framing or lighting is broken.\n"
                % (cat, asset.id, std, BLANK_STD))
            sys.stdout.flush()

    out("%-11s %-24s %-8s %-6s %s\n"
        % ("category", "asset", "px_std", "status", "png"))
    out("-" * 104 + "\n")
    for cat, aid, path, std, _ch, nbytes, blank in rows:
        out("%-11s %-24s %-8.4f %-6s %s (%.1f KB)\n"
            % (cat, aid, std, "BLANK" if blank else "PASS",
               os.path.abspath(path), nbytes / 1024.0))
    out("-" * 104 + "\n")
    out("\nrendered %d/%d category previews, %d blank\n"
        % (len(rows) - len(failures), len(rows), len(failures)))

    out("\nabsolute paths:\n")
    for cat, _aid, path, _s, _c, _b, _bl in rows:
        out("  %s\n" % os.path.abspath(path))
    out("\n")

    if failures:
        out("PREVIEW RENDER FAILED -- blank images: %s\n" % ", ".join(failures))
        sys.stdout.flush()
        raise RuntimeError("blank preview render(s): %s" % ", ".join(failures))
    out("PREVIEW RENDER OK: %d categories, all non-blank\n\n" % len(rows))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.write("\nPREVIEW RENDER FAILED (see traceback above)\n")
        sys.stdout.flush()
        code = 1
    # Explicit: an uncaught exception under `blender --background --python`
    # exits 0, which would report a failed render as success.
    sys.exit(code)
