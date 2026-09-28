"""Category D: street furniture / small city props.

Every prop shares the same authoring rules: Blender-style Z-up, origin at the
centre of the ground footprint with the object resting on z = 0, all geometry
built from the ``core`` kernel primitives (no imported or traced assets).

Parts are split semantically -- frame / housing / lens / glass / wood -- so each
becomes its own submesh material.  Anything that glows (street lamp housing,
traffic lenses, newsstand sign, phone booth interior) gets BOTH a solid
``base_color`` and an ``emissive`` tint, so the shape still reads when the
emissive channel is scaled to zero.
"""

import math

from .. import core as C

# Original palette shared with the building set so the block reads as one city.
CONCRETE = (0.70, 0.68, 0.64)
CONCRETE_DK = (0.50, 0.48, 0.46)
METAL = (0.58, 0.60, 0.62)
METAL_DK = (0.30, 0.32, 0.34)
IRON = (0.24, 0.23, 0.22)
CREAM = (0.96, 0.92, 0.80)
WHITE = (0.95, 0.95, 0.93)
TEAL = (0.16, 0.62, 0.60)
TEAL_DK = (0.11, 0.44, 0.43)
CORAL = (0.96, 0.46, 0.36)
PINK = (0.94, 0.50, 0.61)
MINT = (0.60, 0.88, 0.76)
GRASS_DK = (0.16, 0.40, 0.18)
LEAF_A = (0.26, 0.56, 0.22)
LEAF_B = (0.36, 0.66, 0.27)
BARK = (0.44, 0.35, 0.27)
WOOD = (0.60, 0.42, 0.28)
DARK = (0.09, 0.09, 0.11)
GLASS = (0.26, 0.56, 0.60)
RED = (0.82, 0.10, 0.10)
AMBER = (0.95, 0.55, 0.06)
GREEN = (0.12, 0.72, 0.28)
ORANGE = (0.94, 0.30, 0.10)
PAPER = (0.92, 0.89, 0.80)

LAMP_WARM = (1.00, 0.85, 0.50)
GLOW_COOL = (0.55, 0.88, 0.95)

# Roadworks / parking palette, same register as the rest of the street.
BARRIER_ORANGE = (0.94, 0.40, 0.08)
BARRIER_WHITE = (0.95, 0.94, 0.88)
BEACON_AMBER = (1.00, 0.62, 0.10)
METER_BODY = (0.28, 0.52, 0.56)
METER_TRIM = (0.17, 0.36, 0.39)
METER_FACE = (0.88, 0.90, 0.86)
METER_SLOT = (0.08, 0.09, 0.10)


# --------------------------------------------------------------------------
# small local helpers built on top of the kernel primitives
# --------------------------------------------------------------------------

def _rbox(m, size, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0),
          color=(1.0, 1.0, 1.0)):
    """A box rotated by (rx, ry, rz) DEGREES about its own centre.

    ``C.box`` is axis-aligned only, which is not enough for leaning backrest
    slats, angled roof flaps or the pilasters that have to follow the facets of
    an octagonal pot.  Emitting the same eight corners through the same six
    quads as ``C.box`` keeps the winding (and therefore the volume check)
    identical -- the rotation matrix has determinant +1.
    """
    hx, hy, hz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
    corners = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
               (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]
    ax, ay, az = (math.radians(v) for v in rot)
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    # R = Rz * Ry * Rx
    r = ((cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
         (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
         (-sy, cy * sx, cy * cx))
    o = tuple(center)
    p = []
    for v in corners:
        p.append(C.add(o, (r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
                           r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
                           r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2])))
    m.quad(p[0], p[3], p[2], p[1], color)
    m.quad(p[4], p[5], p[6], p[7], color)
    m.quad(p[0], p[1], p[5], p[4], color)
    m.quad(p[1], p[2], p[6], p[5], color)
    m.quad(p[2], p[3], p[7], p[6], color)
    m.quad(p[3], p[0], p[4], p[7], color)
    return m


def _loft_x(m, prof, stations, color, cap=True):
    """Extrude a CCW (y, z) profile along +X.

    ``prof`` points are ``(y, z)`` pairs in CCW order; ``stations`` is a list of
    ``(x, y_scale)`` pairs.  Mapping ``(y, z)`` into the world that way makes
    the ring CCW seen from +X (Y x Z = +X), which is the right-hand direction
    ``loft`` advances in.
    """
    rings = [[(x, py * sy, pz) for (py, pz) in prof] for (x, sy) in stations]
    return C.loft(m, rings, color, cap_start=cap, cap_end=cap,
                  cap_start_flip=True, cap_end_flip=False)


def _sleeve_x(m, prof, stations, color):
    """Sweep a CCW (y, z) profile along +X, scaling it uniformly per station.

    ``stations`` is ``(x, s)``: at that x the profile is dilated to ``s`` times
    its size about its own centroid, so ``(1.0, 1.04, 1.04, 1.0)`` produces a
    band that rises off the surface and sinks back into it -- a closed solid, so
    neither end cap collapses.
    """
    cy = sum(p[0] for p in prof) / float(len(prof))
    cz = sum(p[1] for p in prof) / float(len(prof))
    rings = []
    for (x, s) in stations:
        rings.append([(x, cy + (py - cy) * s, cz + (pz - cz) * s)
                      for (py, pz) in prof])
    return C.loft(m, rings, color, cap_start_flip=True, cap_end_flip=False)


def _mat(deg):
    """Rotation matrix R = Rz * Ry * Rx from Euler angles in DEGREES."""
    ax, ay, az = (math.radians(v) for v in deg)
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    return ((cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
            (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
            (-sy, cy * sx, cy * cx))


def _place(parts, offset=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0),
           pivot=(0.0, 0.0, 0.0)):
    """Rigid-transform whole parts: rotate about ``pivot``, then translate.

    Every angle is a proper rotation (det +1), so face winding survives
    untouched and no inward-facing shell can be introduced.  Used here to tip
    the parking meter's whole head assembly back 14 deg as one rigid group.
    """
    r = _mat(rot)
    px, py, pz = pivot
    for part in parts:
        part.mesh.pos = [C.add(offset, (
            r[0][0] * (p[0] - px) + r[0][1] * (p[1] - py) + r[0][2] * (p[2] - pz),
            r[1][0] * (p[0] - px) + r[1][1] * (p[1] - py) + r[1][2] * (p[2] - pz),
            r[2][0] * (p[0] - px) + r[2][1] * (p[1] - py) + r[2][2] * (p[2] - pz)))
            for p in part.mesh.pos]
    return parts


def _arc_points(cx, cy, r, a0, a1, n):
    """``n`` samples along a circular arc, endpoints included, no duplicates."""
    return [(cx + r * math.cos(a0 + (a1 - a0) * i / float(n)),
             cy + r * math.sin(a0 + (a1 - a0) * i / float(n)))
            for i in range(n + 1)]


# --------------------------------------------------------------------------
# 1. street light
# --------------------------------------------------------------------------

def _streetlight():
    a = C.Asset("prop_streetlight", "prop")

    # cast base: fluted octagonal plinth
    base = a.part("light_base", base_color=CONCRETE, roughness=0.85)
    C.cylinder(base.mesh, 0.26, 0.22, 8, center=(0, 0, 0.11), color=CONCRETE)
    C.cylinder(base.mesh, 0.30, 0.07, 8, center=(0, 0, 0.035), color=CONCRETE_DK)
    for k in range(8):
        ang = math.pi / 8.0 + k * math.pi / 4.0
        _rbox(base.mesh, (0.07, 0.07, 0.42), center=(0.205 * math.cos(ang),
                                                     0.205 * math.sin(ang), 0.32),
              rot=(0, 0, math.degrees(ang)), color=CONCRETE)
    C.cylinder(base.mesh, 0.19, 0.14, 8, center=(0, 0, 0.50), color=CONCRETE)

    # tapered post with two collars
    post = a.part("light_post", base_color=METAL_DK, metallic=0.35, roughness=0.5)
    C.cylinder(post.mesh, 0.145, 0.55, 10, center=(0, 0, 0.82),
               radius_top=0.095, color=METAL_DK)
    C.cylinder(post.mesh, 0.115, 0.05, 10, center=(0, 0, 1.12), color=METAL)
    C.cylinder(post.mesh, 0.092, 3.05, 10, center=(0, 0, 2.67), color=METAL_DK)
    C.cylinder(post.mesh, 0.105, 0.05, 10, center=(0, 0, 2.95), color=METAL)

    # curved arm sweeping out over the kerb
    arm = a.part("light_arm", base_color=METAL_DK, metallic=0.35, roughness=0.5)
    path = [(0.0, 0.0, 4.18), (0.22, 0.0, 4.52), (0.58, 0.0, 4.72),
            (1.02, 0.0, 4.70), (1.38, 0.0, 4.55), (1.58, 0.0, 4.40)]
    C.tube(arm.mesh, path, 0.072, 8, color=METAL_DK,
           radii=[0.085, 0.078, 0.072, 0.070, 0.068, 0.066])
    _rbox(arm.mesh, (0.16, 0.16, 0.16), center=(0, 0, 4.16), color=METAL_DK)

    # lamp housing -- tapered shade, glowing but still a solid colour
    head = a.part("lamp_housing", base_color=(1.0, 0.90, 0.66),
                  emissive=LAMP_WARM, roughness=0.35)
    C.cylinder(head.mesh, 0.30, 0.24, 12, center=(1.58, 0, 4.28),
               radius_top=0.11, color=(1.0, 0.90, 0.66))
    C.cylinder(head.mesh, 0.30, 0.05, 12, center=(1.58, 0, 4.40),
               color=(0.86, 0.80, 0.62))

    lens = a.part("lamp_lens", base_color=(1.0, 0.94, 0.78), emissive=LAMP_WARM,
                  roughness=0.2)
    C.cylinder(lens.mesh, 0.255, 0.05, 12, center=(1.58, 0, 4.135),
               color=(1.0, 0.94, 0.78))
    C.cylinder(lens.mesh, 0.10, 0.05, 10, center=(1.58, 0, 4.19),
               color=(1.0, 0.97, 0.86))
    return a


# --------------------------------------------------------------------------
# 2. traffic light
# --------------------------------------------------------------------------

def _trafficlight():
    a = C.Asset("prop_trafficlight", "prop")

    base = a.part("signal_base", base_color=CONCRETE, roughness=0.85)
    C.cylinder(base.mesh, 0.24, 0.12, 8, center=(0, 0, 0.06), color=CONCRETE)
    C.cylinder(base.mesh, 0.17, 0.14, 8, center=(0, 0, 0.17), color=CONCRETE_DK)

    post = a.part("signal_post", base_color=METAL_DK, metallic=0.4, roughness=0.5)
    C.cylinder(post.mesh, 0.095, 4.20, 10, center=(0, 0, 2.22), color=METAL_DK)
    C.cylinder(post.mesh, 0.11, 0.06, 10, center=(0, 0, 0.98), color=METAL)
    C.cylinder(post.mesh, 0.065, 1.42, 8, center=(0.71, 0, 4.18), axis="X",
               color=METAL_DK)                      # horizontal mast arm
    C.tube(post.mesh, [(0.02, 0, 3.42), (0.36, 0, 3.86), (0.66, 0, 4.12)],
           0.042, 6, color=METAL_DK)               # diagonal brace
    C.box(post.mesh, (0.22, 0.20, 0.14), center=(0, 0, 4.16), color=METAL_DK)

    housing = a.part("signal_housing", base_color=DARK, roughness=0.45)
    C.box(housing.mesh, (0.38, 0.34, 1.06), center=(1.42, 0, 3.60), color=DARK)
    C.box(housing.mesh, (0.44, 0.10, 0.10), center=(1.42, 0, 4.16), color=METAL_DK)
    C.box(housing.mesh, (0.44, 0.30, 0.08), center=(1.42, 0, 3.03), color=METAL_DK)
    C.box(housing.mesh, (0.26, 0.06, 0.62), center=(1.42, 0.19, 3.60),
          color=METAL_DK)                            # rear access panel

    lens_z = (3.97, 3.62, 3.27)
    lens_c = (RED, AMBER, GREEN)
    lens_e = ((0.95, 0.10, 0.08), (0.95, 0.50, 0.05), (0.0, 0.0, 0.0))
    for i in range(3):
        lit = i < 2                      # red + amber burning, green dark
        p = C.Part("signal_lens_%d" % i, base_color=lens_c[i],
                   emissive=lens_e[i], roughness=0.18)
        C.cylinder(p.mesh, 0.118, 0.07, 12, center=(1.42, -0.20, lens_z[i]),
                   axis="Y", color=lens_c[i])
        a.add(p)

    visor = a.part("signal_visors", base_color=DARK, roughness=0.5)
    for i in range(3):
        _rbox(visor.mesh, (0.30, 0.16, 0.035), center=(1.42, -0.26,
                                                       lens_z[i] + 0.155),
              rot=(-18, 0, 0), color=DARK)
        for sx in (-1, 1):
            _rbox(visor.mesh, (0.035, 0.16, 0.14), center=(1.42 + sx * 0.15,
                                                           -0.26, lens_z[i] + 0.08),
                  color=DARK)
    return a


# --------------------------------------------------------------------------
# 3. palm in a decorative pot
# --------------------------------------------------------------------------

def _frond(m, base, azimuth, length, rise, droop, wmax, thick, color,
           stations=8, seg=6):
    """One drooping palm blade: a flattened, tapering loft along an arc."""
    dx, dy = math.cos(azimuth), math.sin(azimuth)
    pts, rad = [], []
    for i in range(stations):
        u = i / float(stations - 1)
        r = length * u
        z = base[2] + rise * u - droop * (u * u)
        pts.append((base[0] + dx * r, base[1] + dy * r, z))
        w = wmax * (0.34 + 0.66 * math.sin(math.pi * (u ** 0.7))) \
            * ((1.0 - u) ** 0.45)
        rad.append((max(w, 0.024), max(thick * math.sqrt(1.0 - u), 0.013)))

    rings = []
    for i, p in enumerate(pts):
        if i == 0:
            t = C.sub(pts[1], pts[0])
        elif i == len(pts) - 1:
            t = C.sub(pts[-1], pts[-2])
        else:
            t = C.sub(pts[i + 1], pts[i - 1])
        t = C.normalize(t)
        # ``s`` is horizontal and perpendicular to the tangent; s x up == t, which
        # is the right-hand rule loft needs.  The arc never turns vertical, so
        # the cross product below can never collapse.
        s = C.normalize(C.cross((0.0, 0.0, 1.0), t))
        up = C.cross(t, s)
        w, th = rad[i]
        rings.append([C.add(C.add(p, C.mul(s, w * math.cos(a))),
                            C.mul(up, th * math.sin(a)))
                      for a in [2.0 * math.pi * k / seg for k in range(seg)]])
    return C.loft(m, rings, color, cap_start_flip=True, cap_end_flip=False)


def _palm_planter():
    a = C.Asset("prop_palm_planter", "prop")

    pot = a.part("planter_pot", base_color=CREAM, roughness=0.8)
    C.cylinder(pot.mesh, 0.44, 0.10, 8, center=(0, 0, 0.05), color=CONCRETE)
    C.cylinder(pot.mesh, 0.42, 0.58, 8, center=(0, 0, 0.34), radius_top=0.62,
               color=CREAM)
    C.cylinder(pot.mesh, 0.66, 0.13, 8, center=(0, 0, 0.63), color=CREAM)
    for k in range(8):                                  # corner pilasters
        ang = math.pi / 8.0 + k * math.pi / 4.0
        _rbox(pot.mesh, (0.085, 0.085, 0.46),
              center=(0.535 * math.cos(ang), 0.535 * math.sin(ang), 0.33),
              rot=(0, 0, math.degrees(ang)), color=CREAM)
    C.cylinder(pot.mesh, 0.575, 0.06, 8, center=(0, 0, 0.565), color=TEAL)

    soil = a.part("planter_soil", base_color=(0.20, 0.14, 0.10), roughness=1.0)
    C.cylinder(soil.mesh, 0.54, 0.09, 8, center=(0, 0, 0.685), color=(0.20, 0.14, 0.10))

    trunk = C.Part("palm_trunk", base_color=BARK, roughness=0.9, flat=False)
    path = [(0.0, 0.0, 0.66), (0.04, 0.02, 1.34), (0.08, 0.05, 2.04),
            (0.05, 0.03, 2.62)]
    C.tube(trunk.mesh, path, 0.14, 8, smooth=True,
           radii=[0.145, 0.118, 0.100, 0.088], color=BARK)
    a.add(trunk)

    scar_c = C.shade(BARK, 0.82)
    scars = a.part("palm_scars", base_color=scar_c, roughness=0.9)
    for k in range(5):
        C.cylinder(scars.mesh, 0.128 - k * 0.011, 0.05, 8,
                   center=(0.04, 0.02, 0.95 + k * 0.33), color=scar_c)

    crown_c = C.shade(BARK, 0.92)
    crown = a.part("palm_crown", base_color=crown_c, roughness=0.9)
    C.sphere(crown.mesh, 0.15, 10, 5, center=(0.05, 0.03, 2.62), squash=0.85,
             color=crown_c)

    blades = [C.Part("palm_fronds_a", base_color=LEAF_A, roughness=0.75),
              C.Part("palm_fronds_b", base_color=LEAF_B, roughness=0.75)]
    for k in range(7):
        part = blades[k % 2]
        _frond(part.mesh, (0.05, 0.03, 2.60),
               azimuth=math.radians(14.0 + k * 51.4),
               length=1.62 + 0.22 * ((k * 5) % 3) * 0.5,
               rise=0.62, droop=1.34 + 0.10 * (k % 2),
               wmax=0.30, thick=0.021,
               color=part.base_color)
    for p in blades:
        a.add(p)
    return a


# --------------------------------------------------------------------------
# 4. bench
# --------------------------------------------------------------------------

def _bench():
    a = C.Asset("prop_bench", "prop")

    frame = a.part("bench_frame", base_color=METAL_DK, metallic=0.3, roughness=0.6)
    for sx in (-1, 1):
        x = sx * 0.72
        _rbox(frame.mesh, (0.11, 0.62, 0.06), center=(x, 0.01, 0.03),
              color=METAL_DK)
        C.box(frame.mesh, (0.09, 0.09, 0.42), center=(x, -0.20, 0.27),
              color=METAL_DK)
        C.box(frame.mesh, (0.09, 0.11, 0.52), center=(x, 0.22, 0.70),
              color=METAL_DK)
        C.box(frame.mesh, (0.07, 0.44, 0.07), center=(x, 0.01, 0.17),
              color=METAL_DK)
        C.box(frame.mesh, (0.07, 0.07, 0.30), center=(x, -0.02, 0.62),
              color=METAL_DK)
        _rbox(frame.mesh, (0.11, 0.48, 0.055), center=(x, 0.02, 0.795),
              color=METAL_DK)
        _rbox(frame.mesh, (0.15, 0.13, 0.05), center=(x, -0.26, 0.03),
              color=METAL_DK)

    slats = a.part("bench_slats", base_color=TEAL, roughness=0.7)
    for y in (-0.24, -0.12, 0.0, 0.12, 0.24):
        _rbox(slats.mesh, (1.86, 0.115, 0.05), center=(0, y, 0.475),
              color=TEAL)
    for z in (0.60, 0.74, 0.88):
        _rbox(slats.mesh, (1.86, 0.10, 0.05),
              center=(0, 0.20 + 0.12 * (z - 0.45), z), rot=(-6.9, 0, 0),
              color=TEAL)

    trim = a.part("bench_trim", base_color=CREAM, roughness=0.7)
    for sx in (-1, 1):
        _rbox(trim.mesh, (0.045, 0.62, 0.035), center=(sx * 0.72, 0.01, 0.072),
              color=CREAM)
    return a


# --------------------------------------------------------------------------
# 5. trash bin
# --------------------------------------------------------------------------

def _trash_bin():
    a = C.Asset("prop_trash_bin", "prop")

    steel = a.part("bin_body", base_color=TEAL_DK, metallic=0.25, roughness=0.6)
    C.cylinder(steel.mesh, 0.26, 0.07, 12, center=(0, 0, 0.035), color=METAL_DK)
    C.cylinder(steel.mesh, 0.24, 0.74, 14, center=(0, 0, 0.44), radius_top=0.30,
               color=TEAL_DK)
    for k in range(10):
        ang = k * math.pi / 5.0
        _rbox(steel.mesh, (0.045, 0.045, 0.70),
              center=(0.268 * math.cos(ang), 0.268 * math.sin(ang), 0.44),
              rot=(0, 0, math.degrees(ang)), color=TEAL_DK)
    C.cylinder(steel.mesh, 0.315, 0.06, 14, center=(0, 0, 0.845), color=METAL)

    lid = a.part("bin_lid", base_color=METAL, metallic=0.35, roughness=0.45)
    C.cone(lid.mesh, 0.33, 0.22, 12, center=(0, 0, 0.985), radius_top=0.07,
           color=METAL)
    C.cylinder(lid.mesh, 0.075, 0.06, 8, center=(0, 0, 1.115), color=METAL_DK)
    _rbox(lid.mesh, (0.17, 0.05, 0.045), center=(0, -0.10, 1.09),
          rot=(-20, 0, 0), color=METAL_DK)

    trim = a.part("bin_trim", base_color=CORAL, roughness=0.6)
    C.cylinder(trim.mesh, 0.31, 0.055, 14, center=(0, 0, 0.16), color=CORAL)
    C.box(trim.mesh, (0.30, 0.10, 0.05), center=(0, -0.33, 0.055), color=METAL_DK)
    C.box(trim.mesh, (0.07, 0.07, 0.24), center=(0, -0.24, 0.20), color=METAL_DK)
    _rbox(trim.mesh, (0.22, 0.14, 0.03), center=(0, -0.315, 0.115), color=METAL_DK)
    return a


# --------------------------------------------------------------------------
# 6. dumpster
# --------------------------------------------------------------------------

def _dumpster():
    a = C.Asset("prop_dumpster", "prop")

    # Open-topped tapered shell: an open surface, so it declares its outward
    # reference explicitly instead of relying on a signed volume.
    lo = C.rounded_rect(0.96, 0.56, 0.10, 3)
    hi = C.rounded_rect(1.12, 0.66, 0.12, 3)
    body = C.Part("dumpster_shell", base_color=GRASS_DK, roughness=0.65,
                  outward=("point", (0.0, 0.0, 0.60)))
    C.loft(body.mesh, [[(x, y, 0.26) for (x, y) in lo],
                       [(x, y, 1.22) for (x, y) in hi]],
           GRASS_DK, cap_start=True, cap_end=False, cap_start_flip=True)
    a.add(body)

    ribs = a.part("dumpster_ribs", base_color=C.shade(GRASS_DK, 1.18), roughness=0.6)
    for sx in (-1, 1):
        for i in range(5):
            t = -0.60 + i * 0.30
            _rbox(ribs.mesh, (0.07, 0.07, 0.94),
                  center=(sx * 1.055, t * 1.02, 0.74),
                  rot=(0, 0, 0), color=C.shade(GRASS_DK, 1.18))
        _rbox(ribs.mesh, (0.07, 1.34, 0.94), center=(sx * 1.055, 0.0, 0.74),
              color=C.shade(GRASS_DK, 1.18))

    rails = a.part("dumpster_rails", base_color=METAL_DK, metallic=0.3,
                   roughness=0.6)
    for sy in (-1, 1):
        _rbox(rails.mesh, (2.36, 0.10, 0.09), center=(0, sy * 0.68, 1.20),
              color=METAL_DK)
    for sx in (-1, 1):
        _rbox(rails.mesh, (0.10, 1.42, 0.09), center=(sx * 1.14, 0, 1.20),
              color=METAL_DK)
    _rbox(rails.mesh, (2.30, 0.70, 0.07), center=(0, 0.32, 1.255), color=GRASS_DK)
    _rbox(rails.mesh, (2.30, 1.30, 0.07), rot=(50.0, 0, 0),
          center=(0, -0.242, 1.718), color=GRASS_DK)
    for sx in (-1, 1):
        C.box(rails.mesh, (0.10, 0.10, 0.12), center=(sx * 0.90, -0.66, 1.22),
              color=METAL_DK)

    wheels = a.part("dumpster_wheels", base_color=DARK, roughness=0.85)
    for sx in (-1, 1):
        for sy in (-1, 1):
            # seg=12 puts a profile vertex exactly at 270 deg, so the wheel's
            # lowest vertex is at centre.z - r and the bin truly touches z=0.
            # A 10-gon's is at 0.995r and leaves the whole prop hovering 5 mm.
            C.cylinder(wheels.mesh, 0.105, 0.085, 12,
                       center=(sx * 0.80, sy * 0.50, 0.105), axis="X",
                       color=DARK)
            C.box(wheels.mesh, (0.07, 0.14, 0.14), center=(sx * 0.80, sy * 0.50,
                                                           0.20), color=METAL_DK)

    plaque = a.part("dumpster_plaque", base_color=CREAM, roughness=0.7)
    C.box(plaque.mesh, (0.52, 0.04, 0.30), center=(0, -0.60, 0.86),
          color=CREAM)
    C.box(plaque.mesh, (0.62, 0.05, 0.07), center=(0, -0.60, 1.05),
          color=METAL_DK)
    return a


# --------------------------------------------------------------------------
# 7. mailbox
# --------------------------------------------------------------------------

def _mailbox():
    a = C.Asset("prop_mailbox", "prop")

    legs = a.part("mailbox_legs", base_color=METAL_DK, metallic=0.3, roughness=0.6)
    for sx in (-1, 1):
        C.cylinder(legs.mesh, 0.072, 0.94, 10, center=(sx * 0.22, 0, 0.47),
                   color=METAL_DK)
        C.cylinder(legs.mesh, 0.115, 0.05, 10, center=(sx * 0.22, 0, 0.025),
                   color=METAL)
    C.box(legs.mesh, (0.62, 0.09, 0.09), center=(0, 0, 0.86), color=METAL_DK)

    # Rounded-top collection box: a CCW (y, z) profile swept along X.
    prof = [(0.28, 0.92), (0.28, 1.32)]
    prof += _arc_points(0.0, 1.32, 0.28, 0.0, math.pi, 10)[1:]
    prof.append((-0.28, 0.92))
    body = a.part("mailbox_body", base_color=TEAL, metallic=0.15, roughness=0.55)
    _loft_x(body.mesh, prof,
            [(-0.36, 0.92), (-0.31, 1.0), (0.31, 1.0), (0.36, 0.92)],
            TEAL)
    # seam band: the barrel profile dilated, then returned to flush
    _sleeve_x(body.mesh, prof,
              [(-0.06, 1.0), (-0.05, 1.035), (0.05, 1.035), (0.06, 1.0)],
              C.shade(TEAL, 0.72))
    for sx in (-1, 1):                          # corrugation ribs on the sides
        for i in range(4):
            _rbox(body.mesh, (0.055, 0.50, 0.045),
                  center=(sx * 0.345, 0.0, 1.02 + i * 0.14),
                  rot=(0, 0, 0), color=C.shade(TEAL, 1.14))

    door = a.part("mailbox_door", base_color=C.shade(TEAL, 1.12), metallic=0.15,
                  roughness=0.5)
    C.box(door.mesh, (0.62, 0.045, 0.60), center=(0, -0.288, 1.14),
          color=C.shade(TEAL, 1.12))
    C.box(door.mesh, (0.68, 0.05, 0.06), center=(0, -0.292, 1.46),
          color=C.shade(TEAL, 0.82))
    C.box(door.mesh, (0.68, 0.05, 0.06), center=(0, -0.292, 0.90),
          color=C.shade(TEAL, 0.82))
    C.cylinder(door.mesh, 0.035, 0.12, 8, center=(0.22, -0.33, 1.12), axis="Y",
               color=METAL)

    slot = a.part("mailbox_slot", base_color=DARK, roughness=0.9)
    C.box(slot.mesh, (0.44, 0.06, 0.06), center=(0, -0.315, 1.34), color=DARK)
    C.box(slot.mesh, (0.50, 0.04, 0.03), center=(0, -0.318, 1.385),
          color=C.shade(TEAL, 0.6))

    label = a.part("mailbox_label", base_color=PAPER, roughness=0.85)
    C.box(label.mesh, (0.34, 0.02, 0.11), center=(-0.05, -0.315, 1.06),
          color=PAPER)

    flag = a.part("mailbox_flag", base_color=RED, roughness=0.6)
    C.box(flag.mesh, (0.035, 0.035, 0.26), center=(0.30, 0.0, 1.63), color=METAL_DK)
    C.box(flag.mesh, (0.035, 0.16, 0.09), center=(0.30, 0.06, 1.75), color=RED)
    return a


# --------------------------------------------------------------------------
# 8. newsstand
# --------------------------------------------------------------------------

def _newsstand():
    a = C.Asset("prop_newsstand", "prop")

    body = a.part("newsstand_body", base_color=CREAM, roughness=0.75)
    C.box(body.mesh, (2.10, 1.50, 0.12), center=(0, 0, 0.06), color=CONCRETE)
    C.box(body.mesh, (2.02, 1.44, 0.10), center=(0, 0, 0.16), color=TEAL)
    for sx in (-1, 1):
        for sy in (-1, 1):
            C.box(body.mesh, (0.12, 0.12, 2.20),
                  center=(sx * 0.94, sy * 0.66, 1.26), color=CREAM)
    C.box(body.mesh, (2.02, 0.10, 2.06), center=(0, 0.66, 1.17), color=CREAM)
    for sx in (-1, 1):
        C.box(body.mesh, (0.10, 1.32, 2.06), center=(sx * 0.94, 0, 1.17),
              color=CREAM)
    C.box(body.mesh, (1.86, 0.50, 0.94), center=(0, -0.38, 0.63), color=TEAL)
    C.box(body.mesh, (2.00, 0.62, 0.07), center=(0, -0.40, 1.13), color=METAL_DK)
    for z in (1.15, 1.62):
        C.box(body.mesh, (1.80, 0.34, 0.05), center=(0, 0.46, z), color=CONCRETE_DK)

    roof = a.part("newsstand_roof", base_color=METAL, metallic=0.3, roughness=0.5)
    C.box(roof.mesh, (2.34, 1.78, 0.08), center=(0, 0, 2.38), color=CREAM)
    C.box(roof.mesh, (2.26, 1.70, 0.14), center=(0, 0, 2.47), color=METAL)
    for i in range(5):
        C.box(roof.mesh, (0.07, 1.66, 0.06), center=(-0.90 + i * 0.45, 0, 2.56),
              color=C.shade(METAL, 0.82))
    C.box(roof.mesh, (0.08, 0.08, 0.36), center=(-0.62, 0, 2.72), color=METAL_DK)
    C.box(roof.mesh, (0.08, 0.08, 0.36), center=(0.62, 0, 2.72), color=METAL_DK)

    awning = a.part("newsstand_awning", base_color=PINK, roughness=0.6)
    _rbox(awning.mesh, (2.06, 0.88, 0.06), rot=(-16.0, 0, 0),
          center=(0, -0.92, 1.72), color=PINK)
    C.box(awning.mesh, (2.06, 0.05, 0.18), center=(0, -1.31, 1.53), color=CORAL)
    for sx in (-1, 1):
        C.box(awning.mesh, (0.05, 0.88, 0.06),
              center=(sx * 1.01, -0.92, 1.72), color=C.shade(PINK, 0.85))

    sign = a.part("newsstand_sign", base_color=MINT, emissive=(0.25, 0.85, 0.70),
                  roughness=0.35)
    C.box(sign.mesh, (1.62, 0.11, 0.44), center=(0, -0.08, 2.90), color=MINT)
    C.box(sign.mesh, (1.70, 0.05, 0.06), center=(0, -0.14, 2.66), color=METAL_DK)
    C.box(sign.mesh, (0.30, 0.04, 0.22), center=(0, -0.15, 2.90), color=CREAM)

    mags = a.part("newsstand_mags", base_color=PAPER, roughness=0.8)
    for i in range(7):
        x = -0.66 + i * 0.22
        _rbox(mags.mesh, (0.15, 0.035, 0.30),
              center=(x, -0.50 + 0.05 * ((i % 3) - 1), 1.30),
              rot=(0, 0, 0), color=(PAPER if i % 2 else CORAL))
        _rbox(mags.mesh, (0.17, 0.035, 0.26),
              center=(x, -0.52 + 0.04 * ((i % 3) - 1), 1.10),
              rot=(0, 0, 0), color=(MINT if i % 2 else PINK))
    return a


# --------------------------------------------------------------------------
# 9. phone booth
# --------------------------------------------------------------------------

def _phone_booth():
    a = C.Asset("prop_phone_booth", "prop")

    shell = a.part("booth_frame", base_color=TEAL, metallic=0.2, roughness=0.5)
    C.box(shell.mesh, (1.18, 1.18, 0.14), center=(0, 0, 0.07), color=CONCRETE)
    C.box(shell.mesh, (1.08, 1.08, 0.12), center=(0, 0, 0.19), color=METAL_DK)
    for sx in (-1, 1):
        for sy in (-1, 1):
            C.box(shell.mesh, (0.10, 0.10, 2.04), center=(sx * 0.46, sy * 0.46, 1.26),
                  color=TEAL)
    C.box(shell.mesh, (1.08, 1.08, 0.16), center=(0, 0, 2.30), color=TEAL)
    C.box(shell.mesh, (1.24, 1.24, 0.12), center=(0, 0, 2.44), color=METAL)
    C.box(shell.mesh, (0.94, 0.94, 0.10), center=(0, 0, 2.55), color=CREAM)

    finial = a.part("booth_finial", base_color=CORAL, roughness=0.6)
    C.cone(finial.mesh, 0.15, 0.22, 8, center=(0, 0, 2.71), color=CORAL)
    C.cylinder(finial.mesh, 0.045, 0.12, 8, center=(0, 0, 2.88), color=METAL)

    glass = a.part("booth_glass", base_color=GLASS, emissive=(0.10, 0.22, 0.24),
                   roughness=0.12)
    C.box(glass.mesh, (0.84, 0.05, 1.70), center=(0, 0.46, 1.19), color=GLASS)
    for sx in (-1, 1):
        C.box(glass.mesh, (0.05, 0.84, 1.70), center=(sx * 0.46, 0, 1.19),
              color=GLASS)
    C.box(glass.mesh, (0.78, 0.05, 1.62), center=(0, -0.46, 1.15), color=GLASS)

    mullions = a.part("booth_mullions", base_color=CREAM, roughness=0.6)
    for sx in (-1, 1):
        C.box(mullions.mesh, (0.055, 0.07, 1.74), center=(sx * 0.40, -0.47, 1.15),
              color=CREAM)
    C.box(mullions.mesh, (0.86, 0.07, 0.05), center=(0, -0.47, 1.15),
          color=CREAM)
    C.box(mullions.mesh, (0.80, 0.06, 0.05), center=(0, -0.48, 1.66),
          color=CREAM)
    C.box(mullions.mesh, (0.80, 0.06, 0.05), center=(0, -0.48, 0.62),
          color=CREAM)

    interior = a.part("booth_interior", base_color=CREAM, roughness=0.8)
    C.box(interior.mesh, (0.80, 0.30, 0.05), center=(0, 0.28, 0.98),
          color=CONCRETE_DK)
    C.box(interior.mesh, (0.24, 0.16, 0.54), center=(0, 0.36, 1.30),
          color=C.shade(TEAL, 0.72))
    C.box(interior.mesh, (0.10, 0.10, 0.16), center=(0, 0.30, 1.62),
          color=C.shade(TEAL, 0.72))
    C.box(interior.mesh, (0.09, 0.09, 0.26), center=(0, 0.30, 1.80),
          color=DARK)
    C.box(interior.mesh, (0.34, 0.06, 0.26), center=(0, 0.26, 1.80),
          color=CREAM)

    lamp = a.part("booth_lamp", base_color=(1.0, 0.95, 0.82),
                  emissive=GLOW_COOL, roughness=0.2)
    C.cylinder(lamp.mesh, 0.12, 0.07, 10, center=(0, 0, 2.19),
               color=(1.0, 0.95, 0.82))

    sign = a.part("booth_sign", base_color=CORAL, emissive=(0.85, 0.30, 0.20),
                  roughness=0.35)
    C.box(sign.mesh, (0.90, 0.06, 0.22), center=(0, -0.60, 2.44), color=CORAL)
    C.box(sign.mesh, (0.10, 0.06, 0.46), center=(0, -0.62, 2.62), color=CREAM)
    return a


# --------------------------------------------------------------------------
# 10. fire hydrant
# --------------------------------------------------------------------------

def _fire_hydrant():
    a = C.Asset("prop_fire_hydrant", "prop")

    body = a.part("hydrant_body", base_color=RED, roughness=0.55)
    C.cylinder(body.mesh, 0.215, 0.07, 12, center=(0, 0, 0.035), color=METAL_DK)
    C.cylinder(body.mesh, 0.135, 0.58, 12, center=(0, 0, 0.36), color=RED)
    C.cone(body.mesh, 0.205, 0.16, 12, center=(0, 0, 0.73), radius_top=0.105,
           color=RED)
    C.sphere(body.mesh, 0.155, 12, 6, center=(0, 0, 0.80), color=RED)
    C.cylinder(body.mesh, 0.30, 0.05, 12, center=(0, 0, 0.60), color=RED)

    caps = a.part("hydrant_caps", base_color=CREAM, roughness=0.6)
    for sx in (-1, 1):                     # side hose outlets, axis X
        C.cylinder(caps.mesh, 0.062, 0.13, 10, center=(sx * 0.175, 0, 0.47),
                   axis="X", color=CREAM)
        C.cone(caps.mesh, 0.078, 0.07, 10, center=(sx * 0.275, 0, 0.47),
               axis="X", color=CREAM)
    C.cylinder(caps.mesh, 0.058, 0.12, 10, center=(0, -0.175, 0.47), axis="Y",
               color=CREAM)
    C.cone(caps.mesh, 0.074, 0.07, 10, center=(0, -0.27, 0.47), axis="Y",
           color=CREAM)
    C.cylinder(caps.mesh, 0.038, 0.09, 6, center=(0, 0, 0.95), color=METAL)
    C.cone(caps.mesh, 0.055, 0.07, 8, center=(0, 0, 1.02), color=METAL)

    bolts = a.part("hydrant_bolts", base_color=METAL_DK, metallic=0.4,
                   roughness=0.5)
    for k in range(4):
        ang = math.pi / 4.0 + k * math.pi / 2.0
        C.cylinder(bolts.mesh, 0.020, 0.05, 6,
                   center=(0.175 * math.cos(ang), 0.175 * math.sin(ang), 0.085),
                   color=METAL_DK)

    # Chain links as a catenary tube rather than toruses: core.torus()'s
    # axis="Y" branch maps (x, y, z) -> (x, z, y), determinant -1, which mirrors
    # the shell inside out (every face inverted, negative signed volume).  tube()
    # builds its frame from the tangent and is correct on every path.
    chain = a.part("hydrant_chain", base_color=METAL_DK, metallic=0.5,
                   roughness=0.45)
    links = [(0.02, -0.10, 0.66), (0.10, -0.14, 0.56), (0.17, -0.15, 0.49),
             (0.19, -0.16, 0.46)]
    for i in range(3):
        p0 = links[i]
        p1 = links[i + 1]
        C.tube(chain.mesh, [p0, p1], 0.016, 6, color=METAL_DK)
    C.cylinder(chain.mesh, 0.028, 0.05, 6, center=links[0], color=METAL_DK)
    C.cylinder(chain.mesh, 0.028, 0.05, 6, center=links[-1], color=METAL_DK)

    band = a.part("hydrant_band", base_color=CREAM, roughness=0.6)
    C.cylinder(band.mesh, 0.142, 0.09, 12, center=(0, 0, 0.25), color=CREAM)
    return a


# --------------------------------------------------------------------------
# 11. manhole cover -- a ground decal, deliberately wafer thin
# --------------------------------------------------------------------------

def _manhole_cover():
    a = C.Asset("prop_manhole_cover", "prop")

    iron = a.part("cover_iron", base_color=IRON, metallic=0.45, roughness=0.7)
    C.cylinder(iron.mesh, 0.38, 0.024, 24, center=(0, 0, 0.012), color=IRON)

    field = a.part("cover_field", base_color=C.shade(IRON, 1.18), metallic=0.45,
                   roughness=0.62)
    C.cylinder(field.mesh, 0.33, 0.028, 20, center=(0, 0, 0.014),
               color=C.shade(IRON, 1.18))

    ribs = a.part("cover_ribs", base_color=C.shade(IRON, 1.32), metallic=0.45,
                  roughness=0.55)
    for k in range(8):
        ang = k * math.pi / 4.0
        _rbox(ribs.mesh, (0.22, 0.035, 0.030), rot=(0, 0, math.degrees(ang)),
              center=(0.20 * math.cos(ang), 0.20 * math.sin(ang), 0.015),
              color=C.shade(IRON, 1.32))

    hub = a.part("cover_hub", base_color=IRON, metallic=0.45, roughness=0.7)
    C.cylinder(hub.mesh, 0.07, 0.030, 10, center=(0, 0, 0.015), color=IRON)
    return a


# --------------------------------------------------------------------------
# 12. traffic cone
# --------------------------------------------------------------------------

def _traffic_cone():
    a = C.Asset("prop_traffic_cone", "prop")

    base = a.part("cone_base", base_color=ORANGE, roughness=0.7)
    C.box(base.mesh, (0.44, 0.44, 0.05), center=(0, 0, 0.025), color=ORANGE)
    C.box(base.mesh, (0.36, 0.36, 0.035), center=(0, 0, 0.055),
          color=C.shade(ORANGE, 0.88))

    body = a.part("cone_body", base_color=ORANGE, roughness=0.7)
    rings = [[(r * math.cos(a_), r * math.sin(a_), z)
              for a_ in [2.0 * math.pi * i / 12 for i in range(12)]]
             for (r, z) in ((0.165, 0.06), (0.140, 0.17), (0.088, 0.42),
                            (0.038, 0.68))]
    C.loft(body.mesh, rings, ORANGE, cap_start_flip=True, cap_end_flip=False)

    bands = a.part("cone_bands", base_color=WHITE, roughness=0.45)
    for (z0, z1, r0, r1) in ((0.255, 0.345, 0.115, 0.092),
                             (0.455, 0.520, 0.076, 0.058)):
        C.cylinder(bands.mesh, r0, z1 - z0, 12, center=(0, 0, (z0 + z1) * 0.5),
                   radius_top=r1, color=WHITE)
    return a


# --------------------------------------------------------------------------
# 13. construction barrier -- A-frame with a flashing beacon
# --------------------------------------------------------------------------

def _construction_barrier():
    """Orange/white striped A-frame road barrier with an amber lamp.

    Origin at the centre of the ground footprint, resting on z = 0 like every
    other prop, so it drops straight onto the kerb without a runtime tweak.

    The stripes are REAL alternating slabs, not one panel painted with
    face colours.  A single 2 m box is 12 triangles and only two distinct
    centroid X positions, so banding it by X produced one broad orange half
    and one broad cream half.  Interleaved orange and white boards are also
    four times the triangles, so the strip count is deliberately modest.
    """
    a = C.Asset("prop_construction_barrier", "prop")

    HW, HH, HD = 0.62, 0.28, 0.055          # half width / height / plank depth
    span = 2.00                             # foot span along X
    leg_rise = 0.86                         # how high the A-frame legs stand

    # -- feet ---------------------------------------------------------------
    feet = a.part("barrier_feet", base_color=METAL_DK, metallic=0.35,
                  roughness=0.6)
    for sx in (-1, 1):
        _rbox(feet.mesh, (0.30, 0.42, 0.048),
              center=(sx * span * 0.5, 0.0, 0.024), color=METAL_DK)

    # -- the striped plank --------------------------------------------------
    # Built as N side-by-side SLABS, each its own closed box, alternating
    # orange and cream.  Face-colour striping cannot work here: a single box
    # is 12 triangles spanning the whole 2 m panel, so there are only two
    # distinct centroid X positions to band on and the panel came out one
    # broad orange half and one broad cream half -- which is what the first
    # QA render showed.  Real stripes need real faces.
    plank = a.part("barrier_plank", base_color=BARRIER_ORANGE, roughness=0.62)
    n_band = 8
    bw = span / n_band
    zc = leg_rise + 0.22
    for i in range(n_band):
        col = BARRIER_ORANGE if i % 2 == 0 else BARRIER_WHITE
        C.box(plank.mesh, (bw, HD * 2.0, HH * 2.0),
              center=(-span * 0.5 + bw * (i + 0.5), 0.0, zc), color=col)

    # -- A-frame legs -------------------------------------------------------
    legs = a.part("barrier_legs", base_color=BARRIER_WHITE, roughness=0.66)
    for sx in (-1, 1):
        _rbox(legs.mesh, (0.075, 0.075, leg_rise + 0.10),
              center=(sx * span * 0.5, 0.0, (leg_rise + 0.10) * 0.5),
              rot=(0.0, sx * 7.0, 0.0), color=BARRIER_WHITE)
    _rbox(legs.mesh, (span * 0.92, 0.060, 0.060),
          center=(0.0, 0.0, 0.075), color=BARRIER_WHITE)        # lower spreader

    # -- amber beacon on top ------------------------------------------------
    beacon_body = a.part("barrier_beacon", base_color=METAL_DK, metallic=0.4,
                         roughness=0.45)
    C.cylinder(beacon_body.mesh, 0.070, 0.035, 10,
               center=(0.0, 0.0, leg_rise + 0.22 + HH + 0.018),
               color=METAL_DK)
    lamp = a.part("barrier_lamp", base_color=BEACON_AMBER, emissive=BEACON_AMBER,
                  roughness=0.18)
    # Both a solid colour AND an emissive tint, so the lamp still reads as an
    # object when the runtime scales the emissive channel to zero.
    C.cylinder(lamp.mesh, 0.058, 0.070, 10,
               center=(0.0, 0.0, leg_rise + 0.22 + HH + 0.070),
               color=BEACON_AMBER)
    C.cylinder(beacon_body.mesh, 0.062, 0.018, 10,
               center=(0.0, 0.0, leg_rise + 0.22 + HH + 0.114),
               color=METAL_DK)                                   # rain cap
    return a


# --------------------------------------------------------------------------
# 14. parking meter -- slim post, head box, coin slot
# --------------------------------------------------------------------------

def _parking_meter():
    """Kerbside parking meter: slim post, angled head box, coin slot."""
    a = C.Asset("prop_parking_meter", "prop")

    post_h = 1.08
    post = a.part("meter_post", base_color=METER_BODY, metallic=0.45,
                  roughness=0.48)
    C.cylinder(post.mesh, 0.115, 0.055, 10, center=(0.0, 0.0, 0.0275),
               color=METER_TRIM)                                # cast base
    C.cylinder(post.mesh, 0.052, post_h, 10, center=(0.0, 0.0, post_h * 0.5),
               color=METER_BODY)                                # column
    C.cylinder(post.mesh, 0.060, 0.040, 10, center=(0.0, 0.0, 0.075),
               color=METER_TRIM)                                # base collar

    # Head box, tipped back 14 deg so the face looks up at a driver.
    head = C.Part("meter_head", base_color=METER_BODY, metallic=0.45,
                  roughness=0.44)
    _rbox(head.mesh, (0.190, 0.150, 0.290), center=(0.0, 0.0, 0.0),
          color=METER_BODY)
    _rbox(head.mesh, (0.206, 0.166, 0.030), center=(0.0, 0.0, 0.130),
          color=METER_TRIM)                                    # top cap
    C.cylinder(head.mesh, 0.030, 0.110, 8, center=(0.0, 0.0, -0.150),
               color=METER_BODY)                                # neck into post
    _place([head], offset=(0.0, 0.0, post_h + 0.150), rot=(-14.0, 0.0, 0.0))
    a.add(head)

    face = C.Part("meter_face", base_color=METER_FACE, roughness=0.35)
    _rbox(face.mesh, (0.150, 0.014, 0.190), center=(0.0, -0.082, 0.010),
          color=METER_FACE)                                     # display panel
    _place([face], offset=(0.0, 0.0, post_h + 0.150), rot=(-14.0, 0.0, 0.0))
    a.add(face)

    trim = C.Part("meter_trim", base_color=METER_TRIM, metallic=0.5,
                  roughness=0.4)
    # Coin slot, proud of the face by 5 mm: a real recess shadow rather than a
    # coplanar decal that z-fights with the panel behind it.
    _rbox(trim.mesh, (0.020, 0.016, 0.056), center=(0.0, -0.086, -0.088),
          color=METER_SLOT)
    _rbox(trim.mesh, (0.090, 0.014, 0.024), center=(0.0, -0.086, -0.062),
          color=METER_TRIM)                                     # coin shelf lip
    for sx in (-1, 1):                                          # display bezel
        _rbox(trim.mesh, (0.016, 0.016, 0.190),
              center=(sx * 0.083, -0.082, 0.010), color=METER_TRIM)
    _place([trim], offset=(0.0, 0.0, post_h + 0.150), rot=(-14.0, 0.0, 0.0))
    a.add(trim)
    return a


# --------------------------------------------------------------------------

def build_all():
    """Return the ordered list of street-prop assets."""
    return [
        _streetlight(),
        _trafficlight(),
        _palm_planter(),
        _bench(),
        _trash_bin(),
        _dumpster(),
        _mailbox(),
        _newsstand(),
        _phone_booth(),
        _fire_hydrant(),
        _manhole_cover(),
        _traffic_cone(),
        _construction_barrier(),
        _parking_meter(),
    ]
