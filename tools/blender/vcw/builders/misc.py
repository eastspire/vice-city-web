"""Beach furniture + one-off misc props (categories "beach" and "misc").

Eight assets, all procedural: every vertex is generated here from the
``core`` kernel primitives, nothing is imported, traced or measured from a
real-world object.

Authoring rules (see CONTRACT.md): Blender-style Z-up, -Y forward, every asset
centred on X = 0 and resting on Z = 0.

Two kernel facts this module leans on hard:

* ``core.torus`` is only trusted on its default axis, so ring-ish shapes here
  are built from stacked closed solids or lofts instead.
* Every ``rounded_rect`` used as a loft profile keeps a corner radius strictly
  below ``min(hx, hy)``.  ``rounded_rect`` returns a different point count when
  ``r`` is clamped to ``hy`` (the closing point is then dropped), and ``loft``
  requires equal-cardinality rings -- so a clamped radius would blow up with an
  assertion instead of a quietly wrong shell.
"""

import math

from .. import core as C

# --------------------------------------------------------------------------
# palette -- the beach set and the misc set read as one 1980s boardwalk
# --------------------------------------------------------------------------

CORAL = (0.96, 0.46, 0.36)
CORAL_DK = (0.74, 0.30, 0.24)
CREAM = (0.98, 0.94, 0.86)
SAND = (0.90, 0.83, 0.66)
TEAL = (0.16, 0.62, 0.60)
TEAL_DK = (0.10, 0.38, 0.38)
PINK = (0.95, 0.46, 0.62)
MINT = (0.62, 0.88, 0.76)
WHITE = (0.95, 0.95, 0.93)
YELLOW = (0.98, 0.83, 0.22)
CYAN = (0.30, 0.80, 0.90)
MAGENTA = (0.88, 0.24, 0.62)
ORANGE = (0.96, 0.52, 0.14)
GREEN = (0.22, 0.70, 0.34)
DARK = (0.10, 0.10, 0.12)
METAL = (0.60, 0.62, 0.65)
METAL_DK = (0.34, 0.36, 0.39)
CHROME = (0.78, 0.80, 0.82)
GLASS_DK = (0.10, 0.22, 0.28)
WOOD_A = (0.66, 0.48, 0.31)
WOOD_B = (0.58, 0.41, 0.26)
WOOD_C = (0.72, 0.55, 0.37)
WOOD_D = (0.51, 0.35, 0.22)
ASPHALT = (0.30, 0.30, 0.32)
ASPHALT_LT = (0.36, 0.36, 0.38)
PAPER = (0.88, 0.85, 0.76)
PAINT_WHITE = (0.94, 0.94, 0.90)

BOARD_COL = (0.95, 0.93, 0.86)
BOARD_STRIPE = (0.20, 0.58, 0.78)
FIN_COL = (0.90, 0.34, 0.22)


# --------------------------------------------------------------------------
# local helpers (kept tiny -- all real geometry goes through core)
# --------------------------------------------------------------------------

def _mat(deg):
    """Rotation matrix R = Rz * Ry * Rx from Euler angles in DEGREES."""
    ax, ay, az = (math.radians(v) for v in deg)
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    return ((cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
            (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
            (-sy, cy * sx, cy * cx))


def _rbox(m, size, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0),
          color=(1.0, 1.0, 1.0)):
    """A box rotated about its own centre -- ``C.box`` is axis-aligned only."""
    hx, hy, hz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
    corners = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
               (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]
    r = _mat(rot)
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


def _place(parts, offset=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0),
           pivot=(0.0, 0.0, 0.0)):
    """Rigid-transform whole parts: rotate about ``pivot``, then translate.

    Every angle here is a proper rotation (det +1), so face winding survives
    untouched and no inward-facing shell can be introduced.
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


def _drop(parts, z=0.0):
    """Translate a group of parts so the whole group rests on ``z``."""
    lo = None
    for part in parts:
        for p in part.mesh.pos:
            lo = p[2] if lo is None else min(lo, p[2])
    if lo is None:
        return parts
    dz = z - lo
    if abs(dz) < 1e-12:
        return parts
    for part in parts:
        part.mesh.pos = [(p[0], p[1], p[2] + dz) for p in part.mesh.pos]
    return parts


def _shift(parts, offset):
    """Translate whole parts by ``offset`` (pure translation, winding-safe)."""
    dx, dy, dz = offset
    for part in parts:
        part.mesh.pos = [(p[0] + dx, p[1] + dy, p[2] + dz)
                         for p in part.mesh.pos]
    return parts


def _rest(parts, supports, sink=0.05, cap=None):
    """Drop ``parts`` onto whatever ``supports`` already hold, not the ground.

    Freezing an object's bottom at z = 0 makes every element of a "pile" stand
    separately on the pavement, which reads as a tidy row rather than a mess.
    This finds the highest existing vertex whose XY falls inside the incoming
    object's footprint and seats the object on that, minus ``sink`` so it beds
    in slightly instead of hovering on a single point.

    ``cap`` bounds the resting height.  Without it a sequence of rests keeps
    climbing -- each item sits on the previous item's crown -- and the "pile"
    becomes a 2.4 m tower, which is not what a tipped-bin heap looks like.
    """
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    for part in parts:
        for p in part.mesh.pos:
            for k in range(3):
                lo[k] = min(lo[k], p[k])
                hi[k] = max(hi[k], p[k])
    top = None
    for part in supports:
        for p in part.mesh.pos:
            if lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1]:
                top = p[2] if top is None else max(top, p[2])
    if top is None:
        return _drop(parts, 0.0)
    if cap is not None and top - sink > cap:
        return _drop(parts, 0.0)
    return _drop(parts, top - sink)


def _finish(asset, z=0.0):
    _drop(asset.parts, z)
    return asset


def _loft_y(m, prof, stations, color, cap=True):
    """Sweep a CCW ``(x, z)`` profile along +Y.

    ``prof`` is CCW in the usual x-right / z-up plane, whose right-hand normal
    is X x Z = -Y -- the OPPOSITE of the direction the rings advance in, so it
    is reversed here.  Reversed, the ring normal is +Y, ``loft``'s side walls
    come out outward, and the start cap needs ``flip=True`` to face -Y.
    """
    pts = list(prof)[::-1]
    rings = [[(px * sx, y, cz + pz * sz) for (px, pz) in pts]
             for (y, sx, sz, cz) in stations]
    return C.loft(m, rings, color, cap_start=cap, cap_end=cap,
                  cap_start_flip=True, cap_end_flip=False)


# --------------------------------------------------------------------------
# GROUP 1 -- beach props
# --------------------------------------------------------------------------

def _umbrella():
    a = C.Asset("beach_umbrella", "beach")

    pole = a.part("umbrella_pole", base_color=CREAM, roughness=0.45)
    C.cylinder(pole.mesh, 0.028, 2.20, 10, center=(0.0, 0.0, 1.10),
               color=CREAM)
    C.cylinder(pole.mesh, 0.055, 0.07, 10, center=(0.0, 0.0, 0.035),
               color=METAL)                     # ground collar
    C.cylinder(pole.mesh, 0.042, 0.05, 10, center=(0.0, 0.0, 1.24),
               color=METAL)                     # height adjuster

    R = 1.30                     # canopy radius
    Z_RIM = 1.86                 # rim height
    DOME = 0.30                  # apex lift over the rim
    R_IN = 0.14                  # inner radius, around the mast
    THICK = 0.038

    def z_top(r):
        return Z_RIM + DOME * (1.0 - (r / R) ** 2)

    def z_bot(r):
        return z_top(r) - THICK * (1.0 - 0.35 * r / R)

    stripes = [C.Part("umbrella_canopy_a", base_color=CORAL, roughness=0.55),
               C.Part("umbrella_canopy_b", base_color=CREAM, roughness=0.55)]
    r_stations = (R_IN, 0.60 * R, R)
    n_panels = 8
    for i in range(n_panels):
        part = stripes[i % 2]
        a0 = 2.0 * math.pi * i / n_panels
        a1 = 2.0 * math.pi * (i + 1) / n_panels
        c0, s0 = math.cos(a0), math.sin(a0)
        c1, s1 = math.cos(a1), math.sin(a1)
        rings = []
        for r in r_stations:
            zt, zb = z_top(r), z_bot(r)
            # ring order (top_a0, bot_a0, bot_a1, top_a1) gives a right-hand
            # normal along the panel's outward bisector -- the direction the
            # rings advance in, which is what keeps the shell outside-out.
            rings.append([(r * c0, r * s0, zt), (r * c0, r * s0, zb),
                          (r * c1, r * s1, zb), (r * c1, r * s1, zt)])
        C.loft(part.mesh, rings, part.base_color,
               cap_start_flip=True, cap_end_flip=False)
    for p in stripes:
        a.add(p)

    tips = a.part("umbrella_tips", base_color=CREAM, roughness=0.5)
    for i in range(n_panels):
        ang = 2.0 * math.pi * (i + 0.5) / n_panels
        C.cone(tips.mesh, 0.030, 0.13, 6,
               center=(R * math.cos(ang), R * math.sin(ang), Z_RIM - 0.05),
               color=CREAM)

    hub = a.part("umbrella_hub", base_color=METAL, metallic=0.3, roughness=0.4)
    C.cylinder(hub.mesh, 0.075, 0.10, 10, center=(0.0, 0.0, 2.13),
               radius_top=0.03, color=METAL)
    C.sphere(hub.mesh, 0.055, 8, 5, center=(0.0, 0.0, 2.21), color=CREAM)
    return _finish(a)


def _beach_chair():
    """Low slatted lounger, ~1.9 m long, nose toward -Y."""
    a = C.Asset("beach_chair", "beach")

    frame = a.part("chair_frame", base_color=CREAM, roughness=0.4)
    for sx in (-1, 1):
        x = sx * 0.29
        C.tube(frame.mesh, [(x, -0.92, 0.10), (x, -0.55, 0.36),
                            (x, 0.30, 0.36), (x, 0.90, 0.96)],
               0.028, 6, color=CREAM)
        C.tube(frame.mesh, [(x, -0.55, 0.36), (x, -0.61, 0.028)],
               0.026, 6, color=CREAM)
        C.tube(frame.mesh, [(x, 0.30, 0.36), (x, 0.43, 0.028)],
               0.026, 6, color=CREAM)
    for y in (-0.55, 0.30):
        C.cylinder(frame.mesh, 0.022, 0.58, 6, center=(0.0, y, 0.33),
                   axis="X", color=METAL)

    slats_a = a.part("chair_slats_a", base_color=TEAL, roughness=0.55)
    slats_b = a.part("chair_slats_b", base_color=CORAL, roughness=0.55)
    for i in range(6):                                   # flat seat
        part = slats_a if i % 2 == 0 else slats_b
        _rbox(part.mesh, (0.56, 0.105, 0.032),
              center=(0.0, -0.48 + i * 0.15, 0.404), color=part.base_color)
    for i, t in enumerate((0.25, 0.55, 0.85)):            # foot incline
        part = slats_a if i % 2 == 1 else slats_b
        _rbox(part.mesh, (0.56, 0.110, 0.032), rot=(35.0, 0.0, 0.0),
              center=(0.0, -0.92 + 0.37 * t - 0.025, 0.10 + 0.26 * t + 0.036),
              color=part.base_color)
    for i, t in enumerate((0.12, 0.30, 0.48, 0.66, 0.84, 0.96)):
        part = slats_a if i % 2 == 0 else slats_b
        _rbox(part.mesh, (0.56, 0.115, 0.032), rot=(45.0, 0.0, 0.0),
              center=(0.0, 0.30 + 0.60 * t - 0.031, 0.36 + 0.60 * t + 0.031),
              color=part.base_color)

    caps = a.part("chair_caps", base_color=METAL, metallic=0.3, roughness=0.4)
    for sx in (-1, 1):
        for (y, z) in ((-0.61, 0.028), (0.43, 0.028)):
            C.cylinder(caps.mesh, 0.040, 0.03, 8,
                       center=(sx * 0.29, y, z), color=METAL_DK)
    return _finish(a)


def _surfboard():
    """Long flattened lozenge, tapering to a point at both ends.

    Built as a loft of ``rounded_rect`` rings along X (via ``section_rings``),
    with a slight rocker (lift_fn) and a shallow swept fin underneath.  The
    whole board then leans nose-up about 6 degrees so the fin tip is the only
    ground contact -- which is also why the asset is dropped afterwards rather
    than authored at a guessed height.
    """
    a = C.Asset("beach_surfboard", "beach")

    xs = (-1.0, -0.93, -0.80, -0.60, -0.33, 0.0, 0.28, 0.55, 0.78, 0.92, 1.0)

    def half_w(x):
        return max(0.016, 0.25 * (1.0 - x) ** 0.34 * (1.0 + x) ** 0.40)

    def half_t(x):
        return max(0.011, 0.034 * max(0.0, 1.0 - x * x) ** 0.30)

    def rad(x):
        return 0.42 * min(half_w(x), half_t(x))

    def lift(x):
        return 0.05 * x * x + 0.06 * max(0.0, x) ** 3

    deck = a.part("surfboard_deck", base_color=BOARD_COL, roughness=0.30)
    C.loft(deck.mesh, C.section_rings(xs, half_w, half_t, rad, 3, 0.0, lift),
           BOARD_COL, cap_start_flip=True, cap_end_flip=False)
    # Colour only, no extra geometry: a stringer stripe down the spine, darker
    # rails, darker belly.  The stripe predicate keys on the centroid's Y, not
    # its X -- a stringer runs the LENGTH of the board, so it is narrow across
    # the beam.  Testing x instead paints one station-long patch near midships.
    C.recolor_faces_where(
        deck, lambda n, c: n[2] > 0.60 and abs(c[1]) < 0.075, BOARD_STRIPE)
    C.recolor_faces_where(
        deck, lambda n, c: n[2] > 0.60 and abs(c[1]) >= 0.075,
        C.shade(BOARD_COL, 0.88))
    C.recolor_faces_where(
        deck, lambda n, c: abs(n[1]) > 0.70, C.shade(BOARD_COL, 0.76))
    C.recolor_faces_where(
        deck, lambda n, c: n[2] < -0.55, C.shade(BOARD_COL, 0.62))

    fin = a.part("surfboard_fin", base_color=FIN_COL, roughness=0.35)
    _rbox(fin.mesh, (0.30, 0.035, 0.16), center=(-0.40, 0.0, -0.045),
          rot=(0.0, -16.0, 0.0), color=FIN_COL)

    _place(a.parts, rot=(0.0, -6.0, 0.0))
    return _finish(a)


def _boardwalk_plank():
    """One 2.0 x 0.25 x 0.06 m decking plank, bottom face on z = 0."""
    a = C.Asset("beach_boardwalk_plank", "beach")

    half_t = 0.030
    xs = (-1.0, -0.62, -0.20, 0.22, 0.62, 1.0)
    plank = a.part("plank", base_color=WOOD_C, roughness=0.80)
    rings = C.section_rings(xs,
                            lambda x: 0.125,          # half width  -> world Y
                            lambda x: half_t,         # half depth  -> world Z
                            lambda x: 0.012,          # eased long edges
                            2, 0.0, lambda x: half_t)
    C.loft(plank.mesh, rings, WOOD_C, cap_start_flip=True, cap_end_flip=False)

    # Wood tone variation through face colours only: weathered top, two-tone
    # grain down the sides, dark end grain.  Note ``rounded_rect`` hands the
    # side wall back as vertical quads whose normals are +/-Y with |n.z| <=
    # ~0.42, so the side test must NOT also demand |n.z| < 0.55 -- the two
    # conditions are mutually exclusive and silently recolour nothing.
    C.recolor_faces_where(plank, lambda n, c: n[2] > 0.55, WOOD_A)
    C.recolor_faces_where(plank, lambda n, c: abs(n[1]) > 0.70, WOOD_B)
    C.recolor_faces_where(
        plank, lambda n, c: abs(n[1]) > 0.70 and c[0] < -0.60, WOOD_D)
    C.recolor_faces_where(plank, lambda n, c: n[2] < -0.55, WOOD_D)
    C.recolor_faces_where(plank, lambda n, c: abs(n[0]) > 0.55, WOOD_D)
    return _finish(a)


# --------------------------------------------------------------------------
# GROUP 2 -- misc
# --------------------------------------------------------------------------

AWN_RED = (0.90, 0.28, 0.22)
AWN_CREAM = (0.96, 0.92, 0.82)
AWN_TEAL = (0.14, 0.56, 0.56)
AWN_FRAME = (0.40, 0.42, 0.44)
PHONE_SHELL = (0.86, 0.34, 0.28)
PHONE_SHELL_DK = (0.62, 0.22, 0.19)
PHONE_TRIM = (0.90, 0.88, 0.80)
PHONE_DARK = (0.12, 0.12, 0.14)

def _helicopter():
    a = C.Asset("misc_helicopter", "misc")

    cabin_st = [
        (-2.30, 0.16, 0.14, 1.02),
        (-2.10, 0.36, 0.32, 1.03),
        (-1.82, 0.56, 0.52, 1.04),
        (-1.42, 0.70, 0.66, 1.06),
        (-0.95, 0.78, 0.76, 1.07),
        (-0.45, 0.76, 0.72, 1.06),
        (0.02, 0.66, 0.60, 1.04),
        (0.34, 0.48, 0.44, 1.02),
        (0.58, 0.33, 0.31, 1.00),
    ]

    def cabin_rings(stations, dilate=1.0):
        out = []
        for (y, hx, hz, cz) in stations:
            hx, hz = hx * dilate, hz * dilate
            prof = C.rounded_rect(hx, hz, 0.80 * min(hx, hz), 3)
            out.append([(pz, y, cz + px) for (px, pz) in prof])
        return out

    cabin = C.Part("heli_cabin", base_color=CREAM, flat=False, roughness=0.34)
    C.loft(cabin.mesh, cabin_rings(cabin_st), CREAM, smooth=True,
           cap_start_flip=True, cap_end_flip=False)
    C.recolor_faces_where(
        cabin, lambda n, c: n[2] < -0.45 and c[2] < 0.55, METAL_DK)
    a.add(cabin)

    glass = C.Part("heli_glass", base_color=GLASS_DK, flat=False, roughness=0.10)
    C.loft(glass.mesh, cabin_rings(cabin_st[1:6], 1.055), GLASS_DK,
           smooth=True, cap_start_flip=True, cap_end_flip=False)
    a.add(glass)

    boom = C.Part("heli_boom", base_color=CREAM, flat=False, roughness=0.38)
    boom_st = [
        (0.52, 0.31, 0.28, 1.00),
        (1.00, 0.25, 0.22, 1.05),
        (1.60, 0.20, 0.18, 1.12),
        (2.20, 0.16, 0.15, 1.18),
        (2.80, 0.13, 0.12, 1.23),
        (3.28, 0.11, 0.10, 1.27),
    ]
    C.loft(boom.mesh, cabin_rings(boom_st), CREAM, smooth=True,
           cap_start_flip=True, cap_end_flip=False)
    C.recolor_faces_where(
        boom, lambda n, c: c[2] > 2.30 and n[2] < 0.5, CORAL)
    a.add(boom)

    tail = a.part("heli_tail", base_color=CREAM, roughness=0.38)
    _rbox(tail.mesh, (0.09, 0.75, 0.60), center=(0.0, 3.02, 1.66),
          rot=(-14.0, 0.0, 0.0), color=CREAM)
    _rbox(tail.mesh, (0.08, 0.55, 0.44), center=(0.0, 3.20, 2.12),
          rot=(-24.0, 0.0, 0.0), color=CORAL)
    _rbox(tail.mesh, (1.35, 0.26, 0.07), center=(0.0, 2.84, 1.36),
          color=METAL)
    for sx in (-1, 1):
        _rbox(tail.mesh, (0.06, 0.30, 0.26), center=(sx * 0.64, 2.84, 1.48),
              color=CREAM)
    for i in range(4):                                  # tail rotor stubs
        ang = math.radians(45.0 + 90.0 * i)
        _rbox(tail.mesh, (0.05, 0.30, 0.06),
              center=(0.10, 3.32 + 0.16 * math.cos(ang),
                      1.66 + 0.16 * math.sin(ang)),
              rot=(0.0, 0.0, 0.0), color=METAL_DK)

    rotor = a.part("heli_rotor", base_color=METAL_DK, metallic=0.5,
                   roughness=0.38)
    C.cylinder(rotor.mesh, 0.090, 0.46, 10, center=(0.0, -0.55, 2.05),
               color=METAL_DK)
    C.cylinder(rotor.mesh, 0.165, 0.13, 12, center=(0.0, -0.55, 2.30),
               color=METAL)
    C.cylinder(rotor.mesh, 0.115, 0.05, 10, center=(0.0, -0.55, 1.96),
               radius_top=0.16, color=METAL)
    # Blade = 1.42 m half-span, so a 2.84 m disc: about half the fuselage
    # length, which is the proportion a small utility helicopter actually flies.
    for i in range(4):
        ang = math.radians(20.0 + 90.0 * i)
        _rbox(rotor.mesh, (2.84, 0.22, 0.032),
              center=(0.72 * math.cos(ang), -0.55 + 0.72 * math.sin(ang),
                      2.37),
              rot=(4.0, 0.0, math.degrees(ang)), color=DARK)
        _rbox(rotor.mesh, (0.24, 0.24, 0.055),
              center=(0.20 * math.cos(ang), -0.55 + 0.20 * math.sin(ang),
                      2.37),
              rot=(4.0, 0.0, math.degrees(ang)), color=METAL)

    skids = a.part("heli_skids", base_color=METAL, metallic=0.45,
                   roughness=0.40)
    for sx in (-1, 1):
        x = sx * 0.88
        C.tube(skids.mesh, [(x, -2.05, 0.42), (x, -1.78, 0.10),
                            (x, -1.60, 0.035), (x, 0.60, 0.035),
                            (x, 0.82, 0.09)], 0.035, 6, color=METAL)
        for y in (-1.35, 0.15):
            C.tube(skids.mesh, [(sx * 0.55, y, 0.33), (x, y, 0.065)],
                   0.035, 6, color=METAL)
    _rbox(skids.mesh, (1.10, 0.09, 0.09), center=(0.0, -1.35, 0.20),
          rot=(0.0, 0.0, 0.0), color=METAL_DK)

    # The contract puts the origin at the centre of the GROUND FOOTPRINT, and
    # the cabin was authored with its nose at -Y and the tail swinging back to
    # +Y, which leaves the footprint running -2.3..+3.6 in Y.  Shifting by the
    # midpoint puts the origin between the skids under the rotor mast.
    lo, hi = a.bounds()
    _shift(a.parts, (0.0, -(lo[1] + hi[1]) * 0.5, 0.0))
    return _finish(a)


def _helipad():
    """Ground marking built entirely from closed, wafer-thin solids.

    Nothing here is an open plane: every element is a closed box or cylinder
    lying in the XZ footprint, so the exporter's signed-volume outward check
    decides it correctly and no ``outward`` declaration is needed.  The ring is
    28 radial plates rather than a stacked disc, because paint is a 2D band on
    a flat pad -- a solid disc one level higher reads as a wedding cake, not a
    marking.
    """
    a = C.Asset("misc_helipad_marking", "misc")

    pad = a.part("helipad_pad", base_color=ASPHALT, roughness=0.85)
    C.cylinder(pad.mesh, 2.20, 0.030, 24, center=(0.0, 0.0, 0.015),
               color=ASPHALT)

    ring = a.part("helipad_ring", base_color=PAINT_WHITE, roughness=0.70)
    n_seg = 28
    for i in range(n_seg):
        ang = 2.0 * math.pi * i / n_seg
        r_mid = 1.86
        # tangent length is 2*pi*r/n_seg, so consecutive plates meet end to end
        _rbox(ring.mesh, (0.30, 2.0 * math.pi * r_mid / n_seg * 1.06, 0.014),
              center=(r_mid * math.cos(ang), r_mid * math.sin(ang), 0.037),
              rot=(0.0, 0.0, math.degrees(ang) + 90.0), color=PAINT_WHITE)

    letter = a.part("helipad_h", base_color=PAINT_WHITE, roughness=0.70)
    for sx in (-1, 1):
        C.box(letter.mesh, (0.30, 1.40, 0.014),
              center=(sx * 0.52, 0.0, 0.037), color=PAINT_WHITE)
    C.box(letter.mesh, (1.34, 0.30, 0.014), center=(0.0, 0.0, 0.037),
          color=PAINT_WHITE)

    ticks = a.part("helipad_ticks", base_color=PAINT_WHITE, roughness=0.70)
    for i in range(4):
        ang = math.radians(45.0 + 90.0 * i)
        # r = 1.99 keeps the plate's outer corner inside the 2.20 pad
        _rbox(ticks.mesh, (0.36, 0.12, 0.012),
              center=(1.99 * math.cos(ang), 1.99 * math.sin(ang), 0.036),
              rot=(0.0, 0.0, math.degrees(ang)), color=PAINT_WHITE)
    return _finish(a)


def _bin_mesh(m, body, lid_col, wheel):
    """One wheelie bin, built about its own base centre."""
    rings = []
    for (z, hx, hy) in ((0.06, 0.29, 0.25), (0.82, 0.37, 0.31)):
        prof = C.rounded_rect(hx, hy, 0.75 * min(hx, hy), 2)
        rings.append([(x, y, z) for (x, y) in prof])
    C.loft(m, rings, body, cap_start_flip=True, cap_end_flip=False)
    _rbox(m, (0.82, 0.72, 0.07), center=(0.0, 0.0, 0.855), color=lid_col)
    _rbox(m, (0.26, 0.07, 0.05), center=(0.0, 0.0, 0.905), color=METAL_DK)
    _rbox(m, (0.30, 0.20, 0.10), center=(0.0, -0.36, 0.70), color=METAL_DK)
    for sx in (-1, 1):
        C.cylinder(m, 0.070, 0.05, 6, center=(sx * 0.26, 0.26, 0.07),
                   axis="X", color=wheel)


def _bag_mesh(m, body, knot):
    C.sphere(m, 0.44, 10, 6, center=(0.0, 0.0, 0.317), squash=0.72,
             color=body)
    C.cone(m, 0.10, 0.22, 8, center=(0.03, -0.04, 0.60), radius_top=0.035,
           color=knot)
    C.tube(m, [(0.03, -0.04, 0.66), (0.10, -0.12, 0.72), (0.02, -0.20, 0.66)],
           0.028, 5, color=knot)


def _dumpster_pile():
    """Three tipped bins and two bags, each seated on whatever came before.

    ``_rest`` seats every element on the highest support vertex under its own
    footprint, so the heap climbs instead of three bins standing side by side
    on the pavement -- which is what plain ``_drop(..., 0.0)`` produced.
    """
    a = C.Asset("misc_dumpster_pile", "misc")
    heap = []
    # bin_c leans back on the others and bag_a rides on the crest; the rest sit
    # flat on the pavement.  CAP is what keeps the last two from climbing the
    # top of the heap and turning it into a tower.
    CAP = 0.95

    bin_a = C.Part("pile_bin_a", base_color=TEAL_DK, roughness=0.60)
    _bin_mesh(bin_a.mesh, TEAL_DK, TEAL, DARK)
    _place([bin_a], offset=(-0.66, 0.10, 0.0), rot=(0.0, 5.0, 22.0))
    _rest([bin_a], heap)
    a.add(bin_a)
    heap.append(bin_a)

    bin_b = C.Part("pile_bin_b", base_color=(0.72, 0.24, 0.20), roughness=0.60)
    _bin_mesh(bin_b.mesh, (0.72, 0.24, 0.20), CORAL_DK, DARK)
    _place([bin_b], offset=(0.68, 0.34, 0.0), rot=(-92.0, 0.0, -38.0))
    _rest([bin_b], heap, cap=CAP)
    a.add(bin_b)
    heap.append(bin_b)

    bin_c = C.Part("pile_bin_c", base_color=(0.34, 0.36, 0.22), roughness=0.60)
    _bin_mesh(bin_c.mesh, (0.34, 0.36, 0.22), METAL_DK, DARK)
    _place([bin_c], offset=(-0.18, -0.62, 0.0), rot=(34.0, 0.0, 74.0))
    _rest([bin_c], heap)
    a.add(bin_c)
    heap.append(bin_c)

    bag_a = C.Part("pile_bag_a", base_color=(0.16, 0.16, 0.18), flat=False,
                   roughness=0.42)
    _bag_mesh(bag_a.mesh, (0.16, 0.16, 0.18), DARK)
    _place([bag_a], offset=(0.52, -0.52, 0.0), rot=(0.0, 0.0, 14.0))
    _rest([bag_a], heap, sink=0.06, cap=CAP)
    a.add(bag_a)
    heap.append(bag_a)

    bag_b = C.Part("pile_bag_b", base_color=(0.20, 0.28, 0.42), flat=False,
                   roughness=0.42)
    _bag_mesh(bag_b.mesh, (0.20, 0.28, 0.42), TEAL_DK)
    _place([bag_b], offset=(-0.42, 0.42, 0.0), rot=(0.0, 0.0, -24.0))
    _rest([bag_b], heap, sink=0.06, cap=CAP)
    a.add(bag_b)
    heap.append(bag_b)

    crates = [((0.44, 0.30, 0.15), (-0.10, -0.44, 34.0), PAPER),
              ((0.30, 0.24, 0.20), (0.92, -0.22, -22.0), CREAM),
              ((0.26, 0.22, 0.16), (0.02, 0.16, 68.0), (0.86, 0.82, 0.70)),
              ((0.22, 0.18, 0.12), (-0.72, 0.40, 12.0), (0.78, 0.74, 0.62))]
    for ci, ((size, (x, y, roll), col)) in enumerate(crates):
        # Each crate has its own colour, so it needs its own Part; the name is
        # indexed because part names must be unique within an asset.
        box = C.Part("litter_crate_%d" % ci, base_color=col, roughness=0.85)
        _rbox(box.mesh, size, color=col)
        _place([box], offset=(x, y, 0.0), rot=(0.0, 0.0, roll))
        _rest([box], heap, sink=0.06, cap=CAP)
        a.add(box)
    return _finish(a)


def _graffiti_board():
    a = C.Asset("misc_graffiti_board", "misc")

    feet = a.part("board_feet", base_color=METAL_DK, metallic=0.3,
                  roughness=0.6)
    for sx in (-1, 1):
        _rbox(feet.mesh, (0.40, 0.46, 0.16), center=(sx * 0.82, 0.0, 0.08),
              color=METAL_DK)
        _rbox(feet.mesh, (0.12, 0.11, 1.10), center=(sx * 0.82, 0.11, 0.70),
              color=METAL_DK)

    panel = a.part("board_panel", base_color=CREAM, roughness=0.75)
    C.box(panel.mesh, (2.40, 0.10, 1.80), center=(0.0, 0.0, 1.06),
          color=CREAM)

    rails = a.part("board_frame", base_color=(0.26, 0.28, 0.30), roughness=0.6)
    for z in (0.13, 1.99):
        _rbox(rails.mesh, (2.48, 0.15, 0.08), center=(0.0, 0.0, z),
              color=(0.26, 0.28, 0.30))
    for sx in (-1, 1):
        _rbox(rails.mesh, (0.08, 0.15, 1.86), center=(sx * 1.23, 0.0, 1.06),
              color=(0.26, 0.28, 0.30))
    _rbox(rails.mesh, (2.20, 0.06, 0.06), center=(0.0, 0.0, 0.62),
          color=(0.26, 0.28, 0.30))

    # Abstract tag: overlapping colour plates pinned to the front face.
    tag = C.Part("board_tag", base_color=MAGENTA, roughness=0.45)
    plates = [
        ((1.05, 0.030, 0.62), (0.05, -0.065, 1.32), -8.0, MAGENTA),
        ((0.72, 0.030, 0.82), (-0.54, -0.072, 1.28), 12.0, CYAN),
        ((0.60, 0.030, 0.55), (0.63, -0.070, 1.44), 20.0, YELLOW),
        ((0.42, 0.030, 0.90), (0.20, -0.068, 0.98), -14.0, ORANGE),
        ((0.32, 0.030, 0.34), (-0.86, -0.070, 1.64), 26.0, WHITE),
        ((0.52, 0.030, 0.30), (0.56, -0.070, 0.74), -6.0, GREEN),
        ((0.26, 0.030, 0.26), (-0.30, -0.074, 0.62), 18.0, PINK),
        ((0.90, 0.026, 0.07), (-0.10, -0.076, 1.66), -24.0, DARK),
        ((0.74, 0.026, 0.06), (0.30, -0.076, 1.10), 9.0, DARK),
        ((0.58, 0.026, 0.06), (-0.46, -0.076, 0.86), -13.0, CREAM),
    ]
    for (size, center, roll, col) in plates:
        _rbox(tag.mesh, size, center=center, rot=(0.0, roll, 0.0), color=col)
    a.add(tag)

    bolt = a.part("board_bolts", base_color=METAL, metallic=0.4, roughness=0.4)
    for sx in (-1, 1):
        for z in (0.24, 1.06, 1.88):
            C.cylinder(bolt.mesh, 0.035, 0.04, 8,
                       center=(sx * 1.10, -0.070, z), axis="Y", color=METAL)
    return _finish(a)


# --------------------------------------------------------------------------
# GROUP 3 -- storefront misc (awning, street payphone)
# --------------------------------------------------------------------------

def _awning():
    """A sloped striped shop awning on two bracket arms, ~2.5 x 1.2 x 2.0 m.

    The canvas is a single closed slab lofted across six stations, which keeps
    it watertight (so the exporter's signed-volume check decides its
    outwardness with no declared reference) and lets the stripes be per-face
    colour rather than interleaved strips.  That trick only works here because
    the loft subdivides the span into 5 X-bays; the construction barrier's
    plank is a plain box with one face pair, so its stripes had to be real
    geometry.
    """
    a = C.Asset("misc_awning", "misc")

    half_w = 1.20                     # half span along X
    depth = 1.05                      # how far the canvas reaches over -Y
    z_wall, z_edge = 1.92, 1.44       # the slope: 0.48 m over 1.05 m
    thick = 0.055

    # -- the striped canvas -------------------------------------------------
    canvas = a.part("awning_canvas", base_color=AWN_RED, roughness=0.62)
    xs = (-half_w, -half_w * 0.6, -half_w * 0.2, half_w * 0.2,
          half_w * 0.6, half_w)
    rings = []
    for x in xs:
        top = [(x, -depth * (j / 5.0), z_wall + (z_edge - z_wall) * (j / 5.0))
               for j in range(6)]
        bot = [(x, -depth * (j / 5.0),
                z_wall + (z_edge - z_wall) * (j / 5.0) - thick)
               for j in range(6)]
        # The cross-section is a simple CLOSED loop: down the top edge, back up
        # the bottom edge in REVERSE.  Emitting both edges in the same
        # direction instead makes the ring self-overlap -- the two long
        # diagonals cut across the slab and collapse four corner quads into
        # zero-area triangles, which the exporter then (correctly) rejects.
        # Reversed, the loop is counter-clockwise in (y, z), so its right-hand
        # normal is Y x Z = +X -- the direction the rings advance in.
        rings.append(top + bot[::-1])
    C.loft(canvas.mesh, rings, AWN_RED, cap_start_flip=True, cap_end_flip=False)
    # Stripes run DOWN the slope, so the band index follows the face centroid's
    # X -- not its normal, which points every which way across a sloped sheet.
    C.recolor_faces_where(
        canvas, lambda n, c: int((c[0] + half_w) / 0.30) % 2 == 1, AWN_CREAM)
    # Teal valance scallop along the front lip: the bit that says "shopfront".
    C.recolor_faces_where(
        canvas, lambda n, c: n[2] < -0.55, AWN_TEAL)

    # -- frame + bracket arms ----------------------------------------------
    frame = a.part("awning_frame", base_color=AWN_FRAME, metallic=0.35,
                   roughness=0.55)
    _rbox(frame.mesh, (half_w * 2.04, 0.07, 0.075),
          center=(0.0, 0.0, z_wall), color=AWN_FRAME)            # wall rail
    _rbox(frame.mesh, (half_w * 2.04, 0.07, 0.075),
          center=(0.0, -depth, z_edge), color=AWN_FRAME)          # front bar
    # Square-tipped valance panels hanging off the front bar.  They are flat
    # rectangles, NOT scallops -- a scalloped edge needs a real arc of
    # geometry, and naming them that would be a comment the render contradicts.
    for i in range(7):
        x = -half_w * 0.92 + half_w * 1.84 * i / 6.0
        _rbox(frame.mesh, (half_w * 0.30, 0.045, 0.20),
              center=(x, -depth - 0.01, z_edge - 0.16), color=AWN_TEAL)
    for sx in (-1, 1):
        x = sx * (half_w - 0.08)
        # Diagonal stay from the wall down to the front bar -- the arm that
        # makes it an awning rather than a shelf.
        _rbox(frame.mesh, (0.055, 1.12, 0.055),
              center=(x, -depth * 0.5, (z_wall + z_edge) * 0.5 - 0.06),
              rot=(24.0, 0.0, 0.0), color=AWN_FRAME)
        _rbox(frame.mesh, (0.070, 0.070, 0.22),
              center=(x, 0.0, z_wall + 0.10), color=AWN_FRAME)    # wall stub
        _rbox(frame.mesh, (0.055, 0.055, 0.50),
              center=(x, -depth + 0.02, z_edge - 0.22),
              rot=(-20.0, 0.0, 0.0), color=AWN_FRAME)            # front leg
    return _finish(a)


def _street_phone():
    """A kerbside payphone: slim post, rain hood, and a handset on a cord.

    Deliberately NOT a full booth -- ``prop_phone_booth`` already covers the
    enclosed version, and this one is the cheap open-air unit that sits on a
    traffic island.
    """
    a = C.Asset("misc_street_phone", "misc")

    post_h = 1.16
    shell = a.part("phone_post", base_color=PHONE_SHELL, metallic=0.25,
                   roughness=0.58)
    C.cylinder(shell.mesh, 0.20, 0.075, 12, center=(0.0, 0.0, 0.0375),
               color=PHONE_SHELL_DK)                             # cast base
    C.cylinder(shell.mesh, 0.075, post_h, 10, center=(0.0, 0.0, post_h * 0.5),
               color=PHONE_SHELL)
    C.cylinder(shell.mesh, 0.088, 0.050, 10, center=(0.0, 0.0, 0.085),
               color=PHONE_SHELL_DK)                             # base collar

    # -- hood: the little roof over the dial --------------------------------
    hood = a.part("phone_hood", base_color=PHONE_SHELL, roughness=0.55)
    C.box(hood.mesh, (0.46, 0.30, 0.40), center=(0.0, 0.0, post_h + 0.26),
          color=PHONE_SHELL)                                    # body
    _rbox(hood.mesh, (0.54, 0.38, 0.055), center=(0.0, -0.02, post_h + 0.485),
          rot=(7.0, 0.0, 0.0), color=PHONE_SHELL_DK)            # rain hood
    _rbox(hood.mesh, (0.50, 0.34, 0.045), center=(0.0, 0.0, post_h + 0.045),
          color=PHONE_SHELL_DK)                                 # skirt

    trim = a.part("phone_trim", base_color=PHONE_TRIM, roughness=0.5)
    C.box(trim.mesh, (0.30, 0.030, 0.150), center=(0.0, -0.152, post_h + 0.30),
          color=PHONE_TRIM)                                     # dial bezel
    C.box(trim.mesh, (0.34, 0.026, 0.038), center=(0.0, -0.152, post_h + 0.435),
          color=PHONE_TRIM)                                     # number strip
    C.box(trim.mesh, (0.05, 0.026, 0.16), center=(-0.20, -0.152, post_h + 0.05),
          color=PHONE_TRIM)                                     # coin slot door

    # -- handset in its cradle ---------------------------------------------
    handset = a.part("phone_handset", base_color=PHONE_DARK, roughness=0.45)
    _rbox(handset.mesh, (0.085, 0.115, 0.235),
          center=(0.225, -0.115, post_h + 0.30), rot=(0.0, 11.0, 0.0),
          color=PHONE_DARK)                                     # body
    for sz in (0.19, 0.41):                                     # ear + mouth caps
        _rbox(handset.mesh, (0.135, 0.135, 0.085),
              center=(0.245, -0.115, post_h + sz),
              rot=(0.0, 11.0, 0.0), color=PHONE_DARK)
    _rbox(handset.mesh, (0.095, 0.075, 0.075), center=(0.225, -0.060, post_h + 0.30),
          rot=(0.0, 11.0, 0.0), color=PHONE_DARK)                # cradle hook

    # Coiled cord: a short zigzag tube from the handset down to the body.  The
    # zigzag is what makes it read as a cord; a straight tube reads as a wire
    # and, at this scale, as a stray vertex.
    cord = a.part("phone_cord", base_color=PHONE_DARK, roughness=0.6)
    path = []
    for i in range(7):
        t = i / 6.0
        path.append((0.245 - 0.02 * t,
                     -0.060 + 0.012 * math.sin(math.pi * i),
                     post_h + 0.185 - 0.16 * t))
    C.tube(cord.mesh, path, 0.011, 5, color=PHONE_DARK, smooth=False)
    return _finish(a)


# --------------------------------------------------------------------------

def build_all():
    """Return the ordered list: 4 beach props + 6 misc props."""
    return [
        _umbrella(),
        _beach_chair(),
        _surfboard(),
        _boardwalk_plank(),
        _helicopter(),
        _helipad(),
        _dumpster_pile(),
        _graffiti_board(),
        _awning(),
        _street_phone(),
    ]
