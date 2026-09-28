"""Palms: the three roadside silhouettes that line a Miami boulevard.

Every tree is procedural geometry assembled from the ``core`` kernel: a swept,
tapering, slightly curved trunk; darker banded rings left over from the leaf
bases it carried last season; and a crown of drooping blades lofted along their
own arcs.  No imported or traced assets -- the arcs and the ring spacing are
driven by fixed tables so the set is identical on every build.

Authoring is Blender-style Z-up, origin at the centre of the trunk base, the
tree resting on z = 0.  ``build_all`` returns ``palm_tall`` (~9 m, slender,
7 blades), ``palm_short`` (~4.5 m, stout, 6 broad blades) and ``palm_bushy``
(~6 m, 8 blades in two whorls plus a skirt of dead fronds under the crown).
"""

import math

from .. import core as C

# --------------------------------------------------------------------------
# palette -- warm sand bark against two Miami greens
# --------------------------------------------------------------------------

BARK = (0.53, 0.42, 0.30)
BARK_RING = (0.32, 0.24, 0.16)
BARK_SHAFT = (0.45, 0.35, 0.24)

LEAF_A_TOP = (0.33, 0.64, 0.28)
LEAF_A_UND = (0.17, 0.39, 0.18)
LEAF_B_TOP = (0.46, 0.73, 0.31)
LEAF_B_UND = (0.24, 0.47, 0.20)

DEAD_TOP = (0.53, 0.39, 0.21)
DEAD_UND = (0.33, 0.23, 0.12)

COCO = (0.34, 0.24, 0.14)
COCO_CUT = (0.46, 0.34, 0.20)


# --------------------------------------------------------------------------
# trunk curve
#
# One parameterisation serves the whole tree.  Below ``t0`` the trunk is dead
# vertical, and only above it does the sideways offset kick in:
#
#     d(t) = lean * ((max(0, t - t0) / (1 - t0)) ** 2.4)
#
# That vertical bole is not decoration.  core.tube() builds each ring from a
# tangent, and the first tangent is the plain secant from point 0 to point 1 --
# so on a trunk that starts leaning immediately, that ring is already tilted,
# and its lowest vertex lands r * sin(tilt) BELOW z = 0.  On palm_tall that is
# -0.6 mm of geometry under the road: invisible, and exactly the kind of thing
# that makes a tree sink into the pavement.  A vertical lower bole makes the
# first secant exactly +Z, so the base ring is level and its lowest vertex is
# the base cap centre at z = 0.
# --------------------------------------------------------------------------

def _curve(cfg, t):
    """A point on the trunk centreline at normalised height ``t``."""
    s = max(0.0, t - cfg["t0"]) / (1.0 - cfg["t0"])
    d = cfg["lean"] * (s ** 2.4)
    a = cfg["heading"]
    return (d * math.cos(a), d * math.sin(a), cfg["height"] * t)


def _radius(cfg, t):
    """Trunk radius at ``t``: wide at the foot, tapering to the crown."""
    return cfg["r_top"] + (cfg["r_base"] - cfg["r_top"]) * ((1.0 - t) ** 0.85)


def _at(cfg, u):
    """(point, unit tangent, radius) at normalised height ``u``."""
    t = min(1.0, max(0.0, u))
    h = 2.0e-3
    p = _curve(cfg, t)
    lo = _curve(cfg, max(0.0, t - h))
    hi = _curve(cfg, min(1.0, t + h))
    return p, C.normalize(C.sub(hi, lo)), _radius(cfg, t)


def _trunk(a, cfg, name):
    """Swept trunk + banded rings + crown shaft.

    Returns the point the fronds should radiate from.
    """
    trunk = C.Part("%s_trunk" % name, base_color=BARK, roughness=0.92,
                   flat=False)
    n = cfg["stations"]
    # Stations run t = 0 .. 1 inclusive.  Every gap is metres apart, far above
    # tube()'s 1e-6 dedup gate, so path and radii stay the same length and
    # radii[] indexes the rings directly.
    path = [_curve(cfg, i / float(n - 1)) for i in range(n)]
    radii = [_radius(cfg, i / float(n - 1)) for i in range(n)]
    C.tube(trunk.mesh, path, radii[0], cfg["seg"], smooth=True, radii=radii,
           color=BARK)
    a.add(trunk)

    # Leaf-base scars: a short, slightly barrel-shaped collar pushed just proud
    # of the trunk at each scar height, so it reads as a raised band rather
    # than a stripe painted on a cylinder.
    rings = C.Part("%s_rings" % name, base_color=BARK_RING, roughness=0.95)
    count = cfg["rings"]
    # First band clears the ground by its own half-height (plus a hair of
    # tangent tilt), the last sits just under the crown shaft: a scar collar
    # half-buried at z = 0 would poke the asset below the ground plane.
    half = 0.5 * cfg["band_h"]
    u0 = min(0.35, 1.2 * half / cfg["height"])
    for k in range(count):
        u = u0 + (1.0 - 0.015 - u0) * (k / float(count - 1))
        p, tan, rad = _at(cfg, u)
        r0 = rad + cfg["bulge"]
        C.tube(rings.mesh,
               [C.sub(p, C.mul(tan, half)), p, C.add(p, C.mul(tan, half))],
               r0, cfg["seg"], color=BARK_RING,
               radii=[r0 * 0.955, r0, r0 * 0.955])
    a.add(rings)

    # Crown shaft: the boots the fronds emerge from, flaring upward.
    top, _, rtop = _at(cfg, 1.0)
    shaft = C.Part("%s_crown" % name, base_color=BARK_SHAFT,
                   roughness=0.88, flat=False)
    C.cone(shaft.mesh, rtop * 0.94, cfg["shaft"], 10, smooth=True,
           center=(top[0], top[1], top[2] + cfg["shaft"] * 0.5),
           radius_top=rtop * 1.5, color=BARK_SHAFT)
    a.add(shaft)
    return (top[0], top[1], top[2] + cfg["shaft"] * 0.86)


# --------------------------------------------------------------------------
# fronds
# --------------------------------------------------------------------------

def _face_normal(m, i):
    """The geometric normal of face ``i``, computed in isolation."""
    a, b, c = m.faces[i]
    return C.normalize(C.cross(C.sub(m.pos[b], m.pos[a]), C.sub(m.pos[c], m.pos[a])))


def _frond(m, base, azimuth, length, rise, droop, wmax, thick,
           c_top, c_under, stations=9, seg=8, sway=0.0):
    """One palm blade: a flattened, tapering loft along a drooping arc.

    The centreline rises at ``rise`` per unit length and loses it again at
    ``droop`` u^2, so the blade leaves the crown upward, peaks at
    u = rise / (2 * droop) and hangs off the end -- every frond gets its own
    peak height purely from its numbers.  ``sway`` bows the blade sideways in
    its own plane so no two blades mirror each other.

    Cross-sections are ellipses: wide along the horizontal normal ``s``, thin
    along ``up = t x s``.  ``s x up == t`` is the right-hand rule loft() needs
    to keep the shell outward.
    """
    dx, dy = math.cos(azimuth), math.sin(azimuth)
    px, py = -dy, dx
    pts = []
    for i in range(stations):
        u = i / float(stations - 1)
        r = length * u
        s = sway * (u * u)
        pts.append((base[0] + dx * r + px * s,
                    base[1] + dy * r + py * s,
                    base[2] + rise * u - droop * (u * u)))

    rings = []
    for i, p in enumerate(pts):
        if i == 0:
            t = C.sub(pts[1], pts[0])
        elif i == len(pts) - 1:
            t = C.sub(pts[-1], pts[-2])
        else:
            t = C.sub(pts[i + 1], pts[i - 1])
        t = C.normalize(t)
        ref = C.cross((0.0, 0.0, 1.0), t)
        if C.length(ref) < 0.25:
            # A near-vertical tangent collapses cross(z, t); fall back to the
            # blade's own horizontal perpendicular instead.
            ref = (px, py, 0.0)
        s_ax = C.normalize(ref)
        up = C.cross(t, s_ax)
        u = i / float(stations - 1)
        w = wmax * (0.30 + 0.70 * math.sin(math.pi * (u ** 0.62))) \
            * max((1.0 - u) ** 0.5, 0.09)
        th = max(thick * (1.0 - 0.45 * u), 0.35 * thick)
        rings.append([C.add(C.add(p, C.mul(s_ax, w * math.cos(a))),
                            C.mul(up, th * math.sin(a)))
                      for a in [2.0 * math.pi * k / seg for k in range(seg)]])

    first = len(m.faces)
    C.loft(m, rings, c_top, cap_start_flip=True, cap_end_flip=False)
    # loft() emits the side walls first (seg quads per station gap), then the
    # two cap fans; no ring here can collapse, so the split is exact.  Only the
    # side walls get the underside tone -- the caps keep the lit colour.
    for i in range(first, len(m.faces) - 2 * seg):
        if _face_normal(m, i)[2] < 0.0:
            m.fcol[i] = c_under
    return m


def _blade(m, base, az_deg, length, rise, droop, wmax, thick, stations=9,
           seg=8, sway=0.0, top=None, under=None):
    _frond(m, base, math.radians(az_deg), length, rise, droop, wmax, thick,
           top or LEAF_A_TOP, under or LEAF_A_UND, stations, seg, sway)


# --------------------------------------------------------------------------
# 1. palm_tall -- 9 m, slender, 7 blades, coconut cluster
# --------------------------------------------------------------------------

TALL = dict(height=8.30, lean=0.95, heading=18.0, t0=0.30,
            r_base=0.255, r_top=0.135, seg=9, stations=9, rings=13,
            band_h=0.20, bulge=0.026, shaft=0.46)


def _palm_tall():
    a = C.Asset("palm_tall", "palm")
    top = _trunk(a, TALL, "palm_tall")

    ga = C.Part("palm_tall_fronds_a", base_color=LEAF_A_TOP, roughness=0.72)
    gb = C.Part("palm_tall_fronds_b", base_color=LEAF_B_TOP, roughness=0.72)
    for k in range(7):
        part = ga if k % 2 == 0 else gb
        _blade(part.mesh, top,
               az_deg=12.0 + k * 51.43 + (7.0 if k % 2 else -7.0),
               length=3.30 + 0.22 * ((k * 3) % 3),
               rise=1.72 - 0.12 * (k % 2),
               droop=2.68 + 0.16 * (k % 3),
               wmax=0.26, thick=0.030,
               sway=0.20 * (1.0 if k % 2 else -1.0),
               top=LEAF_A_TOP if k % 2 == 0 else LEAF_B_TOP,
               under=LEAF_A_UND if k % 2 == 0 else LEAF_B_UND)
    a.add(ga)
    a.add(gb)

    # Coconut cluster tucked under the shaft, bunched on one side.
    nuts = a.part("palm_tall_coconuts", base_color=COCO, roughness=0.8)
    cz = TALL["height"] - 0.14
    axis, _, _ = _at(TALL, cz / TALL["height"])
    for k in range(6):
        ang = math.radians(95.0 + 17.0 * k)
        d = 0.19 + 0.05 * (k % 3)
        C.sphere(nuts.mesh, 0.105, 8, 4,
                 center=(axis[0] + d * math.cos(ang),
                         axis[1] + d * math.sin(ang),
                         cz + 0.10 * (k % 2)),
                 color=COCO)
    return a


# --------------------------------------------------------------------------
# 2. palm_short -- 4.5 m, stout, 6 broad blades
# --------------------------------------------------------------------------

SHORT = dict(height=3.83, lean=0.42, heading=-35.0, t0=0.24,
             r_base=0.345, r_top=0.215, seg=10, stations=7, rings=9,
             band_h=0.26, bulge=0.032, shaft=0.40)


def _palm_short():
    a = C.Asset("palm_short", "palm")
    top = _trunk(a, SHORT, "palm_short")

    ga = C.Part("palm_short_fronds_a", base_color=LEAF_A_TOP, roughness=0.72)
    gb = C.Part("palm_short_fronds_b", base_color=LEAF_B_TOP, roughness=0.72)
    for k in range(6):
        part = ga if k % 2 == 0 else gb
        _blade(part.mesh, top,
               az_deg=25.0 + k * 60.0 + (9.0 if k % 2 else -9.0),
               length=2.45 + 0.18 * (k % 3),
               rise=1.70 + 0.08 * (k % 2),
               droop=2.48 + 0.18 * (k % 2),
               wmax=0.42, thick=0.040,
               sway=-0.26 if k % 2 else 0.26,
               top=LEAF_A_TOP if k % 2 == 0 else LEAF_B_TOP,
               under=LEAF_A_UND if k % 2 == 0 else LEAF_B_UND)
    a.add(ga)
    a.add(gb)
    return a


# --------------------------------------------------------------------------
# 3. palm_bushy -- 6 m, two whorls of blades plus a dead-frond skirt
# --------------------------------------------------------------------------

BUSHY = dict(height=5.12, lean=0.70, heading=52.0, t0=0.28,
             r_base=0.295, r_top=0.185, seg=9, stations=8, rings=10,
             band_h=0.22, bulge=0.028, shaft=0.44)


def _palm_bushy():
    a = C.Asset("palm_bushy", "palm")
    top = _trunk(a, BUSHY, "palm_bushy")

    ga = C.Part("palm_bushy_fronds_a", base_color=LEAF_A_TOP, roughness=0.72)
    gb = C.Part("palm_bushy_fronds_b", base_color=LEAF_B_TOP, roughness=0.72)
    # Outer whorl: four long blades on the cardinals.
    for k in range(4):
        _blade((ga if k % 2 == 0 else gb).mesh, top,
               az_deg=6.0 + k * 90.0 + 11.0 * (k % 3),
               length=3.10 + 0.14 * (k % 3),
               rise=1.80 - 0.10 * (k % 2),
               droop=2.52 + 0.14 * (k % 3),
               wmax=0.27, thick=0.030,
               sway=0.22 if k % 2 else -0.22,
               top=LEAF_A_TOP if k % 2 == 0 else LEAF_B_TOP,
               under=LEAF_A_UND if k % 2 == 0 else LEAF_B_UND)
    # Inner whorl: four shorter, steeper blades filling the gaps.
    for k in range(4):
        _blade((ga if k % 2 else gb).mesh, top,
               az_deg=44.0 + k * 90.0 - 9.0 * (k % 2),
               length=2.28 + 0.14 * ((k * 3) % 3),
               rise=2.10, droop=2.26 + 0.12 * (k % 3),
               wmax=0.25, thick=0.028,
               sway=-0.18 if k % 2 else 0.18,
               top=LEAF_B_TOP if k % 2 == 0 else LEAF_A_TOP,
               under=LEAF_B_UND if k % 2 == 0 else LEAF_A_UND)
    a.add(ga)
    a.add(gb)

    # Skirt of dead fronds hanging off the trunk just below the crown.
    sz = BUSHY["height"] - 1.05
    axis, _, rad = _at(BUSHY, sz / BUSHY["height"])
    dead = C.Part("palm_bushy_skirt", base_color=DEAD_TOP, roughness=0.9)
    for k in range(5):
        ang = math.radians(30.0 + k * 72.0 + 6.0 * (k % 2))
        push = (axis[0] + (rad + 0.05) * math.cos(ang),
                axis[1] + (rad + 0.05) * math.sin(ang), sz)
        _blade(dead.mesh, push, math.degrees(ang) + 24.0,
               length=1.05 + 0.14 * (k % 3),
               rise=-0.28, droop=1.12 + 0.12 * (k % 3),
               wmax=0.21, thick=0.030,
               stations=7, seg=6, sway=0.14 if k % 2 else -0.14,
               top=DEAD_TOP, under=DEAD_UND)
    a.add(dead)
    return a


def build_all():
    """Return the ordered list of palm assets."""
    return [
        _palm_tall(),
        _palm_short(),
        _palm_bushy(),
    ]
