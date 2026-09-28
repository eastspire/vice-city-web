"""Category B: NEON BAY street vehicles.

Every vehicle shares one kit so the traffic reads as a single fleet:

* a **lofted rounded-rect shell** running along the length axis (``core.
  section_rings`` + ``core.loft`` over ``core.rounded_rect`` box sections --
  never a circle), driven by a silhouette control table plus a wheel-arch /
  nose / tail **sill** function;
* a second loft for the **greenhouse**, which also supplies the windscreen and
  backlight ramps;
* four Y-axis **wheels** (dark tyre + lighter hub) tucked under the arches;
* separate parts for **glass**, **lamps**, **trim** and the tyres;
* **bumpers, mirrors, door cut-lines** as thin inset boxes.

Authoring is Blender-style Z-up with one unit = one metre and the origin at
the centre of the ground footprint.  ``core.section_rings`` maps a profile's
horizontal axis onto world +Y ("vehicle width") and its vertical axis onto
world +Z, so the loft advances along the LENGTH axis X: **-X is the nose,
+X is the tail**, and Y is the width axis.

Nothing is positioned by hand: every surface-mounted detail asks the shell for
its own half-width at that (x, z) -- see :func:`_flank_y` and :func:`_ring` --
so trim cannot drift off the rounded shoulders and become floating slivers.

All geometry is original and procedural -- nothing is imported or traced.
"""

import math

from .. import core as C


# ----------------------------------------------------------------- palette --

TYRE = (0.075, 0.075, 0.085)
HUB = (0.60, 0.62, 0.65)
GLASS = (0.055, 0.090, 0.130)
CHROME = (0.74, 0.77, 0.80)
TRIM_DK = (0.13, 0.14, 0.16)
HEADLIGHT = (1.00, 0.95, 0.76)
TAILLIGHT = (0.80, 0.10, 0.11)

# NEON BAY original liveries (no reference vehicles).
WHITE = (0.93, 0.93, 0.91)
MIAMI_PINK = (0.94, 0.36, 0.52)
MIAMI_TEAL = (0.10, 0.62, 0.62)
CORAL = (0.93, 0.42, 0.26)
TAXI_YELLOW = (0.98, 0.78, 0.10)
COPPER_DK = (0.55, 0.34, 0.16)
POLICE_BLACK = (0.09, 0.10, 0.13)


# ------------------------------------------------------- silhouette reading --
# Control rows are (x, half_width, z_top, corner_radius) sampled along the
# length axis.  The sill line below them comes from the wheel arches and the
# nose/tail lift functions.


def _field(ctrl, x, k):
    """Linear interpolation of column ``k`` of a (x, ...) control table."""
    if x <= ctrl[0][0]:
        return ctrl[0][k]
    for i in range(len(ctrl) - 1):
        a, b = ctrl[i], ctrl[i + 1]
        if a[0] <= x <= b[0]:
            span = b[0] - a[0]
            t = 0.0 if span <= 0.0 else (x - a[0]) / span
            return a[k] + (b[k] - a[k]) * t
    return ctrl[-1][k]


def _arch_z(x, cfg):
    """Sill line: rides at ``sill`` but bulges to ``arch_top`` over each axle."""
    z = cfg["sill"]
    for ax in cfg["axles"]:
        d = abs(x - ax) / cfg["arch_half"]
        if d < 1.0:
            z = max(z, cfg["sill"]
                    + (cfg["arch_top"] - cfg["sill"]) * (1.0 - d * d))
    return z


def _end_z(x, cfg):
    """Sill lift over the nose and the tail (approach / departure angle)."""
    z = cfg["sill"]
    for ex, ez in cfg["ends"]:
        t = abs(x - ex) / cfg["end_span"]
        if t < 1.0 and ez > z:
            z = max(z, cfg["sill"] + (ez - cfg["sill"]) * (1.0 - t))
    return z


def _shell_z0(cfg, x):
    """Bottom of the shell ring at station x."""
    return max(_arch_z(x, cfg), _end_z(x, cfg))


def _arch_xs(cfg):
    """Extra stations so the wheel arch is faceted, not one steep ramp."""
    ah = cfg["arch_half"]
    out = []
    for ax in cfg["axles"]:
        for f in (-1.0, -0.55, 0.0, 0.55, 1.0):
            out.append(ax + ah * f)
    return out


def _ring(ctrl, z0_fn, x):
    """(half_width, half_height, lift, radius, z_bottom, z_top) at station x.

    This mirrors exactly what ``_shell_rings`` hands to ``core.rounded_rect``,
    so a detail placed with it lands on the surface that actually gets lofted.
    """
    z_top = _field(ctrl, x, 2)
    z_bot = min(z0_fn(x), z_top - 0.05)
    hx = max(0.05, _field(ctrl, x, 1))
    hy = max(0.02, (z_top - z_bot) * 0.5)
    # rounded_rect clamps r itself, but clamping here as well keeps every ring
    # at the same point count (a radius clamped to zero drops the arcs and
    # changes the ring cardinality, which loft rejects).
    r = max(0.02, min(_field(ctrl, x, 3), hx * 0.9, hy * 0.9))
    return hx, hy, (z_top + z_bot) * 0.5, r, z_bot, z_top


def _half_w(ring, z):
    """Half-width of a rounded-rect ring at height ``z``."""
    hx, hy, lift, r, _z_bot, _z_top = ring
    py = max(-hy, min(hy, z - lift))
    flat = hy - r
    if abs(py) <= flat:
        return hx
    d = abs(py) - flat
    return (hx - r) + math.sqrt(max(0.0, r * r - d * d))


def _shell_rings(ctrl, z0_fn, extra_xs=(), n_corner=3):
    """Rounded-rect box-section rings along X for a silhouette control table."""
    xs = set(round(r[0], 6) for r in ctrl)
    xs.update(round(x, 6) for x in extra_xs)
    lo, hi = ctrl[0][0], ctrl[-1][0]
    xs = sorted(x for x in xs if lo - 1e-9 <= x <= hi + 1e-9)

    def z_top(x):
        return _field(ctrl, x, 2)

    def z_bot(x):
        return min(z0_fn(x), z_top(x) - 0.05)

    def hx(x):
        return max(0.05, _field(ctrl, x, 1))

    def hy(x):
        return max(0.02, (z_top(x) - z_bot(x)) * 0.5)

    def rad(x):
        return max(0.02, min(_field(ctrl, x, 3), hx(x) * 0.9, hy(x) * 0.9))

    def lift(x):
        return (z_top(x) + z_bot(x)) * 0.5

    return C.section_rings(xs, hx, hy, rad, n_corner=n_corner, lift_fn=lift)


def _flank_y(cfg, x, z):
    """Half-width of the OUTER surface at (x, z): the greenhouse if it covers
    that point, otherwise the body shell."""
    cab = cfg.get("cabin")
    if cab is not None:
        lo, hi = cab["ctrl"][0][0], cab["ctrl"][-1][0]
        if lo - 1e-6 <= x <= hi + 1e-6 and z >= cab["z0"]:
            r = _ring(cab["ctrl"], lambda t: cab["z0"], x)
            if z <= r[5]:
                return _half_w(r, z)
    return _half_w(_ring(cfg["shell"], lambda t: _shell_z0(cfg, t), x), z)


# ------------------------------------------------------------- primitives ---

def _slab(m, pts, push, color):
    """Closed 4-sided slab: quad ``pts`` extruded by ``push``.

    ``pts`` is the OUTER face; the slab grows along ``push`` into the body.
    ``core.loft`` advances ring-to-ring along the ring's right-hand normal, so
    that normal must point along ``push``; this only fixes the corner order.
    The outer cap is then the flipped ``cap_start``.
    """
    pts = [tuple(p) for p in pts]
    p = C.normalize(push)
    n = C.normalize(C.cross(C.sub(pts[1], pts[0]), C.sub(pts[2], pts[0])))
    if C.dot(n, p) < 0.0:
        pts = list(reversed(pts))
    C.loft(m, [pts, [C.add(q, push) for q in pts]], color,
           cap_start_flip=True, cap_end_flip=False)
    return m


def _ramp_glass(m, xa, za, ha, xb, zb, hb, ref, color, lift=0.014, thick=0.05):
    """Windscreen / backlight pane lying on a greenhouse ramp.

    ``xa,za`` is the lower edge, ``xb,zb`` the upper one; ``ref`` is a point
    inside the cabin and is only used to pick which of the two perpendicular
    directions points outward.
    """
    dx, dz = xb - xa, zb - za
    nx, nz = -dz, dx
    n = math.hypot(nx, nz)
    if n < 1e-9:
        return m
    nx, nz = nx / n, nz / n
    mx, mz = (xa + xb) * 0.5, (za + zb) * 0.5
    if nx * (mx - ref[0]) + nz * (mz - ref[1]) < 0.0:
        nx, nz = -nx, -nz
    ex, ez = nx * lift, nz * lift
    quad = [(xa + ex, -ha, za + ez),
            (xa + ex, ha, za + ez),
            (xb + ex, hb, zb + ez),
            (xb + ex, -hb, zb + ez)]
    return _slab(m, quad, (-nx * thick, 0.0, -nz * thick), color)


def _side_glass(m, hw, x1, x2, z1, z2, color, lift=0.014, thick=0.05):
    """One flat side pane on both flanks, sitting just proud of ``hw``."""
    for sy in (-1, 1):
        yy = sy * (hw + lift)
        quad = [(x1, yy, z1), (x2, yy, z1), (x2, yy, z2), (x1, yy, z2)]
        _slab(m, quad, (0.0, -sy * thick, 0.0), color)
    return m


def _bx(part, size, center, color):
    C.box(part.mesh, size, center=center, color=color)
    return part


def _pair(part, size, center, color):
    x, y, z = center
    for sy in (-1, 1):
        C.box(part.mesh, size, center=(x, sy * y, z), color=color)
    return part


def _flank_pair(cfg, part, x, z, size, color, out=0.008, thick=0.020):
    """A pair of thin boxes glued to BOTH flanks at (x, z).

    ``size`` is (dx, dy, dz); ``out`` is the gap between the box's inner face
    and the shell surface at that point, so the detail follows the fenders
    instead of drifting off the rounded shoulder.
    """
    y = _flank_y(cfg, x, z) + out + thick * 0.5
    _pair(part, (size[0], thick, size[2]), (x, y, z), color)
    return part


def _flank_line(cfg, part, x, z0, z1, color, out=0.006, segs=4):
    """A vertical cut-line down both flanks, sampled so it hugs the shell."""
    for i in range(segs):
        za = z0 + (z1 - z0) * (i / float(segs))
        zb = z0 + (z1 - z0) * ((i + 1) / float(segs))
        zm = (za + zb) * 0.5
        y = _flank_y(cfg, x, zm) + out + 0.010
        _pair(part, (0.022, 0.020, zb - za + 0.006), (x, y, zm), color)
    return part


def _flank_strip(cfg, part, x0, x1, z, color, out=0.006, segs=8):
    """A horizontal moulding (beltline) following both flanks from x0 to x1."""
    for i in range(segs):
        xa = x0 + (x1 - x0) * (i / float(segs))
        xb = x0 + (x1 - x0) * ((i + 1) / float(segs))
        xm = (xa + xb) * 0.5
        y = _flank_y(cfg, xm, z) + out + 0.010
        _pair(part, (xb - xa + 0.004, 0.020, 0.026), (xm, y, z), color)
    return part


# ------------------------------------------------------- end-cap features ---

def _cap_ring(cfg, front):
    """The closed cap the loft puts on the nose (front) or tail (rear)."""
    x = cfg["shell"][0][0] if front else cfg["shell"][-1][0]
    return _ring(cfg["shell"], lambda t: _shell_z0(cfg, t), x), x


def _cap_band(ring, x_cap, push, frac_lo, frac_hi, y_frac, depth, bite=0.02):
    """(size, centre) for a panel laid onto the end cap between two height
    fractions.  ``push`` is -1 for the nose, +1 for the tail; ``bite`` is how
    far the panel's inner face reaches back INTO the cap, so lamps and grilles
    are set into the bodywork instead of hovering in front of it."""
    z_bot, z_top = ring[4], ring[5]
    h = z_top - z_bot
    za = z_bot + h * frac_lo
    zb = z_bot + h * frac_hi
    y_out = min(_half_w(ring, za), _half_w(ring, zb)) * y_frac
    y_in = max(0.0, y_out - 0.30)
    size = (depth, y_out - y_in, zb - za)
    center = (x_cap + push * (depth * 0.5 - bite), (y_out + y_in) * 0.5,
              (za + zb) * 0.5)
    return size, center


def _cap_bumper(ring, x_cap, push, height=0.19, depth=0.16, bite=0.055):
    """A bumper bar wrapped around the end cap, as wide as the cap allows.

    ``bite`` is how far the bar's inner face reaches back INTO the body, so the
    bumper is bolted on rather than floating a gap in front of the cap.
    """
    y = _half_w(ring, ring[4] + (ring[5] - ring[4]) * 0.30) * 1.02
    z = ring[4] + (ring[5] - ring[4]) * 0.10
    return (depth, y * 2.0, height), (x_cap + push * (depth * 0.5 - bite),
                                      0.0, z)


# ------------------------------------------------------------ the vehicle ---

def _vehicle(cfg):
    """Assemble one vehicle from its config dictionary."""
    a = C.Asset(cfg["id"], "vehicle")
    paint = cfg["paint"]
    trim_c = cfg.get("trim", CHROME)

    # ---- body shell + greenhouse (one material, one flat part) -------------
    body = a.part("body", base_color=paint, roughness=0.40)
    C.loft(body.mesh,
           _shell_rings(cfg["shell"], lambda x: _shell_z0(cfg, x), _arch_xs(cfg)),
           paint, cap_start_flip=True, cap_end_flip=False)
    cab = cfg.get("cabin")
    if cab:
        C.loft(body.mesh, _shell_rings(cab["ctrl"], lambda x: cab["z0"], ()),
               paint, cap_start_flip=True, cap_end_flip=False)

    # ---- glass ------------------------------------------------------------
    g = cfg["glass"]
    glass = a.part("glass", base_color=GLASS, roughness=0.12, metallic=0.30)
    for key in ("windshield", "backlight"):
        if g.get(key):
            _ramp_glass(glass.mesh, *g[key], g["ref"], GLASS)
    for (x1, x2, z1, z2) in g.get("side", ()):
        _side_glass(glass.mesh, g["side_hw"], x1, x2, z1, z2, GLASS)

    # ---- wheels -----------------------------------------------------------
    wr, ww, track = cfg["wheel"]
    tyres = a.part("tyres", base_color=TYRE, roughness=0.88)
    hubs = a.part("hubs", base_color=HUB, roughness=0.32, metallic=0.65)
    for x in cfg["axles"]:
        for sy in (-1, 1):
            C.cylinder(tyres.mesh, wr, ww, seg=12, center=(x, sy * track, wr),
                       color=TYRE, axis="Y")
            C.cylinder(hubs.mesh, wr * 0.56, ww * 1.10, seg=10,
                       center=(x, sy * track, wr), color=HUB, axis="Y")

    # ---- lamps, laid onto the two end caps -------------------------------
    for front, key, color, emis in (
            (True, "front", HEADLIGHT, (0.85, 0.78, 0.55)),
            (False, "rear", TAILLIGHT, (0.62, 0.06, 0.06))):
        face = cfg["face"][key]
        ring, x_cap = _cap_ring(cfg, front)
        size, center = _cap_band(ring, x_cap, -1.0 if front else 1.0,
                                 *face["lamp"])
        part = a.part(key + "lights", base_color=color, roughness=0.20,
                      emissive=emis)
        _pair(part, size, center, color)

    # ---- trim: bumpers, grille, mirrors, cut-lines, handles ---------------
    trim = a.part("trim", base_color=trim_c, roughness=0.36, metallic=0.55)
    for front, key in ((True, "front"), (False, "rear")):
        ring, x_cap = _cap_ring(cfg, front)
        height, depth, bite = cfg["face"][key].get("bumper",
                                                  (0.19, 0.16, 0.055))
        size, center = _cap_bumper(ring, x_cap, -1.0 if front else 1.0,
                                   height, depth, bite)
        _bx(trim, size, center, cfg.get("bumper_color", trim_c))
    ring, x_cap = _cap_ring(cfg, True)
    size, center = _cap_band(ring, x_cap, -1.0, *cfg["face"]["front"]["grille"])
    _bx(trim, size, center, TRIM_DK)

    mx, mz, msz, m_arm = cfg["mirror"]
    _flank_pair(cfg, trim, mx, mz, (msz[0], 0.0, msz[2]), trim_c,
                out=0.0, thick=0.030)
    _flank_pair(cfg, trim, mx, mz, msz, cfg.get("mirror_color", paint),
                out=m_arm)

    for x in cfg.get("doors", ()):
        z0, z1 = cfg.get("door_span", (0.30, 0.98))
        _flank_line(cfg, trim, x, z0, z1, TRIM_DK)
    for (x, z, dx) in cfg.get("handles", ()):
        _flank_pair(cfg, trim, x, z, (dx, 0.0, 0.046), trim_c)
    if cfg.get("beltline"):
        x0, x1, z = cfg["beltline"]
        _flank_strip(cfg, trim, x0, x1, z, cfg.get("belt_color", trim_c))

    # ---- per-vehicle extras: livery, cargo bed, roof mounts ---------------
    for hook in cfg.get("extras", ()):
        hook(a, cfg)
    return a


# ----------------------------------------------------- silhouette tables ----

_SEDAN_SHELL = [
    (-2.20, 0.600, 0.760, 0.22),
    (-2.06, 0.800, 0.815, 0.20),
    (-1.90, 0.866, 0.860, 0.16),
    (-1.62, 0.878, 0.895, 0.15),
    (-1.34, 0.878, 0.920, 0.14),
    (-1.08, 0.878, 0.945, 0.14),
    (-0.92, 0.868, 1.000, 0.16),
    (-0.55, 0.864, 1.000, 0.16),
    (0.10, 0.864, 1.000, 0.16),
    (0.70, 0.868, 1.000, 0.16),
    (0.94, 0.878, 0.960, 0.14),
    (1.16, 0.878, 0.940, 0.14),
    (1.34, 0.878, 0.925, 0.14),
    (1.58, 0.876, 0.910, 0.14),
    (1.84, 0.864, 0.895, 0.16),
    (2.04, 0.820, 0.875, 0.19),
    (2.20, 0.600, 0.800, 0.22),
]

_SEDAN_CABIN = [
    (-1.02, 0.800, 1.040, 0.045),
    (-0.76, 0.790, 1.400, 0.075),
    (-0.60, 0.784, 1.450, 0.090),
    (0.38, 0.784, 1.450, 0.090),
    (0.54, 0.790, 1.420, 0.075),
    (0.92, 0.800, 1.040, 0.045),
]

_SEDAN_COMMON = {
    "shell": _SEDAN_SHELL,
    "cabin": {"ctrl": _SEDAN_CABIN, "z0": 0.70},
    "sill": 0.26, "ends": ((-2.20, 0.44), (2.20, 0.48)), "end_span": 0.20,
    "arch_top": 0.76, "arch_half": 0.42, "axles": (-1.34, 1.34),
    "wheel": (0.34, 0.22, 0.775),
    "glass": {
        "windshield": (-1.02, 1.040, 0.700, -0.76, 1.400, 0.665),
        "backlight": (0.92, 1.040, 0.700, 0.54, 1.420, 0.670),
        "side": ((-0.56, -0.04, 1.070, 1.325),
                 (0.02, 0.42, 1.070, 1.325)),
        "side_hw": 0.784, "ref": (-0.10, 0.90),
    },
    "face": {
        "front": {"lamp": (0.52, 0.92, 0.74, 0.075),
                  "grille": (0.30, 0.58, 0.50, 0.06)},
        "rear": {"lamp": (0.55, 0.92, 0.80, 0.075),
                 "bumper": (0.19, 0.16, 0.055)},
    },
    "mirror": (-0.80, 1.010, (0.150, 0.085, 0.100), 0.060),
    "doors": (-0.88, 0.06, 0.90),
    "door_span": (0.44, 0.80),
    "handles": ((-0.50, 0.855, 0.17), (0.35, 0.855, 0.17)),
    "beltline": (-0.92, 0.94, 1.045),
}


# ------------------------------------------------------------------ liveries

def _police_livery(a, cfg):
    """Black-and-white cruiser: rocker band and door shields."""
    p = a.part("livery", base_color=POLICE_BLACK, roughness=0.38)
    for sy in (-1, 1):
        n, dx = 9, 2.70 / 9.0
        for i in range(n):
            x = -1.35 + i * dx
            z = 0.380
            y = _flank_y(cfg, x, z) + 0.022
            _bx(p, (dx + 0.004, 0.022, 0.150), (x, sy * y, z), POLICE_BLACK)
        for cx in (-0.44, 0.42):
            for i in range(3):
                z = 0.520 + i * 0.140
                y = _flank_y(cfg, cx, z) + 0.024
                _bx(p, (0.820, 0.024, 0.140), (cx, sy * y, z), POLICE_BLACK)
    return


def _taxi_livery(a, cfg):
    """Cab-yellow checker band along the flanks plus dark door plates."""
    p = a.part("livery", base_color=POLICE_BLACK, roughness=0.45)
    x, i = -0.85, 0
    while x < 0.85 - 1e-9:
        if i % 2 == 0:
            z = 0.845
            y = _flank_y(cfg, x, z) + 0.024
            for sy in (-1, 1):
                _bx(p, (0.185, 0.024, 0.135), (x, sy * y, z), POLICE_BLACK)
        x += 0.185
        i += 1
    for cx in (-0.44, 0.42):
        z = 0.620
        y = _flank_y(cfg, cx, z) + 0.025
        for sy in (-1, 1):
            _bx(p, (0.700, 0.026, 0.300), (cx, sy * y, z), TRIM_DK)
    return


def _lightbar_mount(a, cfg):
    """Low plinth the (separate) light-bar mesh is meant to bolt onto."""
    x = -0.06
    z0 = _field(cfg["cabin"]["ctrl"], x, 2)
    m = a.part("lightbar_mount", base_color=POLICE_BLACK, roughness=0.55)
    _bx(m, (0.620, 1.100, 0.055), (x, 0.0, z0 + 0.028), POLICE_BLACK)
    _bx(m, (0.520, 0.980, 0.022), (x, 0.0, z0 + 0.066), (0.26, 0.28, 0.30))
    for sy in (-1, 1):
        _bx(m, (0.440, 0.050, 0.100), (x, sy * 0.500, z0 - 0.014), TRIM_DK)
    return


def _sign_mount(a, cfg):
    """Small bracket on the roof centre -- the taxi sign bolts onto this."""
    x = -0.30
    z0 = _field(cfg["cabin"]["ctrl"], x, 2)
    m = a.part("sign_mount", base_color=TRIM_DK, roughness=0.55)
    _bx(m, (0.280, 0.240, 0.035), (x, 0.0, z0 + 0.0175), TRIM_DK)
    _bx(m, (0.100, 0.090, 0.055), (x, 0.0, z0 + 0.0625), TRIM_DK)
    return


def _bed(a, cfg):
    """Pickup cargo box: side rails, head-board, tailgate and a bed floor."""
    col = cfg.get("bed_color", (0.30, 0.34, 0.36))
    b = a.part("bed", base_color=col, roughness=0.45)
    ring = _ring(cfg["shell"], lambda t: _shell_z0(cfg, t), 1.50)
    hw = min(_half_w(ring, ring[5] - 0.02), _field(cfg["shell"], 1.50, 1))
    z0, z1 = ring[5] + 0.010, ring[5] + 0.395
    zm, h = (z0 + z1) * 0.5, z1 - z0
    for sy in (-1, 1):
        _bx(b, (1.760, 0.090, h), (1.540, sy * (hw - 0.045), zm), col)
    _bx(b, (0.090, hw * 2.0 - 0.090, h), (0.725, 0.0, zm), col)
    _bx(b, (0.100, hw * 2.0 - 0.090, h), (2.415, 0.0, zm), col)
    for sy in (-1, 1):
        _bx(b, (1.700, hw - 0.500, 0.050), (1.545, sy * (hw - 0.320),
                                           z0 + 0.025), col)
    return


# --------------------------------------------------------------- the fleet --

_COUPED_SHELL = [
    (-2.14, 0.580, 0.740, 0.18),
    (-2.00, 0.800, 0.780, 0.16),
    (-1.82, 0.868, 0.810, 0.12),
    (-1.60, 0.885, 0.840, 0.11),
    (-1.46, 0.885, 0.855, 0.11),
    (-1.20, 0.885, 0.875, 0.11),
    (-1.04, 0.872, 0.920, 0.13),
    (-0.72, 0.868, 0.930, 0.13),
    (-0.10, 0.868, 0.930, 0.13),
    (0.50, 0.870, 0.930, 0.13),
    (1.05, 0.878, 0.920, 0.13),
    (1.28, 0.885, 0.900, 0.12),
    (1.58, 0.885, 0.875, 0.11),
    (1.84, 0.872, 0.845, 0.13),
    (2.06, 0.820, 0.815, 0.16),
    (2.20, 0.600, 0.770, 0.18),
]

_COUPED_CABIN = [
    (-0.92, 0.790, 1.000, 0.050),
    (-0.68, 0.778, 1.260, 0.070),
    (-0.52, 0.772, 1.300, 0.080),
    (0.28, 0.772, 1.300, 0.080),
    (0.56, 0.780, 1.240, 0.070),
    (1.00, 0.792, 0.950, 0.050),
]

_PICKUP_SHELL = [
    (-2.48, 0.660, 0.900, 0.20),
    (-2.34, 0.880, 0.960, 0.18),
    (-2.16, 0.950, 1.000, 0.14),
    (-1.96, 0.960, 1.020, 0.12),
    (-1.76, 0.960, 1.045, 0.12),
    (-1.62, 0.960, 1.060, 0.12),
    (-1.44, 0.960, 1.080, 0.12),
    (-1.24, 0.960, 1.100, 0.12),
    (-1.08, 0.950, 1.140, 0.14),
    (-0.78, 0.940, 1.160, 0.14),
    (-0.30, 0.940, 1.160, 0.14),
    (0.30, 0.945, 1.160, 0.14),
    (0.80, 0.955, 1.160, 0.13),
    (1.10, 0.960, 1.160, 0.12),
    (1.44, 0.960, 1.150, 0.12),
    (1.80, 0.958, 1.140, 0.12),
    (2.10, 0.945, 1.130, 0.13),
    (2.34, 0.910, 1.120, 0.15),
    (2.48, 0.720, 1.060, 0.20),
]

_PICKUP_CABIN = [
    (-1.00, 0.905, 1.220, 0.060),
    (-0.78, 0.888, 1.720, 0.090),
    (-0.62, 0.880, 1.780, 0.100),
    (0.26, 0.880, 1.780, 0.100),
    (0.46, 0.890, 1.720, 0.090),
    (0.62, 0.905, 1.300, 0.060),
]


def build_all():
    """Return the ordered list of vehicle assets."""
    out = []

    # 1. four-door family sedan -- the default NEON BAY cab
    out.append(_vehicle(dict(_SEDAN_COMMON,
                             id="car_sedan", paint=MIAMI_PINK)))

    # 2. two-door sports coupe -- long bonnet, fastback tail
    out.append(_vehicle({
        "id": "car_coupe",
        "paint": MIAMI_TEAL,
        "shell": _COUPED_SHELL,
        "cabin": {"ctrl": _COUPED_CABIN, "z0": 0.66},
        "sill": 0.240, "ends": ((-2.14, 0.40), (2.20, 0.42)), "end_span": 0.18,
        "arch_top": 0.720, "arch_half": 0.40, "axles": (-1.46, 1.28),
        "wheel": (0.320, 0.235, 0.782),
        "glass": {
            "windshield": (-0.92, 1.000, 0.680, -0.68, 1.260, 0.650),
            "backlight": (1.00, 0.950, 0.680, 0.56, 1.240, 0.650),
            "side": ((-0.48, 0.24, 0.975, 1.205),),
            "side_hw": 0.772, "ref": (-0.10, 0.85),
        },
        "face": {
            "front": {"lamp": (0.52, 0.92, 0.74, 0.075),
                      "grille": (0.28, 0.56, 0.50, 0.06)},
            "rear": {"lamp": (0.55, 0.92, 0.80, 0.075),
                     "bumper": (0.18, 0.15, 0.050)},
        },
        "mirror": (-0.86, 0.950, (0.140, 0.080, 0.090), 0.055),
        "doors": (-0.62, 0.60),
        "door_span": (0.40, 0.76),
        "handles": ((-0.28, 0.790, 0.15),),
    }))

    # 3. full-size pickup -- cab forward, open cargo bed aft
    out.append(_vehicle({
        "id": "truck_pickup",
        "paint": CORAL,
        "bed_color": COPPER_DK,
        "shell": _PICKUP_SHELL,
        "cabin": {"ctrl": _PICKUP_CABIN, "z0": 0.95},
        "sill": 0.340, "ends": ((-2.48, 0.560), (2.48, 0.580)),
        "end_span": 0.26,
        "arch_top": 0.860, "arch_half": 0.46, "axles": (-1.44, 1.44),
        "wheel": (0.380, 0.260, 0.800),
        "glass": {
            "windshield": (-1.00, 1.220, 0.770, -0.78, 1.720, 0.740),
            "backlight": (0.62, 1.300, 0.770, 0.46, 1.720, 0.740),
            "side": ((-0.55, 0.20, 1.200, 1.620),),
            "side_hw": 0.880, "ref": (-0.20, 1.30),
        },
        "face": {
            "front": {"lamp": (0.50, 0.90, 0.80, 0.075),
                      "grille": (0.26, 0.44, 0.64, 0.06)},
            "rear": {"lamp": (0.34, 0.66, 0.62, 0.075),
                     "bumper": (0.19, 0.16, 0.055)},
        },
        "bumper_color": (0.66, 0.68, 0.70),
        "mirror": (-0.86, 1.220, (0.190, 0.080, 0.140), 0.070),
        "doors": (-0.60, 0.40),
        "door_span": (0.46, 0.92),
        "handles": ((-0.35, 0.948, 0.19), (0.25, 0.948, 0.19)),
        "extras": (_bed,),
    }))

    # 4. police cruiser -- sedan shell plus a roof light-bar plinth
    out.append(_vehicle(dict(_SEDAN_COMMON,
                             id="car_police", paint=WHITE,
                             bumper_color=(0.20, 0.21, 0.24),
                             extras=(_police_livery, _lightbar_mount))))

    # 5. taxi -- cab yellow, checker band, roof sign bracket
    out.append(_vehicle(dict(_SEDAN_COMMON,
                             id="car_taxi", paint=TAXI_YELLOW,
                             trim=(0.45, 0.46, 0.48),
                             belt_color=(0.20, 0.21, 0.24),
                             bumper_color=(0.22, 0.23, 0.26),
                             extras=(_taxi_livery, _sign_mount))))

    return out
