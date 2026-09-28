"""Category J: in-game MISSION MARKERS -- the objective beacon furniture.

Two assets, both original low-poly geometry generated here from the ``core``
kernel primitives.  Nothing is imported, traced, sampled or measured from a
real-world object, and in particular no game model, texture, font or logo of
any kind is reproduced: this is an original 1980s-Miami-neon take on "a thing
that tells the player where to go".

Authoring rules (inherited from CONTRACT.md): Blender-style Z-up, ``-Y`` is
the front, metres, centred on ``X = 0``.  Unlike a weapon -- whose origin is a
grip so the runtime can parent it to a hand -- a marker is PLANTED in the
world, so both assets put the origin at the centre of the ground footprint
and rest the whole group on ``z = 0``.  That also makes ``marker_coin_ring``
composable with ``misc_helipad_marking``: ring centre at the origin, the coins'
feet on the street.

Kernel facts this module leans on:

* ``C.torus`` is trusted on all three axes -- its ``axis="Y"`` branch
  compensates its own odd-permutation handedness internally (see the winding
  note in ``core.torus``), and the exporter's signed-volume check would catch
  that loudly if it ever regressed.
* A **part is either flat or smooth, never mixed**: ``export.export_part``
  rejects a ``flat=True`` part that contains shared-vertex smooth geometry.
  Every loft here therefore stays ``smooth=False`` and is flattened at export
  time, which is exactly the flat-shaded low-poly look the rest of the set
  uses.  Per-face colour (``face_colors``) carries the hazard bands so they do
  not each have to become their own part.
"""

import math

from .. import core as C

# --------------------------------------------------------------------------
# palette -- read straight off the neon-signboard palette so a marker dropped
# next to a HOTEL sign belongs to the same street
# --------------------------------------------------------------------------

FLAME_YELLOW = (0.98, 0.80, 0.16)
FLAME_AMBER = (0.98, 0.62, 0.10)
FLAME_DARK = (0.16, 0.13, 0.10)        # the black band of the hazard stripes
FLAME_GLOW = (1.00, 0.66, 0.16)
FLAME_CORE = (1.00, 0.92, 0.62)

IRON = (0.26, 0.25, 0.24)
METAL = (0.58, 0.60, 0.62)
METAL_DK = (0.32, 0.34, 0.36)

GOLD = (0.94, 0.74, 0.24)
GOLD_DK = (0.74, 0.54, 0.14)
GOLD_GLOW = (0.34, 0.22, 0.04)         # a faint bloom, not a light source

NEON_TEAL = (0.16, 0.86, 0.78)
NEON_TEAL_DK = (0.08, 0.46, 0.44)

RING_N = 14                            # coins in the ring
RING_R = 0.86                          # ring radius, metres
RING_BAND = 0.22                       # half-width of the painted band
# A coin stands UPRIGHT on its rim, so what seats it on the plate is its
# RADIUS (0.135), not its half-thickness: the disc spans +/-0.135 vertically
# about its own centre, and the lowest point -- the bottom of the rim -- is
# where it touches.  Offsetting the centre by the disc's 0.022 half-depth
# instead sank every coin 0.113 m into the plate, so the ring rendered as
# coins half-buried in the road.
COIN_RADIUS = 0.135


# --------------------------------------------------------------------------
# local helpers -- tiny; all real geometry still goes through core
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
    """A box rotated by (rx, ry, rz) DEGREES about its own centre.

    ``C.box`` is axis-aligned only, which cannot express a coin banked toward
    the ring's centre or a valve canted off the vertical.  Emitting the same
    eight corners through the same six quads as ``C.box`` keeps the winding
    (and therefore the signed-volume check) identical -- the rotation matrix
    is orthogonal with determinant +1.
    """
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
    untouched and no inward-facing shell can be introduced.  The misc and props
    modules each carry their own copy of this helper; it is three lines of
    matrix application and hoisting it into ``core`` would touch every builder.
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


def _merge(dst, src):
    """Append ``src``'s geometry into ``dst``, re-basing the face indices.

    Used to stamp N identical coins into ONE part.  A part is the unit of
    MATERIAL, not of instance, so fourteen coins sharing one gold-and-metal
    look belong to a single part -- and the exporter rejects duplicate part
    NAMES, which a per-coin part would otherwise have forced.
    """
    off = len(dst.mesh.pos)
    dst.mesh.pos.extend(src.mesh.pos)
    dst.mesh.faces.extend(tuple(i + off for i in f) for f in src.mesh.faces)
    dst.mesh.fcol.extend(src.mesh.fcol)
    dst.mesh.skipped += src.mesh.skipped
    return dst


def _ring_pts(r, z, seg=12):
    """One CCW ring of ``seg`` points at height ``z`` -- a loft station."""
    return [(r * math.cos(2.0 * math.pi * i / seg),
             r * math.sin(2.0 * math.pi * i / seg), z)
            for i in range(seg)]


def _drop(asset, z=0.0):
    """Translate the whole asset so its lowest vertex rests on ``z``."""
    lo = None
    for part in asset.parts:
        for p in part.mesh.pos:
            lo = p[2] if lo is None else min(lo, p[2])
    if lo is None:
        return asset
    dz = z - lo
    if abs(dz) > 1e-12:
        for part in asset.parts:
            part.mesh.pos = [(p[0], p[1], p[2] + dz) for p in part.mesh.pos]
    return asset


# --------------------------------------------------------------------------
# 1. marker_flame -- the objective flame cone
# --------------------------------------------------------------------------

def _marker_flame():
    """A ~0.9 x 1.9 x 0.9 m hazard-striped beacon cone with a glowing tip.

    Read order bottom to top: a cast iron foot plate, a small iron hoop, then
    the striped cone, then two offset flame tongues that break the perfectly
    straight silhouette a single lathe gives.  A symmetrical cone reads as a
    party hat; the asymmetry is what makes it read as fire.
    """
    a = C.Asset("marker_flame", "marker")

    # -- foot plate + hoop --------------------------------------------------
    foot = a.part("flame_foot", base_color=IRON, metallic=0.45, roughness=0.62)
    C.cylinder(foot.mesh, 0.44, 0.055, 16, center=(0.0, 0.0, 0.0275),
               color=IRON)
    C.cylinder(foot.mesh, 0.375, 0.030, 14, center=(0.0, 0.0, 0.070),
               color=C.shade(IRON, 0.80))                    # raised collar
    # smooth=False: the hoop shares vertices between its quads (that is what
    # keeps the tube watertight), which a flat part may not contain.
    C.torus(foot.mesh, 0.372, 0.036, 12, 5, center=(0.0, 0.0, 0.098),
            color=METAL_DK, smooth=False)                      # the iron hoop

    # -- striped cone -------------------------------------------------------
    # Eight EVENLY SPACED stations from the collar up to the neck, kept as ONE
    # flat loft so the shell stays watertight and the bands are per-face
    # colour rather than fourteen separate solids.
    #
    # Even spacing is not cosmetic.  The first pass spaced stations to follow
    # the taper, which packed them 70 mm apart near the tip and 200 mm apart
    # at the base; a band index derived from a centroid Z then produced two
    # dark bands in a row up top and a single fat one at the bottom, and the
    # QA render came out with a jagged sawtooth in the upper stripe.  One row
    # of quads = one colour band, so the stripes are now banded by ROW INDEX
    # rather than by a height threshold, which cannot drift no matter how the
    # stations are later retuned.
    body = a.part("flame_body", base_color=FLAME_YELLOW, roughness=0.55)
    z0, z1 = 0.085, 1.560
    rows = 7
    r0, r1 = 0.400, 0.062
    stations = []
    for k in range(rows + 1):
        t = k / float(rows)
        stations.append((r0 + (r1 - r0) * t, z0 + (z1 - z0) * t))
    C.loft(body.mesh, [_ring_pts(r, z, 12) for (r, z) in stations],
           FLAME_YELLOW, cap_start_flip=True, cap_end_flip=False)

    def _row(c):
        """Which loft row a face centroid sits in (0..rows-1).

        FLOOR, not round.  A quad between rows k and k+1 is emitted as two
        triangles whose centroid heights sit at k + 1/3 and k + 2/3 of the way
        across the row -- so ``round`` classifies the second triangle as row
        k+1 and shreds every stripe boundary into a sawtooth.  ``floor`` maps
        both halves of the quad back onto the row they belong to.  Clamped so
        the two end caps land on the first and last real row.
        """
        t = (c[2] - z0) / (z1 - z0) * rows
        if t < 0.0:
            return 0
        k = int(math.floor(t))
        return k if k < rows else rows - 1

    C.recolor_faces_where(
        body, lambda n, c: _row(c) % 2 == 1, FLAME_DARK)
    # Hotter tint low on the sunward flank: face colour only, no geometry.
    # Skipped on the dark rows so the tint never muddies a hazard stripe.
    C.recolor_faces_where(
        body, lambda n, c: _row(c) % 2 == 0 and c[0] < -0.16
        and c[2] < z0 + (z1 - z0) * 0.40, FLAME_AMBER)

    # -- flame tongues ------------------------------------------------------
    # EMISSION lives on the part and a part carries ONE emissive colour, so
    # the hot core and the cooler outer lick are two parts, not two tints.
    glow = a.part("flame_glow", base_color=FLAME_GLOW, emissive=FLAME_GLOW,
                  roughness=0.35)
    C.cone(glow.mesh, 0.076, 0.320, 8, center=(0.0, 0.0, 1.700),
           color=FLAME_GLOW)                                  # main tongue
    lick = C.Part("flame_lick", base_color=FLAME_AMBER,
                  emissive=(1.0, 0.48, 0.10), roughness=0.40)
    C.cone(lick.mesh, 0.042, 0.240, 8, color=FLAME_AMBER)
    _place([lick], offset=(0.118, 0.068, 1.540), rot=(-13.0, 12.0, 0.0))
    a.add(lick)

    core = a.part("flame_core", base_color=FLAME_CORE,
                  emissive=(1.0, 0.88, 0.58), roughness=0.30)
    C.cone(core.mesh, 0.046, 0.210, 8, center=(0.032, -0.020, 1.790),
           color=FLAME_CORE)                                  # hot centre
    return _drop(a)


# --------------------------------------------------------------------------
# 2. marker_coin_ring -- the spinning pickup ring
# --------------------------------------------------------------------------

def _coin_mesh(m, face=GOLD, rim=GOLD_DK):
    """One upright coin built about its own centre, facing along +Y.

    Two stacked discs: a wide rim and a proud boss.  The boss is what stops a
    coin from reading as a washer -- at gameplay distance a single flat disc
    has no highlight to catch and disappears against the road.
    """
    C.cylinder(m, 0.135, 0.028, 10, axis="Y", color=rim)
    C.cylinder(m, 0.094, 0.044, 8, axis="Y", color=face)


def _marker_coin_ring():
    """14 gold coins standing on a neon ring plate, origin at the ring centre.

    The coins face RADIALLY, so the ring reads as one object from any approach
    angle instead of a row of lozenges; spinning it toward the player is then
    purely a rotation about its own centre, which is the transform the runtime
    applies every frame.
    """
    a = C.Asset("marker_coin_ring", "marker")

    # -- neon ring plate on the road ---------------------------------------
    # Radial plates, not one solid disc, for the same reason the helipad ring
    # is 28 separate plates: paint is a 2D band on a flat road and a disc one
    # level higher reads as a wedding cake.  Each plate is a closed box, so
    # the exporter's signed-volume check decides its outwardness and no
    # ``outward`` declaration is needed.
    plate_z = 0.036
    # The coins are stamped with their own centre at z = cz, which puts the
    # bottom of each coin's rim exactly on the plate's top face.  That also
    # means the LOWEST vertex of the whole asset belongs to a coin (its rim
    # touches the plate at z = plate_z), not to the plate itself -- so a plain
    # whole-asset _drop() would be right here only by coincidence of the coin
    # sitting on the plate.  Drop the PLATE, the element that actually touches
    # the street, and let the coins ride on it.
    plate = a.part("ring_plate", base_color=NEON_TEAL, emissive=NEON_TEAL_DK,
                   roughness=0.55)
    n_seg = 20
    r_mid = RING_R
    for i in range(n_seg):
        ang = 2.0 * math.pi * i / n_seg
        # The tangent length 2*pi*r/n_seg makes consecutive plates meet end to
        # end; 1.04 lets them overlap very slightly so no seam shows.
        _rbox(plate.mesh,
              (2.0 * RING_BAND, 2.0 * math.pi * r_mid / n_seg * 1.04, plate_z),
              center=(r_mid * math.cos(ang), r_mid * math.sin(ang),
                      plate_z * 0.5),
              rot=(0.0, 0.0, math.degrees(ang) + 90.0), color=NEON_TEAL)
    # -- the coins ----------------------------------------------------------
    coins = C.Part("ring_coins", base_color=GOLD, emissive=GOLD_GLOW,
                   metallic=0.85, roughness=0.28)
    a.add(coins)                        # registered once, then filled below
    stub = C.Part("coin_stub", base_color=GOLD, roughness=0.28)
    # A coin stands upright on its rim, so its own centre must sit one COIN_RADIUS
    # above the plate -- see COIN_RADIUS for why that is the radius and not the
    # disc's half-thickness.
    cz = plate_z + COIN_RADIUS
    for i in range(RING_N):
        ang = 2.0 * math.pi * i / RING_N
        stub.mesh = C.Mesh()
        _coin_mesh(stub.mesh)
        # Rotating by (ang - 90 deg) maps the coin's +Y face normal onto the
        # outward radial direction (cos ang, sin ang, 0), so every coin looks
        # the same way round the ring.
        _place([stub],
               offset=(RING_R * math.cos(ang), RING_R * math.sin(ang), cz),
               rot=(0.0, 0.0, math.degrees(ang) - 90.0))
        _merge(coins, stub)

    # No anchor studs.  An earlier pass ringed the plate with three metal
    # posts on the midpoints between neighbouring coins; every render of it
    # showed them as grey fingers poking up through the 13-degree gaps, and
    # shortening them just moved the artifact rather than removing it -- in a
    # gap that narrow a post of any visible height reads as a stray sliver.
    # They carried no information the coins and the plate do not already
    # convey, so the two elements that read cleanly are what shipped.
    #
    # Drop ONLY the plate onto z = 0.  A whole-asset _drop() would instead seat
    # the lowest COIN -- whose disc pokes COIN_BOTTOM below the ring plane --
    # on the street, hoisting the plate 0.022 m into the air with it.
    lo = min(p[2] for p in plate.mesh.pos)
    plate.mesh.pos = [(p[0], p[1], p[2] - lo) for p in plate.mesh.pos]
    return a


# --------------------------------------------------------------------------

def build_all():
    """Return the ordered list of mission-marker assets."""
    return [
        _marker_flame(),
        _marker_coin_ring(),
    ]
