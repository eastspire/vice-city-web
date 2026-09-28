"""Handheld weapons + carry pickups: four guns, an armour vest, a med pack.

Everything here is original low-poly blockwork built from the ``core`` kernel
primitives -- no imported or traced geometry.  The guns are stylised
sidearm silhouettes: a few boxes and short cylinders, deliberately readable at
a glance rather than anatomically correct.

AUTHORING CONVENTIONS (inherited from ``CONTRACT.md``, with one deliberate
deviation, see ORIGINS below):

* Z-up, ``-Y`` is the barrel direction, metres, centred on ``X = 0``.
* ORIGINS.  The four guns put their origin at the **centre of the grip**, so a
  renderer can parent them to a hand bone and have the palm land in the right
  place; the barrel then runs toward ``-Y`` from that point.  The two pickups
  are centred on their own bounds.  Because the guns are hand props they are
  NOT authored resting on ``Z = 0`` -- a weapon's origin is a grip, not a
  footprint.

MATERIALS / PARTS.  Each gun is three parts so the silhouette reads with one
material per role: ``*_body`` (painted receiver), ``*_metal`` (bare machined
parts -- barrel, guard, sights) and ``*_grip`` (matte polymer, the pieces a
hand actually touches).  The pickups use the same discipline.

``core.torus`` and ``core.cylinder`` are called on the X and Y axes here.  Both
were verified with ``C.signed_volume(m) > 0`` on this kernel revision (the
``torus`` ``axis="Y"`` branch compensates its own handedness); the axis-Y
cylinder is the cleanest available primitive for a barrel lying along the
authoring forward axis, and the exporter's closed-shell volume check would
reject it loudly if that ever regressed.
"""

import math

from .. import core as C

# ---- palette (shared with props.py so the block reads as one world) --------
GUNMETAL = (0.21, 0.22, 0.24)       # painted receiver
GUNMETAL_LT = (0.32, 0.33, 0.36)
BARREL = (0.30, 0.31, 0.33)         # bare machined steel
BARREL_DK = (0.18, 0.19, 0.20)
POLYMER = (0.11, 0.11, 0.13)        # grip / furniture
POLYMER_LT = (0.17, 0.17, 0.20)
STEEL_BRIGHT = (0.55, 0.57, 0.60)   # trigger guard, springs, pins

WOOD = (0.46, 0.28, 0.16)
WOOD_DK = (0.30, 0.18, 0.10)

OLIVE = (0.26, 0.28, 0.20)          # launcher tube
OLIVE_DK = (0.18, 0.20, 0.14)
ARMO_1 = (0.13, 0.42, 0.44)         # vest ballistic panel
ARMO_2 = (0.09, 0.30, 0.32)
ARMO_TRIM = (0.92, 0.44, 0.18)      # hi-viz trim
ARMO_STRAP = (0.16, 0.17, 0.19)

MED_RED = (0.86, 0.13, 0.14)
MED_RED_DK = (0.62, 0.08, 0.10)
MED_WHITE = (0.94, 0.94, 0.92)
MED_GREY = (0.74, 0.75, 0.74)

# Melee + carry-pickup palette, same neon-boardwalk register as the guns.
BAT_WOOD = (0.72, 0.55, 0.34)
BAT_WOOD_DK = (0.55, 0.40, 0.23)
BAT_TAPE = (0.16, 0.17, 0.19)
BRASS = (0.76, 0.62, 0.28)
GREN_BODY = (0.24, 0.30, 0.22)
GREN_BODY_DK = (0.16, 0.21, 0.15)
NEON_TEAL = (0.16, 0.80, 0.74)
NEON_TEAL_DK = (0.08, 0.44, 0.42)
FLARE_RED = (0.92, 0.28, 0.10)
FLARE_RED_DK = (0.66, 0.16, 0.07)
AMBER_STENCIL = (0.88, 0.72, 0.18)
TANK_WHITE = (0.93, 0.93, 0.89)
BILL_GREEN = (0.30, 0.56, 0.34)
BILL_GREEN_DK = (0.22, 0.44, 0.27)
BILL_TAN = (0.80, 0.72, 0.52)
BILL_TAN_DK = (0.68, 0.60, 0.42)


# ---------------------------------------------------------------------------
# local helpers on top of the kernel primitives
# ---------------------------------------------------------------------------

def _rbox(m, size, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0),
          color=(1.0, 1.0, 1.0)):
    """Box rotated by (rx, ry, rz) DEGREES about its own centre.

    ``C.box`` is axis-aligned only, which cannot express a pistol grip raked
    back 18-20 deg, a stock comb sloping down to the butt, or a canted
    magazine.  Emitting the same eight corners through the same six quads as
    ``C.box`` keeps the winding identical -- the rotation matrix is
    orthogonal with determinant +1, so signed volume is unchanged.
    """
    hx, hy, hz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
    corners = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
               (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]
    ax, ay, az = (math.radians(v) for v in rot)
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    r = ((cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
         (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
         (-sy, cy * sx, cy * cx))
    o = tuple(center)
    p = []
    for v in corners:
        p.append((o[0] + r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
                  o[1] + r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
                  o[2] + r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2]))
    m.quad(p[0], p[3], p[2], p[1], color)
    m.quad(p[4], p[5], p[6], p[7], color)
    m.quad(p[0], p[1], p[5], p[4], color)
    m.quad(p[1], p[2], p[6], p[5], color)
    m.quad(p[2], p[3], p[7], p[6], color)
    m.quad(p[3], p[0], p[4], p[7], color)
    return m


def _shift(part, offset):
    """Translate every vertex of ``part``.

    A LOCAL reimplementation of ``core.translate_part``, which is broken: it
    mutates vertices with ``p[0] += ox`` while every entry in ``Mesh.pos`` is
    an immutable tuple, so it raises TypeError on any non-empty mesh.  No
    other builder calls it, so the bug has never been hit.  Rebuilt as a pure
    list comprehension rather than an in-place edit.

    ``C.translate_part`` is listed in CONTRACT.md, so this is worth reporting
    upstream: the fix there is one line -- ``part.mesh.pos = [(p[0] + ox,
    p[1] + oy, p[2] + oz) for p in part.mesh.pos]``.
    """
    ox, oy, oz = offset
    part.mesh.pos = [(p[0] + ox, p[1] + oy, p[2] + oz) for p in part.mesh.pos]
    return part


def _center(asset):
    """Translate every part so the asset's bounds centre lands on the origin.

    Used by the two pickups, whose contract is "origin at its centre".  Doing
    it on the finished geometry rather than by hand-tuned constants means a
    late tweak to a trim strip cannot silently slide the asset off its own
    pivot.
    """
    lo, hi = asset.bounds()
    off = (0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2]))
    for p in asset.parts:
        _shift(p, (-off[0], -off[1], -off[2]))
    return asset


def _torso_ring(z, hx, hy, r, n_corner=3):
    """One CCW (x, y) torso cross-section at height ``z``.

    ``rounded_rect`` is already CCW in XY, which is exactly the right-hand
    order ``C.loft`` needs while advancing along +Z, so the rings can be fed
    to ``loft`` unchanged.  The rounded corners are what curve the vest's front
    and back panels: a hard rectangle would read as a crate.
    """
    return [(px, py, z) for (px, py) in C.rounded_rect(hx, hy, r, n_corner)]


def _trigger_guard(m, y_front, z_top, color, w=0.018, bar=0.011, drop=0.040,
                   span=0.050, thick=0.010):
    """An open trigger guard: front post + bottom rail, 3-sided.

    Deliberately not a closed frame -- a filled loop would be 12 triangles of
    solid where two thin bars read the same and cost half.  ``y_front`` is the
    guard's forward post, ``span`` how far back the bottom rail reaches.
    """
    _rbox(m, (w, bar, drop), center=(0.0, y_front, z_top - drop * 0.5),
          color=color)
    _rbox(m, (w, span, thick),
          center=(0.0, y_front + span * 0.5 - bar * 0.5, z_top - drop),
          color=color)
    return m


# ---------------------------------------------------------------------------
# 1. pistol -- ~0.20 m, origin at the centre of the grip
# ---------------------------------------------------------------------------

def _pistol():
    a = C.Asset("wep_pistol", "weapon")

    # Grip centre IS the origin, raked back 18 deg: local (0,0,+h) lands at
    # -Y (forward) so the top of the grip tucks under the frame and the
    # bottom trails to +Y the way a raked grip does.
    grip = a.part("pistol_grip", base_color=POLYMER, roughness=0.62)
    _rbox(grip.mesh, (0.030, 0.034, 0.100), rot=(18.0, 0.0, 0.0), color=POLYMER)
    for k in range(3):                               # finger-groove relief
        _rbox(grip.mesh, (0.034, 0.008, 0.008),
              center=(0.0, 0.006 + k * 0.001, -0.022 + k * 0.018),
              rot=(18.0, 0.0, 0.0), color=POLYMER_LT)

    # slide + frame, one block, the mass of the silhouette
    body = a.part("pistol_body", base_color=GUNMETAL, metallic=0.25,
                  roughness=0.48)
    C.box(body.mesh, (0.030, 0.160, 0.040), center=(0.0, -0.040, 0.066),
          color=GUNMETAL)
    C.box(body.mesh, (0.026, 0.034, 0.030), center=(0.0, 0.016, 0.050),
          color=GUNMETAL)                              # frame / dust cover
    _rbox(body.mesh, (0.026, 0.030, 0.026), center=(0.0, 0.041, 0.062),
          color=GUNMETAL_LT)                           # rear sight / hammer
    for k in range(4):                                 # slide serrations
        _rbox(body.mesh, (0.032, 0.005, 0.034),
              center=(0.0, 0.018 + k * 0.008, 0.068), color=GUNMETAL_LT)

    metal = a.part("pistol_metal", base_color=STEEL_BRIGHT, metallic=0.6,
                   roughness=0.35)
    C.cylinder(metal.mesh, 0.0095, 0.024, 10, center=(0.0, -0.132, 0.066),
               axis="Y", color=BARREL)                # muzzle / barrel tip
    C.cylinder(metal.mesh, 0.0055, 0.014, 8, center=(0.0, -0.144, 0.066),
               axis="Y", color=BARREL_DK)             # bore
    C.box(metal.mesh, (0.008, 0.030, 0.008), center=(0.0, -0.126, 0.090),
          color=STEEL_BRIGHT)                          # front sight blade
    _trigger_guard(metal.mesh, -0.046, 0.048, STEEL_BRIGHT,
                   w=0.018, bar=0.011, drop=0.040, span=0.056)
    _rbox(metal.mesh, (0.013, 0.012, 0.026), center=(0.0, -0.024, 0.032),
          rot=(10.0, 0.0, 0.0), color=STEEL_BRIGHT)   # trigger
    return a


# ---------------------------------------------------------------------------
# 2. SMG -- ~0.45 m, origin at the centre of the grip
# ---------------------------------------------------------------------------

def _smg():
    a = C.Asset("wep_smg", "weapon")

    grip = a.part("smg_grip", base_color=POLYMER, roughness=0.62)
    _rbox(grip.mesh, (0.026, 0.034, 0.105), rot=(20.0, 0.0, 0.0), color=POLYMER)
    # 8 deg forward cant: the magazine bottom kicks toward -Y.
    _rbox(grip.mesh, (0.022, 0.055, 0.170), center=(0.0, -0.062, -0.050),
          rot=(-8.0, 0.0, 0.0), color=POLYMER)        # magazine
    _rbox(grip.mesh, (0.028, 0.018, 0.174), center=(0.0, -0.070, -0.052),
          rot=(-8.0, 0.0, 0.0), color=POLYMER_LT)     # mag floor plate
    _rbox(grip.mesh, (0.026, 0.040, 0.078), center=(0.0, -0.180, 0.008),
          color=POLYMER)                               # foregrip

    body = a.part("smg_body", base_color=GUNMETAL, metallic=0.25, roughness=0.48)
    C.box(body.mesh, (0.040, 0.170, 0.072), center=(0.0, -0.045, 0.068),
          color=GUNMETAL)                              # receiver
    C.box(body.mesh, (0.034, 0.105, 0.050), center=(0.0, -0.182, 0.068),
          color=GUNMETAL_LT)                           # handguard
    C.box(body.mesh, (0.026, 0.026, 0.026), center=(0.0, 0.052, 0.068),
          color=GUNMETAL)                              # buffer tube stub
    _rbox(body.mesh, (0.028, 0.090, 0.055), center=(0.0, 0.108, 0.062),
          color=GUNMETAL)                              # stock
    C.box(body.mesh, (0.034, 0.012, 0.060), center=(0.0, 0.152, 0.060),
          color=GUNMETAL_LT)                           # butt plate
    C.box(body.mesh, (0.030, 0.030, 0.014), center=(0.0, -0.140, 0.108),
          color=GUNMETAL_LT)                           # top rail
    C.box(body.mesh, (0.012, 0.034, 0.020), center=(0.0, 0.012, 0.112),
          color=GUNMETAL_LT)                           # charging handle

    metal = a.part("smg_metal", base_color=STEEL_BRIGHT, metallic=0.6,
                   roughness=0.35)
    C.cylinder(metal.mesh, 0.013, 0.030, 10, center=(0.0, -0.249, 0.068),
               axis="Y", color=BARREL)                # barrel shroud tip
    C.cylinder(metal.mesh, 0.012, 0.022, 10, center=(0.0, -0.275, 0.068),
               axis="Y", color=BARREL)                # muzzle
    C.cylinder(metal.mesh, 0.006, 0.026, 8, center=(0.0, -0.285, 0.068),
               axis="Y", color=BARREL_DK)             # bore
    C.box(metal.mesh, (0.008, 0.008, 0.018), center=(0.0, -0.228, 0.108),
          color=STEEL_BRIGHT)                          # front post
    C.box(metal.mesh, (0.026, 0.008, 0.018), center=(0.0, -0.100, 0.108),
          color=STEEL_BRIGHT)                          # rear aperture
    _trigger_guard(metal.mesh, -0.048, 0.040, STEEL_BRIGHT,
                   w=0.018, bar=0.011, drop=0.038, span=0.052)
    _rbox(metal.mesh, (0.013, 0.012, 0.026), center=(0.0, -0.026, 0.026),
          rot=(10.0, 0.0, 0.0), color=STEEL_BRIGHT)   # trigger
    return a


# ---------------------------------------------------------------------------
# 3. shotgun -- ~0.75 m, origin at the centre of the grip
# ---------------------------------------------------------------------------

def _shotgun():
    a = C.Asset("wep_shotgun", "weapon")

    grip = a.part("shotgun_grip", base_color=WOOD, roughness=0.7)
    _rbox(grip.mesh, (0.026, 0.036, 0.108), rot=(20.0, 0.0, 0.0), color=WOOD)
    for k in range(3):
        _rbox(grip.mesh, (0.030, 0.007, 0.008),
              center=(0.0, 0.004 + k * 0.001, -0.026 + k * 0.019),
              rot=(20.0, 0.0, 0.0), color=WOOD_DK)
    # pump / forend, ribbed, the tell that reads "shotgun" not "rifle"
    _rbox(grip.mesh, (0.046, 0.135, 0.052), center=(0.0, -0.192, 0.044),
          color=WOOD)
    for k in range(6):
        _rbox(grip.mesh, (0.050, 0.007, 0.050),
              center=(0.0, -0.243 + k * 0.020, 0.044), color=WOOD_DK)

    body = a.part("shotgun_body", base_color=GUNMETAL, metallic=0.25,
                  roughness=0.5)
    C.box(body.mesh, (0.038, 0.170, 0.060), center=(0.0, -0.010, 0.048),
          color=GUNMETAL)                              # receiver
    # Stock: -6 deg about X drops the butt end, so the comb slopes down to the
    # heel the way a gunstock does instead of reading as a second box.
    _rbox(body.mesh, (0.034, 0.215, 0.058), center=(0.0, 0.182, 0.055),
          rot=(-6.0, 0.0, 0.0), color=GUNMETAL)
    _rbox(body.mesh, (0.040, 0.026, 0.072), center=(0.0, 0.290, 0.042),
          rot=(-6.0, 0.0, 0.0), color=POLYMER)          # recoil pad
    C.box(body.mesh, (0.030, 0.010, 0.030), center=(0.0, 0.070, 0.044),
          color=GUNMETAL_LT)                           # receiver-to-stock joint

    metal = a.part("shotgun_metal", base_color=STEEL_BRIGHT, metallic=0.6,
                   roughness=0.35)
    C.cylinder(metal.mesh, 0.016, 0.400, 12, center=(0.0, -0.235, 0.055),
               axis="Y", color=BARREL)                # barrel
    C.cylinder(metal.mesh, 0.019, 0.022, 12, center=(0.0, -0.424, 0.055),
               axis="Y", color=BARREL_DK)             # muzzle ring
    C.cylinder(metal.mesh, 0.013, 0.380, 10, center=(0.0, -0.205, 0.035),
               axis="Y", color=BARREL_DK)             # magazine tube
    C.cylinder(metal.mesh, 0.010, 0.018, 8, center=(0.0, -0.404, 0.035),
               axis="Y", color=STEEL_BRIGHT)          # tube end cap
    C.sphere(metal.mesh, 0.006, 8, 4, center=(0.0, -0.432, 0.076),
             color=STEEL_BRIGHT)                      # bead sight
    _trigger_guard(metal.mesh, -0.044, 0.042, STEEL_BRIGHT,
                   w=0.018, bar=0.012, drop=0.038, span=0.052)
    _rbox(metal.mesh, (0.013, 0.012, 0.026), center=(0.0, -0.022, 0.026),
          rot=(10.0, 0.0, 0.0), color=STEEL_BRIGHT)   # trigger
    return a


# ---------------------------------------------------------------------------
# 4. rocket launcher -- ~1.0 m, origin at the centre of the grip
# ---------------------------------------------------------------------------

def _rocket_launcher():
    a = C.Asset("wep_rocket_launcher", "weapon")

    grip = a.part("rpg_grip", base_color=POLYMER, roughness=0.6)
    _rbox(grip.mesh, (0.030, 0.040, 0.115), rot=(18.0, 0.0, 0.0), color=POLYMER)
    _rbox(grip.mesh, (0.030, 0.130, 0.030), center=(0.0, 0.170, 0.012),
          color=POLYMER)                               # shoulder-rest strut
    _rbox(grip.mesh, (0.060, 0.030, 0.092), center=(0.0, 0.250, 0.028),
          rot=(-10.0, 0.0, 0.0), color=POLYMER_LT)     # shoulder pad
    _trigger_guard(grip.mesh, -0.052, 0.062, POLYMER,
                   w=0.020, bar=0.013, drop=0.044, span=0.058)
    _rbox(grip.mesh, (0.014, 0.013, 0.028), center=(0.0, -0.028, 0.040),
          rot=(10.0, 0.0, 0.0), color=POLYMER_LT)      # trigger

    body = a.part("rpg_body", base_color=OLIVE, metallic=0.2, roughness=0.62)
    C.cylinder(body.mesh, 0.052, 0.740, 14, center=(0.0, -0.265, 0.075),
               axis="Y", color=OLIVE)                  # launch tube
    C.cone(body.mesh, 0.055, 0.060, 14, center=(0.0, 0.135, 0.075),
           radius_top=0.038, axis="Y", color=OLIVE_DK)  # rear venturi
    C.box(body.mesh, (0.020, 0.660, 0.014), center=(0.0, -0.265, 0.134),
          color=OLIVE_DK)                              # top rail
    for k in range(4):                                  # tube bands
        C.cylinder(body.mesh, 0.057, 0.018, 14, center=(0.0, -0.470 + k * 0.180,
                                                        0.075),
                   axis="Y", color=OLIVE_DK)

    metal = a.part("rpg_metal", base_color=STEEL_BRIGHT, metallic=0.65,
                   roughness=0.32)
    # Flared muzzle: cone radius is the -Y end, radius_top the +Y end, so the
    # flare opens toward the muzzle the way it should.
    C.cone(metal.mesh, 0.100, 0.075, 14, center=(0.0, -0.6675, 0.075),
           radius_top=0.055, axis="Y", color=BARREL)    # flare
    C.cylinder(metal.mesh, 0.104, 0.016, 14, center=(0.0, -0.713, 0.075),
               axis="Y", color=BARREL_DK)             # muzzle lip
    C.cylinder(metal.mesh, 0.052, 0.014, 14, center=(0.0, -0.635, 0.075),
               axis="Y", color=BARREL_DK)             # blast-shield collar
    C.cylinder(metal.mesh, 0.040, 0.190, 10, center=(0.0, 0.135, 0.075),
               axis="Y", color=BARREL_DK)             # breech
    C.box(metal.mesh, (0.030, 0.090, 0.044), center=(0.0, 0.165, 0.075),
          color=BARREL)                                # sight bracket
    C.cylinder(metal.mesh, 0.020, 0.030, 10, center=(0.0, 0.130, 0.098),
               axis="Z", color=STEEL_BRIGHT)            # open sight
    C.box(metal.mesh, (0.040, 0.016, 0.020), center=(0.0, -0.210, 0.152),
          color=STEEL_BRIGHT)                          # optic body
    return a


# ---------------------------------------------------------------------------
# 5. armour vest -- 0.45 wide x 0.55 tall x 0.22 deep, centred
# ---------------------------------------------------------------------------

def _armor_vest():
    a = C.Asset("pickup_armor_vest", "pickup")

    # Envelope target: 0.45 wide (X) x 0.55 tall (Z) x 0.22 deep (Y).  The
    # shell is authored at full size, then _center() lands the bounds centre
    # on the origin -- neck opening above, hem below.
    #
    # Torso shell: rounded cross-sections lofted up the Z axis.  The rounded
    # corners ARE the "slightly curved front and back panels" -- a hard
    # rectangle here reads as a crate, not a body armour carrier.  The half
    # depth is held at 0.106 so the shell alone spans 0.212 of the 0.22
    # budget and the pouches / trim fit inside it.
    shell = a.part("vest_shell", base_color=ARMO_1, roughness=0.66)
    C.loft(shell.mesh,
           [_torso_ring(-0.272, 0.155, 0.068, 0.048),
            _torso_ring(-0.200, 0.192, 0.086, 0.060),
            _torso_ring(-0.060, 0.210, 0.098, 0.068),
            _torso_ring(0.070, 0.216, 0.100, 0.068),
            _torso_ring(0.170, 0.206, 0.092, 0.062),
            _torso_ring(0.228, 0.188, 0.076, 0.050)],
           ARMO_1, cap_start_flip=True, cap_end_flip=False)

    plate = a.part("vest_plate", base_color=ARMO_2, roughness=0.5)
    C.box(plate.mesh, (0.290, 0.026, 0.370), center=(0.0, -0.088, -0.020),
          color=ARMO_2)                                 # front ballistic plate
    C.box(plate.mesh, (0.310, 0.020, 0.350), center=(0.0, 0.088, -0.020),
          color=ARMO_2)                                 # back plate
    for sx in (-1, 1):                                  # magazine pouches
        C.box(plate.mesh, (0.092, 0.044, 0.150),
              center=(sx * 0.096, -0.090, -0.180), color=ARMO_2)
        C.box(plate.mesh, (0.100, 0.010, 0.022),
              center=(sx * 0.096, -0.104, -0.110), color=ARMO_TRIM)

    straps = a.part("vest_straps", base_color=ARMO_STRAP, roughness=0.72)
    # Two shoulder straps arch over the top leaving a neck gap between them.
    for sx in (-1, 1):
        _rbox(straps.mesh, (0.078, 0.186, 0.028),
              center=(sx * 0.126, 0.0, 0.252), color=ARMO_STRAP)
    C.box(straps.mesh, (0.180, 0.030, 0.070), center=(0.0, 0.072, 0.244),
          color=ARMO_STRAP)                             # back neck yoke
    for sx in (-1, 1):                                  # side cummerbunds
        C.box(straps.mesh, (0.018, 0.160, 0.200), center=(sx * 0.207, 0.0, 0.020),
              color=ARMO_STRAP)
        C.box(straps.mesh, (0.026, 0.056, 0.046), center=(sx * 0.205, 0.0, 0.020),
              color=ARMO_TRIM)

    trim = a.part("vest_trim", base_color=ARMO_TRIM, roughness=0.55)
    C.box(trim.mesh, (0.146, 0.010, 0.024), center=(0.0, -0.101, 0.075),
          color=ARMO_TRIM)                              # hi-viz chest flash
    C.box(trim.mesh, (0.146, 0.010, 0.024), center=(0.0, 0.098, 0.075),
          color=ARMO_TRIM)
    for sx in (-1, 1):                                  # shoulder yoke tabs
        C.box(trim.mesh, (0.048, 0.124, 0.011), center=(sx * 0.126, 0.0, 0.270),
              color=ARMO_TRIM)
    return _center(a)


# ---------------------------------------------------------------------------
# 6. first-aid pack -- 0.30 wide x 0.20 tall x 0.12 deep, centred
# ---------------------------------------------------------------------------

def _health_pack():
    a = C.Asset("pickup_health_pack", "pickup")

    # Envelope target: 0.30 wide (X) x 0.20 tall (Z) x 0.12 deep (Y).
    #
    # Every fitting is kept INSIDE that envelope rather than allowed to grow
    # it, which is why the numbers here are hand-checked instead of derived:
    # hinges proud of the case add 20-30 mm to a 300 mm box, and a handle
    # that arcs above the lid adds it in Z.  The handle is a low-profile loop
    # whose bar caps exactly flush with the case top (0.100), and the cross
    # straddles the front wall rather than standing off it.
    case = a.part("med_case", base_color=MED_RED, roughness=0.55)
    C.box(case.mesh, (0.300, 0.116, 0.200), color=MED_RED)           # body
    C.box(case.mesh, (0.300, 0.116, 0.030), center=(0.0, 0.0, 0.085),
          color=MED_RED_DK)                                          # lid
    C.box(case.mesh, (0.300, 0.118, 0.012), center=(0.0, 0.0, 0.064),
          color=MED_RED_DK)                                          # clasp band
    C.box(case.mesh, (0.280, 0.106, 0.014), center=(0.0, 0.0, -0.093),
          color=MED_RED_DK)                                          # base plinth
    for sx in (-1, 1):                                  # handle posts
        C.box(case.mesh, (0.020, 0.034, 0.024), center=(sx * 0.036, 0.0, 0.082),
              color=MED_RED_DK)
    C.box(case.mesh, (0.092, 0.034, 0.006), center=(0.0, 0.0, 0.097),
          color=MED_RED_DK)                                          # handle bar

    trim = a.part("med_trim", base_color=MED_WHITE, roughness=0.5)
    # The cross: two slim boxes straddling the front (-Y) wall, 5 mm proud of
    # it and 5 mm sunk in, so it is a real protruding inlay rather than a
    # coplanar decal that would z-fight with the case face.
    C.box(trim.mesh, (0.042, 0.010, 0.130), center=(0.0, -0.057, -0.006),
          color=MED_WHITE)
    C.box(trim.mesh, (0.130, 0.010, 0.042), center=(0.0, -0.057, -0.006),
          color=MED_WHITE)
    for sx in (-1, 1):                                  # lid clasps
        C.box(trim.mesh, (0.030, 0.010, 0.048), center=(sx * 0.098, -0.058, 0.046),
              color=MED_WHITE)
    C.box(trim.mesh, (0.130, 0.008, 0.016), center=(0.0, -0.057, -0.080),
          color=MED_WHITE)                              # lower label bar

    metal = a.part("med_metal", base_color=MED_GREY, metallic=0.5,
                   roughness=0.4)
    for sx in (-1, 1):                                  # side hinges
        C.cylinder(metal.mesh, 0.008, 0.024, 8,
                   center=(sx * 0.136, 0.0, 0.048), axis="X", color=MED_GREY)
    return _center(a)


# ---------------------------------------------------------------------------
# 7. bat -- ~1.05 m, origin at the centre of the grip
# ---------------------------------------------------------------------------

def _bat():
    """A taped-handle ball bat / steel pipe hybrid, ~0.85 m end to end.

    Same origin contract as the four guns: the origin is the centre of the
    GRIP, not of the bat, so a runtime can parent it to a hand bone exactly as
    it parents a pistol.  The club points along +Z so the silhouette reads
    upright in the weapon preview instead of edge-on.
    """
    a = C.Asset("wep_bat", "weapon")

    grip = a.part("bat_grip", base_color=BAT_WOOD, roughness=0.72)
    # Tapered handle: three stations so it is not a plain dowel.
    C.loft(grip.mesh,
           [[(0.030 * math.cos(t), 0.030 * math.sin(t), z)
             for t in [2.0 * math.pi * i / 8 for i in range(8)]]
            for z in (-0.150, -0.040, 0.075)],
           BAT_WOOD_DK, cap_start_flip=True, cap_end_flip=False)
    # Grip tape: four bands, alternating shade, so the hand position reads.
    for k, z in enumerate((-0.128, -0.086, -0.044, -0.002)):
        C.cylinder(grip.mesh, 0.0345, 0.030, 8, center=(0.0, 0.0, z),
                   color=BAT_TAPE if k % 2 == 0 else C.shade(BAT_TAPE, 1.35))

    body = a.part("bat_body", base_color=BAT_WOOD, roughness=0.62)
    # Barrel -> barrel swell -> blunt cap, as one closed flat loft.
    C.loft(body.mesh,
           [[(0.031 * math.cos(t), 0.031 * math.sin(t), z)
             for t in [2.0 * math.pi * i / 10 for i in range(10)]]
            for z in (0.078, 0.185, 0.330, 0.455, 0.530, 0.556)],
           BAT_WOOD, cap_start_flip=True, cap_end_flip=False)
    # One lacquered band near the throat, face colour only.
    C.recolor_faces_where(
        body, lambda n, c: 0.178 < c[2] < 0.226, FLARE_RED)

    knob = a.part("bat_knob", base_color=BAT_TAPE, roughness=0.55)
    C.cylinder(knob.mesh, 0.036, 0.022, 8, center=(0.0, 0.0, -0.160),
               color=BAT_TAPE)                                # butt end cap
    C.cylinder(knob.mesh, 0.030, 0.016, 8, center=(0.0, 0.0, 0.562),
               color=C.shade(BAT_TAPE, 1.2))                 # strike face ring
    return a


# ---------------------------------------------------------------------------
# 8. grenade -- ~0.11 m, origin at the centre of the body
# ---------------------------------------------------------------------------

def _grenade():
    """A frag with a pull ring and a safety pin.

    Origin at the body's centre, which is where the runtime's hand/throw
    transform expects a point-throw primitive -- unlike the guns, this is not
    meant to be sighted down its own length.
    """
    a = C.Asset("wep_grenade", "weapon")

    body = a.part("gren_body", base_color=GREN_BODY, roughness=0.55)
    # Classic two-dome silhouette: sphere, waist, sphere.  A single sphere
    # reads as a ball bearing.
    C.sphere(body.mesh, 0.0425, 12, 6, center=(0.0, 0.0, 0.0075), squash=0.86,
             color=GREN_BODY)
    C.sphere(body.mesh, 0.0425, 12, 6, center=(0.0, 0.0, -0.0075), squash=0.86,
             color=GREN_BODY)
    C.cylinder(body.mesh, 0.0335, 0.030, 10, center=(0.0, 0.0, 0.0),
               color=GREN_BODY_DK)                            # moulded waist
    # Fuse assembly: neck, collar, striker cap.
    C.cylinder(body.mesh, 0.0175, 0.020, 8, center=(0.0, 0.0, 0.047),
               color=GREN_BODY_DK)
    cap = a.part("gren_cap", base_color=STEEL_BRIGHT, metallic=0.6,
                 roughness=0.35)
    C.cylinder(cap.mesh, 0.0165, 0.014, 8, center=(0.0, 0.0, 0.062),
               color=STEEL_BRIGHT)                            # striker cap
    C.cylinder(cap.mesh, 0.0055, 0.030, 6, center=(0.010, 0.0, 0.080),
               axis="X", color=STEEL_BRIGHT)                  # pin, pulled out

    ring = a.part("gren_ring", base_color=BRASS, metallic=0.7, roughness=0.32)
    # Pull ring hanging off the pin's eye, plane of the ring facing the user.
    C.torus(ring.mesh, 0.0135, 0.0030, 10, 5, center=(0.024, 0.0, 0.076),
            color=BRASS, axis="Y", smooth=False)
    # Muzzle-authored sight band, a face-colour detail that gives the olive
    # dome a top and a bottom at gameplay distance.
    C.recolor_faces_where(
        body, lambda n, c: n[2] < -0.72, GREN_BODY_DK)
    return a


# ---------------------------------------------------------------------------
# 9. ammo box -- 0.34 wide x 0.16 tall x 0.20 deep, centred
# ---------------------------------------------------------------------------

def _ammo_box():
    """Olive steel ammo can with a latching lid.

    Origin centred like the other pickups: a pickup is spawned in the world,
    not parented to a hand.
    """
    a = C.Asset("pickup_ammo_box", "pickup")

    body = a.part("ammo_body", base_color=OLIVE, metallic=0.30, roughness=0.60)
    C.box(body.mesh, (0.340, 0.200, 0.128), color=OLIVE)          # tin
    C.box(body.mesh, (0.348, 0.208, 0.020), center=(0.0, 0.0, 0.064),
          color=OLIVE_DK)                                          # lid flange
    C.box(body.mesh, (0.348, 0.208, 0.014), center=(0.0, 0.0, -0.070),
          color=OLIVE_DK)                                          # base flange
    # Pressed side ribs: face colour only, three per side.
    C.recolor_faces_where(
        body, lambda n, c: abs(n[1]) > 0.70 and abs(c[2]) < 0.030,
        OLIVE_DK)

    latch = a.part("ammo_latch", base_color=MED_GREY, metallic=0.55,
                   roughness=0.42)
    for sx in (-1, 1):
        C.box(latch.mesh, (0.030, 0.014, 0.052),
              center=(sx * 0.112, -0.104, 0.028), color=MED_GREY)
        C.cylinder(latch.mesh, 0.008, 0.026, 6,
                   center=(sx * 0.112, -0.112, 0.006), axis="Y",
                   color=STEEL_BRIGHT)                             # catch pin
    C.box(latch.mesh, (0.150, 0.014, 0.012), center=(0.0, -0.104, 0.060),
          color=MED_GREY)                                          # hasp

    stencil = a.part("ammo_stencil", base_color=AMBER_STENCIL, roughness=0.7)
    # Struck through 5 mm proud of the front wall so it reads as paint on tin
    # rather than a coplanar decal that z-fights with the body face.
    C.box(stencil.mesh, (0.180, 0.010, 0.026), center=(0.0, -0.101, -0.014),
          color=AMBER_STENCIL)
    C.box(stencil.mesh, (0.120, 0.010, 0.016), center=(0.0, -0.101, -0.042),
          color=AMBER_STENCIL)
    for sx in (-1, 1):                                             # stencil dot
        C.cylinder(stencil.mesh, 0.014, 0.010, 8,
                   center=(sx * 0.128, -0.101, -0.014), axis="Y",
                   color=AMBER_STENCIL)
    handle = a.part("ammo_handle", base_color=MED_GREY, metallic=0.5,
                    roughness=0.45)
    C.box(handle.mesh, (0.130, 0.030, 0.014), center=(0.0, 0.0, 0.084),
          color=MED_GREY)                                          # carry bar
    for sx in (-1, 1):
        C.box(handle.mesh, (0.016, 0.030, 0.024),
              center=(sx * 0.058, 0.0, 0.076), color=MED_GREY)
    return _center(a)


# ---------------------------------------------------------------------------
# 10. cash stack -- 0.30 wide x 0.11 tall x 0.20 deep, centred
# ---------------------------------------------------------------------------

def _cash_stack():
    """Three banded bundles of notes plus a loose wad on top.

    Banded bundles are how the shape stays legible at 20 m: a single block
    reads as a brick, three stepped bands read as money.
    """
    a = C.Asset("pickup_cash_stack", "pickup")

    notes = a.part("cash_notes", base_color=BILL_GREEN, roughness=0.85)
    bands = a.part("cash_bands", base_color=BILL_TAN, roughness=0.9)
    # (size, centre, colour) per bundle, deliberately stepped in X and Y so the
    # pile is not three identical bricks in a row.
    bundles = (
        ((0.300, 0.196, 0.030), (-0.006, -0.052, -0.044), BILL_GREEN),
        ((0.272, 0.180, 0.026), (0.028, 0.040, -0.016), BILL_GREEN_DK),
        ((0.240, 0.162, 0.022), (-0.030, 0.026, 0.012), BILL_GREEN),
        ((0.170, 0.130, 0.020), (0.034, -0.026, 0.033), BILL_TAN_DK),
    )
    for size, centre, col in bundles:
        C.box(notes.mesh, size, center=centre, color=col)
    # Rubber bands: one hoop around each of the three lower bundles.  A band is
    # two thin straps, not a closed ring, so the bundle shows through.
    for size, centre, _col in bundles[:3]:
        hx, hy, hz = size[0] * 0.5, size[1] * 0.5, size[2]
        for sx in (-1, 1):
            C.box(bands.mesh, (0.014, hy * 2.04, hz * 1.06),
                  center=(centre[0] + sx * hx * 0.52, centre[1], centre[2]),
                  color=BILL_TAN)
        C.box(bands.mesh, (hx * 2.04, 0.014, hz * 1.06),
              center=(centre[0], centre[1] + hy * 0.52, centre[2]),
              color=BILL_TAN)
    return _center(a)


# ---------------------------------------------------------------------------
# 11. oxygen tank -- 0.16 wide x 0.56 tall x 0.16 deep, centred
# ---------------------------------------------------------------------------

def _o2_tank():
    """Scuba cylinder with a valve wheel and a carry handle."""
    a = C.Asset("pickup_o2_tank", "pickup")

    body = a.part("tank_body", base_color=TANK_WHITE, metallic=0.35,
                  roughness=0.42)
    # Rounded-shoulder cylinder: three lofted rings plus a spherical dome cap
    # reads far better than a flat-ended tube.
    C.loft(body.mesh,
           [[(r * math.cos(t), r * math.sin(t), z)
             for t in [2.0 * math.pi * i / 12 for i in range(12)]]
            for (r, z) in ((0.062, -0.250), (0.072, -0.190), (0.072, 0.140),
                           (0.062, 0.196))],
           TANK_WHITE, cap_start_flip=True, cap_end_flip=False)
    C.sphere(body.mesh, 0.062, 12, 5, center=(0.0, 0.0, 0.196), squash=0.92,
             color=TANK_WHITE)
    C.sphere(body.mesh, 0.062, 12, 5, center=(0.0, 0.0, -0.250), squash=0.92,
             color=TANK_WHITE)
    # Two teal wrap bands -- the only place the pickup category gets neon.
    for z in (-0.090, 0.050):
        C.cylinder(body.mesh, 0.0735, 0.034, 12, center=(0.0, 0.0, z),
                   color=NEON_TEAL)

    valve = a.part("tank_valve", base_color=BRASS, metallic=0.7, roughness=0.34)
    C.cylinder(valve.mesh, 0.026, 0.048, 8, center=(0.0, 0.0, 0.268),
               color=BRASS)                                       # valve body
    C.cylinder(valve.mesh, 0.0125, 0.036, 8, center=(0.0, 0.0, 0.308),
               color=BRASS)                                       # stem
    C.cylinder(valve.mesh, 0.0145, 0.050, 6,
               center=(0.034, 0.0, 0.308), axis="X", color=BRASS)  # outlet
    C.torus(valve.mesh, 0.030, 0.0068, 12, 5, center=(0.0, 0.0, 0.330),
            color=BRASS, smooth=False)                            # hand wheel

    handle = a.part("tank_handle", base_color=STEEL_BRIGHT, metallic=0.6,
                    roughness=0.38)
    # The carry handle arcs over the crown and clears the valve wheel by
    # 30 mm, which is why it is a tube rather than a box.
    C.tube(handle.mesh, [(-0.058, 0.0, 0.150), (-0.046, 0.0, 0.238),
                         (0.0, 0.0, 0.268), (0.046, 0.0, 0.238),
                         (0.058, 0.0, 0.150)], 0.0105, 6, color=STEEL_BRIGHT,
           smooth=False)
    for sx in (-1, 1):                                             # handle feet
        C.cylinder(handle.mesh, 0.013, 0.020, 6, center=(sx * 0.058, 0.0, 0.144),
                   color=STEEL_BRIGHT)
    return _center(a)


# ---------------------------------------------------------------------------
# 12. flare pack -- 0.26 wide x 0.07 tall x 0.18 deep, centred
# ---------------------------------------------------------------------------

def _flare_pack():
    """A bundle of five road flares, rubber-banded."""
    a = C.Asset("pickup_flare_pack", "pickup")

    tubes = a.part("flare_tubes", base_color=FLARE_RED, roughness=0.62)
    caps = a.part("flare_caps", base_color=FLARE_RED_DK, roughness=0.55)
    # Five tubes, 4 in the bottom row and 1 riding the crease of the two
    # middle ones -- the pile-up that makes a flat box read as cylinders.
    layout = ((-0.082, -0.046), (-0.006, -0.046), (0.070, -0.046),
              (-0.044, 0.004), (0.032, 0.004), (-0.004, 0.050))
    for i, (x, y) in enumerate(layout):
        r = 0.0235 if i < 4 else 0.0215
        z = -0.006 if i < 4 else 0.026
        C.cylinder(tubes.mesh, r, 0.176, 8, center=(x, y, z),
                   axis="Y", color=FLARE_RED)
        # Cap at the -Y end and a paper band near the middle.
        C.cylinder(caps.mesh, r * 1.03, 0.020, 8, center=(x, y - 0.082, z),
                   axis="Y", color=FLARE_RED_DK)
    for i, (x, y) in enumerate(layout):
        r = 0.0235 if i < 4 else 0.0215
        z = -0.006 if i < 4 else 0.026
        C.cylinder(caps.mesh, r * 1.04, 0.016, 8, center=(x, y + 0.030, z),
                   axis="Y", color=AMBER_STENCIL)

    strap = a.part("flare_strap", base_color=(0.14, 0.15, 0.16), roughness=0.85)
    C.box(strap.mesh, (0.246, 0.020, 0.128), center=(0.0, -0.014, 0.0),
          color=(0.14, 0.15, 0.16))                              # lower band
    C.box(strap.mesh, (0.246, 0.020, 0.070), center=(0.0, -0.030, 0.050),
          color=(0.20, 0.21, 0.22))                              # upper band
    C.box(strap.mesh, (0.030, 0.048, 0.012), center=(0.0, -0.030, 0.086),
          color=BAT_TAPE)                                         # buckle
    return _center(a)


# ---------------------------------------------------------------------------

def build_all():
    """Return the ordered list of weapon + pickup assets."""
    return [
        _pistol(),
        _smg(),
        _shotgun(),
        _rocket_launcher(),
        _bat(),
        _grenade(),
        _armor_vest(),
        _health_pack(),
        _ammo_box(),
        _cash_stack(),
        _o2_tank(),
        _flare_pack(),
    ]
