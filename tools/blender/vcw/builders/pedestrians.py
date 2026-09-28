"""Category C: low-poly pedestrians.

Every figure is built as a set of SEPARATE closed shells -- one ``Part`` per
joint segment -- so a renderer can pose the result by rotating each part
about its own joint.  Nothing is merged: ``head``, ``torso``,
``upper_arm_L`` ... ``lower_leg_R`` are distinct parts, and the exporter emits
them as separate submeshes with their own materials.

RIG / A-POSE
    Authoring is Blender-style: +X right, -Y forward (the way the figure
    faces), +Z up, metres.  The origin is the centre of the soles at z = 0.
    Arms hang down and angle slightly away from the torso, legs are straight
    and set a little apart, so every joint axis is unambiguous.

    Two hand-authored shells per figure (torso, hair) plus the eight limbs
    are verified by :func:`_seal`, which flips any closed shell whose signed
    volume came out negative.  ``loft`` winding depends on the direction the
    rings advance relative to the profile's right-hand normal, so the flip is
    the cheap way to guarantee the exporter's outward-facing check passes
    without every station table having to be reasoned about by hand.
"""

import math

from .. import core as C

# --------------------------------------------------------------------------
# palettes -- one per figure, all four instantly separable at a glance
# --------------------------------------------------------------------------

SKIN_SUIT = (0.87, 0.69, 0.56)
SKIN_DRESS = (0.94, 0.78, 0.65)
SKIN_WORK = (0.76, 0.58, 0.44)
SKIN_STREET = (0.62, 0.45, 0.34)

SUIT_CLOTH = (0.15, 0.17, 0.26)      # charcoal-navy jacket
SUIT_SHIRT = (0.95, 0.95, 0.92)
SUIT_TIE = (0.70, 0.13, 0.17)
SUIT_TROUSER = (0.12, 0.13, 0.19)
SUIT_SHOE = (0.08, 0.07, 0.06)
SUIT_HAIR = (0.20, 0.14, 0.10)

DRESS_CLOTH = (0.94, 0.23, 0.45)     # hot pink
DRESS_TRIM = (0.14, 0.63, 0.66)      # teal
DRESS_SHOE = (0.20, 0.63, 0.62)
DRESS_HAIR = (0.93, 0.79, 0.36)      # blonde

WORK_DENIM = (0.28, 0.43, 0.66)
WORK_SHIRT = (0.86, 0.67, 0.36)      # dusty tan
WORK_STRAP = (0.24, 0.37, 0.58)
WORK_BOOT = (0.34, 0.23, 0.14)
WORK_CAP = (0.96, 0.60, 0.13)
WORK_HAIR = (0.18, 0.13, 0.10)

STREET_TOP = (0.32, 0.63, 0.52)      # oversized sea-green hoodie
STREET_TROUSER = (0.20, 0.21, 0.25)  # baggy charcoal cargos
STREET_SHOE = (0.94, 0.94, 0.92)
STREET_CAP = (0.85, 0.28, 0.28)
STREET_HAIR = (0.12, 0.10, 0.10)


# --------------------------------------------------------------------------
# shell helpers -- every one of them returns a CLOSED, outward-wound shell
# --------------------------------------------------------------------------

def _shells(m):
    """Split a mesh into index-connected groups of faces.

    A Part may legitimately hold several disjoint closed shells -- a torso
    plus its neck cylinder, a head plus its nose wedge.  ``core.is_closed``
    reports False for those (the shells share no vertices), so a whole-mesh
    signed volume would just be a SUM over shells and a positive total can
    hide one inverted shell.  Splitting on shared vertex indices lets each
    shell be judged on its own.

    Note the shells are returned whether or not they are watertight, so
    callers must gate on :func:`C.is_closed` before using volume as a
    winding test -- see :func:`_seal`.
    """
    parent = {}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(len(m.pos)):
        parent[i] = i
    for (a, b, c) in m.faces:
        union(a, b)
        union(a, c)
    groups = {}
    for fi, (a, b, c) in enumerate(m.faces):
        groups.setdefault(find(a), []).append(fi)
    return list(groups.values())


def _seal(part):
    """Guarantee every closed shell in ``part`` is wound outward.

    ``core.loft`` emits side walls correctly only when the rings advance
    along the profile's right-hand normal; ``core.cylinder`` and
    ``core.tube`` compensate by flipping the start cap.  Rather than hand-
    check every station table, each connected shell whose signed volume came
    out negative is reversed in place.  Faces and face colours are reversed
    together so colours stay bound to their triangle.
    """
    m = part.mesh
    if not m.faces:
        return part
    for group in _shells(m):
        sub = C.Mesh()
        # remap only this shell's vertices into a standalone mesh
        remap = {}
        for fi in group:
            tri = []
            for v in m.faces[fi]:
                if v not in remap:
                    remap[v] = len(sub.pos)
                    sub.pos.append(m.pos[v])
                tri.append(remap[v])
            sub.faces.append(tuple(tri))
            sub.fcol.append(m.fcol[fi])
        if not C.is_closed(sub):
            # Not watertight -- an open surface or a stack of flat triangles
            # with no interior.  Signed volume is meaningless here (a lone
            # box face integrates to 0.0), so flipping on its sign would
            # silently invert perfectly good faces.  Leave it alone.
            continue
        if C.signed_volume(sub) >= 0.0:
            continue
        # Reverse each face in place.  The face keeps its index, so its
        # face_colors entry stays bound to it and needs no reordering.
        for fi in group:
            a, b, c = m.faces[fi]
            m.faces[fi] = (a, c, b)
    return part


def _ring(m, stations, color, seg=10):
    """Loft stacked circles along +Z.

    ``stations`` is ``(z, radius)`` or ``(z, radius, y_offset)``.
    """
    rings = []
    for st in stations:
        z, r = st[0], st[1]
        yo = st[2] if len(st) > 2 else 0.0
        rings.append([(r * math.cos(2.0 * math.pi * k / seg),
                       yo + r * math.sin(2.0 * math.pi * k / seg), z)
                      for k in range(seg)])
    C.loft(m, rings, color, cap_start=True, cap_end=True, smooth=False)


def _box_ring(m, stations, color, n=2):
    """Loft stacked rounded rectangles along +Z.

    ``stations`` is ``(z, half_x, half_y, corner)``.  This is the torso /
    pelvis workhorse: a rounded-rect section reads as shoulders and hips at
    low poly counts where a cylinder reads as a barrel.
    """
    rings = []
    for (z, hx, hy, r) in stations:
        rings.append([(px, py, z) for (px, py) in C.rounded_rect(hx, hy, r, n)])
    C.loft(m, rings, color, cap_start=True, cap_end=True, smooth=False)


def _bar(m, p0, p1, hx, hy, color, n=1):
    """A closed rectangular-section bar between two world points."""
    t = C.normalize(C.sub(p1, p0))
    ref = (0.0, 0.0, 1.0) if abs(C.dot(t, (0.0, 0.0, 1.0))) < 0.9 \
        else (1.0, 0.0, 0.0)
    side = C.normalize(C.cross(ref, t))
    up = C.cross(t, side)
    rings = []
    for p in (tuple(p0), tuple(p1)):
        rings.append([C.add(p, C.add(C.mul(side, px), C.mul(up, py)))
                      for (px, py) in C.rounded_rect(hx, hy, min(hx, hy) * 0.35, n)])
    C.loft(m, rings, color, cap_start=True, cap_end=True, smooth=False)


def _limb(m, p0, p1, r_top, r_bottom, color, seg=6):
    """A tapered limb segment: a cylinder whose two radii differ.

    ``r_top`` is the radius at ``p0`` (the joint end) and ``r_bottom`` the
    radius at ``p1``, which is what gives the silhouette its taper.
    """
    C.tube(m, [tuple(p0), tuple(p1)], r_bottom, seg=seg, color=color,
           caps=True, smooth=False, radii=[r_top, r_bottom])


def _dome(m, stations, color, seg=10):
    """Hair / hat crown: circles that shrink to a near point on top."""
    _ring(m, stations, color, seg=seg)


def _head(asset, z, r, squash, skin, hair_stations, hair_color,
          hair_long=False):
    """Head sphere (with a nose marking the -Y facing) plus a hair part.

    ``Asset.part()`` attaches the part already, so neither block may call
    ``asset.add()`` again -- a part listed twice ships as two submeshes with
    the same name, which collides in any renderer that keys its rig off part
    names.
    """
    head = asset.part("head", base_color=skin)
    C.sphere(head.mesh, r, seg_u=8, seg_v=5, center=(0.0, 0.0, z),
             color=skin, squash=squash, smooth=False)
    # a small wedge on -Y so the facing is readable from any distance
    C.box(head.mesh, (r * 0.34, r * 0.42, r * 0.34),
          center=(0.0, -r * 0.94, z - r * 0.16), color=skin)

    hair = asset.part("hair", base_color=hair_color)
    _dome(hair.mesh, hair_stations, hair_color, 10)
    if hair_long:
        # two side falls plus a back panel reaching mid-back
        for sx in (-1, 1):
            _bar(hair.mesh, (sx * r * 0.88, r * 0.22, z + r * 0.35),
                 (sx * r * 0.80, r * 0.55, z - r * 2.10),
                 r * 0.46, r * 0.34, hair_color)
        C.box(hair.mesh, (r * 1.55, r * 0.34, r * 2.00),
              center=(0.0, r * 0.74, z - r * 1.30), color=hair_color)
    _seal(hair)


def _shoe(asset, name, x, y, w, d, h, color):
    """A shoe: a closed wedge planted on z = 0, toe toward -Y."""
    p = asset.part(name, base_color=color)
    C.box(p.mesh, (w, d, h), center=(x, y, h * 0.5), color=color)
    # a slightly raised toe cap so the foot reads directionally
    C.box(p.mesh, (w * 0.94, d * 0.34, h * 0.72),
          center=(x, y - d * 0.40, h * 0.36), color=C.shade(color, 0.82))
    return _seal(p)


# --------------------------------------------------------------------------
# the figure assembly
# --------------------------------------------------------------------------

def _figure(spec):
    """Assemble one pedestrian from its skeleton + palette table."""
    a = C.Asset(spec["id"], "pedestrian")
    j = spec["joints"]

    sh_z, sh_x = j["shoulder_z"], j["shoulder_x"]
    el_z, el_x = j["elbow_z"], j["elbow_x"]
    wr_z, wr_x = j["wrist_z"], j["wrist_x"]
    hip_z, hip_x = j["hip_z"], j["hip_x"]
    kn_z, kn_x = j["knee_z"], j["knee_x"]
    an_z, an_x = j["ankle_z"], j["ankle_x"]
    arm_y = j.get("arm_y", 0.0)
    leg_y = j.get("leg_y", 0.0)

    ra0, ra1, ra2 = spec["arm_r"]
    rl0, rl1, rl2 = spec["leg_r"]
    arm_c = spec["arm_color"]
    leg_c = spec["leg_color"]

    # ---- torso: pelvis, waist, ribcage, shoulders, neck -------------------
    torso = a.part("torso", base_color=spec["torso_color"])
    _box_ring(torso.mesh, spec["torso_stations"], spec["torso_color"], 2)
    C.cylinder(torso.mesh, j["neck_r"], j["neck_top"] - j["shoulder_z"] * 0.96,
               seg=8, center=(0.0, 0.0, (j["neck_top"]
                                          + j["shoulder_z"] * 0.96) * 0.5),
               color=spec["skin"])
    for extra in spec.get("torso_detail", ()):
        extra(torso.mesh)
    _seal(torso)

    # ---- head + hair -----------------------------------------------------
    _head(a, spec["head_z"], spec["head_r"], spec["head_squash"],
          spec["skin"], spec["hair_stations"], spec["hair_color"],
          hair_long=spec.get("long_hair", False))

    # ---- optional silhouette-defining extras ------------------------------
    for extra in spec.get("extras", ()):
        extra(a)

    # ---- arms: upper_arm -> lower_arm, both slightly off the body --------
    for side, sx in (("L", 1), ("R", -1)):
        p = a.part("upper_arm_%s" % side, base_color=arm_c)
        _limb(p.mesh, (sx * sh_x, arm_y, sh_z), (sx * el_x, arm_y, el_z),
              ra0, ra1, arm_c, 6)
        _seal(p)

        p = a.part("lower_arm_%s" % side, base_color=arm_c)
        _limb(p.mesh, (sx * el_x, arm_y, el_z), (sx * wr_x, arm_y, wr_z),
              ra1, ra2, arm_c, 6)
        _seal(p)

    # ---- legs: upper_leg -> lower_leg, straight and a little apart -------
    for side, sx in (("L", 1), ("R", -1)):
        p = a.part("upper_leg_%s" % side, base_color=leg_c)
        _limb(p.mesh, (sx * hip_x, leg_y, hip_z), (sx * kn_x, leg_y, kn_z),
              rl0, rl1, leg_c, 6)
        _seal(p)

        p = a.part("lower_leg_%s" % side, base_color=leg_c)
        _limb(p.mesh, (sx * kn_x, leg_y, kn_z), (sx * an_x, leg_y, an_z),
              rl1, rl2, leg_c, 6)
        _seal(p)

    # ---- feet ------------------------------------------------------------
    sw, sd, sh_ = spec["shoe"]
    for side, sx in (("L", 1), ("R", -1)):
        _shoe(a, "shoe_%s" % side, sx * an_x, -sd * 0.5 + leg_y * 0.0,
              sw, sd, sh_, spec["shoe_color"])

    return a


# --------------------------------------------------------------------------
# 1. business suit -- narrowest silhouette, jacket + tie
# --------------------------------------------------------------------------

def _suit_tie(m):
    """Narrow tie plus two collar wings, drawn into the torso shell."""
    C.box(m, (0.055, 0.030, 0.26), center=(0.0, -0.100, 1.330),
          color=SUIT_TIE)
    for sx in (-1, 1):
        C.box(m, (0.052, 0.030, 0.10), center=(sx * 0.042, -0.096, 1.428),
              color=SUIT_SHIRT)


def _ped_suit():
    j = dict(shoulder_z=1.425, shoulder_x=0.185,
             elbow_z=1.105, elbow_x=0.228,
             wrist_z=0.845, wrist_x=0.262,
             hip_z=0.945, hip_x=0.104,
             knee_z=0.485, knee_x=0.107,
             ankle_z=0.095, ankle_x=0.111,
             neck_r=0.050, neck_top=1.560)
    spec = dict(
        id="ped_suit",
        skin=SKIN_SUIT, torso_color=SUIT_CLOTH, arm_color=SUIT_CLOTH,
        leg_color=SUIT_TROUSER, shoe_color=SUIT_SHOE,
        hair_color=SUIT_HAIR,
        joints=j,
        head_z=1.638, head_r=0.098, head_squash=1.14,
        hair_stations=[(1.605, 0.105), (1.672, 0.100), (1.716, 0.074),
                       (1.744, 0.034)],
        torso_stations=[
            (0.845, 0.140, 0.104, 0.050),   # jacket hem
            (0.980, 0.152, 0.110, 0.052),
            (1.100, 0.146, 0.101, 0.046),   # waist, the suit's pinch
            (1.210, 0.164, 0.108, 0.052),
            (1.320, 0.178, 0.110, 0.058),   # chest
            (1.412, 0.182, 0.102, 0.066),   # shoulders
            (1.470, 0.062, 0.058, 0.050),   # neck
        ],
        arm_r=(0.058, 0.048, 0.041),
        leg_r=(0.090, 0.074, 0.056),
        shoe=(0.098, 0.250, 0.072),
        torso_detail=(_suit_tie,),
    )
    return _figure(spec)


# --------------------------------------------------------------------------
# 2. dress -- narrow bodice over an A-line cone skirt, bare arms and legs
# --------------------------------------------------------------------------

def _ped_dress():
    def skirt(a):
        """Truncated cone flaring out from the waist -- the dress read."""
        p = a.part("skirt", base_color=DRESS_CLOTH)
        top_z, bot_z = 1.055, 0.585
        C.cone(p.mesh, 0.305, top_z - bot_z, seg=14,
               center=(0.0, 0.0, (top_z + bot_z) * 0.5),
               color=DRESS_CLOTH, radius_top=0.152)
        _seal(p)

        hem = a.part("skirt_hem", base_color=DRESS_TRIM)
        C.cylinder(hem.mesh, 0.312, 0.055, seg=14,
                   center=(0.0, 0.0, bot_z + 0.030), color=DRESS_TRIM)
        _seal(hem)

    j = dict(shoulder_z=1.400, shoulder_x=0.156,
             elbow_z=1.086, elbow_x=0.196,
             wrist_z=0.826, wrist_x=0.228,
             hip_z=0.930, hip_x=0.096,
             knee_z=0.472, knee_x=0.099,
             ankle_z=0.092, ankle_x=0.102,
             neck_r=0.044, neck_top=1.530)
    spec = dict(
        id="ped_dress",
        skin=SKIN_DRESS, torso_color=DRESS_CLOTH, arm_color=SKIN_DRESS,
        leg_color=SKIN_DRESS, shoe_color=DRESS_SHOE,
        hair_color=DRESS_HAIR,
        joints=j,
        head_z=1.612, head_r=0.094, head_squash=1.13,
        long_hair=True,
        hair_stations=[(1.582, 0.101), (1.648, 0.096), (1.692, 0.070),
                       (1.720, 0.032)],
        torso_stations=[
            (0.940, 0.118, 0.088, 0.044),   # skirt/waist join
            (1.040, 0.122, 0.088, 0.042),   # natural waist
            (1.150, 0.134, 0.092, 0.046),
            (1.270, 0.152, 0.100, 0.054),   # bust
            (1.390, 0.156, 0.094, 0.060),   # shoulders
            (1.452, 0.056, 0.052, 0.046),
        ],
        arm_r=(0.044, 0.037, 0.031),
        leg_r=(0.078, 0.062, 0.046),
        shoe=(0.082, 0.212, 0.056),
        extras=(skirt,),
    )
    return _figure(spec)


# --------------------------------------------------------------------------
# 3. work overalls -- widest torso, denim bib + shoulder straps
# --------------------------------------------------------------------------

def _ped_overalls():
    def bib(a):
        """Denim chest bib and four straps: the overalls read."""
        p = a.part("bib", base_color=WORK_DENIM)
        C.box(p.mesh, (0.285, 0.036, 0.300), center=(0.0, -0.118, 1.310),
              color=WORK_DENIM)
        C.box(p.mesh, (0.052, 0.036, 0.110), center=(0.0, -0.118, 1.472),
              color=WORK_DENIM)
        _seal(p)

        s = a.part("straps", base_color=WORK_STRAP)
        for sx in (-1, 1):
            _bar(s.mesh, (sx * 0.100, -0.120, 1.180), (sx * 0.112, -0.062, 1.455),
                 0.030, 0.017, WORK_STRAP)
            _bar(s.mesh, (sx * 0.100, 0.100, 1.180), (sx * 0.112, 0.062, 1.455),
                 0.030, 0.017, WORK_STRAP)
        # waistband buckle
        C.box(s.mesh, (0.070, 0.040, 0.060), center=(0.0, -0.110, 1.040),
              color=(0.72, 0.66, 0.38))
        _seal(s)

        cap = a.part("cap", base_color=WORK_CAP)
        C.sphere(cap.mesh, 0.108, seg_u=10, seg_v=3, center=(0.0, 0.004, 1.716),
                 color=WORK_CAP, squash=0.52, smooth=False)
        _box_ring(cap.mesh, [(1.640, 0.112, 0.104, 0.055)], WORK_CAP, 2)
        C.box(cap.mesh, (0.190, 0.150, 0.020), center=(0.0, -0.140, 1.660),
              color=WORK_CAP)
        _seal(cap)

        tool = a.part("tool_belt", base_color=WORK_BOOT)
        C.box(tool.mesh, (0.300, 0.220, 0.060), center=(0.0, 0.020, 1.010),
              color=WORK_BOOT)
        _seal(tool)

    # shoulder_x must sit at or OUTSIDE the torso's half-width at shoulder
    # height (the 1.408 station below is hx = 0.206): a pivot buried in the
    # ribcage makes the upper arm sweep through the chest when it rotates.
    j = dict(shoulder_z=1.400, shoulder_x=0.222,
             elbow_z=1.072, elbow_x=0.264,
             wrist_z=0.800, wrist_x=0.296,
             hip_z=0.935, hip_x=0.122,
             knee_z=0.478, knee_x=0.124,
             ankle_z=0.098, ankle_x=0.126,
             neck_r=0.058, neck_top=1.540)
    spec = dict(
        id="ped_overalls",
        skin=SKIN_WORK, torso_color=WORK_SHIRT, arm_color=WORK_SHIRT,
        leg_color=WORK_DENIM, shoe_color=WORK_BOOT,
        hair_color=WORK_HAIR,
        joints=j,
        head_z=1.618, head_r=0.100, head_squash=1.13,
        hair_stations=[(1.600, 0.070), (1.632, 0.078), (1.652, 0.072),
                       (1.664, 0.040)],
        torso_stations=[
            (0.860, 0.168, 0.122, 0.060),   # work shirt hem over the hips
            (0.990, 0.186, 0.128, 0.062),
            (1.110, 0.180, 0.122, 0.056),   # waist under the bib
            (1.240, 0.198, 0.130, 0.062),   # bulkier ribcage
            (1.340, 0.208, 0.132, 0.066),   # chest
            (1.408, 0.206, 0.122, 0.072),   # squared-off work shoulders
            (1.470, 0.068, 0.062, 0.052),
        ],
        arm_r=(0.072, 0.061, 0.053),
        leg_r=(0.108, 0.090, 0.070),
        shoe=(0.118, 0.278, 0.096),
        extras=(bib,),
    )
    return _figure(spec)


# --------------------------------------------------------------------------
# 4. streetwear -- drop shoulders, hoodie mass, baggy cargo legs
# --------------------------------------------------------------------------

def _ped_streetwear():
    def hoodie(a):
        """Hood bunched at the neck plus a kangaroo pocket."""
        h = a.part("hood", base_color=STREET_TOP)
        _box_ring(h.mesh, [
            (1.360, 0.070, 0.062, 0.055),
            (1.430, 0.120, 0.100, 0.070),
            (1.520, 0.132, 0.112, 0.078),
            (1.580, 0.108, 0.092, 0.070),
            (1.612, 0.060, 0.052, 0.048),
        ], STREET_TOP, 2)
        _seal(h)

        pk = a.part("pocket", base_color=C.shade(STREET_TOP, 0.86))
        C.box(pk.mesh, (0.290, 0.040, 0.170), center=(0.0, -0.136, 1.055),
              color=C.shade(STREET_TOP, 0.86))
        _seal(pk)

        cap = a.part("cap", base_color=STREET_CAP)
        _ring(cap.mesh, [
            (1.652, 0.112, 0.004), (1.700, 0.116, 0.004),
            (1.736, 0.104, 0.004), (1.756, 0.072, 0.004),
            (1.764, 0.028, 0.004),
        ], STREET_CAP, 10)
        # backwards brim, so it sits behind the head (+Y) and still reads
        C.box(cap.mesh, (0.176, 0.130, 0.018), center=(0.0, 0.148, 1.668),
              color=C.shade(STREET_CAP, 0.85))
        _seal(cap)

    # heavy drop shoulders; the pivot clears the 1.390 station (hx = 0.238)
    j = dict(shoulder_z=1.372, shoulder_x=0.256,
             elbow_z=1.038, elbow_x=0.302,
             wrist_z=0.762, wrist_x=0.336,
             hip_z=0.915, hip_x=0.132,
             knee_z=0.462, knee_x=0.136,
             ankle_z=0.100, ankle_x=0.140,
             neck_r=0.060, neck_top=1.520)
    spec = dict(
        id="ped_streetwear",
        skin=SKIN_STREET, torso_color=STREET_TOP, arm_color=STREET_TOP,
        leg_color=STREET_TROUSER, shoe_color=STREET_SHOE,
        hair_color=STREET_HAIR,
        joints=j,
        head_z=1.598, head_r=0.098, head_squash=1.14,
        hair_stations=[(1.586, 0.074), (1.618, 0.082), (1.638, 0.076),
                       (1.650, 0.044)],
        torso_stations=[
            (0.760, 0.196, 0.132, 0.066),   # boxy hoodie hem
            (0.920, 0.216, 0.140, 0.070),
            (1.080, 0.206, 0.132, 0.064),
            (1.230, 0.224, 0.142, 0.070),
            (1.330, 0.236, 0.146, 0.076),   # widest point of the silhouette
            (1.390, 0.238, 0.136, 0.084),
            (1.452, 0.072, 0.066, 0.054),
        ],
        arm_r=(0.082, 0.070, 0.058),       # sleeves swallow the arms
        leg_r=(0.124, 0.112, 0.088),       # baggy, barely-tapering cargo leg
        shoe=(0.130, 0.288, 0.112),
        extras=(hoodie,),
    )
    return _figure(spec)


def build_all():
    """Return the ordered list of pedestrian assets."""
    return [_ped_suit(), _ped_dress(), _ped_overalls(), _ped_streetwear()]
