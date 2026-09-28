"""Neon sign kit -- eight generic storefront / facade signs.

Every sign is assembled from one shared kit, so the set reads as the signage
of a single city block:

* ``frame``  -- dark metal backing plate plus a border rail that stands proud
  on the FRONT (-Y) only, leaving the back flush for wall mounting.
* ``panel``  -- the near-black sign face the tubes are bent onto.
* ``tube``   -- the glowing letterforms.  Each letter is a small stroke
  alphabet laid out on a 3 x 5 cell grid and swept as a real 3D tube
  (``core.tube``, parallel-transport frames), so the neon has diameter and
  catches light like bent glass instead of reading as a flat decal.
* ``accent`` -- a border loop plus an underline / arrow / double rule, all in
  a second neon colour.

The stroke alphabet is deliberately square and grid-locked (a 7-segment /
dot-matrix feel) so no font is needed and every stroke lands on one baseline.
Only generic dictionary words are used -- HOTEL, BAR, DINER, PIZZA, TROPIC,
CLUB, ARCADE, MOTEL.  No trademarks, brands or logos; all geometry is
original and built from maths.

ORIGIN: the sign faces -Y so it mounts flat against a facade whose front is
-Y, is centred on X = 0, and the lowest point of the frame sits at z = 0.
Envelope 2.40 m wide x 1.20 m tall x ~0.26-0.28 m deep (the tubes stand
proud of the sign face).
"""

from .. import core as C

# Neon colours -- original palette, no reference imagery.
PINK = (1.00, 0.15, 0.55)
CYAN = (0.10, 0.90, 1.00)
YELLOW = (1.00, 0.85, 0.20)
ORANGE = (1.00, 0.45, 0.10)

# ---- plate / frame kit (metres) ----------------------------------------
PLATE_W = 2.40
PLATE_H = 1.20
PLATE_D = 0.18                    # plate thickness
RAIL = 0.090                      # border rail thickness (square section)
RAIL_PROUD = 0.020                # how far the rail stands proud on -Y
PANEL_T = 0.020                   # sign-face panel thickness
PANEL_PROUD = 0.020               # panel stands proud of the plate front
PANEL_D = PLATE_D + PANEL_PROUD   # panel spans from the plate back forward
PANEL_FRONT = -(PLATE_D * 0.5 + PANEL_PROUD)   # -Y face of the sign face
PANEL_INSET = RAIL                # panel meets the rails' inner faces

# Tubes rest ON the sign face, sunk a hair so no face is exactly coplanar.
TUBE_EMBED = 0.004
TUBE_R_MIN = 0.024
TUBE_R_MAX = 0.034

# Accent (border loop / underline / arrow) tube radius.
ACCENT_R = 0.024
TUBE_SEG = 8

# Dark metal backing / near-black sign face.
BACK = (0.10, 0.11, 0.13)
FACE = (0.045, 0.05, 0.062)

# ---------------------------------------------------------------------------
# Stroke alphabet.  Every glyph is a tuple of polylines on a 3 (x) by 5 (z)
# grid, origin at the baseline-left of the letter's own box.  The layout
# keeps lh = lw * 5/3, so both grid axes share one scale and the strokes stay
# isotropic however the word is fitted.
# ---------------------------------------------------------------------------
GLYPHS = {
    "A": (((0.0, 0.0), (1.5, 5.0)),
          ((1.5, 5.0), (3.0, 0.0)),
          ((0.6, 2.0), (2.4, 2.0))),
    # 7-segment style: the right side is split at the waist so the bowl reads
    # as B rather than as a closed 8.
    "B": (((0.0, 0.0), (0.0, 5.0)),
          ((0.0, 5.0), (2.4, 5.0)),
          ((2.4, 5.0), (2.4, 2.5)),
          ((2.4, 2.5), (2.4, 0.0)),
          ((0.0, 2.5), (2.4, 2.5)),
          ((0.0, 0.0), (2.4, 0.0))),
    "C": (((2.9, 4.1), (2.9, 5.0), (0.0, 5.0),
           (0.0, 0.0), (2.9, 0.0), (2.9, 0.9)),),
    "D": (((0.0, 0.0), (0.0, 5.0)),
          ((0.0, 5.0), (1.7, 5.0), (2.9, 3.4),
           (2.9, 1.6), (1.7, 0.0), (0.0, 0.0))),
    "E": (((0.0, 0.0), (0.0, 5.0)),
          ((0.0, 5.0), (2.8, 5.0)),
          ((0.0, 2.5), (2.35, 2.5)),
          ((0.0, 0.0), (2.8, 0.0))),
    "H": (((0.0, 0.0), (0.0, 5.0)),
          ((3.0, 0.0), (3.0, 5.0)),
          ((0.0, 2.5), (3.0, 2.5))),
    "I": (((1.5, 0.0), (1.5, 5.0)),
          ((0.4, 5.0), (2.6, 5.0)),
          ((0.4, 0.0), (2.6, 0.0))),
    "L": (((0.0, 0.0), (0.0, 5.0)),
          ((0.0, 0.0), (2.8, 0.0))),
    "M": (((0.0, 0.0), (0.0, 5.0), (1.5, 2.7),
           (3.0, 5.0), (3.0, 0.0)),),
    "N": (((0.0, 0.0), (0.0, 5.0), (3.0, 0.0), (3.0, 5.0)),),
    "O": (((0.0, 0.0), (3.0, 0.0), (3.0, 5.0),
           (0.0, 5.0), (0.0, 0.0)),),
    "P": (((0.0, 0.0), (0.0, 5.0)),
          ((0.0, 5.0), (2.4, 5.0)),
          ((2.4, 5.0), (2.4, 2.5)),
          ((0.0, 2.5), (2.4, 2.5))),
    "R": (((0.0, 0.0), (0.0, 5.0)),
          ((0.0, 5.0), (2.4, 5.0)),
          ((2.4, 5.0), (2.4, 2.5)),
          ((0.0, 2.5), (2.4, 2.5)),
          ((2.4, 2.5), (3.0, 0.0))),
    "T": (((1.5, 0.0), (1.5, 5.0)),
          ((0.0, 5.0), (3.0, 5.0))),
    "U": (((0.0, 5.0), (0.0, 0.0), (3.0, 0.0), (3.0, 5.0)),),
    "Z": (((0.0, 5.0), (3.0, 5.0), (0.0, 0.0), (3.0, 0.0)),),
}

# Letter gap as a fraction of letter width.
GAP_RATIO = 0.26

# The accent border loop's centre line.  Text is fitted INSIDE this ring, so
# a word can never collide with the loop however tall it is.
BORDER_X = PLATE_W * 0.5 - 0.115
BORDER_Z0 = 0.115
BORDER_Z1 = PLATE_H - 0.115

# Minimum air between the lit text and the border loop.
EDGE_CLEAR = 0.045
# Vertical band reserved under the text for the accent underline / arrow.
ACCENT_BAND = 0.150


def _glyph_polys(letter, x0, z0, u):
    """Place one glyph's polylines into world (x, z) at cell size ``u``."""
    return [[(x0 + px * u, z0 + pz * u) for (px, pz) in poly]
            for poly in GLYPHS[letter]]


def _tube(part, pts2d, y, radius, color):
    """Sweep one 2D (x, z) polyline into a real 3D tube on the sign face."""
    return C.tube(part.mesh, [(px, y, pz) for (px, pz) in pts2d], radius,
                  seg=TUBE_SEG, color=color, smooth=False)


def _tube_y(radius):
    """Y of a tube of ``radius`` resting on the sign face (half sunk in)."""
    return PANEL_FRONT + TUBE_EMBED - radius


def _fit(word, lw_cap):
    """Lay the word out inside the border ring.

    Returns ``(lw, gap, total, z_base, r, z_lo, z_hi)``, where ``z_lo`` /
    ``z_hi`` are the band the lit tube envelope occupies.  That band is
    clamped to the free area BEFORE the accent is placed, which is what keeps
    an underline or arrowhead from ever striking through a tall word.
    """
    n = len(word)
    clear_x = BORDER_X - ACCENT_R - EDGE_CLEAR
    clear_z1 = BORDER_Z1 - ACCENT_R - EDGE_CLEAR
    clear_z0 = BORDER_Z0 + ACCENT_R + ACCENT_BAND
    avail_w = clear_x * 2.0
    avail_h = clear_z1 - clear_z0

    # the glyph box is 3 x 5 cells, so the height limit is lw = lh * 3/5
    lw = min(avail_w / (n + (n - 1) * GAP_RATIO), avail_h * 0.6, lw_cap)
    lh = lw * 5.0 / 3.0
    gap = lw * GAP_RATIO
    total = n * lw + (n - 1) * gap

    r = max(TUBE_R_MIN, min(TUBE_R_MAX, lw * 0.095))
    # Centre the GLYPHS in the free band, then report the envelope they
    # actually need (axis +/- tube radius), clamped to the band.
    z_axis = clear_z0 + (avail_h - lh) * 0.5
    z_lo = max(clear_z0, z_axis - r)
    z_hi = min(clear_z1, z_axis + lh + r)
    # if the envelope was clamped, slide the word so it stays centred
    z_base = z_axis + ((z_lo + z_hi) * 0.5 - (z_axis + lh * 0.5))
    return lw, gap, total, z_base, r, z_lo, z_hi


def _sign(asset_id, word, neon, accent, accent_kind="underline",
          lw_cap=0.30, back=BACK):
    """Assemble one neon sign."""
    a = C.Asset(asset_id, "sign")

    # ---- frame: backing plate + border rail ------------------------------
    frame = a.part("frame", base_color=back, roughness=0.45, metallic=0.65)
    rail_c = C.shade(back, 2.1)
    C.box(frame.mesh, (PLATE_W, PLATE_D, PLATE_H),
          center=(0.0, 0.0, PLATE_H * 0.5), color=back,
          colors={"+y": C.shade(back, 0.7), "-y": C.shade(back, 0.85)})
    # rails run from the plate's back (flush, so it mounts flat on a wall) to
    # RAIL_PROUD in front of it
    rail_depth = PLATE_D * 0.5 + RAIL_PROUD
    rail_y = PLATE_D * 0.5 - rail_depth * 0.5
    for sgn in (-1, 1):
        z = PLATE_H - RAIL * 0.5 if sgn > 0 else RAIL * 0.5
        C.box(frame.mesh, (PLATE_W, rail_depth, RAIL),
              center=(0.0, rail_y, z), color=rail_c)
        C.box(frame.mesh, (RAIL, rail_depth, PLATE_H - RAIL * 2.0),
              center=(sgn * (PLATE_W * 0.5 - RAIL * 0.5), rail_y,
                      PLATE_H * 0.5), color=rail_c)

    # ---- sign face: near-black panel standing proud of the plate ---------
    panel = a.part("panel", base_color=FACE, roughness=0.85)
    C.box(panel.mesh,
          (PLATE_W - PANEL_INSET * 2.0, PANEL_D, PLATE_H - PANEL_INSET * 2.0),
          center=(0.0, PANEL_FRONT + PANEL_D * 0.5, PLATE_H * 0.5),
          color=FACE)

    # ---- tube: the lit word ---------------------------------------------
    lw, gap, total, z_base, r, z_lo, _z_hi = _fit(word, lw_cap)
    u = lw / 3.0
    tube = a.part("tube", base_color=C.shade(neon, 0.72), emissive=neon,
                  roughness=0.22)
    tube_c = C.shade(neon, 0.9)
    x = -total * 0.5
    for ch in word:
        for poly in _glyph_polys(ch, x, z_base, u):
            _tube(tube, poly, _tube_y(r), r, tube_c)
        x += lw + gap

    # ---- accent: border loop + underline / arrow / double rule ---------
    acc = a.part("accent", base_color=C.shade(accent, 0.72), emissive=accent,
                 roughness=0.22)
    ar = ACCENT_R
    ay = _tube_y(ar)
    acc_c = C.shade(accent, 0.9)

    # Closed border loop: tube() drops a repeated end point, so the two end
    # caps meet at the corner and read as a neon weld joint.
    _tube(acc, [(-BORDER_X, BORDER_Z0), (BORDER_X, BORDER_Z0),
               (BORDER_X, BORDER_Z1), (-BORDER_X, BORDER_Z1),
               (-BORDER_X, BORDER_Z0)], ay, ar, acc_c)

    # The underline / arrow lives in the gap between the ring's inner face and
    # the text's envelope.  Sizing it from THAT gap -- not a fixed height --
    # is what stops a tall word like BAR from being struck through.
    gap_lo = BORDER_Z0 + ar
    z_acc = (gap_lo + z_lo) * 0.5
    half = (z_lo - gap_lo) * 0.5
    hw = min(total * 0.5 + 0.10, BORDER_X - ar - 0.06)

    if accent_kind == "underline":
        _tube(acc, [(-hw, z_acc), (hw, z_acc)], ay, ar, acc_c)
    elif accent_kind == "double":
        # Two rules centred in the band, so both stay clear of the ring's
        # inner face below and the text above.  The offset is derived from
        # the band itself (not a fixed constant) so the pair keeps a visible
        # air gap on every word, however tall that word is.
        dz = max(0.0, min(0.60 * half, half - ar))
        _tube(acc, [(-hw, z_acc + dz), (hw, z_acc + dz)], ay, ar, acc_c)
        _tube(acc, [(-hw * 0.80, z_acc - dz), (hw * 0.80, z_acc - dz)],
              ay, ar, acc_c)
    elif accent_kind == "arrow":
        # Shaft and head sized to the same gap, so the head can never touch
        # the letters.  The shaft stops at the head's base and the head's
        # tip is inset from the ring, keeping the pair inside the border.
        a_half = max(0.0, min(half - ar, hw * 0.16))
        a_len = min(2.2 * a_half, hw * 0.45)
        tip = hw - a_len * 0.30
        base = tip - a_len
        _tube(acc, [(-hw, z_acc), (base, z_acc)], ay, ar, acc_c)
        _tube(acc, [(base, z_acc + a_half), (tip, z_acc),
                   (base, z_acc - a_half)], ay, ar, acc_c)
    else:
        raise ValueError("unknown accent kind %r" % (accent_kind,))

    return a


def build_all():
    """Return the eight neon sign assets, in street order.

    ``lw_cap`` is the per-sign letter-width ceiling.  Short words are capped
    generously so they do not read as undersized beside long ones; long words
    simply fall back to whatever the border ring allows.
    """
    return [
        _sign("sign_hotel", "HOTEL", PINK, CYAN, "underline", lw_cap=0.300,
              back=(0.11, 0.10, 0.13)),
        _sign("sign_bar", "BAR", YELLOW, PINK, "arrow", lw_cap=0.420,
              back=(0.13, 0.10, 0.09)),
        _sign("sign_diner", "DINER", CYAN, ORANGE, "double", lw_cap=0.300,
              back=(0.09, 0.12, 0.14)),
        _sign("sign_pizza", "PIZZA", ORANGE, CYAN, "arrow", lw_cap=0.300,
              back=(0.14, 0.09, 0.08)),
        _sign("sign_tropic", "TROPIC", CYAN, YELLOW, "underline", lw_cap=0.250,
              back=(0.08, 0.13, 0.14)),
        _sign("sign_club", "CLUB", PINK, CYAN, "arrow", lw_cap=0.370,
              back=(0.13, 0.08, 0.13)),
        _sign("sign_arcade", "ARCADE", YELLOW, PINK, "underline", lw_cap=0.250,
              back=(0.10, 0.11, 0.15)),
        _sign("sign_motel", "MOTEL", ORANGE, CYAN, "arrow", lw_cap=0.300,
              back=(0.12, 0.11, 0.09)),
    ]
