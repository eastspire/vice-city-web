"""Category A: Art Deco / pastel Miami buildings.

All buildings share one facade kit so the set reads as a single city block:
vertical pilasters, horizontal string courses, stepped setbacks, balcony
parapets, a roof parapet, rooftop water tank + AC units, and a blank neon sign
backing board.  Colour and proportions vary per variant.

Origin is the centre of the ground footprint at z = 0 (so the JSON export puts
the building base exactly on y = 0).  ``-Y`` is the street-facing front.
"""

import math

from .. import core as C

# Miami pastels + Art Deco cream/teal. Original palette, no reference images.
PALETTE = {
    "pink":      (0.96, 0.55, 0.66),
    "mint":      (0.62, 0.90, 0.79),
    "teal":      (0.20, 0.66, 0.64),
    "coral":     (0.98, 0.51, 0.40),
    "cream":     (0.97, 0.91, 0.76),
    "apricot":   (0.99, 0.75, 0.53),
    "lilac":     (0.79, 0.71, 0.93),
    "aqua":      (0.45, 0.82, 0.88),
    "sand":      (0.93, 0.84, 0.66),
    "white":     (0.96, 0.96, 0.94),
}

CONCRETE = (0.72, 0.70, 0.66)
CONCRETE_DK = (0.55, 0.53, 0.50)
ROOF = (0.42, 0.44, 0.47)
GLASS = (0.16, 0.32, 0.42)
GLASS_LIT = (0.98, 0.86, 0.52)
TRIM = (0.98, 0.96, 0.90)
AWNING = (0.93, 0.25, 0.36)
METAL = (0.62, 0.64, 0.66)
METAL_DK = (0.38, 0.40, 0.42)


def _shrub(color, k=1.0):
    return tuple(min(1.0, c * k) for c in color)


def building(asset_id, key, width, depth, floors, floor_h=3.4,
             setbacks=(), pilasters=4, balcony_rows=(), cornice=True,
             ground_accent=True, base_color=None, trim=None):
    """Assemble one Art Deco building.

    ``setbacks`` is a sequence of ``(height_above_ground, scale)`` -- above the
    given height the footprint shrinks to ``scale`` of its original half-extent,
    which produces the classic stepped tower silhouette.
    """
    a = C.Asset(asset_id, "building")
    wall = base_color or PALETTE[key]
    trim_c = trim or TRIM

    def half_at(z):
        """Footprint half-extents at height z, after any setbacks."""
        s = 1.0
        for h, sc in setbacks:
            if z >= h:
                s = min(s, sc)
        return width * 0.5 * s, depth * 0.5 * s

    total_h = floors * floor_h

    # ---- main shaft ------------------------------------------------------
    shaft = a.part("shaft", base_color=wall)
    lo = 0.0
    hi = total_h
    C.box(shaft.mesh, (width, depth, total_h), center=(0, 0, total_h * 0.5),
          color=wall, colors={"-z": CONCRETE_DK})
    _ = (lo, hi, half_at)

    # vertical pilasters -- the signature Deco fluting.
    # All pilasters share ONE part: they are the same colour and material, and
    # one part keeps the part count sane on a 12-storey facade.
    pil = C.Part("pilasters", base_color=_shrub(wall, 1.06))
    for i in range(pilasters):
        t = (i + 0.5) / pilasters - 0.5          # -0.5 .. +0.5
        x = t * width * 0.94
        w = width * 0.030
        d = 0.14
        C.box(pil.mesh, (w, depth + d, total_h * 0.94),
              center=(x, 0, total_h * 0.47), color=_shrub(wall, 1.06))
    a.add(pil)

    # horizontal string courses every floor (one shared part)
    courses = C.Part("courses", base_color=trim_c)
    for f in range(floors):
        z = f * floor_h
        sc = 1.0
        for h, s in setbacks:
            if z >= h:
                sc = min(sc, s)
        w2, d2 = width * sc, depth * sc
        C.box(courses.mesh, (w2 + 0.16, d2 + 0.16, 0.16), center=(0, 0, z),
              color=trim_c)

    # ---- setback shoulders: the exposed roof of each lower step ----------
    setback_slab = C.Part("setback_slabs", base_color=ROOF)
    terrace_rail = C.Part("terrace_rails", base_color=CONCRETE)
    for h, s in sorted(setbacks):
        prev = 1.0
        for h2, s2 in sorted(setbacks):
            if h2 < h:
                prev = s2
        if prev <= s:
            continue
        w_out, d_out = width * prev, depth * prev
        w_in, d_in = width * s, depth * s
        slab_w = (w_out + w_in) * 0.5
        slab_d = (d_out + d_in) * 0.5
        C.box(setback_slab.mesh, (slab_w, slab_d, 0.22), center=(0, 0, h - 0.11),
              color=ROOF)
        # a small parapet wall around the exposed terrace
        pw = (w_out - w_in) * 0.5
        for sgn in (-1, 1):
            C.box(terrace_rail.mesh, (w_out, 0.16, 0.52),
                  center=(0, sgn * (d_out * 0.5 - 0.08), h + 0.26),
                  color=CONCRETE)
            C.box(terrace_rail.mesh, (0.16, d_out - 0.32, 0.52),
                  center=(sgn * (w_out * 0.5 - 0.08), 0, h + 0.26),
                  color=CONCRETE)
        del pw

    if setback_slab.mesh.faces:
        a.add(setback_slab)
        a.add(terrace_rail)

    # ---- windows ---------------------------------------------------------
    win_rows = []
    for f in range(floors):
        z = f * floor_h + floor_h * 0.5
        sc = 1.0
        for h, s in setbacks:
            if z >= h:
                sc = min(sc, s)
        win_rows.append((f, z, sc))

    windows = C.Part("windows", base_color=GLASS, roughness=0.25)
    windows_side = C.Part("windows_side", base_color=GLASS, roughness=0.25)
    for f, z, sc in win_rows:
        w2, d2 = width * sc, depth * sc
        cols = max(2, int(round(w2 / 2.4)))
        for cidx in range(cols):
            t = (cidx + 0.5) / cols - 0.5
            x = t * w2 * 0.86
            # south (front, -Y) and north (+Y) facades.
            # All windows of an asset share ONE part: per-instance parts would
            # all share the name "window" and collide when a renderer binds
            # materials by name.  Lit and unlit windows differ only by face
            # colour, which face_colors already expresses.
            for sgn in (-1, 1):
                lit = ((f + cidx * 3 + (1 if sgn < 0 else 5)) % 7) < 2
                col = GLASS_LIT if lit else GLASS
                ww, wh = 1.05, 1.55
                y = sgn * (d2 * 0.5 + 0.03)
                C.box(windows.mesh, (ww, 0.10, wh), center=(x, y, z),
                      color=col,
                      colors={"+y" if sgn > 0 else "-y": col})
            # east/west ends
            for sgn in (-1, 1):
                C.box(windows_side.mesh, (0.10, 0.9, 1.45),
                      center=(sgn * (w2 * 0.5 + 0.03), t * d2 * 0.7, z),
                      color=GLASS)

    a.add(windows)
    a.add(windows_side)

    # ---- balcony parapets ------------------------------------------------
    balconies = C.Part("balconies", base_color=trim_c)
    for f in balcony_rows:
        z0 = f * floor_h + floor_h * 0.18
        sc = 1.0
        for h, s in setbacks:
            if z0 >= h:
                sc = min(sc, s)
        w2, d2 = width * sc, depth * sc
        n_bal = max(2, int(round(w2 / 3.2)))
        for i in range(n_bal):
            t = (i + 0.5) / n_bal - 0.5
            x = t * w2 * 0.84
            y = -(d2 * 0.5 + 0.42)
            # one shared "balcony" part for the whole building
            C.box(balconies.mesh, (2.1, 0.84, 0.10), center=(x, y, z0),
                  color=trim_c)
            C.box(balconies.mesh, (2.1, 0.10, 0.78),
                  center=(x, y - 0.37, z0 + 0.44), color=trim_c)
            C.box(balconies.mesh, (0.10, 0.84, 0.78),
                  center=(x - 1.0, y, z0 + 0.44), color=trim_c)
            C.box(balconies.mesh, (0.10, 0.84, 0.78),
                  center=(x + 1.0, y, z0 + 0.44), color=trim_c)

    if balcony_rows:
        a.add(balconies)

    # ---- ground floor: awning + shopfront band ---------------------------
    if ground_accent:
        # height 0.85 centred at height/2 -> bottom exactly on z = 0.  A centre
        # of 0.42 sinks the band's bottom 5 mm below the ground plane, which the
        # verifier catches on every building.
        band_h = 0.85
        band = C.Part("ground_band", base_color=_shrub(wall, 0.72))
        C.box(band.mesh, (width + 0.10, depth + 0.10, band_h),
              center=(0, 0, band_h * 0.5), color=_shrub(wall, 0.72))
        a.add(band)

        aw = C.Part("awning", base_color=AWNING)
        C.box(aw.mesh, (width * 0.46, 1.5, 0.16),
              center=(0, -(depth * 0.5 + 0.72), 2.35), color=AWNING)
        C.box(aw.mesh, (width * 0.46, 0.10, 0.42),
              center=(0, -(depth * 0.5 + 1.44), 2.13), color=AWNING)
        a.add(aw)

    # ---- roof: parapet, water tank, AC units, sign backing ---------------
    top_s = 1.0
    for _h, s in setbacks:
        top_s = min(top_s, s)
    rw, rd = width * top_s, depth * top_s

    par = C.Part("parapet", base_color=trim_c)
    ph = 0.75
    for sgn in (-1, 1):
        C.box(par.mesh, (rw + 0.24, 0.18, ph),
              center=(0, sgn * (rd * 0.5 + 0.03), total_h + ph * 0.5),
              color=trim_c)
        C.box(par.mesh, (0.18, rd - 0.30, ph),
              center=(sgn * (rw * 0.5 + 0.03), 0, total_h + ph * 0.5),
              color=trim_c)
    a.add(par)

    if cornice:
        cor = C.Part("cornice", base_color=trim_c)
        C.box(cor.mesh, (rw + 0.36, rd + 0.36, 0.22),
              center=(0, 0, total_h + 0.11), color=trim_c)
        a.add(cor)

    # rooftop water tank on a steel frame.  The barrel is smooth-shaded, so it
    # lives in its OWN part -- a Part carries a single `flat` flag, and mixing a
    # smooth loft with flat primitives in one part makes the flat ones inherit
    # averaged normals they must not have.
    tank_r = min(rw, rd) * 0.22
    tx = rw * 0.24
    ty = rd * 0.16
    tz = total_h + 0.22
    legs = C.Part("tank_legs", base_color=METAL_DK, metallic=0.5)
    for sx in (-1, 1):
        for sy in (-1, 1):
            C.box(legs.mesh, (0.10, 0.10, 0.95),
                  center=(tx + sx * tank_r * 0.62, ty + sy * tank_r * 0.62,
                          tz + 0.47), color=METAL_DK)
    a.add(legs)

    tank = C.Part("water_tank", base_color=(0.62, 0.52, 0.40), roughness=0.85,
                  flat=False)
    C.cylinder(tank.mesh, tank_r, 1.5, 10, center=(tx, ty, tz + 0.95 + 0.75),
               color=(0.62, 0.52, 0.40), smooth=True)
    a.add(tank)

    tank_lid = C.Part("water_tank_lid", base_color=(0.48, 0.40, 0.32),
                      roughness=0.85)
    C.cone(tank_lid.mesh, tank_r * 1.04, 0.34, 10,
           center=(tx, ty, tz + 0.95 + 1.5 + 0.17), color=(0.48, 0.40, 0.32))
    a.add(tank_lid)

    # rooftop AC condensers
    acs = C.Part("ac_units", base_color=METAL)
    for i in range(3):
        ax = -rw * 0.28 + i * rw * 0.26
        ay = -rd * 0.22
        C.box(acs.mesh, (1.15, 0.95, 0.80), center=(ax, ay, tz + 0.40),
              color=METAL)
        C.cylinder(acs.mesh, 0.30, 0.10, 8,
                   center=(ax, ay, tz + 0.85), color=METAL_DK)
    a.add(acs)

    # blank neon sign backing board on the front facade (signs attach to this)
    sb = C.Part("sign_board", base_color=METAL_DK, roughness=0.8)
    sb_w = min(rw * 0.52, 4.2)
    sb_y = -(rd * 0.5 + 0.12)
    sb_z = total_h * 0.72
    C.box(sb.mesh, (sb_w, 0.22, 2.1), center=(0, sb_y, sb_z), color=METAL_DK)
    for sgn in (-1, 1):
        C.box(sb.mesh, (0.10, 0.34, 0.10),
              center=(sgn * sb_w * 0.42, sb_y + 0.02, sb_z - 0.95),
              color=METAL_DK)
    a.add(sb)
    return a


# --------------------------------------------------------------------------
# the 12 building variants
# --------------------------------------------------------------------------

def build_all():
    """Return the ordered list of building assets."""
    out = []

    # 1. classic pastel Deco hotel, 5 storeys, mild setback
    out.append(building(
        "bldg_deco_pink", "pink", 12.0, 10.0, 5,
        setbacks=((11.0, 0.78),), pilasters=5, balcony_rows=(1, 2, 3, 4)))

    # 2. tall teal tower with a pronounced ziggurat top
    out.append(building(
        "bldg_deco_teal", "teal", 11.0, 9.0, 9,
        setbacks=((14.0, 0.82), (23.0, 0.62), (27.0, 0.44)), pilasters=4,
        balcony_rows=(2, 4, 6)))

    # 3. wide cream apartment block, 4 storeys, no setback
    out.append(building(
        "bldg_cream_block", "cream", 18.0, 11.0, 4,
        pilasters=7, balcony_rows=(1, 2, 3)))

    # 4. mint corner shop, 3 storeys
    out.append(building(
        "bldg_mint_shop", "mint", 10.0, 8.0, 3,
        pilasters=4, balcony_rows=(2,)))

    # 5. coral art-deco hall with a single setback and awning
    out.append(building(
        "bldg_coral_hall", "coral", 14.0, 12.0, 4,
        setbacks=((9.5, 0.80),), pilasters=5, balcony_rows=(1, 3)))

    # 6. apricot low-rise motel block, 2 storeys
    out.append(building(
        "bldg_apricot_motel", "apricot", 16.0, 9.0, 2,
        pilasters=6, balcony_rows=(1,)))

    # 7. lilac slim high-rise, 12 storeys, double setback
    out.append(building(
        "bldg_lilac_tower", "lilac", 9.5, 8.5, 12,
        setbacks=((20.0, 0.85), (32.0, 0.70)), pilasters=4,
        balcony_rows=(3, 5, 7, 9, 11)))

    # 8. aqua aquarium-style block with deep balconies
    out.append(building(
        "bldg_aqua_arcade", "aqua", 15.0, 12.0, 5,
        pilasters=6, balcony_rows=(1, 2, 3, 4)))

    # 9. sand-coloured 6-storey mid-rise
    out.append(building(
        "bldg_sand_midrise", "sand", 13.0, 10.0, 6,
        setbacks=((13.0, 0.88),), pilasters=5, balcony_rows=(2, 3, 4, 5)))

    # 10. white deco landmark with a ziggurat crown
    out.append(building(
        "bldg_white_landmark", "white", 12.5, 12.5, 10,
        setbacks=((16.0, 0.85), (26.0, 0.66), (31.0, 0.46)), pilasters=4,
        balcony_rows=(2, 4, 6, 8)))

    # 11. pink twin-setback 7 storey
    out.append(building(
        "bldg_pink_terrace", "pink", 14.0, 10.0, 7,
        setbacks=((10.0, 0.86), (19.0, 0.68)), pilasters=5,
        balcony_rows=(1, 3, 5)))

    # 12. teal industrial-loft, 3 storeys, wide and shallow
    out.append(building(
        "bldg_teal_loft", "teal", 20.0, 9.0, 3,
        pilasters=8, balcony_rows=(2,)))

    return out
