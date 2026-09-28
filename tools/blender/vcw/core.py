"""Geometry kernel for the NEON BAY procedural asset pipeline.

Pure Python -- deliberately does NOT import bpy, so every builder stays
importable and unit-testable outside Blender.

AUTHORING CONVENTION (Blender-style):
    +X = right,  -Y = forward (Blender front view),  +Z = up,  1 unit = 1 m.
    Every asset is authored centred on X=0 with its origin resting on Z=0
    (buildings / vehicles / props) or on the soles (pedestrians).

EXPORT CONVENTION (Y-up JSON, see assets/SCHEMA.md):
    json = (x, z, -y)   -- i.e. +Z forward, +Y up, +X unchanged.
    This matrix has determinant +1, so handedness is preserved.
"""

import math

EPS = 1e-12


# --------------------------------------------------------------------------
# vector helpers
# --------------------------------------------------------------------------

def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def length(v):
    return math.sqrt(dot(v, v))


def normalize(v):
    n = length(v)
    if n < EPS:
        return (0.0, 0.0, 1.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t)


def mix_color(c0, c1, t):
    return lerp(c0, c1, t)


def shade(c, k):
    """Multiply a colour by k, clamped to 0..1."""
    return (min(1.0, c[0] * k), min(1.0, c[1] * k), min(1.0, c[2] * k))


# --------------------------------------------------------------------------
# Mesh
# --------------------------------------------------------------------------

class Mesh(object):
    """Accumulates triangles for one Part.

    Two kinds of geometry end up here:

    * ``tri`` / ``quad`` append three *fresh* vertices per triangle.  That is
      exactly flat shading, and it guarantees every stored normal equals its own
      triangle's geometric normal.
    * ``loft`` / ``tube`` can share vertices between neighbouring quads and
      registers them in a "smooth group"; those vertices get area-averaged
      normals and the owning Part must set ``flat = False``.
    """

    __slots__ = ("pos", "faces", "fcol", "_smooth", "skipped")

    def __init__(self):
        self.pos = []
        self.faces = []
        self.fcol = []
        self._smooth = []   # list of vertex-index groups to average over
        self.skipped = 0    # degenerate triangles dropped (must stay 0)

    # -- low level ---------------------------------------------------------

    def _push_face(self, idx, color, smooth_group=None):
        a, b, c = (self.pos[i] for i in idx)
        n = cross(sub(b, a), sub(c, a))
        if length(n) < EPS:
            self.skipped += 1
            return False
        self.faces.append(tuple(idx))
        self.fcol.append(tuple(color))
        if smooth_group is not None:
            smooth_group.append(idx)
        return True

    def tri(self, a, b, c, color):
        """Flat triangle from three world-space points, CCW seen from outside."""
        i = len(self.pos)
        self.pos.append(tuple(a))
        self.pos.append(tuple(b))
        self.pos.append(tuple(c))
        return self._push_face((i, i + 1, i + 2), color)

    def quad(self, a, b, c, d, color):
        """Flat quad a-b-c-d (CCW seen from outside) as two triangles."""
        self.tri(a, b, c, color)
        self.tri(a, c, d, color)

    def tri_idx(self, ia, ib, ic, color, smooth_group=None):
        return self._push_face((ia, ib, ic), color, smooth_group)

    def quad_idx(self, ia, ib, ic, idd, color, smooth_group=None):
        self._push_face((ia, ib, ic), color, smooth_group)
        self._push_face((ia, ic, idd), color, smooth_group)

    @property
    def nverts(self):
        return len(self.pos)

    @property
    def ntris(self):
        return len(self.faces)

    # -- normals -----------------------------------------------------------

    def face_normals(self):
        out = []
        for (a, b, c) in self.faces:
            va, vb, vc = self.pos[a], self.pos[b], self.pos[c]
            out.append(normalize(cross(sub(vb, va), sub(vc, va))))
        return out

    def vertex_normals(self):
        """(face_normals, vertex_normals).

        A vertex inside a registered smooth group averages only the faces of
        that group.  A vertex that belongs to a SINGLE face -- and, critically,
        any vertex in a mesh with no smooth groups at all -- receives that
        face's exact geometric normal.

        The second rule is the important one for flat parts.  Primitives like
        :func:`cone` and :func:`loft` share vertices between neighbouring
        triangles on purpose (that is what makes them watertight), so a naive
        "average all incident faces" rule silently smooths a cone's base rim and
        its apex even though the part is flagged ``flat=True``.
        """
        fn = self.face_normals()
        inc = [[] for _ in self.pos]
        for fi, f in enumerate(self.faces):
            for i in f:
                inc[i].append(fi)

        # With no smooth groups at all, every vertex is flat: hand back the
        # face normal and skip averaging entirely.
        if not self._smooth:
            out = [(0.0, 0.0, 1.0)] * len(self.pos)
            for fi, f in enumerate(self.faces):
                for i in f:
                    out[i] = fn[fi]
            return fn, out

        group_of = {}
        for grp in self._smooth:
            for i in grp:
                group_of.setdefault(i, grp)

        out = []
        for vi in range(len(self.pos)):
            faces = inc[vi]
            if not faces:
                out.append((0.0, 0.0, 1.0))
                continue
            if len(faces) == 1:
                out.append(fn[faces[0]])
                continue
            grp = group_of.get(vi)
            if grp is not None:
                sibs = set(grp)
                allowed = [f for f in faces
                           if any(v in sibs for v in self.faces[f])]
                faces = allowed or faces
            acc = (0.0, 0.0, 0.0)
            for f in faces:
                acc = add(acc, fn[f])
            out.append(normalize(acc) if length(acc) > EPS else fn[faces[0]])
        return fn, out

    def bounds(self):
        if not self.pos:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        lo = [float("inf")] * 3
        hi = [float("-inf")] * 3
        for p in self.pos:
            for k in range(3):
                if p[k] < lo[k]:
                    lo[k] = p[k]
                if p[k] > hi[k]:
                    hi[k] = p[k]
        return tuple(lo), tuple(hi)

    def flatten(self):
        """Return a copy where every triangle owns its own three vertices.

        Flat shading is not a per-face flag on shared geometry -- it is a
        statement that each triangle gets its own vertices and its own normal.
        A vertex shared by four triangles (a cone's base rim, a cylinder's end
        cap edge) can only carry one averaged normal, so it can only ever look
        smooth.  Splitting is the only representation that is honest.

        Watertightness is index-based and is lost here by definition; the
        geometry is unchanged, so any winding/volume check still holds and can
        be run against the original mesh before splitting.
        """
        out = Mesh()
        for fi, f in enumerate(self.faces):
            base = len(out.pos)
            for k in f:
                out.pos.append(self.pos[k])
            out.faces.append((base, base + 1, base + 2))
            out.fcol.append(self.fcol[fi])
        return out

    def copy(self):
        out = Mesh()
        out.pos = list(self.pos)
        out.faces = list(self.faces)
        out.fcol = list(self.fcol)
        return out


# --------------------------------------------------------------------------
# Part / Asset
# --------------------------------------------------------------------------

class Part(object):
    """One named chunk of an asset with its own material settings.

    ``outward`` declares how "this face points out of the asset" is decided for
    OPEN surfaces, where signed volume says nothing because there is no
    enclosed region:

    * ``("point", (x, y, z))`` -- every face normal must point away from this
      interior reference point (a concave shell, a lamp shade, ...).
    * ``("dir", (x, y, z))``   -- every face normal must point this way (a sign
      face, a one-sided decal, a ground marking).

    Leave it ``None`` for closed shells: the exporter then verifies outwardness
    from the sign of the signed volume, which needs no declared reference.
    Flat parts still duplicate a vertex per triangle, so that check applies to
    every one of their triangles individually.
    """

    def __init__(self, name, base_color=(0.8, 0.8, 0.8), emissive=(0.0, 0.0, 0.0),
                 flat=True, roughness=0.7, metallic=0.0, outward=None):
        self.name = name
        self.base_color = tuple(base_color)
        self.emissive = tuple(emissive)
        self.flat = bool(flat)
        self.roughness = float(roughness)
        self.metallic = float(metallic)
        self.outward = outward
        self.mesh = Mesh()

    def set_color(self, color):
        self.base_color = tuple(color)
        self.mesh.fcol = [tuple(color)] * self.mesh.ntris
        return self

    def ntris(self):
        return self.mesh.ntris


def signed_volume(m):
    """Signed volume via the divergence theorem: > 0 means outward winding."""
    v = 0.0
    for (a, b, c) in m.faces:
        v += dot(m.pos[a], cross(m.pos[b], m.pos[c])) / 6.0
    return v


def is_closed(m):
    """True when every edge is shared by exactly two faces (index-based).

    Only meaningful for shared-vertex shells; flat parts duplicate a vertex per
    triangle by design, so every edge there is used exactly once and this
    returns False.  Use :func:`closed_components` for a check that also covers
    flat parts.
    """
    from collections import Counter
    if not m.pos:
        return False
    cnt = Counter()
    for f in m.faces:
        for i in range(3):
            a, b = f[i], f[(i + 1) % 3]
            if a == b:
                return False
            cnt[(a, b) if a < b else (b, a)] += 1
    return all(v == 2 for v in cnt.values())


def _weld(pos, tol=1e-6):
    """Map vertex indices to welded ids by rounded position."""
    key_of = {}
    remap = [0] * len(pos)
    for i, p in enumerate(pos):
        k = (round(p[0] / tol), round(p[1] / tol), round(p[2] / tol))
        if k not in key_of:
            key_of[k] = len(key_of)
        remap[i] = key_of[k]
    return remap


def closed_components(m, tol=1e-6):
    """Split a mesh into position-welded closed shells.

    Flat parts store three vertices per triangle, so index-based manifold tests
    cannot see that the 12 triangles of a box form a closed shell.  Welding by
    POSITION first recovers the true connectivity: each component that is
    closed (every welded edge used by exactly two faces) and has a positive
    signed volume is correctly wound and outward-facing.

    Returns ``(n_closed, n_inverted, n_open)``.
    """
    from collections import Counter, defaultdict
    if not m.faces:
        return (0, 0, 0)
    remap = _weld(m.pos, tol)

    # split faces into components linked by welded edges
    parent = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    faces_of = defaultdict(list)
    parent = {}
    for fi, f in enumerate(m.faces):
        for i in f:
            w = remap[i]
            parent.setdefault(w, w)
    for fi, f in enumerate(m.faces):
        w = [remap[i] for i in f]
        for k in range(3):
            union(w[k], w[(k + 1) % 3])

    comp_faces = defaultdict(list)
    for fi, f in enumerate(m.faces):
        w = [remap[i] for i in f]
        comp_faces[find(w[0])].append(fi)

    n_closed = n_inv = n_open = 0
    for cid, fl in comp_faces.items():
        edge_count = Counter()
        vol = 0.0
        for fi in fl:
            f = m.faces[fi]
            w = [remap[i] for i in f]
            for k in range(3):
                a, b = w[k], w[(k + 1) % 3]
                edge_count[(a, b) if a < b else (b, a)] += 1
            vol += dot(m.pos[f[0]], cross(m.pos[f[1]], m.pos[f[2]])) / 6.0
        if edge_count and all(v == 2 for v in edge_count.values()):
            n_closed += 1
            if vol <= 0.0:
                n_inv += 1
        else:
            n_open += 1
    return (n_closed, n_inv, n_open)


def has_smooth_geometry(m):
    """True when the mesh contains any shared-vertex (smoothable) surface."""
    return bool(m._smooth)


class Asset(object):
    def __init__(self, asset_id, category):
        self.id = asset_id
        self.category = category
        self.parts = []

    def add(self, part):
        self.parts.append(part)
        return part

    def part(self, name, **kw):
        return self.add(Part(name, **kw))

    def bounds(self):
        lo = [float("inf")] * 3
        hi = [float("-inf")] * 3
        any_vert = False
        for p in self.parts:
            if not p.mesh.pos:
                continue
            any_vert = True
            a, b = p.mesh.bounds()
            for k in range(3):
                lo[k] = min(lo[k], a[k])
                hi[k] = max(hi[k], b[k])
        if not any_vert:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        return tuple(lo), tuple(hi)


# --------------------------------------------------------------------------
# profiles
# --------------------------------------------------------------------------

def rounded_rect(hx, hy, r, n=4):
    """CCW rounded rectangle in XY.

    Corner circle centres are inset by r ((hx-r, hy-r), not (hx, hy)) and each
    corner is walked as a true arc, never as a chord.
    """
    r = max(0.0, min(r, hx, hy))
    if r <= EPS:
        return [(hx, -hy), (hx, hy), (-hx, hy), (-hx, -hy)]
    pts = []
    corners = ((hx - r, hy - r, 0.0),
               (-(hx - r), hy - r, math.pi * 0.5),
               (-(hx - r), -(hy - r), math.pi),
               (hx - r, -(hy - r), math.pi * 1.5))
    for cx, cy, a0 in corners:
        for i in range(n + 1):
            a = a0 + math.pi * 0.5 * (i / float(n))
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    # The final arc lands exactly on the first arc's start point; drop it so
    # the closed loop has no zero-length edge (which would loft into a
    # degenerate quad that gets silently discarded).
    if len(pts) > 1 and abs(pts[-1][0] - pts[0][0]) < 1e-12 \
            and abs(pts[-1][1] - pts[0][1]) < 1e-12:
        pts.pop()
    return pts


def circle(r, seg, phase=0.0):
    """CCW circle in XY."""
    return [(r * math.cos(phase + 2.0 * math.pi * i / seg),
             r * math.sin(phase + 2.0 * math.pi * i / seg))
            for i in range(seg)]


def section_rings(xs, hx_fn, hy_fn, r_fn, n_corner=4, y_off=0.0, lift_fn=None):
    """Lofted box-section running along +X: one rounded-rect ring per station.

    Each station's profile lives in the plane x = station and is mapped as
    ``(px, py) -> (X, Y, Z) = (station, y_off + px, py + lift)`` -- i.e. the
    profile's horizontal axis becomes world +Y (vehicle width) and its vertical
    axis becomes world +Z (height).  Rings come out CCW seen from +X, which is
    what :func:`loft` expects when advancing along +X.
    """
    rings = []
    for x in xs:
        lift = 0.0 if lift_fn is None else lift_fn(x)
        prof = rounded_rect(hx_fn(x), hy_fn(x), r_fn(x), n_corner)
        rings.append([(x, y_off + px, py + lift) for (px, py) in prof])
    return rings


# --------------------------------------------------------------------------
# general 3D shell builder
# --------------------------------------------------------------------------

def _cap_from_indices(m, idx, color, flip=False):
    """Fan-cap a closed loop of EXISTING vertex indices about a new centre.

    Reusing the ring's own vertices (rather than appending a duplicate copy) is
    what keeps the shell topologically watertight: duplicate positions would
    leave the ring's perimeter edges used by exactly one face each, i.e. a
    T-junction that no index-based manifold check -- and no renderer doing
    edge-morphing or normal-averaging across the seam -- can see.
    """
    n = len(idx)
    c = (0.0, 0.0, 0.0)
    for i in idx:
        c = add(c, m.pos[i])
    ci = len(m.pos)
    m.pos.append(mul(c, 1.0 / float(n)))
    order = list(idx)
    if flip:
        order.reverse()
    for k in range(n):
        m.tri_idx(order[k], order[(k + 1) % n], ci, color)


def _cap_from_points(m, pts3, color, flip=False):
    """Fan-cap a closed 3D ring whose vertices are NOT already in the mesh."""
    start = len(m.pos)
    for p in pts3:
        m.pos.append(tuple(p))
    _cap_from_indices(m, list(range(start, start + len(pts3))), color, flip)


def loft(m, rings, color, cap_start=True, cap_end=True, smooth=False,
         cap_start_color=None, cap_end_color=None, cap_start_flip=False,
         cap_end_flip=False):
    """Stack equal-cardinality rings of 3D points into a closed shell.

    Rings must run in the same rotational direction.  With ``cap_end_flip``
    controlling which way the end caps point, the caller is responsible for
    overall winding; every builder here passes rings it has verified.

    ``smooth=True`` shares side-wall vertices between rings so normals
    interpolate; the owning Part must then be ``flat=False``.
    """
    n = len(rings[0])
    for r in rings:
        assert len(r) == n, "loft rings must have equal point counts"
    base = len(m.pos)
    for r in rings:
        for p in r:
            m.pos.append(tuple(p))
    group = None
    if smooth:
        group = list(range(base, base + n * len(rings)))
        m._smooth.append(group)
    for s in range(len(rings) - 1):
        a = base + s * n
        b = base + (s + 1) * n
        for j in range(n):
            j2 = (j + 1) % n
            m.quad_idx(a + j, a + j2, b + j2, b + j, color, group)
    if cap_start:
        _cap_from_indices(m, list(range(base, base + n)),
                          color if cap_start_color is None else cap_start_color,
                          flip=cap_start_flip)
    if cap_end:
        _cap_from_indices(m, list(range(base + n * (len(rings) - 1),
                                       base + n * len(rings))),
                          color if cap_end_color is None else cap_end_color,
                          flip=cap_end_flip)
    return m


def box(m, size, center=(0.0, 0.0, 0.0), color=(0.8, 0.8, 0.8), colors=None):
    """Axis-aligned box.  ``colors`` may override per face with keys
    '+x','-x','+y','-y','+z','-z'."""
    sx, sy, sz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
    cx, cy, cz = center
    p = [(cx - sx, cy - sy, cz - sz), (cx + sx, cy - sy, cz - sz),
         (cx + sx, cy + sy, cz - sz), (cx - sx, cy + sy, cz - sz),
         (cx - sx, cy - sy, cz + sz), (cx + sx, cy - sy, cz + sz),
         (cx + sx, cy + sy, cz + sz), (cx - sx, cy + sy, cz + sz)]
    c = colors or {}

    def col(k):
        return c.get(k, color)

    m.quad(p[0], p[3], p[2], p[1], col("-z"))
    m.quad(p[4], p[5], p[6], p[7], col("+z"))
    m.quad(p[0], p[1], p[5], p[4], col("-y"))
    m.quad(p[1], p[2], p[6], p[5], col("+x"))
    m.quad(p[2], p[3], p[7], p[6], col("+y"))
    m.quad(p[3], p[0], p[4], p[7], col("-x"))
    return m


def _axis_place(profile, along, center, axis):
    """Place an XY profile onto the plane ``along`` along ``axis``.

    ``along`` is an ABSOLUTE coordinate on the build axis, while ``center``
    offsets the two perpendicular axes.  Callers mix the centre's own component
    into ``along`` themselves; doing it here for the perpendicular axes only is
    what previously let ``cylinder(center=(1, 2, 3))`` come out at z = -1..+1
    instead of z = 2..4 -- invisible to any volume or normal test, because both
    are position-independent.

    The mapping keeps the loop's right-hand normal pointing along +axis, so it
    always agrees with the direction :func:`loft` advances in.  Skipping that
    rule silently turns the shell inside-out (the ``axis="Y"`` case: mapping
    (px, py) -> (X, Z) gives X x Z = -Y, the opposite of +Y).
    """
    out = []
    for px, py in profile:
        if axis == "Z":
            v = (center[0] + px, center[1] + py, along)      # X x Y = +Z
        elif axis == "X":
            v = (along, center[1] + px, center[2] + py)      # Y x Z = +X
        else:
            v = (center[0] + py, along, center[2] + px)      # Z x X = +Y
        out.append(v)
    return out


def _axis_offset(center, axis):
    """The ``center`` component that lies on the build axis."""
    return center[2] if axis == "Z" else (center[0] if axis == "X" else center[1])


def cylinder(m, radius, height, seg=12, center=(0.0, 0.0, 0.0),
             color=(0.8, 0.8, 0.8), radius_top=None, caps=True, smooth=False,
             phase=0.0, axis="Z"):
    """Cylinder / cone / truncated cone centred on ``center`` along ``axis``.

    Rings are CCW when viewed from the +axis end, so the side walls and the
    +axis end cap come out CCW-from-outside and the -axis cap is flipped.
    """
    rt = radius if radius_top is None else radius_top
    mid = _axis_offset(center, axis)
    lo = _axis_place(circle(radius, seg, phase), mid - height * 0.5, center, axis)
    hi = _axis_place(circle(rt, seg, phase), mid + height * 0.5, center, axis)
    return loft(m, [lo, hi], color,
                cap_start=caps, cap_end=caps, smooth=smooth,
                cap_start_flip=True, cap_end_flip=False)


def cone(m, radius, height, seg=10, center=(0.0, 0.0, 0.0),
         color=(0.8, 0.8, 0.8), smooth=False, axis="Z", caps=True,
         radius_top=0.0):
    """Cone with a real base cap and a true apex point.

    The apex is a single shared vertex, not a tiny ring: a near-zero-radius ring
    costs seg extra degenerate triangles AND shares its vertices with the side
    wall, so area-averaging drags the apex normals off the cone surface and the
    flat-normal invariant fails.  A real apex also renders to a clean point.

    ``radius_top > 0`` gives a truncated cone instead (both ends flat-capped).
    """
    c = tuple(center)
    mid = _axis_offset(c, axis)

    base_ring = _axis_place(circle(radius, seg), mid - height * 0.5, c, axis)
    base = len(m.pos)
    for p in base_ring:
        m.pos.append(p)

    if radius_top <= 0.0:
        # True apex: the side wall is a fan from the base ring to one point.
        tip_i = len(m.pos)
        m.pos.append(_axis_place(((0.0, 0.0),), mid + height * 0.5, c, axis)[0])
        for j in range(seg):
            m.tri_idx(base + j, base + (j + 1) % seg, tip_i, color)
        if caps:
            _cap_from_indices(m, list(range(base, base + seg)), color, flip=True)
        return m

    # Truncated: loft the two rings, sharing vertices so `smooth` works.
    top_ring = _axis_place(circle(radius_top, seg), mid + height * 0.5, c, axis)
    tbase = len(m.pos)
    for p in top_ring:
        m.pos.append(p)
    group = list(range(base, tbase + seg)) if smooth else None
    if smooth:
        m._smooth.append(group)
    for j in range(seg):
        j2 = (j + 1) % seg
        m.quad_idx(base + j, base + j2, tbase + j2, tbase + j, color, group)
    if caps:
        _cap_from_indices(m, list(range(base, base + seg)), color, flip=True)
        _cap_from_indices(m, list(range(tbase, tbase + seg)), color, flip=False)
    return m


def tube(m, path, radius, seg=6, color=(0.8, 0.8, 0.8), caps=True, smooth=False,
         radii=None):
    """Sweep a circle along a 3D polyline using parallel-transport frames.

    The frame is (tangent, side, up) built so that side x up = tangent.  Loft
    advances ring-to-ring along the tangent, so the rings must satisfy the
    same right-hand rule; a left-handed frame fills the tube inside-out.
    """
    pts = []
    for p in path:
        p = tuple(p)
        if not pts or length(sub(p, pts[-1])) > 1e-6:
            pts.append(p)
    if len(pts) < 2:
        return m

    tangents = []
    for i in range(len(pts)):
        if i == 0:
            t = sub(pts[1], pts[0])
        elif i == len(pts) - 1:
            t = sub(pts[-1], pts[-2])
        else:
            t = sub(pts[i + 1], pts[i - 1])
        tangents.append(normalize(t))

    t0 = tangents[0]
    ref = (0.0, 0.0, 1.0) if abs(dot(t0, (0.0, 0.0, 1.0))) < 0.9 else (1.0, 0.0, 0.0)
    side = normalize(cross(ref, t0))
    rings = []
    for i, p in enumerate(pts):
        t = tangents[i]
        if length(cross(t, side)) < 1e-7:
            side = normalize(cross((0.0, 0.0, 1.0), t))
        side = normalize(cross(side, t))
        up = normalize(cross(t, side))
        r = radius if radii is None else radii[i]
        rings.append([add(add(p, mul(side, r * math.cos(a))), mul(up, r * math.sin(a)))
                      for a in [2.0 * math.pi * k / seg for k in range(seg)]])

    return loft(m, rings, color, cap_start=caps, cap_end=caps, smooth=smooth,
                cap_start_flip=True, cap_end_flip=False)


def sphere(m, radius, seg_u=10, seg_v=6, center=(0.0, 0.0, 0.0),
           color=(0.8, 0.8, 0.8), squash=1.0, smooth=False):
    """UV sphere; ``squash`` scales the local Z axis into an ellipsoid.

    The two pole rows collapse to a single vertex each.  Repeating a point
    seg_u times instead (the obvious implementation) yields seg_u zero-area
    triangles at each pole, which the mesh silently drops -- losing real
    surface area and skewing the volume test.
    """
    cx, cy, cz = center
    base = len(m.pos)

    def row(v, n):
        """Vertices of latitude row v; a single vertex at the poles."""
        t = math.pi * v / seg_v
        z = radius * squash * math.cos(t)
        r = radius * math.sin(t)
        if n == 1:
            return [_append(m, (cx, cy, cz + z))]
        start = len(m.pos)
        for i in range(n):
            m.pos.append((cx + r * math.cos(2 * math.pi * i / n),
                          cy + r * math.sin(2 * math.pi * i / n), cz + z))
        return list(range(start, start + n))

    rows = [row(0, 1)] + [row(v, seg_u) for v in range(1, seg_v)] + [row(seg_v, 1)]
    group = None
    if smooth:
        group = list(range(base, len(m.pos)))
        m._smooth.append(group)
    for a, b in zip(rows, rows[1:]):
        # Rows DESCEND (+Z pole -> -Z pole), so advancing `b` travels -Z, while
        # loft()'s contract is "advance along the ring's right-hand direction"
        # (+Z here).  The mid quads are therefore mirrored relative to a
        # +Z-ascending loft: emit (b[j], b[j2], a[j2], a[j]), not
        # (a[j], a[j2], b[j2], b[j]).
        #
        # The DIAGONAL must stay a cross edge (b[j]-a[j2]) either way.  Putting
        # it on a vertical or ring edge instead makes that edge shared by three
        # faces -- a non-manifold seam that still renders as a clean sphere.
        # The pole fans are unaffected by the row order and keep the natural
        # outward winding.
        if len(a) == 1:                      # top pole fan
            for j in range(seg_u):
                m.tri_idx(a[0], b[j], b[(j + 1) % seg_u], color, group)
        elif len(b) == 1:                    # bottom pole fan
            for j in range(seg_u):
                m.tri_idx(a[j], b[0], a[(j + 1) % seg_u], color, group)
        else:
            for j in range(seg_u):
                j2 = (j + 1) % seg_u
                m.quad_idx(b[j], b[j2], a[j2], a[j], color, group)
    return m


def _append(m, p):
    m.pos.append(p)
    return len(m.pos) - 1


def torus(m, R, r, seg_u=12, seg_v=6, center=(0.0, 0.0, 0.0),
          color=(0.8, 0.8, 0.8), axis="Z", smooth=True):
    """Torus from explicit loops (bmesh.ops.create_torus does not exist).

    Winding: advancing ``i`` sweeps the ring CCW in the plane normal to +axis
    while advancing ``j`` sweeps the tube CCW as seen from the ring centre, so
    ``(di, dj)`` is a left-handed pair and the quad must be emitted as
    ``(i,j) (i,j2) (i2,j2) (i2,j)`` to stay outward.  Reversing it fills the
    tube inside-out while still rendering a closed surface.
    """
    cx, cy, cz = center

    def place(i, j):
        a = 2.0 * math.pi * i / seg_u
        b = 2.0 * math.pi * j / seg_v
        rr = R + r * math.cos(b)
        x, y, z = rr * math.cos(a), rr * math.sin(a), r * math.sin(b)
        if axis == "Z":
            return (cx + x, cy + y, cz + z)
        if axis == "X":
            return (cx + z, cy + x, cz + y)
        # Y: swap Y and Z.  That is an ODD permutation (det -1), so the mapped
        # loop's right-hand normal points along -Y while the rings advance along
        # +Y -- the mirror of the Z case.  The quad emitter below reverses its
        # winding to compensate; negating the coordinates instead only rotates
        # the shell without fixing the handedness, leaving signed volume
        # negative and every face normal pointing into the tube.
        return (cx + x, cy + z, cz + y)

    base = len(m.pos)
    for i in range(seg_u):
        for j in range(seg_v):
            m.pos.append(place(i, j))
    group = None
    if smooth:
        group = list(range(base, base + seg_u * seg_v))
        m._smooth.append(group)
    # Winding, determined by measuring signed volume for all three axes rather
    # than by reasoning about permutation parity:
    #   order A = (i,j) (i,j2) (i2,j2) (i2,j)   -> outward on the Y axis
    #   order B = (i,j2) (i,j) (i2,j) (i2,j2)   -> outward on X and Z
    # The Y mapping (x, z, y) is the only ODD permutation of the three, and it
    # is also the only one needing order A.  Using the same order for all three
    # leaves Y watertight but inside-out: a closed shell whose signed volume is
    # negative, i.e. every face normal points into the tube.
    order_a = (axis == "Y")
    for i in range(seg_u):
        i2 = (i + 1) % seg_u
        for j in range(seg_v):
            j2 = (j + 1) % seg_v
            if order_a:
                m.quad_idx(base + i * seg_v + j, base + i * seg_v + j2,
                           base + i2 * seg_v + j2, base + i2 * seg_v + j,
                           color, group)
            else:
                m.quad_idx(base + i * seg_v + j2, base + i * seg_v + j,
                           base + i2 * seg_v + j, base + i2 * seg_v + j2,
                           color, group)
    return m


def ribbon(m, pts, width, up=(0.0, 0.0, 1.0), color=(0.8, 0.8, 0.8)):
    """A flat strip of constant ``width`` following a 3D polyline."""
    up = normalize(up)
    left, right = [], []
    for i, p in enumerate(pts):
        if i == 0:
            t = sub(pts[1], pts[0])
        elif i == len(pts) - 1:
            t = sub(pts[-1], pts[-2])
        else:
            t = sub(pts[i + 1], pts[i - 1])
        t = normalize(t)
        n = normalize(cross(t, up))
        left.append(add(tuple(p), mul(n, width * 0.5)))
        right.append(add(tuple(p), mul(n, -width * 0.5)))
    for i in range(len(pts) - 1):
        m.quad(left[i], left[i + 1], right[i + 1], right[i], color)
    return m


# --------------------------------------------------------------------------
# part-level utilities
# --------------------------------------------------------------------------

def mirror_x(part, name_suffix="_m"):
    """Duplicate a part mirrored across X=0; face winding is reversed."""
    out = Part(part.name + name_suffix, part.base_color, part.emissive,
               part.flat, part.roughness, part.metallic)
    out.mesh.pos = [(-p[0], p[1], p[2]) for p in part.mesh.pos]
    out.mesh.faces = [(a, c, b) for (a, b, c) in part.mesh.faces]
    out.mesh.fcol = list(part.mesh.fcol)
    return out


def rotate_z(part, angle_deg, name_suffix="_r"):
    """Duplicate a part rotated about its own Z origin."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    out = Part(part.name + name_suffix, part.base_color, part.emissive,
               part.flat, part.roughness, part.metallic)
    out.mesh.pos = [(p[0] * ca - p[1] * sa, p[0] * sa + p[1] * ca, p[2])
                    for p in part.mesh.pos]
    out.mesh.faces = list(part.mesh.faces)
    out.mesh.fcol = list(part.mesh.fcol)
    return out


def translate_part(part, offset):
    """In-place translate every vertex of a part."""
    ox, oy, oz = offset
    for p in part.mesh.pos:
        p[0] += ox
        p[1] += oy
        p[2] += oz
    return part


def recolor_faces(part, key, color, tol=1e-4):
    """Recolor faces whose current color matches ``key`` within ``tol``."""
    n = 0
    for i, c in enumerate(part.mesh.fcol):
        if (abs(c[0] - key[0]) <= tol and abs(c[1] - key[1]) <= tol
                and abs(c[2] - key[2]) <= tol):
            part.mesh.fcol[i] = tuple(color)
            n += 1
    return n


def recolor_faces_where(part, predicate, color):
    """Recolor faces where ``predicate(face_normal, centroid)`` is True."""
    m = part.mesh
    fn = m.face_normals()
    n = 0
    for i, (a, b, c) in enumerate(m.faces):
        va, vb, vc = m.pos[a], m.pos[b], m.pos[c]
        cen = mul(add(add(va, vb), vc), 1.0 / 3.0)
        if predicate(fn[i], cen):
            m.fcol[i] = tuple(color)
            n += 1
    return n
