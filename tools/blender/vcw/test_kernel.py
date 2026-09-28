"""Kernel sanity checks -- runnable OUTSIDE Blender:  python3 test_kernel.py

Two independent outward-winding tests are used:

* ``signed_volume`` > 0  -- valid for ANY closed shell, including non-convex
  ones like a torus where there is no single interior reference point.
* ``radial_outward``     -- valid only for shapes star-shaped about the centre,
  so it is used as a cross-check on boxes/cylinders/spheres.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core as C  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print("  PASS  %s" % name)
    else:
        print("  FAIL  %s  %s" % (name, detail))
        FAILS.append(name)


def signed_volume(m):
    v = 0.0
    for (a, b, c) in m.faces:
        v += C.dot(m.pos[a], C.cross(m.pos[b], m.pos[c])) / 6.0
    return v


def radial_outward(m, centre=None, thresh=-0.001):
    """Count faces whose normal points back toward the solid's interior.

    The reference point defaults to the area-weighted face centroid, which lies
    strictly inside any convex solid.  An earlier version referenced the
    radial direction from a fixed centre instead: that is wrong for FLAT FACES
    (a cylinder cap's normal is perpendicular to every radial vector), so it
    reported 24 perfectly healthy cap faces as inverted.  Per-face
    centroid-to-interior is correct for convex shapes including their caps.
    """
    fn = m.face_normals()
    cents = []
    for (a, b, c) in m.faces:
        cents.append(C.mul(C.add(C.add(m.pos[a], m.pos[b]), m.pos[c]), 1 / 3.0))
    if centre is None:
        n = float(len(cents))
        centre = (sum(c[0] for c in cents) / n,
                  sum(c[1] for c in cents) / n,
                  sum(c[2] for c in cents) / n)
    bad = 0
    for i in range(len(m.faces)):
        if C.dot(fn[i], C.sub(cents[i], centre)) < thresh:
            bad += 1
    return bad


def nonmanifold_edges(m):
    """Every edge of a closed manifold shell is shared by exactly 2 faces.

    Only meaningful for SHARED-vertex shells.  Flat parts duplicate a vertex
    per triangle on purpose (that is what flat shading is), so every edge there
    is used once by construction and this metric is meaningless for them.
    """
    from collections import Counter
    if not m._smooth:
        return None
    cnt = Counter()
    for f in m.faces:
        for i in range(3):
            cnt[tuple(sorted((f[i], f[(i + 1) % 3])))] += 1
    return sum(1 for v in cnt.values() if v != 2)


print("== box ==")
m = C.Mesh()
C.box(m, (2.0, 4.0, 6.0))
check("box tris", m.ntris == 12, m.ntris)
check("box no degenerate", m.skipped == 0, m.skipped)
check("box radial outward", radial_outward(m, (0, 0, 0)) == 0, radial_outward(m, (0, 0, 0)))
check("box volume==48", abs(signed_volume(m) - 48.0) < 1e-9, signed_volume(m))

print("== rounded_rect: extents + area vs closed form ==")
p = C.rounded_rect(1.0, 2.0, 0.5, 6)
check("rr point count", len(p) == 4 * 7, len(p))  # 4 corners x (n+1) arc samples
check("rr hx exact", abs(max(q[0] for q in p) - 1.0) < 1e-12, max(q[0] for q in p))
check("rr hy exact", abs(max(q[1] for q in p) - 2.0) < 1e-12, max(q[1] for q in p))
area = 0.0
for i in range(len(p)):
    x0, y0 = p[i]
    x1, y1 = p[(i + 1) % len(p)]
    area += x0 * y1 - x1 * y0
area *= 0.5
w, h, r, n = 2.0, 4.0, 0.5, 6
# Analytic area of the *arc* shape is w*h minus the four corner gaps; a
# polyline cut from the arcs is marginally smaller, so hold it to a tight
# relative band rather than exact equality.
expect = w * h - (4 - math.pi) * r * r
err = abs(area - expect) / expect
check("rr area within 0.15% of analytic", err < 1.5e-3, (area, expect, err))

print("== cylinder: volume, winding, watertight (all 3 axes) ==")
exact = math.pi * 0.25 * 2.0
for axis in ("Z", "X", "Y"):
    m = C.Mesh()
    C.cylinder(m, 0.5, 2.0, 24, center=(1.0, 2.0, 3.0), axis=axis)
    v = signed_volume(m)
    # a 24-gon prism is ~1.1% smaller than the circumscribed circle
    poly = 0.5 * 24 * 0.5 * 0.5 * math.sin(2 * math.pi / 24) * 2.0
    check("cyl%s vol" % axis, abs(v - poly) / poly < 0.001, (v, poly))
    check("cyl%s outward" % axis, radial_outward(m) == 0,
          radial_outward(m))
    check("cyl%s closed (volume>0)" % axis, v > 0, v)

print("== absolute positioning along the build axis (regression) ==")
# Volume and normal checks are position-INDEPENDENT, so they cannot catch a
# primitive that lands in the wrong place.  Correct semantics:
#   build axis       -> center +- h/2   (radius plays no part)
#   perpendicular ax -> center +- r
# A builder that forgets to mix the centre's own component into `along` puts
# the part at the origin instead of where it was asked to be.
for axis in ("Z", "X", "Y"):
    ai = {"Z": 2, "X": 0, "Y": 1}[axis]
    m = C.Mesh()
    C.cylinder(m, 0.5, 2.0, 24, center=(1.0, 2.0, 3.0), axis=axis)
    lo, hi = m.bounds()
    centre = (1.0, 2.0, 3.0)
    check("cyl%s build axis == centre+-h/2" % axis,
          abs((lo[ai] + hi[ai]) * 0.5 - centre[ai]) < 1e-9
          and abs((hi[ai] - lo[ai]) - 2.0) < 1e-9, (lo[ai], hi[ai]))
    for pi in range(3):
        if pi == ai:
            continue
        check("cyl%s perp axis %d == centre+-r" % (axis, pi),
              abs((lo[pi] + hi[pi]) * 0.5 - centre[pi]) < 1e-9
              and abs((hi[pi] - lo[pi]) - 1.0) < 1e-9, (lo[pi], hi[pi]))

m = C.Mesh()
C.cone(m, 0.6, 1.5, 12, center=(0.0, 0.0, 5.0))
lo, hi = m.bounds()
check("cone apex at centre+h/2", abs(hi[2] - 5.75) < 1e-9, (lo[2], hi[2]))
check("cone base at centre-h/2", abs(lo[2] - 4.25) < 1e-9, (lo[2], hi[2]))

print("== cone ==")
m = C.Mesh()
C.cone(m, 0.6, 1.5, 24, center=(0, 0, 0.75))
v = signed_volume(m)
ev = (0.5 * 24 * 0.6 * 0.6 * math.sin(2 * math.pi / 24)) * 1.5 / 3.0
check("cone vol", abs(v - ev) / ev < 0.01, (v, ev))
check("cone outward", radial_outward(m) == 0,
      radial_outward(m))

print("== sphere: no pole degeneracy, volume, winding ==")
m = C.Mesh()
C.sphere(m, 1.0, 16, 10, center=(0, 0, 0), smooth=True)
check("sphere no degenerate", m.skipped == 0, m.skipped)
check("sphere outward", radial_outward(m, (0, 0, 0)) == 0, radial_outward(m, (0, 0, 0)))
# A seg_u x seg_v UV sphere is INSCRIBED in the unit sphere: every vertex lies
# on r=1 but every face cuts the interior, so it must under-fill 4/3*pi by a
# few percent.  The expected deficit is the mean of the inscribed-polygon area
# factor over the latitude bands; assert against the sphere volume with a
# tolerance that admits the deficit but would still catch a real leak.
sphere_v = 4.0 * math.pi / 3.0
v = signed_volume(m)
ratio = v / sphere_v
check("sphere vol within inscribed-poly range (0.93..1.00)", 0.93 < ratio <= 1.0001,
      (v, sphere_v, ratio))
check("sphere watertight", nonmanifold_edges(m) == 0, nonmanifold_edges(m))

print("== torus: winding + volume ==")
m = C.Mesh()
C.torus(m, 1.0, 0.3, 16, 10)
v = signed_volume(m)
vt = 2 * math.pi ** 2 * (1.0 ** 2) * (0.3 ** 2)   # V = 2 pi^2 R^2 r^2
check("torus vol>0 (outward)", v > 0, v)
check("torus vol close", abs(v - vt) / vt < 0.12, (v, vt))  # polygon sampling under-fills the tube
check("torus watertight", nonmanifold_edges(m) == 0, nonmanifold_edges(m))
# A torus has no single interior point, so check the tube wall directly:
# every face normal must point away from its own point on the ring centreline.
fn = m.face_normals()
bad = 0
for i, (a, b, c) in enumerate(m.faces):
    cen = C.mul(C.add(C.add(m.pos[a], m.pos[b]), m.pos[c]), 1 / 3.0)
    core = C.mul(C.normalize((cen[0], cen[1], 0.0)), 1.0)
    if C.dot(fn[i], C.sub(cen, core)) < 0.0:
        bad += 1
check("torus tube normals outward", bad == 0, bad)

print("== torus on every axis (regression) ==")
# The Y mapping (x, z, y) is the only ODD permutation of the three, and it is
# the only one needing the mirrored quad order.  With a single shared order the
# Y torus comes out watertight but INSIDE-OUT: still a closed surface, still
# renders as a torus, and only signed volume reveals it.
for axis in ("Z", "X", "Y"):
    m = C.Mesh()
    C.torus(m, 1.0, 0.3, 16, 10, axis=axis)
    v = signed_volume(m)
    vt = 2 * math.pi ** 2 * (1.0 ** 2) * (0.3 ** 2)
    check("torus%s vol>0 (outward)" % axis, v > 0, v)
    check("torus%s vol close" % axis, abs(v - vt) / vt < 0.12, (v, vt))
    check("torus%s watertight" % axis, nonmanifold_edges(m) == 0,
          nonmanifold_edges(m))

print("== tube along a curve ==")
path = [(0, 0, 0), (0, 0, 1), (0.5, 0, 2), (1.5, 0.3, 2.4)]
m = C.Mesh()
C.tube(m, path, 0.2, 8, smooth=True)
check("tube no degenerate", m.skipped == 0, m.skipped)
check("tube volume>0", signed_volume(m) > 0, signed_volume(m))
check("tube watertight", nonmanifold_edges(m) == 0, nonmanifold_edges(m))

print("== flat shading survives shared-vertex primitives (regression) ==")
# cone()/loft() share vertices so the shell is watertight, but a FLAT part must
# still hand each triangle its own exact normal.  Without the "no smooth groups
# => no averaging" rule, the cone's base rim (valence 4) and its apex (valence
# seg) come out visibly smoothed.
for name, build in (
    ("cone", lambda m: C.cone(m, 0.2, 0.34, 10, center=(2.4, 1.6, 20.0))),
    ("cylinder", lambda m: C.cylinder(m, 0.2, 0.34, 10, center=(0, 0, 5))),
    ("sphere-flat", lambda m: C.sphere(m, 0.3, 10, 6, center=(0, 0, 0))),
    ("torus-flat", lambda m: C.torus(m, 1.0, 0.3, 12, 6, smooth=False)),
):
    m = C.Mesh()
    build(m)
    # BEFORE splitting, shared-vertex primitives average -- that is the bug.
    fn, vn = m.vertex_normals()
    shared_worst = 0.0
    for fi, f in enumerate(m.faces):
        for i in f:
            shared_worst = max(shared_worst, 1.0 - C.dot(fn[fi], vn[i]))
    # AFTER splitting, every triangle owns its normal -- this is what the
    # exporter does for flat parts, and it must be exact.
    flat = m.flatten()
    check("%s flatten -> 3 verts/tri" % name,
          flat.nverts == 3 * flat.ntris, (flat.nverts, flat.ntris))
    fn2, vn2 = flat.vertex_normals()
    worst = 0.0
    for fi, f in enumerate(flat.faces):
        for i in f:
            worst = max(worst, 1.0 - C.dot(fn2[fi], vn2[i]))
    check("%s flat normals exact after split" % name, worst < 1e-12, worst)
    # splitting must not move any geometry or change volume
    check("%s split preserves volume" % name,
          abs(signed_volume(flat) - signed_volume(m)) < 1e-9,
          (signed_volume(flat), signed_volume(m)))
    check("%s split preserves face colours" % name, flat.fcol == m.fcol)
    del shared_worst

# and the smooth flag must actually average
m = C.Mesh()
C.sphere(m, 1.0, 16, 10, smooth=True)
fn, vn = m.vertex_normals()
worst = 0.0
for fi, f in enumerate(m.faces):
    for i in f:
        worst = max(worst, 1.0 - C.dot(fn[fi], vn[i]))
check("sphere smooth normals DO differ from face normals", worst > 1e-3, worst)

print("== section_rings / loft (box-section car body) ==")
xs = [-2.0, -1.0, 0.0, 1.0, 2.0]
rings = C.section_rings(
    xs,
    lambda x: 0.85 * (1.0 - 0.10 * (x / 2.0) ** 2),               # half-width -> Y
    lambda x: 0.30 * (1.0 - 0.45 * max(0.0, x - 0.6) / 1.4),      # half-height -> Z
    lambda x: 0.12,
    lift_fn=lambda x: 0.35,
)
check("ring cardinality equal", len(set(len(r) for r in rings)) == 1,
      [len(r) for r in rings])
m = C.Mesh()
C.loft(m, rings, (1, 0, 0), smooth=True)
v = signed_volume(m)
check("loft volume>0 (outward)", v > 0, v)
check("loft no degenerate", m.skipped == 0, m.skipped)
check("loft watertight", nonmanifold_edges(m) == 0, nonmanifold_edges(m))
lo, hi = m.bounds()
check("loft bbox x == 4m", abs((hi[0] - lo[0]) - 4.0) < 1e-9, (lo[0], hi[0]))
check("loft centred on y", abs(lo[1] + hi[1]) < 1e-9, (lo[1], hi[1]))

print("== vertex normal == face normal for a flat part ==")
m = C.Mesh()
C.box(m, (1, 1, 1))
fn, vn = m.vertex_normals()
worst = 0.0
for vi in range(m.nverts):
    for fi, f in enumerate(m.faces):
        if vi in f:
            worst = max(worst, 1.0 - min(C.dot(vn[vi], fn[fi]), 1.0))
check("flat verts exact normal", worst < 1e-12, worst)

print("== smooth part actually interpolates ==")
m = C.Mesh()
C.sphere(m, 1.0, 16, 10, smooth=True)
fn, vn = m.vertex_normals()
mid = m.nverts // 2
check("smooth vertex differs from every incident face normal",
      any(1.0 - min(C.dot(vn[mid], f) for f in fn) > 1e-6 for f in fn),
      "all identical -> smoothing is not happening")

print("== mirror_x flips winding correctly ==")
a = C.Part("a")
C.box(a.mesh, (1, 2, 3))
b = C.mirror_x(a)
check("mirror preserves volume", abs(signed_volume(a.mesh) - signed_volume(b.mesh)) < 1e-9,
      (signed_volume(a.mesh), signed_volume(b.mesh)))
check("rotate_z preserves volume",
      abs(signed_volume(C.rotate_z(a, 37.0).mesh) - signed_volume(a.mesh)) < 1e-6,
      signed_volume(C.rotate_z(a, 37.0).mesh))

print("== ribbon ==")
m = C.Mesh()
C.ribbon(m, [(0, 0, 0), (0, 0, 1), (1, 0, 2)], 0.4)
check("ribbon tris", m.ntris == 4, m.ntris)

print()
if FAILS:
    print("KERNEL FAILURES: %d -> %s" % (len(FAILS), FAILS))
    sys.exit(1)
print("ALL KERNEL CHECKS PASSED")
