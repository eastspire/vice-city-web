"""Asset -> JSON exporter (Y-up) and manifest writer.

The renderer consumes these files, so the export contract is deliberately
strict and self-checking:

* **Handedness.** Authoring is Blender-style Z-up; the export matrix is
  ``json = (x, z, -y)`` so +Y is up and +Z is forward.  Its determinant is +1,
  so winding order is preserved exactly -- a CCW face from outside stays CCW.
* **CCW.** Every triangle's stored normal is recomputed from its own three
  positions and asserted to agree with the vertex normal.  For flat parts the
  vertices are not shared, so this is an exact equality check rather than a
  tolerance one.
* **Units.** 1 Blender unit = 1 metre; positions are written as plain metres.
* **Colours.** Linear 0..1 floats.  Emission is a separate per-part colour; an
  all-zero emissive means the part does not glow.
"""

import json
import os

from . import core as C

# Authoring (Z-up, Blender front = -Y) -> export (Y-up, forward = +Z).
#   json.x =  x
#   json.y =  z   (up)
#   json.z = -y   (forward)
# det = +1  => right-handed, winding preserved.
_M = ((1.0, 0.0, 0.0),
      (0.0, 0.0, 1.0),
      (0.0, -1.0, 0.0))


def to_y_up(v):
    """Apply the Z-up -> Y-up export matrix to a 3D vector."""
    return (_M[0][0] * v[0] + _M[0][1] * v[1] + _M[0][2] * v[2],
            _M[1][0] * v[0] + _M[1][1] * v[1] + _M[1][2] * v[2],
            _M[2][0] * v[0] + _M[2][1] * v[1] + _M[2][2] * v[2])


def _r(v, nd=6):
    """Round for compact JSON; keeps -0.0 out of the file."""
    out = round(float(v), nd)
    return 0.0 if out == 0 else out


def _rv(v, nd=6):
    return [_r(v[0], nd), _r(v[1], nd), _r(v[2], nd)]


def _rc(c, nd=4):
    """Colours are written at 4 decimals -- plenty for 8-bit output."""
    return [round(min(1.0, max(0.0, float(c[0]))), nd),
            round(min(1.0, max(0.0, float(c[1]))), nd),
            round(min(1.0, max(0.0, float(c[2]))), nd)]


class ExportError(AssertionError):
    pass


def verify_outward(part, asset_id):
    """Assert every face of ``part`` points out of the asset.

    Three regimes, in decreasing order of strength:

    1. ``outward=("point", p)`` / ``("dir", d)`` -- declared by the builder.
       Needed for genuinely open surfaces, where there is no interior to measure.
    2. Closed shell(s) -- every position-welded component must have a POSITIVE
       signed volume.  Valid for concave and holed shapes alike, and it checks
       EVERY triangle, not a sample.
    3. Neither -- reported as unchecked (a genuine open shell).

    The component test welds vertices by position first, so it works on flat
    parts too.  An index-based ``is_closed`` would call every flat part open --
    it stores three vertices per triangle -- and silently skip the majority of
    the geometry.
    """
    mesh = part.mesh
    fn = mesh.face_normals()

    if part.outward is not None:
        kind, ref = part.outward
        r = tuple(ref)
        if kind == "point":
            bad = 0
            for i, (a, b, c) in enumerate(mesh.faces):
                cen = C.mul(C.add(C.add(mesh.pos[a], mesh.pos[b]), mesh.pos[c]),
                            1.0 / 3.0)
                d = C.sub(cen, r)
                if C.length(d) < 1e-9:
                    continue
                if C.dot(fn[i], C.normalize(d)) < -1e-6:
                    bad += 1
            if bad:
                raise ExportError("%s/%s: %d faces point inward w.r.t. %s"
                                  % (asset_id, part.name, bad, r))
            return "declared-point"
        if kind == "dir":
            d = C.normalize(r)
            bad = sum(1 for i in range(len(fn)) if C.dot(fn[i], d) <= 0.0)
            if bad:
                raise ExportError("%s/%s: %d faces do not point along %s"
                                  % (asset_id, part.name, bad, r))
            return "declared-dir"

    n_closed, n_inverted, n_open = C.closed_components(mesh)
    if n_inverted:
        raise ExportError(
            "%s/%s: %d of %d closed shells have negative signed volume -- "
            "those faces are wound inward" % (asset_id, part.name, n_inverted,
                                              n_closed))
    if n_closed:
        return "closed(%d shells%s)" % (n_closed,
                                        ", %d open" % n_open if n_open else "")
    return "open-shell"


def export_part(part, asset_id, nd=6):
    """Serialise one Part to the JSON schema, asserting the invariants."""
    mesh = part.mesh
    if not mesh.faces:
        raise ExportError("%s/%s: part has no faces" % (asset_id, part.name))
    if mesh.skipped:
        raise ExportError("%s/%s: %d degenerate triangles were dropped"
                          % (asset_id, part.name, mesh.skipped))
    if len(mesh.fcol) != len(mesh.faces):
        raise ExportError("%s/%s: face_colors (%d) != faces (%d)"
                          % (asset_id, part.name, len(mesh.fcol), len(mesh.faces)))

    face_normals, vertex_normals = mesh.vertex_normals()

    # --- the hard invariant: faces wind CCW seen from outside ----------------
    #
    # Deliberately compares against something OTHER than the stored vertex
    # normal.  On a shared-vertex primitive a vertex normal is an AVERAGE over
    # incident faces (a cone's apex averages all 8 side triangles toward +Z),
    # so dot(face_normal, vertex_normal) can go negative on perfectly correct
    # geometry and equally can mask a genuinely inverted face.  The only sound
    # test at this stage is geometric: does each triangle's own normal point
    # away from the solid?  verify_outward() answers exactly that.
    for fi, (ia, ib, ic) in enumerate(mesh.faces):
        n = C.cross(C.sub(mesh.pos[ib], mesh.pos[ia]),
                    C.sub(mesh.pos[ic], mesh.pos[ia]))
        if C.length(n) < 1e-12:
            raise ExportError("%s/%s face %d is degenerate"
                              % (asset_id, part.name, fi))

    outward_status = verify_outward(part, asset_id)

    # A Part carries ONE flat flag.  Mixed smooth/flat geometry inside one part
    # would smear the flat surfaces, so reject it before it silently ships.
    if C.has_smooth_geometry(mesh) and part.flat:
        raise ExportError(
            "%s/%s: part is flagged flat but contains smooth-lofted geometry "
            "(%d shared-vertex triangle groups). Split it, or set flat=False."
            % (asset_id, part.name, len(mesh._smooth)))

    # Flat shading means each triangle owns its vertices.  Primitives share
    # vertices to stay watertight, so a flat part MUST be split here or a
    # vertex shared by four triangles would carry one averaged normal and the
    # surface would render smooth despite flat=true.  Verify first, split after.
    work = mesh.flatten() if part.flat else mesh
    if part.flat:
        face_normals, vertex_normals = work.vertex_normals()
        for fi, f in enumerate(work.faces):
            gn = face_normals[fi]
            for k in f:
                if C.dot(gn, vertex_normals[k]) < 0.999:
                    raise ExportError(
                        "%s/%s face %d: flat normal not exact after split "
                        "(dot=%.6f)" % (asset_id, part.name, fi,
                                        C.dot(gn, vertex_normals[k])))
    mesh = work

    positions = [to_y_up(p) for p in mesh.pos]
    normals = [to_y_up(n) for n in vertex_normals]
    faces = [list(f) for f in mesh.faces]
    bounds_min, bounds_max = mesh.bounds()
    bmin = to_y_up(bounds_min)
    bmax = to_y_up(bounds_max)
    bounds_min = tuple(min(bmin[k], bmax[k]) for k in range(3))
    bounds_max = tuple(max(bmin[k], bmax[k]) for k in range(3))

    return {
        "name": part.name,
        "base_color": _rc(part.base_color),
        "emissive": _rc(part.emissive),
        "roughness": _r(part.roughness, 3),
        "metallic": _r(part.metallic, 3),
        "positions": [_rv(p, nd) for p in positions],
        "normals": [_rv(n, nd) for n in normals],
        "faces": faces,
        "face_colors": [_rc(c) for c in mesh.fcol],
        "flat": bool(part.flat),
        "tri_count": len(faces),
        "outward_check": outward_status,
    }


def export_asset(asset):
    """Serialise a whole Asset to the JSON schema."""
    # Part names must be unique WITHIN an asset.  A renderer that builds one
    # material per part name (the common case) silently drops or merges every
    # duplicate, so a per-instance part like one "window" per window quietly
    # loses its per-instance colours.  Fail the build instead.
    seen = {}
    dupes = {}
    for p in asset.parts:
        if p.name in seen:
            dupes.setdefault(p.name, []).append(p)
        else:
            seen[p.name] = p
    if dupes:
        detail = ", ".join("'%s' x%d" % (k, len(v) + 1)
                           for k, v in sorted(dupes.items())[:5])
        raise ExportError(
            "%s: %d duplicate part name(s) (%s). Give each part a unique name, "
            "or merge same-named parts into one Part." % (asset.id, len(dupes), detail))

    parts = [export_part(p, asset.id) for p in asset.parts if p.mesh.faces]
    if not parts:
        raise ExportError("%s: no non-empty parts" % asset.id)
    lo, hi = asset.bounds()
    tlo, thi = to_y_up(lo), to_y_up(hi)
    bmin = tuple(min(tlo[k], thi[k]) for k in range(3))
    bmax = tuple(max(tlo[k], thi[k]) for k in range(3))

    # Re-derive the bounds from the exported positions so a mismatch is
    # impossible to miss downstream.
    gmin = [float("inf")] * 3
    gmax = [float("-inf")] * 3
    for p in parts:
        for v in p["positions"]:
            for k in range(3):
                gmin[k] = min(gmin[k], v[k])
                gmax[k] = max(gmax[k], v[k])
    for k in range(3):
        if abs(gmin[k] - bmin[k]) > 1e-5 or abs(gmax[k] - bmax[k]) > 1e-5:
            raise ExportError("%s: declared bounds %s != actual %s"
                              % (asset.id, (bmin, bmax), (gmin, gmax)))

    return {
        "id": asset.id,
        "category": asset.category,
        "y_up": True,
        "bounds": {"min": _rv(bmin), "max": _rv(bmax)},
        "part_count": len(parts),
        "tri_count": sum(p["tri_count"] for p in parts),
        "parts": parts,
    }


def write_asset(asset, out_dir, nd=6):
    doc = export_asset(asset)
    path = os.path.join(out_dir, "%s.json" % asset.id)
    with open(path, "w") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    return doc, path


def write_manifest(entries, out_dir, version=1,
                   generator="tools/blender/build_assets.py"):
    manifest = {
        "version": version,
        "generator": generator,
        "up_axis": "Y",
        "unit": "meter",
        "asset_count": len(entries),
        "tri_count": sum(e["tri_count"] for e in entries),
        "categories": sorted(set(e["category"] for e in entries)),
        "assets": entries,
    }
    path = os.path.join(out_dir, "manifest.json")
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    return manifest, path
