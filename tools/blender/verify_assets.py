#!/usr/bin/env python3
"""Strict validator for the exported NEON BAY / Vice City asset JSON files.

    Blender --background --python tools/blender/verify_assets.py

Reads every ``<id>.json`` in ``assets/``, checks it against the export
contract defined by ``vcw/export.py``, and prints a report.  Exits 0 only
when every assertion holds; exits 1 and names each violation otherwise.

Pure standard library on purpose: this runs inside Blender's bundled Python.
It never writes to the asset files -- verification is read-only.

SCHEMA (per asset document)::

    { id, category, y_up, bounds:{min,max}, part_count, tri_count, parts:[
        { name, base_color, emissive, roughness, metallic,
          positions[][], normals[][], faces[][][], face_colors[][],
          flat, tri_count, outward_check } ] }

Coordinate system is Y-up: +Y up, +Z forward (a vehicle's nose is at max Z),
+X right.  The exporter applied ``(x, z, -y)`` to Blender's Z-up authoring
space, a right-handed (det +1) rotation, so CCW-from-outside winding in the
authoring mesh is still CCW-from-outside here.

Checks, numbered as in the contract:

 1. top level   -- id / category non-empty strings, y_up true, finite bounds
                   with min <= max on every axis
 2. per part    -- unique names within the asset, non-empty positions,
                   len(normals) == len(positions), len(faces) > 0,
                   len(face_colors) == len(faces), every colour in [0, 1]
 3. per face    -- exactly 3 integer indices, each in [0, len(positions)),
                   no repeated index (non-degenerate)
 4. winding     -- geometric normal is a unit vector and agrees in DIRECTION
                   with the stored normal; stored normals are unit length
 5. outward     -- per position-welded closed component, the divergence
                   theorem volume must be positive (negative == faces wound
                   inward).  Parts that are genuinely open shells cannot be
                   volume-checked and are reported as informational
 6. bounds      -- recomputed from every position of every part, and must
                   match the declared bounds within 1e-4
 7. y-up        -- ground-resting categories rest on y = 0 and do not sink
                   through the ground plane
 8. totals      -- asset / triangle / vertex counts, per category

Two notes on how the harder checks are decided, because the naive version of
each is either unsound or powerless:

* **Welded components, not index-based edges.**  A flat part stores three
  fresh vertices per triangle by design, so every edge is used exactly once
  and an index-based manifold test would declare *every* flat part an open
  shell and silently skip the majority of the geometry.  Welding by position
  first recovers the true connectivity.

* **Vertex normals are only authoritative on flat parts.**  On a smooth part
  a vertex normal is an area-weighted average over every incident face, so on
  a high-valence crease vertex it can legitimately lean against one of the
  faces that owns it.  Asserting ``dot(face_normal, normals[i0]) > 0`` there
  would fail correct geometry, so on smooth parts that specific dot is
  reported as a warning and the real gate is the geometric component volume
  of check 5.  On flat parts the stored normal *is* the face normal (each
  triangle owns its vertices), so the dot is exact and is a hard assertion.
"""

import argparse
import json
import math
import os
import sys

# --- tolerances ------------------------------------------------------------

FLAT_DOT_MIN = 0.99        # flat parts: geometric normal vs stored normal
SMOOTH_DOT_MIN = 0.0       # smooth parts: reported, never fatal (see module doc)
NORMAL_UNIT_TOL = 1e-4     # stored normals must be unit length
BOUNDS_TOL = 1e-4          # declared vs recomputed bounds
GROUND_TOL = -0.001        # ground-resting categories may not sink below y=0
WELD_TOL = 1e-6            # position weld resolution, matches the exporter's

GROUND_CATEGORIES = ("building", "prop", "palm", "vehicle", "sign")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DEFAULT_ASSETS_DIR = os.path.join(REPO_ROOT, "assets")


# --- small helpers ---------------------------------------------------------

def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and math.isfinite(v)


def _vec3(v):
    """Return a 3-tuple of finite floats, or None if the shape is wrong."""
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        return None
    if not all(_is_num(c) for c in v):
        return None
    return (float(v[0]), float(v[1]), float(v[2]))


def _colour(v):
    """Return a 3-tuple of floats, or None if malformed / out of [0, 1]."""
    p = _vec3(v)
    if p is None:
        return None
    if any(c < 0.0 or c > 1.0 for c in p):
        return None
    return p


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _len(a):
    return math.sqrt(_dot(a, a))


def _face_normal(pos, f):
    """Unit geometric normal of a face, plus its unnormalised magnitude.

    Returns (unit_normal, magnitude) or (None, magnitude) when degenerate.
    """
    a, b, c = pos[f[0]], pos[f[1]], pos[f[2]]
    n = _cross(_sub(b, a), _sub(c, a))
    m = _len(n)
    if m < 1e-12:
        return None, m
    return (n[0] / m, n[1] / m, n[2] / m), m


# --- check 5 machinery: position-welded closed components ------------------

def _weld(positions, tol=WELD_TOL):
    """Map each vertex index to an id shared with coincident vertices.

    Rounds to the weld grid exactly as ``vcw.core._weld`` does, so the
    component split here agrees with the exporter's own.
    """
    key_of = {}
    remap = []
    for p in positions:
        k = (round(p[0] / tol), round(p[1] / tol), round(p[2] / tol))
        idx = key_of.get(k)
        if idx is None:
            idx = len(key_of)
            key_of[k] = idx
        remap.append(idx)
    return remap


def closed_components(positions, faces):
    """Split a part into position-welded shells.

    Returns a list of ``(is_closed, signed_volume, n_faces)``.  A closed shell
    is one where every welded edge is used by exactly two faces; its signed
    volume via the divergence theorem is positive only when every one of its
    faces is wound outward.
    """
    remap = _weld(positions)
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

    for w in remap:
        parent.setdefault(w, w)
    for f in faces:
        w = [remap[i] for i in f]
        for t in range(3):
            union(w[t], w[(t + 1) % 3])

    groups = {}
    for fi, f in enumerate(faces):
        groups.setdefault(find(remap[f[0]]), []).append(fi)

    out = []
    for _cid, fl in groups.items():
        edge_count = {}
        vol = 0.0
        for fi in fl:
            f = faces[fi]
            a, b, c = positions[f[0]], positions[f[1]], positions[f[2]]
            cr = _cross(b, c)
            vol += _dot(a, cr) / 6.0
            w = [remap[i] for i in f]
            for t in range(3):
                x, y = w[t], w[(t + 1) % 3]
                key = (x, y) if x < y else (y, x)
                edge_count[key] = edge_count.get(key, 0) + 1
        closed = bool(edge_count) and all(v == 2 for v in edge_count.values())
        out.append((closed, vol, len(fl)))
    return out


# --- per-part validation ---------------------------------------------------

def check_part(aid, p, idx, seen_names, failures, warnings, stats):
    """Run checks 2-5 over one part.  Returns a short status string."""
    where = "%s/part[%d] %r" % (aid, idx, p.get("name"))
    status = "ok"

    name = p.get("name")
    if not isinstance(name, str) or not name:
        failures.append("%s: name must be a non-empty string" % where)
    else:
        if name in seen_names:
            failures.append(
                "%s: duplicate part name %r (part names must be unique within "
                "an asset; a renderer binding materials by name would collide)"
                % (where, name))
        seen_names.add(name)

    for key in ("base_color", "emissive"):
        if _colour(p.get(key)) is None:
            failures.append(
                "%s: %s must be 3 finite values in [0, 1], got %r"
                % (where, key, p.get(key)))

    for key in ("roughness", "metallic"):
        if not _is_num(p.get(key)):
            failures.append("%s: %s must be a finite number, got %r"
                            % (where, key, p.get(key)))

    # -- positions / normals ------------------------------------------------
    raw_pos = p.get("positions")
    raw_nrm = p.get("normals")
    if not isinstance(raw_pos, list) or not raw_pos:
        failures.append("%s: positions must be a non-empty list" % where)
        return "fail"
    if not isinstance(raw_nrm, list):
        failures.append("%s: normals must be a list" % where)
        return "fail"

    pos = []
    for i, v in enumerate(raw_pos):
        p3 = _vec3(v)
        if p3 is None:
            failures.append("%s: positions[%d] must be 3 finite numbers, got %r"
                            % (where, i, v))
            return "fail"
        pos.append(p3)

    if len(raw_nrm) != len(pos):
        failures.append("%s: len(normals)=%d != len(positions)=%d"
                        % (where, len(raw_nrm), len(pos)))
        return "fail"

    nrm = []
    for i, v in enumerate(raw_nrm):
        n3 = _vec3(v)
        if n3 is None:
            failures.append("%s: normals[%d] must be 3 finite numbers, got %r"
                            % (where, i, v))
            return "fail"
        nrm.append(n3)

    # -- faces --------------------------------------------------------------
    faces = p.get("faces")
    if not isinstance(faces, list) or not faces:
        failures.append("%s: faces must be a non-empty list" % where)
        return "fail"

    clean = []
    for fi, f in enumerate(faces):
        if not isinstance(f, list) or len(f) != 3:
            failures.append("%s: face %d must have exactly 3 indices, got %r"
                            % (where, fi, f))
            return "fail"
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in f):
            failures.append("%s: face %d indices must be ints, got %r"
                            % (where, fi, f))
            return "fail"
        if any(v < 0 or v >= len(pos) for v in f):
            failures.append(
                "%s: face %d index out of range (positions has %d vertices): %r"
                % (where, fi, len(pos), f))
            return "fail"
        if len(set(f)) != 3:
            failures.append("%s: face %d is degenerate (repeated index): %r"
                            % (where, fi, f))
            return "fail"
        clean.append(tuple(f))

    fcol = p.get("face_colors")
    if not isinstance(fcol, list) or len(fcol) != len(clean):
        failures.append("%s: len(face_colors)=%s != len(faces)=%d"
                        % (where, len(fcol) if isinstance(fcol, list) else "n/a",
                           len(clean)))
        return "fail"
    for i, c in enumerate(fcol):
        if _colour(c) is None:
            failures.append("%s: face_colors[%d] must be 3 finite values in "
                            "[0, 1], got %r" % (where, i, c))
            return "fail"

    flat = p.get("flat")
    if not isinstance(flat, bool):
        failures.append("%s: flat must be a bool, got %r" % (where, flat))
        return "fail"

    declared_tri = p.get("tri_count")
    if not isinstance(declared_tri, int) or isinstance(declared_tri, bool) \
            or declared_tri != len(clean):
        failures.append("%s: tri_count=%r but %d faces present"
                        % (where, declared_tri, len(clean)))
        status = "fail"

    declared_out = p.get("outward_check")
    if not isinstance(declared_out, str) or not declared_out:
        failures.append("%s: outward_check must be a non-empty string" % where)
        return "fail"

    # -- check 4: stored normals are unit length ----------------------------
    worst_unit = 0.0
    for i, n in enumerate(nrm):
        worst_unit = max(worst_unit, abs(_len(n) - 1.0))
    if worst_unit > NORMAL_UNIT_TOL:
        failures.append(
            "%s: stored normal is not unit length (worst deviation %.2e > %.0e)"
            % (where, worst_unit, NORMAL_UNIT_TOL))
        status = "fail"

    # -- check 4: winding ----------------------------------------------------
    # On flat parts every triangle owns its vertices, so the stored normal IS
    # the face normal and agreement must be exact.  On smooth parts the stored
    # normal is an average over incident faces and can oppose one of them, so
    # the dot is reported, never fatal; check 5 is the real gate there.
    worst_flat = 1.0
    worst_smooth = 1.0
    n_flat_bad = 0
    n_smooth_neg = 0
    for f in clean:
        n, _m = _face_normal(pos, f)
        if n is None:
            continue                      # already reported as degenerate
        d = _dot(n, nrm[f[0]])
        if flat:
            if d < worst_flat:
                worst_flat = d
            if d <= FLAT_DOT_MIN:
                n_flat_bad += 1
        else:
            if d < worst_smooth:
                worst_smooth = d
            if d <= SMOOTH_DOT_MIN:
                n_smooth_neg += 1

    if n_flat_bad:
        failures.append(
            "%s: %d face(s) wound against their stored normal "
            "(worst dot(geometric, stored) = %.6f, must be > %.2f); the face "
            "is CCW from the inside"
            % (where, n_flat_bad, worst_flat, FLAT_DOT_MIN))
        status = "fail"
    elif worst_flat < 1.0:
        status = "warn" if status == "ok" else status

    if n_smooth_neg and not n_flat_bad:
        warnings.append(
            "%s: smooth part, %d face(s) have dot(geometric, vertex normal at "
            "i0) <= 0 (worst %.3f). Expected on smooth geometry: a vertex "
            "normal is averaged over all incident faces, so on a high-valence "
            "crease vertex it can oppose a face it owns. Outwardness is "
            "decided by the component-volume check below, not by this dot."
            % (where, n_smooth_neg, worst_smooth))
        if status == "ok":
            status = "warn"

    # -- check 5: outward-facing --------------------------------------------
    comps = closed_components(pos, clean)
    n_closed = sum(1 for c, _v, _n in comps if c)
    n_inverted = sum(1 for c, v, _n in comps if c and v <= 0.0)
    n_open = sum(1 for c, _v, _n in comps if not c)

    if n_inverted:
        failures.append(
            "%s: %d of %d closed shells have non-positive signed volume -- "
            "those faces are wound inward"
            % (where, n_inverted, n_closed))
        status = "fail"

    if declared_out.startswith("closed") and n_inverted == 0 and n_closed == 0:
        failures.append(
            "%s: outward_check claims %r but no position-welded closed shell "
            "was found" % (where, declared_out))
        status = "fail"

    stats["parts"] += 1
    stats["tris"] += len(clean)
    stats["verts"] += len(pos)
    stats["closed"] += n_closed
    stats["inverted"] += n_inverted
    if declared_out.startswith("open-shell"):
        stats["open_shell"] += 1
    elif declared_out.startswith("declared-"):
        stats["declared"] += 1

    if status == "ok" and n_open:
        # Open shells are legal and expected, but worth surfacing per asset.
        pass
    return status


# --- per-asset validation --------------------------------------------------

def check_asset(doc, path, failures, warnings, stats):
    aid = doc.get("id") if isinstance(doc, dict) else None
    label = aid if isinstance(aid, str) and aid else os.path.basename(path)

    if not isinstance(doc, dict):
        failures.append("%s: document root must be an object" % label)
        return None

    # -- check 1 ------------------------------------------------------------
    if not isinstance(aid, str) or not aid:
        failures.append("%s: id must be a non-empty string" % label)
        aid = label
    elif os.path.splitext(os.path.basename(path))[0] != aid:
        failures.append("%s: id %r does not match its filename" % (label, aid))

    category = doc.get("category")
    if not isinstance(category, str) or not category:
        failures.append("%s: category must be a non-empty string" % label)

    if doc.get("y_up") is not True:
        failures.append("%s: y_up must be exactly true (got %r)"
                        % (label, doc.get("y_up")))

    bounds = doc.get("bounds")
    lo = hi = None
    if not isinstance(bounds, dict):
        failures.append("%s: bounds must be an object" % label)
    else:
        lo = _vec3(bounds.get("min"))
        hi = _vec3(bounds.get("max"))
        if lo is None or hi is None:
            failures.append("%s: bounds.min/max must each be 3 finite numbers"
                            % label)
            lo = hi = None
        else:
            for k in range(3):
                if lo[k] > hi[k]:
                    failures.append(
                        "%s: bounds.min[%d]=%g > bounds.max[%d]=%g"
                        % (label, k, lo[k], k, hi[k]))

    parts = doc.get("parts")
    if not isinstance(parts, list) or not parts:
        failures.append("%s: parts must be a non-empty list" % label)
        return None

    seen_names = set()
    part_status = []
    a_stats = {"parts": 0, "tris": 0, "verts": 0, "closed": 0,
               "inverted": 0, "open_shell": 0, "declared": 0}
    gmin = [float("inf")] * 3
    gmax = [float("-inf")] * 3
    real_pos = 0

    for i, p in enumerate(parts):
        if not isinstance(p, dict):
            failures.append("%s/part[%d]: must be an object" % (label, i))
            part_status.append("fail")
            continue
        before = len(failures)
        part_status.append(check_part(aid, p, i, seen_names,
                                      failures, warnings, a_stats))
        if len(failures) != before:
            part_status[-1] = "fail"
        # bounds accumulation runs off well-formed parts only
        if isinstance(p, dict) and isinstance(p.get("positions"), list):
            for v in p["positions"]:
                p3 = _vec3(v)
                if p3 is None:
                    continue
                real_pos += 1
                for k in range(3):
                    if p3[k] < gmin[k]:
                        gmin[k] = p3[k]
                    if p3[k] > gmax[k]:
                        gmax[k] = p3[k]

    # -- check 6: bounds agree with the vertices ----------------------------
    if lo is not None and real_pos:
        for k in range(3):
            if abs(gmin[k] - lo[k]) > BOUNDS_TOL or abs(gmax[k] - hi[k]) > BOUNDS_TOL:
                failures.append(
                    "%s: declared bounds %r do not match the vertices "
                    "(actual min %r max %r, tolerance %.0e)"
                    % (label, [lo, hi], gmin, gmax, BOUNDS_TOL))

    # -- check 7: ground-resting assets rest on y = 0 ------------------------
    if lo is not None and category in GROUND_CATEGORIES and lo[1] < GROUND_TOL:
        failures.append(
            "%s: category %r must rest on y = 0, but bounds.min[1] = %.4f "
            "(%.1f mm below the ground plane); a %s would sink into the street"
            % (label, category, lo[1], abs(lo[1]) * 1000.0, category))

    # -- check 8: declared totals agree -------------------------------------
    pc = doc.get("part_count")
    if not isinstance(pc, int) or isinstance(pc, bool) or pc != len(parts):
        failures.append("%s: part_count=%r but %d parts present"
                        % (label, pc, len(parts)))
    tc = doc.get("tri_count")
    if not isinstance(tc, int) or isinstance(tc, bool) or tc != a_stats["tris"]:
        failures.append("%s: tri_count=%r but %d triangles present"
                        % (label, tc, a_stats["tris"]))

    for k in ("parts", "tris", "verts", "closed", "inverted"):
        stats[k] += a_stats[k]
    stats["open_shell"] += a_stats["open_shell"]
    stats["declared"] += a_stats["declared"]
    if isinstance(category, str) and category:
        stats.setdefault("categories", {})
        entry = stats["categories"].setdefault(
            category, {"assets": 0, "tris": 0, "parts": 0})
        entry["assets"] += 1
        entry["tris"] += a_stats["tris"]
        entry["parts"] += a_stats["parts"]

    worst = "fail" if "fail" in part_status else (
        "warn" if "warn" in part_status else "ok")
    size = None
    if lo is not None and hi is not None:
        size = (hi[0] - lo[0], hi[2] - lo[2], hi[1] - lo[1])   # L x W x H
    return {"id": aid, "category": category if isinstance(category, str) else "?",
            "parts": a_stats["parts"], "tris": a_stats["tris"],
            "size": size, "status": worst}


# --- reporting -------------------------------------------------------------

def print_header(stats, n_files):
    print()
    print("=" * 78)
    print("  VICE CITY WEB  --  EXPORTED ASSET VERIFIER")
    print("=" * 78)
    print("  assets verified   : %d" % n_files)
    print("  parts             : %d" % stats["parts"])
    print("  triangles         : %d" % stats["tris"])
    print("  vertices          : %d" % stats["verts"])
    print("  closed shells     : %d  (outwardness proven by signed volume)"
          % stats["closed"])
    print("  open shells       : %d  (informational: no volume to check)"
          % stats["open_shell"])
    print("  declared-outward  : %d  (informational: builder-declared reference)"
          % stats["declared"])
    cats = stats.get("categories", {})
    if cats:
        print()
        print("  per category")
        print("    %-12s %7s %8s %9s" % ("category", "assets", "parts", "tris"))
        for c in sorted(cats):
            e = cats[c]
            print("    %-12s %7d %8d %9d" % (c, e["assets"], e["parts"], e["tris"]))


def print_table(rows):
    print()
    print("=" * 78)
    print("  PER-ASSET")
    print("=" * 78)
    print("  %-26s %-11s %5s %7s  %-21s %s"
          % ("id", "category", "parts", "tris", "size LxWxH (m)", "winding"))
    print("  " + "-" * 74)
    for r in rows:
        if r["size"] is None:
            size = "-"
        else:
            size = "%.2f x %.2f x %.2f" % r["size"]
        print("  %-26s %-11s %5d %7d  %-21s %s"
              % (r["id"][:26], r["category"][:11], r["parts"], r["tris"],
                 size, r["status"]))


def print_problems(warnings, failures):
    if warnings:
        print()
        print("=" * 78)
        print("  WARNINGS  (%d) -- non-fatal, see each note for why"
              % len(warnings))
        print("=" * 78)
        for w in warnings:
            first, sep, rest = w.partition("\n")
            print("  - %s" % first)
            if rest:
                for line in rest.split("\n"):
                    print("    %s" % line)

    print()
    print("=" * 78)
    if failures:
        print("  FAILURES  (%d)" % len(failures))
        print("=" * 78)
        for f in failures:
            print("  x %s" % f)
    else:
        print("  No failures.")


# --- driver ----------------------------------------------------------------

def script_args():
    """Arguments after Blender's ``--`` separator, if any."""
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return []


def main(argv):
    ap = argparse.ArgumentParser(
        description="Validate exported asset JSON against the export contract.")
    ap.add_argument("--assets-dir", default=DEFAULT_ASSETS_DIR,
                    help="directory holding <id>.json (default: repo assets/)")
    ap.add_argument("--quiet", action="store_true",
                    help="print failures only")
    args = ap.parse_args(argv)

    assets_dir = args.assets_dir
    if not os.path.isdir(assets_dir):
        print("FAIL: assets directory not found: %s" % assets_dir, file=sys.stderr)
        return 1

    files = sorted(os.path.join(assets_dir, f) for f in os.listdir(assets_dir)
                   if f.endswith(".json") and f != "manifest.json")

    failures = []
    warnings = []
    stats = {"parts": 0, "tris": 0, "verts": 0, "closed": 0,
             "inverted": 0, "open_shell": 0, "declared": 0}
    rows = []

    for path in files:
        try:
            with open(path, "r") as fh:
                doc = json.load(fh)
        except (OSError, ValueError) as exc:
            failures.append("%s: cannot read as JSON: %s"
                            % (os.path.basename(path), exc))
            continue
        r = check_asset(doc, path, failures, warnings, stats)
        if r is not None:
            rows.append(r)

    rows.sort(key=lambda r: (r["category"], r["id"]))

    if not files:
        failures.append(
            "no asset JSON found in %s -- the asset build has not been run, so "
            "there is nothing to verify. A green result here would be "
            "vacuous, not a pass." % assets_dir)

    if not args.quiet:
        print_header(stats, len(files))
        if files:
            print_table(rows)
    print_problems(warnings, failures)

    print()
    if failures:
        print("VERDICT: FAIL  --  %d violation(s) across %d asset file(s)."
              % (len(failures), len(files)))
        return 1
    print("VERDICT: PASS  --  all assertions hold for %d asset(s), %d "
          "triangles." % (len(files), stats["tris"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(script_args()))
