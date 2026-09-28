"""Negative control: does the exporter actually REJECT an inverted part?

Run from a fresh process (module caching in a long-lived interpreter will
serve stale code and give a false answer):
    python3 negative_control.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vcw import core as C            # noqa: E402
from vcw import export               # noqa: E402
from vcw.builders import props      # noqa: E402

fails = []


def case(name, fn):
    try:
        fn()
        print("  FAIL  %s -- was NOT rejected" % name)
        fails.append(name)
    except export.ExportError as e:
        print("  PASS  %s -- rejected: %s" % (name, str(e)[:110]))


def flip(part):
    part.mesh.faces = [(x, z, y) for (x, y, z) in part.mesh.faces]
    return part


def get(asset_id):
    return [a for a in props.build_all() if a.id == asset_id][0]


print("== baseline (must export cleanly) ==")
for aid in ("prop_traffic_cone", "prop_bench", "prop_streetlight"):
    a = get(aid)
    try:
        export.export_asset(a)
        print("  PASS  %s exports clean" % aid)
    except Exception as e:  # noqa: BLE001
        print("  FAIL  %s -> %s" % (aid, e))
        fails.append(aid)

print("== negative controls (each MUST be rejected) ==")

def export_flipped(asset_id, idx):
    """Flip one part's winding and export; must raise ExportError."""
    asset = get(asset_id)
    asset.parts[idx].mesh.faces = [(x, z, y)
                                   for (x, y, z) in asset.parts[idx].mesh.faces]
    return export.export_asset(asset)


# Flip every part of each asset in turn -- every single one must be rejected.
for aid in ("prop_traffic_cone", "prop_bench", "prop_streetlight",
            "prop_manhole_cover", "prop_trafficlight"):
    asset = get(aid)
    rejected = 0
    total = 0
    for i, p in enumerate(asset.parts):
        if not p.mesh.faces:
            continue
        total += 1
        try:
            export_flipped(aid, i)
        except export.ExportError:
            rejected += 1
        except Exception:      # noqa: BLE001
            pass
    ok = rejected == total and total > 0
    print("  %-4s %-20s %d/%d flipped parts rejected"
          % ("PASS" if ok else "FAIL", aid, rejected, total))
    if not ok:
        fails.append(aid)


print("== kernel-level inversion detection ==")
for name, mk in (("box", lambda m: C.box(m, (1, 1, 1))),
                 ("cylinder", lambda m: C.cylinder(m, 0.5, 2.0, 12)),
                 ("torus", lambda m: C.torus(m, 1.0, 0.3, 12, 8))):
    m = C.Mesh()
    mk(m)
    before = C.closed_components(m)
    m.faces = [(x, z, y) for (x, y, z) in m.faces]
    after = C.closed_components(m)
    ok = before[1] == 0 and after[1] > 0
    print("  %-4s %-9s before=%s after=%s  %s"
          % (name, "PASS" if ok else "FAIL", before, after,
             "" if ok else "<-- inversion NOT detected"))
    if not ok:
        fails.append(name)

print()
if fails:
    print("NEGATIVE CONTROL FAILURES: %s" % fails)
    sys.exit(1)
print("ALL NEGATIVE CONTROLS PASSED (inverted geometry IS rejected)")
