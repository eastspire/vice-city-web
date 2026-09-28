"""Build entry point: every vcw builder -> assets/*.json + assets/manifest.json.

Run headless:

    /Applications/Blender.app/Contents/MacOS/Blender --background \
        --python tools/blender/build_assets.py

This is the ONLY writer of the exported JSON.  It re-runs every builder in
``vcw/builders`` rather than reading anything off disk, so a manifest entry can
never describe geometry that differs from what the builders currently produce.

Invariant policy: this script never swallows an error.  A failed export
(export.ExportError), a duplicate asset id, a degenerate builder or a missing
part all propagate, and :func:`main` turns any of them into exit code 1.  The
reason is mechanical, not stylistic: Blender runs ``--python`` scripts inside a
wrapper that catches the exception, prints a traceback and STILL EXITS 0.  A
silent zero would hand a half-written asset set to whatever runs next.
"""

import os
import sys
import traceback

# Blender executes this file with the repository root -- not tools/blender -- on
# sys.path, so put our own directory there to make ``vcw`` importable.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from vcw import export                                          # noqa: E402
from vcw.builders import (                                     # noqa: E402
    buildings, markers, misc, palms, pedestrians, props, signs, vehicles,
    weapons,
)

# Every builder module, in a stable order so the report and the manifest assets
# array read the same way on every run.
BUILDER_MODULES = (
    ("buildings", buildings),
    ("vehicles", vehicles),
    ("pedestrians", pedestrians),
    ("props", props),
    ("signs", signs),
    ("palms", palms),
    ("misc", misc),
    ("weapons", weapons),
    ("markers", markers),
)

# CONTRACT.md: hard per-asset budget.
MAX_TRIS_PER_ASSET = 8000


def repo_root():
    """Repository root = two levels above tools/blender."""
    root = os.path.dirname(os.path.dirname(HERE))
    if not os.path.isdir(root):
        raise RuntimeError("repo root %s does not exist" % root)
    return root


def collect_assets():
    """Run every builder and return (assets, per-module counts).

    Raises on a duplicate asset id -- two builders claiming the same id means
    one JSON file would silently overwrite the other.
    """
    assets = []
    per_module = []
    owner = {}
    for name, module in BUILDER_MODULES:
        if not hasattr(module, "build_all"):
            raise RuntimeError("builder %s exposes no build_all()" % name)
        built = module.build_all()
        if not built:
            raise RuntimeError("builder %s returned no assets" % name)
        per_module.append((name, len(built)))
        for asset in built:
            if not asset.id:
                raise RuntimeError("builder %s produced an asset with no id"
                                   % name)
            if asset.id in owner:
                raise RuntimeError(
                    "duplicate asset id %r: produced by %r and %r"
                    % (asset.id, owner[asset.id], name))
            owner[asset.id] = name
            assets.append(asset)
    return assets, per_module


def build(assets_dir):
    """Export every asset, write the manifest, return (entries, report_rows)."""
    assets_dir = os.path.abspath(assets_dir)
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(os.path.join(assets_dir, "preview"), exist_ok=True)

    assets, per_module = collect_assets()

    entries = []
    report_rows = []
    total_bytes = 0
    for asset in assets:
        doc, path = export.write_asset(asset, assets_dir)
        size = os.path.getsize(path)
        total_bytes += size

        if doc["tri_count"] <= 0:
            raise RuntimeError("%s: exported zero triangles" % asset.id)
        if doc["tri_count"] > MAX_TRIS_PER_ASSET:
            raise RuntimeError("%s: %d triangles exceeds the %d budget"
                               % (asset.id, doc["tri_count"],
                                  MAX_TRIS_PER_ASSET))
        if doc["part_count"] <= 0:
            raise RuntimeError("%s: exported zero parts" % asset.id)

        lo, hi = asset.bounds()
        size_m = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
        entries.append({
            "id": asset.id,
            "category": asset.category,
            "file": os.path.basename(path),
            "tri_count": doc["tri_count"],
            "part_count": doc["part_count"],
            "bounds": doc["bounds"],
        })
        report_rows.append((asset.id, asset.category, doc["tri_count"],
                            doc["part_count"], size_m, size))

    _, manifest_path = export.write_manifest(entries, assets_dir)
    total_bytes += os.path.getsize(manifest_path)

    return entries, report_rows, per_module, total_bytes, manifest_path


def report(entries, report_rows, per_module, total_bytes, manifest_path,
           assets_dir):
    """Print the human-readable build report."""
    out = sys.stdout.write
    out("\n=== NEON BAY asset build ===\n\n")
    out("%-26s %-12s %7s %6s  %s\n"
        % ("asset", "category", "tris", "parts", "size (m)  W x D x H"))
    out("-" * 74 + "\n")
    for aid, cat, tris, parts, size_m, nbytes in report_rows:
        out("%-26s %-12s %7d %6d  %5.2f x %5.2f x %5.2f   %6.1f KB\n"
            % (aid, cat, tris, parts, size_m[0], size_m[1], size_m[2],
               nbytes / 1024.0))
    out("-" * 74 + "\n")

    total_tris = sum(r[2] for r in report_rows)
    out("\nper-category totals:\n")
    cats = {}
    for aid, cat, tris, parts, _s, _b in report_rows:
        t, c, p = cats.get(cat, (0, 0, 0))
        cats[cat] = (t + 1, c + tris, p + parts)
    for cat in sorted(cats):
        n, tris, parts = cats[cat]
        out("  %-12s %3d assets %7d tris %6d parts\n" % (cat, n, tris, parts))

    out("\nbuilders:\n")
    for name, n in per_module:
        out("  %-14s %3d assets\n" % (name, n))

    out("\nTOTALS\n")
    out("  assets           %d\n" % len(entries))
    out("  triangles        %d\n" % total_tris)
    out("  bytes written    %d (%.2f MB)\n" % (total_bytes,
                                               total_bytes / (1024.0 * 1024.0)))
    out("  manifest         %s\n" % manifest_path)
    out("  assets dir       %s\n" % os.path.abspath(assets_dir))
    out("\nBUILD OK: %d assets, %d triangles\n\n" % (len(entries), total_tris))
    sys.stdout.flush()


def main():
    root = repo_root()
    assets_dir = os.path.join(root, "assets")
    entries, report_rows, per_module, total_bytes, manifest_path = build(
        assets_dir)
    report(entries, report_rows, per_module, total_bytes, manifest_path,
           assets_dir)
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.write("\nBUILD FAILED -- assets/ may be incomplete\n")
        sys.stdout.flush()
        code = 1
    # Explicit: an uncaught exception under `blender --background --python`
    # exits 0, which would hide the failure from any caller checking $?.
    sys.exit(code)
