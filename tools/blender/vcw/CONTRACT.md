# Builder authoring contract (NEON BAY asset pipeline)

Read this before writing a builder. It is the contract every category already
satisfies — follow it and your file will be picked up without touching anything
else.

## Shape of a builder module

File: `tools/blender/vcw/builders/<category>.py`
Exports exactly one function: `build_all() -> list[core.Asset]`

```python
from .. import core as C

def build_all():
    return [_sedan(), _coupe(), ...]
```

A builder MUST NOT import `bpy`, `json`, or `os`. It only builds geometry and
returns `Asset` objects. The exporter, the manifest and the renders are handled
by `build_assets.py`.

## Coordinate systems

**Authoring (Blender-style, Z-up):**
- `+X` right, `-Y` forward, `+Z` up, `1 unit = 1 metre`.
- Every asset is centred on `X = 0`, resting on `Z = 0`.
- The origin sits where a game would want to place the thing:
  buildings/vehicles/props → centre of the ground footprint;
  pedestrians → centre of the soles.

**Export (Y-up):** the exporter applies `(x, z, -y)` (det +1, handedness
preserved). So authoring `-Y` becomes JSON `+Z`. Vehicles and pedestrians face
authoring `-Y`.

## `core` primitives you should use

Vectors: `add sub mul dot cross normalize length lerp shade mix_color`
Shapes: `box cylinder cone sphere torus tube loft section_rings rounded_rect
        circle ribbon`
Parts: `Part Asset mirror_x rotate_z translate_part recolor_faces
       recolor_faces_where`
Checks: `signed_volume is_closed has_smooth_geometry`

Signatures worth knowing:

```python
C.box(m, size, center=(0,0,0), color=(r,g,b), colors={'+x':..,'-z':..,..})
C.cylinder(m, radius, height, seg=12, center=(0,0,0), color=col,
           radius_top=None, caps=True, smooth=False, axis='Z')
C.cone(m, radius, height, seg=10, center=(0,0,0), color=col,
       caps=True, radius_top=0.0)            # radius_top>0 => truncated
C.sphere(m, radius, seg_u=10, seg_v=6, center=(0,0,0), color=col,
         squash=1.0, smooth=False)
C.torus(m, R, r, seg_u=12, seg_v=6, center=(0,0,0), color=col, axis='Z')
C.tube(m, path, radius, seg=6, color=col, caps=True, smooth=False)
C.loft(m, rings, color, cap_start=True, cap_end=True, smooth=False)
C.rounded_rect(hx, hy, r, n=4)   # CCW 2D profile in XY
C.ribbon(m, pts, width, up=(0,0,1), color=col)
```

Every one of these appends triangles to `m` and returns `m`, so they chain.

### Two rules that are easy to get wrong

1. **`smooth=` must match the Part's `flat` flag.** A `Part` carries one `flat`
   boolean. Build smooth geometry only into a `Part(..., flat=False)`. The
   exporter rejects a flat part containing smooth geometry.
2. **Never put an open surface in a Part without telling it what "outward" is.**
   The exporter verifies every face points out of the asset, but for an open
   surface there is no interior to measure. Pass it explicitly:
   ```python
   p = C.Part("sign_face", flat=True, outward=("dir", (0, -1, 0)))   # faces -Y
   p = C.Part("lamp_shade", flat=True, outward=("point", (0, 0, 1.0)))  # around a point
   ```
   Leave `outward=None` for closed shells (boxes, cylinders, lofts) — the
   exporter uses signed volume then, which is exact.

## Parts vs colours

A `Part` is the unit of material. `base_color` is the part's material colour;
`face_colors[i]` is the colour of face `i` and may differ (windows, tyres,
lights). `emissive` is a part-level glow colour — all zeros means no glow.

Prefer splitting into a few semantically meaningful parts (`body`, `glass`,
`tyres`, `lights`) over one part with many colours. Each part becomes a
submesh with its own material at render time.

## Budget

- Total across ALL assets must stay under ~150,000 triangles.
- Hard budget per asset: **8,000 triangles**. Aim for 300–3,500.
- Prefer one `Part` holding many triangles over many single-triangle parts.
  A building has 132 parts and that is already more than it needs — reuse a
  single part and switch `face_colors` when faces differ only in colour.

## Verify before you claim it works

```bash
cd tools/blender
python3 -c "
import sys; sys.path.insert(0,'.')
from vcw.builders import <module>
from vcw import export
tot=0
for a in <module>.build_all():
    d = export.export_asset(a)   # raises on any invariant violation
    lo,hi = d['bounds']['min'], d['bounds']['max']
    tot += d['tri_count']
    assert d['tri_count'] <= 8000, (a.id, d['tri_count'])
    print('%-24s %5d tris  %5.2f x %5.2f x %5.2f m' % (a.id, d['tri_count'], hi[0]-lo[0], hi[2]-lo[2], hi[1]-lo[1]))
print('TOTAL', tot)
"
```

That script raises on: inverted faces, non-exact flat normals, degenerate
triangles, `face_colors`/`faces` length mismatch, and bounds that disagree with
the actual vertices. If it prints without raising, your builder is correct.

Style: no licence headers, no asset-store references. Original geometry only.
