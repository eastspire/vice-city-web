# Blender asset pipeline — NEON BAY

Everything in `assets/` is generated from the Python modules in this directory.
There are no imported models, no traced geometry, no downloaded textures, no
fonts and no logos: every triangle is computed from `vcw/core.py` at build time.

```
tools/blender/
├── build_assets.py        entry point -- writes assets/*.json + manifest.json
├── render_previews.py     renders assets/preview/*.png with Blender EEVEE
├── verify_assets.py       asserts every schema invariant; non-zero on failure
├── negative_control.py    proves the verifier actually rejects bad geometry
├── README.md              this file
└── vcw/
    ├── CONTRACT.md        authoring contract for adding a new category
    ├── core.py            geometry kernel (pure Python, no bpy)
    ├── export.py          asset -> Y-up JSON, with invariant enforcement
    ├── test_kernel.py     60+ kernel self-tests (runs outside Blender)
    └── builders/
        ├── buildings.py   12   Art Deco / pastel Miami
        ├── vehicles.py     5   sedan, coupe, pickup, police, taxi
        ├── pedestrians.py  4   suit, dress, overalls, streetwear (rigged)
        ├── props.py      12   street furniture
        ├── signs.py       8   neon signs
        ├── palms.py       3   tall, short, bushy
        ├── misc.py        8   beach props + helicopter/helipad/graffiti
        └── weapons.py     6   4 guns + 2 pickups
```

## Requirements

Blender **4.5 LTS** at `/Applications/Blender.app/Contents/MacOS/Blender`.
Nothing else — the builders are pure Python and import only the standard library
and `vcw`.

## Regenerating everything

Run from the repository root (`/Users/sqs/code/vice-city-web`).

```bash
BL=/Applications/Blender.app/Contents/MacOS/Blender

# 1. Build every asset  ->  assets/*.json, assets/manifest.json
$BL --background --python tools/blender/build_assets.py

# 2. Render one preview per category  ->  assets/preview/*.png
$BL --background --python tools/blender/render_previews.py

# 3. Verify every invariant            ->  exits non-zero on any violation
$BL --background --python tools/blender/verify_assets.py
```

All three exit `0` on success. Each prints a per-asset table and a totals block.

### Current state

| | |
|---|---|
| Assets | 58 across 10 categories |
| Triangles | 68,140 (budget: 150,000) |
| Parts | 447 |
| Vertices | 199,291 |
| JSON on disk | ~10.1 MB |
| Previews | 10 PNGs, all non-blank |

## Tests that need no Blender

```bash
cd tools/blender/vcw && python3 test_kernel.py      # 60+ geometry assertions
cd tools/blender        && python3 negative_control.py
```

`test_kernel.py` covers signed volume (outward winding), watertightness,
absolute positioning, flat-vs-smooth normal behaviour, and the Y-up export
matrix. `negative_control.py` deliberately inverts parts and confirms the
verifier rejects them — without it, "the verifier passes" could just mean "the
verifier does nothing".

## How the pipeline is structured

### Coordinate systems

Assets are **authored** in Blender-style Z-up (`+X` right, `−Y` forward,
`+Z` up, 1 unit = 1 m) because that is what Blender and the preview renderer
want. `export.py` converts to **Y-up** with `(x, z, −y)` — determinant `+1`, so
handedness and winding order survive untouched. So in the JSON:

- `y` is **height**
- `z` is **forward** (a vehicle's nose is at max `z`)
- every asset's origin sits where a game would place it (buildings and vehicles
  on the ground, pedestrians at the soles, weapons at the grip)

See `assets/SCHEMA.md` for the full contract.

### The flat-shaded representation

`flat: true` is the default aesthetic. It means **every triangle owns its three
vertices**, which is the only representation in which a shared vertex can carry
its own face's normal instead of an average. Primitives still share vertices
internally so shells stay watertight; `Mesh.flatten()` splits them at export
time, and the exporter asserts the result is bit-exact.

### How outwardness is proven

Signed volume via the divergence theorem, over **position-welded** closed
components:

```
vol = Σ dot(p0, cross(p1, p2)) / 6      > 0  ⟹  outward
```

Welding by position matters. An index-based manifold test calls every flat part
open — it stores three vertices per triangle — which silently skips the
outwardness check on the majority of the geometry. The welded test covers
2,796 of 2,827 components; the remaining 30 are genuinely open shells (a
one-sided decal, an awning underside) and are reported as such rather than
pretended to be verified.

This set has produced two real bugs that a render would not have caught: a
torus whose `axis="Y"` branch inverted the whole shell while still looking like
a torus, and a building ground-floor band sitting 5 mm below `z = 0`.

## Adding a new category

1. Read `vcw/CONTRACT.md`.
2. Add `vcw/builders/<name>.py` exposing `build_all() -> list[Asset]`.
3. Register the module in `build_assets.py` and `render_previews.py`.
4. Run the three commands above.

Check it in isolation first — this raises on any invariant violation:

```bash
cd tools/blender && python3 -c "
import sys; sys.path.insert(0,'.')
from vcw.builders import <name>
from vcw import export
for a in <name>.build_all():
    d = export.export_asset(a)     # raises on inverted faces, dup names, ...
    lo, hi = d['bounds']['min'], d['bounds']['max']
    print('%-24s %5d tris  %5.2f x %5.2f x %5.2f m'
          % (a.id, d['tri_count'], hi[0]-lo[0], hi[2]-lo[2], hi[1]-lo[1]))
"
```

Budget: ≤ 8,000 triangles per asset; the whole set must stay under 150,000.

## Blender 4.5 API notes

- The EEVEE engine id is **`BLENDER_EEVEE_NEXT`**. `"BLENDER_EEVEE"` does not
  exist in 4.5 and raises on assignment.
- Principled BSDF inputs are `Base Color`, `Emission Color`,
  `Emission Strength`, `Roughness`, `Metallic`. (`Clearcoat` → `Coat Weight`,
  `Transmission` → `Transmission Weight`.)
- **`blender --background --python` exits 0 on an uncaught exception.** Both
  entry points therefore catch, print the traceback and call `sys.exit(1)`
  explicitly — otherwise "fails loudly on error" is not actually true.
- Preview cameras are framed from the asset's **bounding sphere**
  (`dist = radius / sin(half_fov) * 1.25`), never a hard-coded distance: a
  hard-coded distance silently puts the camera inside the model once the model
  grows. Each render is then checked for blankness by pixel standard deviation,
  and a camera aimed away from the subject produces std ≈ 0.003, which the
  renderer reports as `BLANK` and fails on.

## Content note

Every asset is an original procedural model. The neon signs use generic English
words (HOTEL, BAR, DINER, PIZZA, TROPIC, CLUB, ARCADE, MOTEL) rendered with a
hand-authored stroke alphabet — no real trademark, brand, typeface or game
asset appears anywhere in this pipeline.
