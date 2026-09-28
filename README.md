# vice-city-web

An original open-world browser game in the spirit of the classic 80s Miami crime
sandbox — every asset is procedurally modelled from scratch in Blender, and the
engine is [euv](https://github.com/euv-dev/euv) (Rust → WebAssembly).

No Rockstar / GTA assets, textures, audio, logos or fonts are used. Everything here
is original work.

## Stack

| Layer | Choice |
| --- | --- |
| 3D assets | Blender 4.5 LTS headless (`bpy`), procedural, low-poly flat-shaded |
| Asset format | JSON mesh (positions / normals / faces / per-face colours) |
| Language | Rust 2024 |
| UI + engine | `euv` / `euv-engine` 0.28 (virtual DOM, reactive signals, WebGL + Canvas2D backends) |
| Build target | `wasm32-unknown-unknown` |
| Deploy | GitHub Pages via GitHub Actions |

## Layout

```
assets/           generated meshes + manifest.json + SCHEMA.md
assets/preview/   Blender preview renders, one per category
tools/blender/    the bpy scripts that generate assets/
src/              the game
www/              static site shell (index.html) + built wasm bundle
.github/workflows/pages.yml   build + deploy pipeline
```

## Regenerating the assets

```bash
/Applications/Blender.app/Contents/MacOS/Blender \
  --background --python tools/blender/build_assets.py
```

## Building the game

```bash
cargo build --release --target wasm32-unknown-unknown
wasm-bindgen --target web --no-typescript \
  --out-dir www/pkg --out-name vice_city_web \
  target/wasm32-unknown-unknown/release/vice_city_web.wasm
```

Or just:

```bash
tools/preview.sh
```

which builds and serves `www/` on <http://localhost:8765>.

## Live

<https://eastspire.github.io/vice-city-web/>

## Licence

MIT.
