#!/usr/bin/env bash
# 构建 WASM 并起一个静态服务器预览 www/。
#
#   tools/preview.sh            # 构建 + 起服务器(端口 8765)
#   tools/preview.sh --serve    # 跳过构建,只起服务器
#
# assets/ 会以符号链接的形式挂进 www/,这样 index.html 里的相对路径
# `assets/xxx.json` 在本地预览和 GitHub Pages 上完全一致。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${VCW_PREVIEW_PORT:-8765}"
TARGET_DIR="${CARGO_TARGET_DIR:-/Users/sqs/Library/Caches/cargo-target/vcweb}"
BIN_NAME="vice_city_web"

cd "$ROOT"

SERVE_ONLY=0
if [[ "${1:-}" == "--serve" ]]; then
  SERVE_ONLY=1
fi

if [[ "$SERVE_ONLY" -eq 0 ]]; then
  echo "==> cargo build (release, wasm32-unknown-unknown)"
  CARGO_TARGET_DIR="$TARGET_DIR" cargo build --release --target wasm32-unknown-unknown

  echo "==> wasm-bindgen --target web -> www/pkg/"
  wasm-bindgen \
    --target web \
    --no-typescript \
    --out-dir www/pkg \
    --out-name "$BIN_NAME" \
    "$TARGET_DIR/wasm32-unknown-unknown/release/${BIN_NAME}.wasm"

  ls -la www/pkg/
fi

# 让 www/assets/ 指向仓库里的 assets/,保持相对路径一致。
if [[ ! -e www/assets ]]; then
  ln -s ../assets www/assets
fi

# 释放已占用的端口。
if lsof -ti "tcp:$PORT" >/dev/null 2>&1; then
  echo "==> port $PORT busy, stopping previous server"
  kill "$(lsof -ti "tcp:$PORT")" || true
  sleep 1
fi

echo "==> serving www/ on http://localhost:$PORT/"
echo "    (Ctrl-C to stop)"
cd www
exec python3 -m http.server "$PORT" --bind 127.0.0.1
