#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
EE4745_CACHE="${EE4745_CACHE:-$(cd .. && pwd)/ee4745_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$EE4745_CACHE/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$EE4745_CACHE/xdg}"
mkdir -p "$EE4745_CACHE" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
python verify_gradients.py
python train_fashion.py --cache-dir "$EE4745_CACHE" --output-dir . --threads 2 --device cpu
python pretrained_cam.py --cache-dir "$EE4745_CACHE" --output-dir . --threads 2
python make_report.py
