#!/usr/bin/env bash
set -euo pipefail

# =============================
# Config (override if you want)
# =============================
SAM_DIR="${SAM_DIR:-$HOME/models/sam}"
SAM_FILE="sam_vit_h_4b8939.pth"
SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/${SAM_FILE}"

echo "SAM model directory: ${SAM_DIR}"
mkdir -p "${SAM_DIR}"

# =============================
# Install Python dependencies
# =============================
echo "Installing Python deps into current conda/venv..."
pip install --upgrade \
  gradio pillow numpy matplotlib pydantic ollama ultralytics opencv-python \
  segment-anything

# Optional: install PyTorch if missing (CPU wheels by default)
python - <<'PY' || {
  echo "Installing PyTorch (CPU) since it's not detected..."
  pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
}
import importlib, sys
try:
    importlib.import_module("torch")
    importlib.import_module("torchvision")
    print("PyTorch is already installed.")
except Exception as e:
    sys.exit(1)
PY

# =============================
# Pull Ollama vision models
# =============================
for model in "llama3.2-vision:latest" "llava:latest"; do
  echo "Pulling Ollama model: $model"
  ollama pull "$model" || {
    echo "Warning: failed to pull $model. Continuing..."
  }
done

# =============================
# Warm-up YOLO (download yolov8n.pt)
# =============================
python - <<'PY'
from ultralytics import YOLO
try:
    model = YOLO("yolov8n.pt")  # auto-downloads if missing
    _ = model.model
    print("YOLO weights ready (yolov8n).")
except Exception as e:
    print("YOLO warm-up failed:", e)
PY

# =============================
# Download SAM ViT-H checkpoint
# =============================
DEST="${SAM_DIR}/${SAM_FILE}"
if [[ -f "${DEST}" ]]; then
  echo "SAM checkpoint already present: ${DEST}"
else
  echo "Downloading SAM ViT-H checkpoint to ${DEST} ..."
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 2 -o "${DEST}.tmp" "${SAM_URL}"
    mv "${DEST}.tmp" "${DEST}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${DEST}.tmp" "${SAM_URL}"
    mv "${DEST}.tmp" "${DEST}"
  else
    echo "Error: neither curl nor wget found. Install one and re-run." >&2
    exit 1
  fi
fi

# Quick sanity check: can we import SAM and see the file?
python - <<PY
import os
print("SAM .pth exists:", os.path.exists("${DEST}"), "->", "${DEST}")
try:
    import segment_anything as _sam  # noqa
    print("segment-anything import: OK")
except Exception as e:
    print("segment-anything import FAILED:", e)
PY

echo
echo "== Setup complete! =="
echo "SAM checkpoint: ${DEST}"
echo
echo "Make sure 'ollama serve' is running, then run:"
echo "  python visual_cot_video.py --port 7860"
echo
echo "In the Image tab, set:"
echo "  • SAM checkpoint path: ${DEST}"
echo "  • SAM model type: vit_h"
