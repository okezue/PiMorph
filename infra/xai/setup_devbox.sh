#!/usr/bin/env bash
# Runs ON the xAI devbox (fou / h100-sandbox). Installs the PiMorph environment under
# /data/okebell/pimorph and fetches code, tiles and checkpoints (presigned URLs in urls.txt)
# plus the large public datasets directly from their sources.
# Usage: bash setup_devbox.sh urls.txt
set -euo pipefail
URLS="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
ROOT=/data/okebell/pimorph
mkdir -p "$ROOT/downloads" "$ROOT/PiMorph"
cd "$ROOT"

if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$PATH"

echo "== fetch presigned objects =="
while read -r name url; do
  [ -z "$name" ] && continue
  [ -f "downloads/$name" ] && { echo "have $name"; continue; }
  echo "fetching $name"; curl -sS -L --retry 3 -o "downloads/$name" "$url"
done < "$URLS"

echo "== code =="
cd "$ROOT/PiMorph" && tar xzf ../downloads/code.tar.gz && cd "$ROOT"
mkdir -p PiMorph/data/tiles PiMorph/models PiMorph/runs
for t in downloads/tiles_*.tar; do [ -f "$t" ] && tar xf "$t" -C PiMorph/data/tiles; done
for m in downloads/*.pt; do [ -f "$m" ] && cp "$m" PiMorph/models/; done
find PiMorph/data/tiles -name '._*' -delete || true

echo "== python env =="
cd "$ROOT/PiMorph"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev,ml,complex]" pycocotools imageio
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu128 "torch==2.7.*" "torchvision==0.22.*"
uv pip install --python .venv/bin/python "cellpose>=4.0"
.venv/bin/python -c "import torch; print('torch', torch.__version__, torch.version.cuda, 'gpus', torch.cuda.device_count(), torch.cuda.get_device_name(0))"

echo "== public datasets =="
mkdir -p data/LIVECell/images data/neurips_cellseg data/cornea_cells
LC=https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021
for f in livecell_coco_train.json livecell_coco_val.json livecell_coco_test.json; do
  [ -f "data/LIVECell/$f" ] || curl -sS -L -o "data/LIVECell/$f" "$LC/annotations/LIVECell/$f"
done
if [ ! -d data/LIVECell/images/images ] && [ -z "$(ls -A data/LIVECell/images 2>/dev/null)" ]; then
  curl -sS -L -o data/LIVECell/images.zip "$LC/images.zip" && (cd data/LIVECell/images && unzip -q ../images.zip) && rm data/LIVECell/images.zip
fi
Z=https://zenodo.org/records/10719375/files
for f in Training-labeled.zip Tuning.zip; do
  [ -d "data/neurips_cellseg/${f%.zip}" ] && continue
  curl -sS -L -o "data/neurips_cellseg/$f" "$Z/$f?download=1" && (cd data/neurips_cellseg && unzip -q "$f" && rm "$f")
done
ls data/neurips_cellseg
[ -d data/cornea_cells/labels ] || git clone -q --depth 1 https://github.com/svdeepak99/U-Net_Segmentation-Cornea_Cells.git data/cornea_cells
echo "LIVECell images: $(find data/LIVECell/images -name '*.tif' | wc -l)"
echo "NeurIPS labeled: $(ls data/neurips_cellseg/Training-labeled/images 2>/dev/null | wc -l) tuning: $(ls data/neurips_cellseg/Tuning/images 2>/dev/null | wc -l)"
echo "setup done"
