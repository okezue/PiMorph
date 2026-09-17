#!/usr/bin/env bash
# Runs ON the training instance (Deep Learning AMI, Ubuntu 22.04) as user ubuntu.
# No AWS credentials are placed on the instance: code and data arrive through presigned
# S3 URLs listed in urls.txt (one "<name> <url>" per line, uploaded next to this script).
# Usage: bash bootstrap.sh urls.txt
set -euo pipefail
URLS="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"

sudo apt-get update -qq && sudo apt-get install -y -qq libgl1 unzip >/dev/null
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

mkdir -p ~/work/PiMorph ~/work/downloads && cd ~/work
while read -r name url; do
  [ -z "$name" ] && continue
  echo "fetching $name"
  curl -sS -L --retry 3 -o "downloads/$name" "$url"
done < "$URLS"

cd ~/work/PiMorph
tar xzf ../downloads/code.tar.gz
mkdir -p data/tiles
for t in ../downloads/tiles_*.tar; do
  [ -f "$t" ] && tar xf "$t" -C data/tiles
done
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev,ml,complex]" "torch>=2.3" "torchvision>=0.18" "cellpose>=4.0" pycocotools imageio
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
du -sh data/tiles/* || true
echo "bootstrap done"
