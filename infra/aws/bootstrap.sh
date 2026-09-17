#!/usr/bin/env bash
# Runs ON the training instance (Deep Learning AMI, Ubuntu 22.04) as user ubuntu.
# Installs uv, unpacks the source tarball and syncs the training tiles from S3.
# Usage: bash bootstrap.sh <bucket> <code_tarball_key>
set -euo pipefail
BUCKET="$1"; CODE_KEY="$2"

sudo apt-get update -qq && sudo apt-get install -y -qq libgl1 unzip >/dev/null
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

mkdir -p ~/work/PiMorph && cd ~/work/PiMorph
aws s3 cp "s3://${BUCKET}/${CODE_KEY}" code.tar.gz --only-show-errors
tar xzf code.tar.gz && rm code.tar.gz
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev,ml,complex,aws]" "torch>=2.3" "torchvision>=0.18" "cellpose>=4.0" pycocotools imageio
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

mkdir -p data/tiles
aws s3 sync "s3://${BUCKET}/tiles/" data/tiles/ --only-show-errors
du -sh data/tiles/* || true
echo "bootstrap done"
