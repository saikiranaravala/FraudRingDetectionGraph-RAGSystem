#!/usr/bin/env bash
# Render.com build script
# Installs CPU-only PyTorch first (avoids the 2 GB CUDA wheel),
# then installs the rest of requirements.txt.

set -e

echo "==> Installing CPU-only PyTorch..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "==> Installing project dependencies..."
pip install -r requirements.txt

echo "==> Build complete."
