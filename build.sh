#!/usr/bin/env bash
# Render.com build script
# fastembed uses ONNX Runtime (no PyTorch) — fits Render free plan (512 MB).
# PyTorch is NOT installed on Render; GNN training runs locally only.

set -e

echo "==> Installing project dependencies..."
pip install -r requirements.txt

echo "==> Build complete."
