#!/bin/bash
# Install private PyTorch and IsaacGym wheels for a uv project
# Usage: ./install_wheels.sh <project_path>
# Example: ./install_wheels.sh /workspace/genesis-speed-benchmark/project_isaacgym

set -e

PROJECT_PATH="${1:-/workspace/genesis-speed-benchmark/project_isaacgym}"
WHEELS_DIR="/workspace/wheels"

if [ ! -d "$PROJECT_PATH" ]; then
    echo "Error: Project directory not found: $PROJECT_PATH"
    exit 1
fi

echo "=========================================="
echo "Installing wheels to project: $PROJECT_PATH"
echo "=========================================="

cd "$PROJECT_PATH"

# Install private PyTorch wheels first
echo "Installing private PyTorch wheels..."
uv pip install "$WHEELS_DIR"/torch-*.whl
uv pip install "$WHEELS_DIR"/torchvision-*.whl

# Install IsaacGym wheel
echo "Installing IsaacGym wheel..."
uv pip install "$WHEELS_DIR"/isaacgym-*.whl

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo "Verify installation:"
echo "  uv run python -c 'import torch; print(f\"PyTorch {torch.__version__}\")'"
echo "  uv run python -c 'import isaacgym; print(\"IsaacGym OK\")'"
echo ""
