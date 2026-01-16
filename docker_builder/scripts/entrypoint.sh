#!/bin/bash
# Entrypoint script for Motphys Benchmark Base Image
# Displays system information and available environments on startup

set -e

# Display system banner
echo ""
echo "=========================================="
echo "Motphys Benchmark Base Image"
echo "=========================================="
echo "Container started at: $(date)"
echo ""

# Display GPU info if available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (GPU info unavailable)"
    echo ""
fi

echo "=========================================="
echo "Pre-configured uv Environments"
echo "=========================================="
echo ""
echo "Available environments at /workspace/env/:"
echo ""

# Check and display isaacgym environment
if [ -d "/workspace/env/isaacgym/.venv" ]; then
    ISAACGYM_PYTHON=$(/workspace/env/isaacgym/.venv/bin/python --version 2>/dev/null || echo "Unknown")
    echo "  [OK] isaacgym - $ISAACGYM_PYTHON"
    echo "       Usage: uv_isaacgym run python your_script.py"
else
    echo "  [--] isaacgym - Not initialized"
fi
echo ""

# Check and display motrixsim environment
if [ -d "/workspace/env/motrixsim/.venv" ]; then
    MOTRIXSIM_PYTHON=$(/workspace/env/motrixsim/.venv/bin/python --version 2>/dev/null || echo "Unknown")
    echo "  [OK] motrixsim - $MOTRIXSIM_PYTHON"
    echo "       Usage: uv_motrixsim run python your_script.py"
else
    echo "  [--] motrixsim - Not initialized"
fi
echo ""

# Check and display genesis environment
if [ -d "/workspace/env/genesis/.venv" ]; then
    GENESIS_PYTHON=$(/workspace/env/genesis/.venv/bin/python --version 2>/dev/null || echo "Unknown")
    echo "  [OK] genesis - $GENESIS_PYTHON"
    echo "       Usage: uv_genesis run python your_script.py"
else
    echo "  [--] genesis - Not initialized"
fi
echo ""

# Check and display mjwarp environment
if [ -d "/workspace/env/mjwarp/.venv" ]; then
    MJWARP_PYTHON=$(/workspace/env/mjwarp/.venv/bin/python --version 2>/dev/null || echo "Unknown")
    echo "  [OK] mjwarp - $MJWARP_PYTHON"
    echo "       Usage: uv_mjwarp run python your_script.py"
else
    echo "  [--] mjwarp - Not initialized"
fi
echo ""

echo "=========================================="
echo "Additional Resources"
echo "=========================================="
echo ""
echo "IsaacGym source: /workspace/third_party/isaacgym/"
echo "  (For editable install: pip install -e /workspace/third_party/isaacgym/python)"
echo ""
echo "Pre-built wheels: /workspace/wheels/"
if [ -d "/workspace/wheels" ]; then
    ls /workspace/wheels/*.whl 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/^/  - /' || echo "  (no wheels found)"
fi
echo ""

echo "=========================================="
echo "VSCode Integration"
echo "=========================================="
echo ""
echo "To use these environments in VSCode:"
echo "  1. Open Command Palette (Ctrl+Shift+P)"
echo "  2. Select 'Python: Select Interpreter'"
echo "  3. Choose from:"
echo "     - /workspace/env/isaacgym/.venv/bin/python"
echo "     - /workspace/env/motrixsim/.venv/bin/python"
echo "     - /workspace/env/genesis/.venv/bin/python"
echo "     - /workspace/env/mjwarp/.venv/bin/python"
echo ""
echo "=========================================="
echo ""

# Execute the command passed to docker run (or default to bash)
exec "$@"
