#!/bin/bash
# Universal PyTorch wheel builder with sm_120 (RTX 50-series) support
# Based on: https://blog.csdn.net/m0_56706433/article/details/148902144
# Can be run on host machine or inside Docker

set -e

# Configuration
PYTORCH_VERSION="${PYTORCH_VERSION:-v2.3.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-v0.18.1}"
PYTHON_CMD="${PYTHON_CMD:-python3.8}"
BUILD_DIR="${BUILD_DIR:-/tmp/pytorch_build}"
OUTPUT_DIR="${OUTPUT_DIR:-.}"
MAX_JOBS="${MAX_JOBS:-8}"

echo "=========================================="
echo "PyTorch sm_120 Wheel Builder"
echo "=========================================="
echo "Configuration:"
echo "  PyTorch:     $PYTORCH_VERSION"
echo "  TorchVision: $TORCHVISION_VERSION"
echo "  Python:      $PYTHON_CMD"
echo "  Build dir:   $BUILD_DIR"
echo "  Output dir:  $OUTPUT_DIR"
echo "  Max jobs:    $MAX_JOBS"
echo "=========================================="
echo ""

# Check Python availability
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    echo "ERROR: $PYTHON_CMD not found"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version)
echo "Using: $PYTHON_VERSION"
echo ""

# Create build directory
mkdir -p "$BUILD_DIR"
mkdir -p "$OUTPUT_DIR"
cd "$BUILD_DIR"

echo "Step 1/5: Installing build dependencies..."
$PYTHON_CMD -m pip install --no-cache-dir \
    setuptools \
    wheel \
    pyyaml \
    numpy \
    typing_extensions \
    cmake \
    ninja

echo ""
echo "Step 2/5: Checking PyTorch ($PYTORCH_VERSION)..."
if [ ! -d "$BUILD_DIR/pytorch" ]; then
    echo "Cloning PyTorch (this may take 5-10 minutes)..."
    git clone --branch "$PYTORCH_VERSION" --depth 1 \
        --recurse-submodules --shallow-submodules \
        https://github.com/pytorch/pytorch.git
    echo "✓ PyTorch cloned successfully"
else
    echo "✓ Using cached PyTorch source at $BUILD_DIR/pytorch"
    cd "$BUILD_DIR/pytorch"

    if [ "${SKIP_PYTORCH_RESET}" = "1" ]; then
        echo "⚠ SKIP_PYTORCH_RESET=1, skipping git reset (debug mode)"
    else
        echo "Resetting to clean state (git checkout + clean)..."
        # Restore all tracked files to their original state
        git checkout . 2>/dev/null || true
        # Remove ALL untracked files including those in .gitignore (build artifacts, etc.)
        git clean -fdx
        # Verify branch
        CURRENT_BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || git describe --tags --exact-match 2>/dev/null || echo "detached")
        echo "✓ Reset complete. Branch: $CURRENT_BRANCH"
    fi
fi

cd "$BUILD_DIR/pytorch"

echo ""
echo "Step 3/5: Applying sm_120 patch (following CSDN blog)..."

# Patch 1: cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake (line 227)
SELECT_ARCH_FILE="cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake"
if [ -f "$SELECT_ARCH_FILE" ]; then
    echo "Checking $SELECT_ARCH_FILE before patching..."
    echo "Lines 225-235 before patch:"
    sed -n '225,235p' "$SELECT_ARCH_FILE" | cat -n

    if grep -q "Unknown CUDA Architecture Name" "$SELECT_ARCH_FILE"; then
        echo ""
        echo "Applying SM 120 patch using sed..."

        # Patch 1: Replace the error message with SM 120 support
        # Find the line with "Unknown CUDA Architecture Name" and replace it
        sed -i '/message(SEND_ERROR "Unknown CUDA Architecture Name/c\        set(arch_bin 12.0)\n        set(arch_ptx 12.0)' "$SELECT_ARCH_FILE"
        echo "✓ Replaced error message with arch_bin/arch_ptx 12.0"

        # Patch 2: Remove the "arch_bin wasn't set" error check
        # Delete the if(NOT arch_bin) block (3 lines)
        sed -i '/if(NOT arch_bin)/,/endif()/d' "$SELECT_ARCH_FILE"
        echo "✓ Removed arch_bin validation check"

        echo ""
        echo "Verifying patch result..."
        echo "Lines 225-235 after patch:"
        sed -n '225,235p' "$SELECT_ARCH_FILE" | cat -n

        if grep -q "set(arch_bin 12.0)" "$SELECT_ARCH_FILE"; then
            echo "✓ PATCH SUCCESS: Found 'set(arch_bin 12.0)'"
        else
            echo "✗ PATCH FAILED: 'set(arch_bin 12.0)' not found!"
            exit 1
        fi
    else
        echo "✓ $SELECT_ARCH_FILE already patched or doesn't need patching"
    fi
else
    echo "✗ ERROR: $SELECT_ARCH_FILE not found!"
    exit 1
fi

# Patch 2: Dockerfile (if exists, for gencode)
if [ -f "Dockerfile" ]; then
    if ! grep -q "12.0" Dockerfile; then
        echo "Patching Dockerfile for SM 120..."
        sed -i 's/"3.5 5.2 6.0 6.1 7.0+PTX 8.0"/"3.5 5.2 6.0 6.1 7.0+PTX 8.0 12.0"/' Dockerfile
        echo "✓ Patched Dockerfile"
    fi
fi

echo ""
echo "=========================================="
echo "All patches applied successfully!"
echo "=========================================="

echo ""
echo "Step 4/5: Building PyTorch (2-4 hours)..."
echo "CUDA architectures: 8.0;8.6;8.9;9.0;12.0"
echo ""

export CMAKE_BUILD_TYPE=Release
export USE_CUDA=1
export USE_CUDNN=0
export USE_MKLDNN=0
export BUILD_TEST=0
export USE_FBGEMM=0
export USE_NNPACK=0
export USE_QNNPACK=0
export MAX_JOBS="$MAX_JOBS"
# Include 12.0 as per CSDN blog instructions
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"

# Enable ccache for faster rebuilds
if command -v ccache &> /dev/null; then
    # Use cache directory if BUILD_DIR is under /cache
    if [[ "$BUILD_DIR" == /cache/* ]]; then
        export CCACHE_DIR="/cache/ccache"
    else
        export CCACHE_DIR="$BUILD_DIR/.ccache"
    fi
    mkdir -p "$CCACHE_DIR"
    echo "Using ccache at: $CCACHE_DIR"
fi

$PYTHON_CMD setup.py bdist_wheel

echo ""
echo "Step 5/5: Building torchvision..."
cd "$BUILD_DIR"
if [ ! -d "$BUILD_DIR/vision" ]; then
    echo "Cloning torchvision..."
    git clone --branch "$TORCHVISION_VERSION" --depth 1 \
        https://github.com/pytorch/vision.git
    echo "✓ Torchvision cloned successfully"
else
    echo "✓ Using cached torchvision source"
    cd "$BUILD_DIR/vision"

    if [ "${SKIP_TORCHVISION_RESET}" = "1" ]; then
        echo "⚠ SKIP_TORCHVISION_RESET=1, skipping git reset (debug mode)"
    else
        echo "Resetting to clean state (git checkout + clean)..."
        git checkout . 2>/dev/null || true
        git clean -fdx
        echo "✓ Reset complete"
    fi
fi

cd "$BUILD_DIR/vision"

echo "Installing PyTorch wheel (required by torchvision build)..."
$PYTHON_CMD -m pip install --no-cache-dir "$BUILD_DIR/pytorch/dist/"torch-*.whl
echo "✓ PyTorch installed"

echo ""
echo "Building torchvision wheel..."
$PYTHON_CMD setup.py bdist_wheel

echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="

# Copy wheels to output directory
cp "$BUILD_DIR/pytorch/dist/"*.whl "$OUTPUT_DIR/"
cp "$BUILD_DIR/vision/dist/"*.whl "$OUTPUT_DIR/"

echo ""
echo "Wheels saved to: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/*.whl
