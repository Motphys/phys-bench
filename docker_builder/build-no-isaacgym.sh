#!/bin/bash
# Build script for multi-simulator Docker environment WITHOUT IsaacGym
# No PyTorch compilation required - much faster build time!

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="motphys-bench"
IMAGE_TAG="no-isaacgym"

echo "=========================================="
echo "Motphys Benchmark Image Builder (No IsaacGym)"
echo "=========================================="
echo "Building Docker image: $IMAGE_NAME:$IMAGE_TAG"
echo ""
echo "This image includes pre-initialized uv environments for:"
echo "  - motrixsim (Python 3.12)"
echo "  - genesis (Python 3.12)"
echo "  - mjwarp (Python 3.12)"
echo "  - isaacsim (Python 3.11)"
echo ""
echo "Excluded:"
echo "  - isaacgym (requires custom PyTorch compilation)"
echo ""
echo "Estimated build time: 15-30 minutes"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found. Please install Docker first."
    exit 1
fi


echo ""
echo "Building Docker image..."
echo "Note: uv environments will be pre-initialized during build"
echo ""

# Build with BuildKit for better caching and performance
DOCKER_BUILDKIT=1 docker build \
    --tag "$IMAGE_NAME:$IMAGE_TAG" \
    --progress=plain \
    -f Dockerfile.no-isaacgym \
    ..

echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo "Image: $IMAGE_NAME:$IMAGE_TAG"
echo ""
echo "To run the container:"
echo "  docker run --gpus all -it $IMAGE_NAME:$IMAGE_TAG"
echo ""
echo "Or use run-no-isaacgym.sh script"
echo ""
echo "Available engine aliases in container:"
echo "  uv_motrixsim  - MotrixSim benchmark"
echo "  uv_genesis    - Genesis benchmark"
echo "  uv_mjwarp     - MuJoCo Warp benchmark"
echo "  uv_isaacsim   - IsaacSim 5.0 benchmark"
