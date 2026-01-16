#!/bin/bash
# Build script for multi-simulator Docker environment
# Supports Isaac Gym, Genesis, and MJX with RTX 50-series (sm_120)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="docker.mp/motphys-bench-base"
IMAGE_TAG="latest"

echo "=========================================="
echo "Motphys Benchmark Base Image Builder"
echo "=========================================="
echo "Building Docker image: $IMAGE_NAME:$IMAGE_TAG"
echo ""
echo "This base image includes pre-initialized uv environments for:"
echo "  - isaacgym (Python 3.8)"
echo "  - motrixsim (Python 3.12)"
echo "  - genesis (Python 3.12)"
echo "  - mjwarp (Python 3.12)"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found. Please install Docker first."
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "WARNING: NVIDIA Docker runtime not available or GPU not detected"
    echo "Make sure you have:"
    echo "  1. NVIDIA GPU driver installed"
    echo "  2. NVIDIA Container Toolkit installed"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Building Docker image..."
echo "This may take 15-30 minutes (or 3-4 hours if building PyTorch)"
echo "Note: uv environments will be pre-initialized during build"
echo ""

# Build with BuildKit for better caching and performance
# Cache is managed by Docker BuildKit (use 'docker builder prune' to clean)
DOCKER_BUILDKIT=1 docker build \
    --tag "$IMAGE_NAME:$IMAGE_TAG" \
    --progress=plain \
    -f Dockerfile \
    ..

echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo "Image: $IMAGE_NAME:$IMAGE_TAG"
echo ""
echo "To run the container:"
echo "  ./run.sh"
echo ""
echo "Or manually:"
echo "  docker run --gpus all -it $IMAGE_NAME:$IMAGE_TAG"
