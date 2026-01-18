#!/bin/bash
# Run script for multi-simulator Docker environment WITHOUT IsaacGym

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="motphys-bench"
IMAGE_TAG="no-isaacgym"
CONTAINER_NAME="motphys-bench-no-isaacgym"

echo "=========================================="
echo "Motphys Benchmark Container (No IsaacGym)"
echo "=========================================="

# Check if container is already running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '$CONTAINER_NAME' already exists."
    read -p "Remove and recreate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker rm -f "$CONTAINER_NAME"
    else
        echo "Attaching to existing container..."
        docker start "$CONTAINER_NAME"
        docker exec -it "$CONTAINER_NAME" /bin/bash
        exit 0
    fi
fi

echo "Starting container: $CONTAINER_NAME"
echo ""

# Run container with GPU support
docker run --gpus all \
    --name "$CONTAINER_NAME" \
    --hostname motphys-bench \
    --interactive \
    --tty \
    --rm \
    --shm-size=16gb \
    --network host \
    -v "$SCRIPT_DIR/../:/workspace/host" \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    "$IMAGE_NAME:$IMAGE_TAG" \
    /bin/bash
