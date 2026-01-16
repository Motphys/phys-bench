#!/bin/bash
# Run script for multi-simulator Docker environment

set -e

IMAGE_NAME="docker.mp/motphys-bench-base"
IMAGE_TAG="latest"
CONTAINER_NAME="motphys-bench-base"

echo "=========================================="
echo "Starting Multi-Simulator Environment"
echo "=========================================="

# Check if image exists
if ! docker image inspect "$IMAGE_NAME:$IMAGE_TAG" &> /dev/null; then
    echo "ERROR: Docker image not found: $IMAGE_NAME:$IMAGE_TAG"
    echo "Please run './build.sh' first"
    exit 1
fi

# Check if container already running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '$CONTAINER_NAME' already exists."

    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container is running. Attaching..."
        docker exec -it "$CONTAINER_NAME" /bin/bash
    else
        echo "Container is stopped. Starting..."
        docker start "$CONTAINER_NAME"
        docker exec -it "$CONTAINER_NAME" /bin/bash
    fi
    exit 0
fi

# Get current directory for mounting
HOST_WORKDIR="$(pwd)/../"  # Mount parent directory

# Setup DISPLAY for X11/Wayland
if [ -z "$DISPLAY" ]; then
    echo "DISPLAY not set, detecting..."
    # Try common XWayland displays
    if [ -S /tmp/.X11-unix/X0 ]; then
        export DISPLAY=:0
        echo "Set DISPLAY=:0"
    elif [ -S /tmp/.X11-unix/X1 ]; then
        export DISPLAY=:1
        echo "Set DISPLAY=:1"
    else
        echo "Warning: No X11 display found, GUI may not work"
    fi
fi

# Allow X11 access for Docker
echo ""
echo "Setting up X11 access..."
xhost +local:docker > /dev/null 2>&1 || echo "Warning: xhost command failed (continuing anyway)"

echo ""
echo "Configuration:"
echo "  Image: $IMAGE_NAME:$IMAGE_TAG"
echo "  Container: $CONTAINER_NAME"
echo "  Mount: $HOST_WORKDIR -> /workspace/host"
echo "  Display: $DISPLAY"
echo ""

# Build docker run command
DOCKER_CMD="docker run -it --rm \
    --name $CONTAINER_NAME \
    --gpus all \
    --network host \
    --ipc host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $HOST_WORKDIR:/workspace/host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw"

# Add XAUTHORITY mount if file exists
if [ -f "$HOME/.Xauthority" ]; then
    DOCKER_CMD="$DOCKER_CMD -e XAUTHORITY=/root/.Xauthority -v $HOME/.Xauthority:/root/.Xauthority:ro"
    echo "  X11 Auth: $HOME/.Xauthority"
else
    echo "  X11 Auth: None (using xhost)"
fi

DOCKER_CMD="$DOCKER_CMD $IMAGE_NAME:$IMAGE_TAG"

# Run container
eval $DOCKER_CMD

echo ""
echo "Container exited."
