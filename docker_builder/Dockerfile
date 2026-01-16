# Base image for physics simulator benchmarks (MotrixSim, Genesis, MJWarp)
# Image: docker.mp/motphys-bench-base:latest
#
# Provides pre-initialized uv environments at /workspace/env/:
# - /workspace/env/isaacgym (Python 3.8)
# - /workspace/env/motrixsim (Python 3.12)
# - /workspace/env/genesis (Python 3.12)
# - /workspace/env/mjwarp (Python 3.12)
#
# Includes isaacgym source code at /workspace/third_party/isaacgym for editable installs
# Supports RTX 50-series (sm_120) via custom PyTorch build

# ==============================================================================
# Stage 1: PyTorch Builder (Python 3.8 + sm_120 support)
# ==============================================================================
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS pytorch-builder

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install build dependencies (exclude cmake, will install older version via pip)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    ccache \
    libopenblas-dev \
    libopenmpi-dev \
    ninja-build \
    software-properties-common \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.8 and pip (deadsnakes doesn't include ensurepip)
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install CMake 3.18 via pip (older version compatible with protobuf 2.x)
RUN python3.8 -m pip install --no-cache-dir cmake==3.18.4 && \
    cmake --version

# Copy build script and run it
# Cache bust: change SCRIPT_VERSION to force rebuild from this point
COPY docker_builder/scripts/build_pytorch_wheel.sh /tmp/build_pytorch_wheel.sh
RUN chmod +x /tmp/build_pytorch_wheel.sh

WORKDIR /tmp

# Build PyTorch with persistent cache (managed by Docker BuildKit)
# Use 'docker builder prune --filter id=pytorch-cache' to clean up cache
# Debug: Set --build-arg SKIP_PYTORCH_RESET=1 and/or SKIP_TORCHVISION_RESET=1 to skip git reset
ARG SKIP_PYTORCH_RESET=1
ARG SKIP_TORCHVISION_RESET=1
RUN --mount=type=cache,target=/cache,id=pytorch-cache \
    PYTHON_CMD=python3.8 \
    BUILD_DIR=/cache/pytorch_build \
    OUTPUT_DIR=/tmp/wheels \
    MAX_JOBS=16 \
    SKIP_PYTORCH_RESET=${SKIP_PYTORCH_RESET} \
    SKIP_TORCHVISION_RESET=${SKIP_TORCHVISION_RESET} \
    /tmp/build_pytorch_wheel.sh

# ==============================================================================
# Stage 2: Final Runtime Image
# ==============================================================================
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install system dependencies (including runtime libraries for PyTorch and OpenGL for Isaac Gym)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cuda-cupti-12-8 \
    curl \
    git \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa \
    libglvnd0 \
    libglew2.2 \
    libglfw3 \
    mesa-utils \
    libglib2.0-0 \
    libopenblas0 \
    libopenmpi3 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    software-properties-common \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.8 and 3.12
# Note: Python 3.12+ no longer has distutils (deprecated and removed)
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.8 python3.8-dev python3.8-distutils python3.8-venv \
    python3.12 python3.12-dev python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create workspace directory
WORKDIR /workspace

# Copy pre-built PyTorch wheels from builder stage
RUN mkdir -p /workspace/wheels
COPY --from=pytorch-builder /tmp/wheels/*.whl /workspace/wheels/

# Copy third_party (includes IsaacGym source code)
COPY docker_builder/third_party/ /workspace/third_party/

# Copy project configuration assets (pyproject.toml + uv.lock)
COPY docker_builder/assets/ /workspace/assets/

# Copy utility scripts
COPY docker_builder/scripts/ /workspace/scripts/
RUN chmod +x /workspace/scripts/*.sh

# Initialize uv environments for all projects
RUN mkdir -p /workspace/env

# Initialize isaacgym environment (Python 3.8)
RUN mkdir -p /workspace/env/isaacgym && \
    cp /workspace/assets/project_isaacgym/pyproject.toml /workspace/env/isaacgym/ && \
    cd /workspace/env/isaacgym && \
    uv sync --python 3.8

# Initialize other environments (Python 3.12)
RUN for proj in motrixsim genesis mjwarp; do \
    mkdir -p /workspace/env/$proj && \
    cp /workspace/assets/project_$proj/pyproject.toml /workspace/env/$proj/ && \
    cd /workspace/env/$proj && \
    uv sync --python 3.12; \
    done

# Setup uv project aliases (no need to activate, just run: uv_isaacgym run python xxx.py)
RUN echo 'alias uv_isaacgym="uv --project /workspace/env/isaacgym"' >> /root/.bashrc && \
    echo 'alias uv_motrixsim="uv --project /workspace/env/motrixsim"' >> /root/.bashrc && \
    echo 'alias uv_genesis="uv --project /workspace/env/genesis"' >> /root/.bashrc && \
    echo 'alias uv_mjwarp="uv --project /workspace/env/mjwarp"' >> /root/.bashrc

WORKDIR /workspace

# Use entrypoint script to display system info and run benchmarks on startup
ENTRYPOINT ["/workspace/scripts/entrypoint.sh"]
CMD ["/bin/bash"]
