# Multi-Simulator Physics Benchmark Docker Environment

Lightweight Docker environment supporting multiple physics simulators including Isaac Gym, Genesis, MJX, and Motrixsim, with native support for NVIDIA RTX 50 series GPUs (sm_120).

## Design Philosophy

**Simple and Flexible Infrastructure**:
- Provides **two Python virtual environments** (py38/py312)
- **No pre-installed simulator dependencies** (managed by user projects)
- Provides pre-compiled PyTorch wheels (with sm_120 support)
- Allows users to freely manage dependencies in their projects

## Features

- ✅ **2 venvs**: py38 (Isaac Gym) / py312 (Genesis/MJX/others)
- ✅ **RTX 50 series support**: sm_120 compute capability (RTX 5090/5080/5070)
- ✅ **Lightweight**: No hardcoded dependencies at Docker layer
- ✅ **Pre-compiled PyTorch**: Avoid repeated compilation, accelerate deployment
- ✅ **Unified scripts**: Host and container share compilation scripts

## System Requirements

### Host Machine
- **Operating System**: Ubuntu 24.04 (recommended) or 22.04
- **GPU**: NVIDIA RTX 50 series (or 30/40 series)
- **Driver**: NVIDIA Driver >= 570 (for RTX 50)
- **Docker**: >= 20.10
- **NVIDIA Container Toolkit**: Installed
- **Disk Space**: 20GB (image) + 50GB (compilation cache, if compiling PyTorch)

### Environment Check
```bash
# Check GPU
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## Quick Start
```bash
cd docker_builder
./build.sh

./run.sh
```

Or use docker-compose:
```bash
docker compose up -d
docker exec -it benchmark-env bash
```


## Manual Setup
### 1. Prepare Resources (Optional)

#### Download Isaac Gym (Optional)
```bash
# Download from NVIDIA official website
# https://developer.nvidia.com/isaac-gym

cp ~/Downloads/IsaacGym_Preview_4_Package.tar.gz docker_builder/assets/
```

#### Compile PyTorch Wheel (Recommended)
build.sh will compile automatically

Or if you have a pre-compiled wheel, copy directly:
```bash
cp path/to/torch-*.whl docker_builder/wheels/
```

**Note**: If the wheels directory is empty, Docker will compile PyTorch inside the container.


## Usage

### Quick Test (without pyproject.toml)

```bash
# Activate py38 environment
py38

# Install dependencies
pip install /workspace/wheels/torch-*.whl numpy scipy

# Test Isaac Gym
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f'Compute capability: {cap[0]}.{cap[1]} (sm_{cap[0]}{cap[1]})')
"
```

```bash
# Activate py312 environment
py312

# Install Genesis
pip install genesis-world

# Test
python -c "import genesis as gs; print(gs.__version__)"
```

## Python Version Selection

Based on simulator support:

| Simulator | Python Support | Use venv |
|-----------|----------------|----------|
| Isaac Gym | 3.6-3.8 | **py38** |
| Genesis | 3.10-3.12 (3.13 not supported) | **py312** |
| MJX | 3.10-3.13 | **py312** |

### Container Structure

```
/workspace/
├── venvs/                     # Two pre-built venvs
│   ├── py38/                  # Python 3.8 (Isaac Gym)
│   └── py312/                 # Python 3.12 (Genesis/MJX)
│
├── wheels/                    # Pre-compiled wheels
│   ├── torch-*-cp38-*.whl
│   └── torchvision-*.whl
│
├── assets/                    # Resource files
│
└── host/                      # Mounted host directory
    └── (your project)
```

## Usage Examples

### Isaac Gym Project

```bash
# Create project
mkdir /workspace/host/isaacgym_test
cd /workspace/host/isaacgym_test

# Create pyproject.toml
cat > pyproject.toml << 'EOF'
[project]
name = "isaacgym-test"
requires-python = ">=3.8,<3.9"
dependencies = ["numpy<1.24", "scipy"]

[tool.uv]
find-links = ["/workspace/wheels"]
EOF

# Initialize environment
uv venv --python 3.8
uv sync

# Install PyTorch (uv will find wheel from find-links)
uv add torch torchvision

# Test
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### Genesis Project

```bash
# Create project
mkdir /workspace/host/genesis_test
cd /workspace/host/genesis_test

# Create pyproject.toml
cat > pyproject.toml << 'EOF'
[project]
name = "genesis-test"
requires-python = ">=3.12"
dependencies = ["genesis-world", "numpy", "matplotlib"]
EOF

# Initialize environment
uv venv --python 3.12
uv sync

# Test
uv run python -c "import genesis as gs; gs.init()"
```

### MJX Project

```bash
# Create project
mkdir /workspace/host/mjx_test
cd /workspace/host/mjx_test

# Create pyproject.toml
cat > pyproject.toml << 'EOF'
[project]
name = "mjx-test"
requires-python = ">=3.12"
dependencies = ["jax[cuda12]", "mujoco>=3.0", "mujoco-mjx>=3.0"]
EOF

# Initialize environment
uv venv --python 3.12
uv sync

# Test
uv run python -c "import jax; print(jax.devices())"
```

## Technical Details

### PyTorch sm_120 Patch
Based on [CSDN Blog](https://blog.csdn.net/m0_56706433/article/details/148902144):
- Modify `cmake/public/cuda.cmake`
- Add `-gencode arch=compute_120,code=sm_120`
- Set `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"` during compilation

### Why Not Use Conda?
- Using system Python directly in containers is more lightweight
- uv is faster (10-100x vs pip/conda)
- pyproject.toml is more aligned with Python standards
- Reduces image size (~5GB difference)

### Unified Compilation Script
`scripts/build_pytorch_wheel.sh` is used for both:
- Host compilation: `OUTPUT_DIR=./wheels ./scripts/build_pytorch_wheel.sh`
- Docker build: Dockerfile calls the same script

Avoids duplicate maintenance of compilation logic.

## Troubleshooting

### Build Fails: CUDA not found
Ensure using `nvidia/cuda:12.8.0-devel` base image (not runtime).

### Runtime: GPU not detected
```bash
# Check NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# If fails, reinstall toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Isaac Gym: Import Error
Ensure `assets/IsaacGym_Preview_4_Package.tar.gz` exists and is placed before building.

### PyTorch Compilation Timeout
Pre-compile on host machine:
```bash
cd docker_builder
OUTPUT_DIR=./wheels ./scripts/build_pytorch_wheel.sh
```

### Genesis: Python 3.13 Incompatible
Genesis does not support Python 3.13 (Taichi dependency issue). Use py312 environment.

## Performance Optimization

### Reduce Image Size
- Use multi-stage builds (implemented)
- Clean apt cache (implemented)
- Only install runtime CUDA libraries (final image already uses runtime)

### Accelerate Build
- Use Docker BuildKit (enabled in build.sh)
- Enable ccache (configured in compilation script)
- Use pre-compiled wheels

### Runtime Optimization
run.sh already includes optimization options:
- `--ipc host`: Shared memory acceleration
- `--network host`: Reduce network overhead
- `--ulimit memlock=-1`: Allow locking more memory

## References

- [Isaac Gym Official Documentation](https://developer.nvidia.com/isaac-gym)
- [Genesis Simulator](https://github.com/Genesis-Embodied-AI/Genesis)
- [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html)
- [PyTorch sm_120 Support (CSDN Blog)](https://blog.csdn.net/m0_56706433/article/details/148902144)
- [uv Documentation](https://docs.astral.sh/uv/)

## Python Version Support Research

- **Genesis**: [Python 3.10-3.12 supported, 3.13 not supported](https://github.com/Genesis-Embodied-AI/Genesis/issues/513)
- **MJX**: [Python 3.10-3.13 all supported](https://pypi.org/project/mujoco-mjx/)

## License

This Docker configuration is under MIT License. Each simulator follows its respective license.
