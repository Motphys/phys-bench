# Benchmarks

This directory contains physics simulation benchmarks that compare the quality and accuracy of different physics engines using standardized robotic manipulation tasks.

## Grasp Benchmarks

The grasp benchmarks test a robotic arm's ability to pick up and hold objects while subjecting the gripper to shaking motions. This tests the stability and accuracy of contact physics, friction modeling, and constraint solving.

### Test Scenario

The benchmark implements a **Franka Panda robot arm** performing a pick-and-shake task:

1. **Initial Phase** (0-1s): Move from initial position to lifted position
2. **Approach Phase** (1-2s): Lower arm to object
3. **Grasp Phase** (2-3s): Close gripper on object
4. **Lift Phase** (3-4s): Lift object off the ground
5. **Shake Phase** (4-20s): Shake arm with random perturbations to test grasp stability
6. **Verification**: Check if object remains above ground threshold

### Success Criteria

- **Pass**: Object remains elevated (z > threshold) after 20 seconds
- **Fail**: Object falls below threshold during the shake phase

### Test Objects

Three objects with different shapes and physical properties:

| Object   | Description            | Challenge                            |
| -------- | ---------------------- | ------------------------------------ |
| `cube`   | Cube geometry          | Flat surfaces, edge contacts         |
| `ball`   | Spherical geometry     | Curved surface, rolling tendency     |
| `bottle` | Complex cylinder shape | Cylindrical grasp, varying diameters |

## Docker Setup (Recommended)

A unified Docker image is available with pre-configured environments for all physics engines, supporting RTX 50 series (sm_120).

### Features

- **Unified PyTorch Build**: Pre-compiled PyTorch with sm_120 support for Python 3.8/3.12
- **Unified IsaacGym**: Source code at `/workspace/third_party/isaacgym` with editable install support
- **Unified Environment Management**: 4 pre-initialized uv environments, ready to use

### Installation

Clone the repository with submodules (includes isaacgym):

```bash
git clone --recurse-submodules https://github.com/motphys/phys-bench.git
cd phys-bench
```

If you've already cloned the repository, initialize submodules:

```bash
git submodule update --init --recursive
```

### Quick Start

#### 1. Build the Image

```bash
cd docker_builder
./build.sh
```

> First build takes ~3-4 hours (compiling PyTorch), subsequent builds take ~15-30 minutes

#### 2. Run the Container

```bash
cd docker_builder
./run.sh
```

#### 3. Connect with VSCode

1. Install VSCode extension: **Dev Containers** or **Remote - Containers**
2. After container is running, in VSCode select **Attach to Running Container** → `motphys-bench-base`
3. Open folder: `/workspace/host/`
4. Select Python interpreter (Ctrl+Shift+P):
   - `/workspace/env/isaacgym/.venv/bin/python`
   - `/workspace/env/motrixsim/.venv/bin/python`
   - `/workspace/env/genesis/.venv/bin/python`
   - `/workspace/env/mjwarp/.venv/bin/python`

### Pre-configured Environments

| Environment | Python Version | Purpose               | Alias          |
| ----------- | -------------- | --------------------- | -------------- |
| isaacgym    | 3.8            | IsaacGym benchmark    | `uv_isaacgym`  |
| motrixsim   | 3.12           | MotrixSim benchmark   | `uv_motrixsim` |
| genesis     | 3.12           | Genesis benchmark     | `uv_genesis`   |
| mjwarp      | 3.12           | MuJoCo Warp benchmark | `uv_mjwarp`    |

### Running Benchmarks in Docker

Using aliases (recommended):

```bash
# Genesis benchmark
uv_genesis run python grasp-bench/test_genesis_cube.py -B 1024

# MotrixSim benchmark
uv_motrixsim run python grasp-bench/test_motrixsim_cube.py -B 1024

# IsaacGym benchmark
uv_isaacgym run python grasp-bench/test_isaacgym_cube.py -B 1024

# MJWarp benchmark
uv_mjwarp run python grasp-bench/test_mjwarp_cube.py -B 1024
```

Using full commands:

```bash
uv --project /workspace/env/genesis run python script.py
uv --project /workspace/env/motrixsim run python script.py
```

### Docker Image Information

- **Image Name**: `docker.mp/motphys-bench-base:latest`
- **Container Name**: `motphys-bench-base`
- **Base Image**: `nvidia/cuda:12.8.0-runtime-ubuntu22.04`
- **CUDA Version**: 12.8
- **Supported Architectures**: sm_80, sm_86, sm_89, sm_90, sm_120 (RTX 50 series)

### Docker Compose (Optional)

```bash
cd docker_builder
docker-compose up -d
docker exec -it motphys-bench-base /bin/bash
```

## Local Installation (Alternative)

### Prerequisites

All dependencies are managed through uv's optional dependency groups.

### All Dependencies

To install all benchmark dependencies at once:

```bash
uv sync --all-extras
```

This installs dependencies for Motrix, MuJoCo, and Genesis benchmarks.

## Running the Benchmarks (Local Installation)

### Speed Benchmarks

Speed benchmarks test FPS performance with batched environments:

```bash
# Genesis benchmark (1024 parallel environments)
uv run grasp-bench/test_genesis_cube.py -B 1024
uv run grasp-bench/test_genesis_ball.py -B 1024
uv run grasp-bench/test_genesis_bottle.py -B 1024

# MotrixSim benchmark
uv run grasp-bench/test_motrixsim_cube.py -B 1024
uv run grasp-bench/test_motrixsim_ball.py -B 1024

# With visualization (slower)
uv run grasp-bench/test_genesis_cube.py -B 1024 -v

# With random actions
uv run grasp-bench/test_genesis_cube.py -B 1024 -r
```

### Quality Benchmarks

Quality benchmarks test grasp stability and physics accuracy:

```bash
# Test with cube (default)
uv run grasp/grasp_shaking_test_{engine}.py --object=cube
```

### Running All Tests

To run all grasp benchmarks across different engines, objects, and configurations:

```bash
# Run all tests with default settings
uv run grasp/run_all_grasp_tests.py

# Run tests for specific engines
uv run grasp/run_all_grasp_tests.py --engines mujoco,motrix

# Run tests for specific objects
uv run grasp/run_all_grasp_tests.py --objects cube,bottle

# Use custom timestep values
uv run grasp/run_all_grasp_tests.py --dt-values 0.002

# Disable shake test (use slip test instead)
uv run grasp/run_all_grasp_tests.py --no-shake
```

This will:

- Run tests across all engine/object/DT combinations
- Generate video recordings for each test
- Create a comprehensive HTML comparison report at `output/comparison_report.html`

#### View Test Report

**[📊 Click here to view the latest test report](https://htmlpreview.github.io/?https://raw.githubusercontent.com/Motphys/phys-bench/refs/heads/main/output/comparison_report.html)**

The report includes:

- **Engine Overview**: Success rate cards and configuration matrix for each physics engine
- **Detailed Results**: Object-by-object breakdown with video evidence
- **Interactive Navigation**: Quick tabs to jump between test objects
- **Performance Metrics**: Drop times, success rates, and cross-engine comparisons

To generate a new report locally:

```bash
uv run grasp/generate_report.py
```

## Command-Line Flags

### Speed Benchmark Flags

| Flag     | Type    | Default | Description                                        |
| -------- | ------- | ------- | -------------------------------------------------- |
| `-B`     | int     | `1`     | Batch size (number of parallel environments)       |
| `-v`     | boolean | `False` | Enable visualization                               |
| `-r`     | boolean | `False` | Enable random actions (50Hz control frequency)     |
| `-m`     | boolean | `False` | Move along trajectory (cube only)                  |
| `--mjcf` | boolean | `False` | Use mjx_panda.xml instead of panda.xml (cube only) |

### Quality Benchmark Flags

| Flag       | Type    | Default | Description                                                           |
| ---------- | ------- | ------- | --------------------------------------------------------------------- |
| `--object` | string  | `cube`  | Object to grasp. Choices: `cube`, `ball`, `bottle`                    |
| `--shake`  | boolean | `True`  | Enable arm shaking after grasping. Set to `False` for slip-grasp test |
| `--record` | boolean | `False` | Record simulation to MP4 video file                                   |
| `--dt`     | float   | `0.002` | Simulation timestep in seconds                                        |
| `--mjx`    | boolean | `False` | Use MJX-compatible XML files for the Franka robot model               |
| `-V`       | boolean | `False` | Visualize simulation in a window (same as `--visual`)                 |

## Expected Output

### Speed Benchmark Output

```
Warmup Phase 1: Grasping (100 steps)...
Warmup Phase 2: Lifting (100 steps)...
Benchmark: 500 steps...
per env: 1,234.56 FPS
total  : 1,264,189.44 FPS
```

The output shows:

- **per env**: FPS for a single environment (physics simulation speed)
- **total**: Aggregate FPS across all parallel environments (throughput)

### Quality Benchmark Output

#### Successful Test

```
✅ The shaking-grasp-cube-test passed.
```

The object was successfully grasped and remained above the threshold throughout the shake phase.

#### Failed Test

```
❌ The shaking-grasp-cube-test failed.
```

The object fell below the threshold during the shake phase, indicating grasp instability.

#### Video Output

When `--record=True`, an MP4 video is saved showing the simulation: `{engine}_grasp_shake_{object}.mp4`

## Interpreting Results

These benchmarks test different aspects of physics simulation:

### Speed Benchmarks

- **Per-env FPS**: Single environment simulation speed (physics engine efficiency)
- **Total FPS**: Aggregate throughput across all parallel environments (scalability)
- **Scalability**: How well does the engine scale with batch size?
- **GPU Utilization**: Efficient use of GPU resources for parallel simulation

### Quality Benchmarks

- **Contact Stability**: How well does the engine maintain contact between gripper and object?
- **Friction Modeling**: Are friction forces computed accurately to prevent slipping?
- **Constraint Solving**: Does the solver handle contact constraints robustly?
- **Numerical Stability**: Does the simulation remain stable under perturbation?

Comparing results across different physics engines can help validate simulation accuracy and identify areas for improvement.
