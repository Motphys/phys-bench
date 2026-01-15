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

## Prerequisites

All dependencies are managed through uv's optional dependency groups. Install from the `motrixsim-py` directory.

### All Dependencies

To install all benchmark dependencies at once:

```bash
uv sync --all-extras
```

This installs dependencies for Motrix, MuJoCo, and Genesis benchmarks.

## Running the Benchmarks

Test with the Motrix physics engine (this project):

```bash

# Test with cube (default)
uv run grasp/grasp_shaking_test_{engine}.py --object=cube
```

## Command-Line Flags

All benchmark scripts support the following flags:

| Flag       | Type    | Default | Description                                                           |
| ---------- | ------- | ------- | --------------------------------------------------------------------- |
| `--object` | string  | `cube`  | Object to grasp. Choices: `cube`, `ball`, `bottle`                    |
| `--shake`  | boolean | `True`  | Enable arm shaking after grasping. Set to `False` for slip-grasp test |
| `--record` | boolean | `False` | Record simulation to MP4 video file                                   |
| `--dt`     | float   | `0.002` | Simulation timestep in seconds                                        |
| `--mjx`    | boolean | `False` | Use MJX-compatible XML files for the Franka robot model               |

## Expected Output

### Successful Test

```
✅ The shaking-grasp-cube-test passed.
```

The object was successfully grasped and remained above the threshold throughout the shake phase.

### Failed Test

```
❌ The shaking-grasp-cube-test failed.
```

The object fell below the threshold during the shake phase, indicating grasp instability.

### Video Output

When `--record=True`, an MP4 video is saved showing the simulation: `{engine}_grasp_shake_{object}.mp4`

## Interpreting Results

These benchmarks test different aspects of physics simulation:

- **Contact Stability**: How well does the engine maintain contact between gripper and object?
- **Friction Modeling**: Are friction forces computed accurately to prevent slipping?
- **Constraint Solving**: Does the solver handle contact constraints robustly?
- **Numerical Stability**: Does the simulation remain stable under perturbation?

Comparing results across different physics engines can help validate simulation accuracy and identify areas for improvement.
