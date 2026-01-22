# Gyroscope Precession Benchmarks

The gyroscope precession benchmarks test rotational dynamics, angular momentum conservation, and rigid body physics through a spinning gyroscope simulation. This tests the accuracy of rotational physics, collision handling, and numerical stability in constrained rotational motion.

## Scenario

The benchmark implements a **gyroscope** performing precession and nutation:

1. **Setup**: Gyroscope with three rigid bodies (main cylinder body, slender axis, spherical tip)
2. **Initial Conditions**: Axis tilted at non-perpendicular angle to ground
3. **Initial Spin**: Angular velocity applied along the rotation axis
4. **Free Motion**: Gyroscope moves under gravity and ground friction
5. **Verification**: Observe precession, nutation, and stability over time

## Expected Behavior

Due to conservation of angular momentum:

- **Precession**: Slow rotation of the rotation axis around the vertical
- **Nutation**: Small oscillations superimposed on the precession
- **Stability**: Faster initial spin → smaller nutation amplitude
- **Failure**: Too slow spin → excessive nutation → ground contact → loss of stability

## Test

| Spin Speed | Expected Behavior                     | Challenge                                   |
| ---------- | ------------------------------------- | ------------------------------------------- |
| 100 rad/s  | Stable precession, minimal nutation   | Large angular momentum, collision stability |
| 50 rad/s   | Moderate precession and nutation      | Balanced rotational dynamics                |
| 20 rad/s   | Large nutation, potential instability | Maintaining spin stability                  |

## Running Tests

```bash
# Run gyroscope precession tests
cd gyro_precession

# Genesis benchmark
uv_genesis run python gyro_precession_test_genesis.py

# MotrixSim benchmark
uv_motrixsim run python gyro_precession_test_motrix.py

# MuJoCo benchmark
uv_mjwarp run python gyro_precession_test_mujoco.py

# IsaacSim benchmark
uv_isaacsim run python gyro_precession_test_isaacsim.py
```

#### Run All Tests

To run all gyroscope precession benchmarks across different engines and configurations:

```bash
cd gyro_precession

# Run all tests with default settings
uv run python run_all_gyro_precession_tests.py

# Run tests for specific engines
uv run python run_all_gyro_precession_tests.py --engines genesis,motrix

# Use custom timestep values
uv run python run_all_gyro_precession_tests.py --dt-values 0.002,0.005
```

This will:

- Run tests across all engine/DT combinations
- Generate video recordings for each test
- Create a comprehensive HTML comparison report at `output/gyro_precession/comparison_report.html`

#### View Test Report

**[📊 Click here to view the latest test report](https://htmlpreview.github.io/?https://raw.githubusercontent.com/Motphys/phys-bench/refs/heads/main/output/gyro_precession/comparison_report.html)**

# Results Analysis

In terms of simulation stability, both MotrixSim and Genesis can maintain good posture even with large simulation time steps. In the scenario with an initial spin speed of 100, the angular momentum is relatively high, and Mujoco and IsaacSim showed varying degrees of collision solver failure.
