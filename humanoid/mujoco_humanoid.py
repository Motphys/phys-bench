#!/usr/bin/env python3
"""
MuJoCo Humanoid Benchmark

CPU parallel simulation of multiple humanoid robots using simple mj_step() loops.
Supports 1-10 humanoids per environment with multi-threaded batch processing.
Matches the implementation pattern of motrix_humanoid.py for consistency.
"""

import argparse
import json
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
from mujoco import rollout


def get_humanoid_positions(n_humanoids):
    """Calculate humanoid positions with 1.2m spacing along x-axis."""
    positions = []
    for i in range(n_humanoids):
        positions.append((i * 1.2, 0, 0))
    return positions


def create_multi_humanoid_scene(n_humanoids, positions):
    """Generate MJCF model with multiple humanoids using MuJoCo spec API.

    Args:
        n_humanoids: Number of humanoid robots to create
        positions: List of (x, y, z) tuples for each humanoid

    Returns:
        Compiled MjModel with multiple humanoids
    """
    humanoid_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../assets/humanoid.xml")
    )

    # Load first humanoid as base scene
    spec = mujoco.MjSpec.from_file(humanoid_path)

    # Attach additional humanoids if N > 1
    if n_humanoids > 1:
        for i in range(1, n_humanoids):
            # Create a frame for this humanoid at the desired position
            frame = spec.worldbody.add_frame()
            frame.name = f"humanoid{i}_frame"
            frame.pos = positions[i]

            # Load humanoid spec
            humanoid_copy = mujoco.MjSpec.from_file(humanoid_path)

            # Attach to the frame with prefix
            spec.attach(humanoid_copy, prefix=f"humanoid{i}_", frame=frame)

    # Compile the spec to a model
    model = spec.compile()
    return model


def main():
    parser = argparse.ArgumentParser(description="MuJoCo Humanoid Benchmark")
    parser.add_argument(
        "-N",
        type=int,
        default=1,
        choices=range(1, 101),
        help="Number of humanoids (1-101)",
    )
    parser.add_argument(
        "-B", type=int, default=1, help="Batch size (parallel environments)"
    )
    parser.add_argument("-T", type=int, default=1, help="Number of CPU threads")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps")
    parser.add_argument("--benchmark", type=int, default=500, help="Benchmark steps")

    parser.add_argument(
        "-v",
        action="store_true",
        default=False,
        help="Enable visualization (single env only)",
    )

    args = parser.parse_args()

    n_humanoids = args.N
    n_envs = args.B
    n_threads = args.T
    warmup_steps = args.warmup
    benchmark_steps = args.benchmark

    ########################## Create Model ##########################
    positions = get_humanoid_positions(n_humanoids)
    model = create_multi_humanoid_scene(n_humanoids, positions)
    model.opt.timestep = 0.005
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT  # For stability

    ########################## Create Data List for Threads ##########################
    data_list = [mujoco.MjData(model) for _ in range(n_threads)]

    ########################## Warmup Phase ##########################
    print(
        f"Configuration: B={n_envs} envs, T={n_threads} threads, N={n_humanoids} humanoids"
    )
    humanoid_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../assets/humanoid.xml")
    )
    print(f"Model: {humanoid_path}")
    print(
        f"Warmup: {n_humanoids} humanoids to standing position ({warmup_steps} steps)..."
    )

    # Simple warmup loop (no timeout check)
    for i in range(warmup_steps):
        for data in data_list:
            mujoco.mj_step(model, data)

    ########################## Benchmark Phase ##########################
    print(f"Benchmark: {benchmark_steps} steps with {n_humanoids} humanoids...")

    if args.v and n_envs == 1:
        # Visualization mode
        with mujoco.viewer.launch_passive(model, data_list[0]) as viewer:
            for step in range(benchmark_steps):
                mujoco.mj_step(model, data_list[0])
                viewer.sync()
                time.sleep(model.opt.timestep)
    else:
        # Benchmark mode using rollout
        try:
            # Get state size and capture initial state after warmup
            state_size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            initial_state = np.zeros((state_size,), dtype=np.float64)
            mujoco.mj_getState(
                model, data_list[0], initial_state, mujoco.mjtState.mjSTATE_FULLPHYSICS
            )
            initial_state = np.tile(initial_state, (n_envs, 1))

            step_state_out = np.zeros((n_envs, 1, state_size), dtype=np.float64)

            t0 = time.perf_counter()
            for step in range(benchmark_steps):
                rollout.rollout(model, data_list, initial_state, None, state=step_state_out)
                initial_state = step_state_out[:, 0, :]
                elapsed = time.perf_counter() - t0
                # Timeout check: if cumulative time exceeds (step+1) * 2 seconds
                if elapsed > (step + 1) * 2.0:
                    error_data = {
                        "status": "error",
                        "error_code": "TIMEOUT",
                        "error_message": f"Timeout at step {step+1}: {elapsed:.2f}s > {(step+1)*2.0:.2f}s",
                        "per_env_fps": 0.0,
                        "total_fps": 0.0
                    }
                    print(json.dumps(error_data))
                    sys.exit(1)
            t1 = time.perf_counter()

            elapsed = t1 - t0
            per_env_fps = benchmark_steps / elapsed
            total_fps = per_env_fps * n_envs

            print(f"per env: {per_env_fps:,.2f} FPS")
            print(f"total  : {total_fps:,.2f} FPS")
        except Exception as e:
            error_data = {
                "status": "error",
                "error_code": "BENCHMARK_ERROR",
                "error_message": f"{type(e).__name__}: {str(e)}",
                "per_env_fps": 0.0,
                "total_fps": 0.0
            }
            print(json.dumps(error_data))
            sys.exit(1)


if __name__ == "__main__":
    main()
