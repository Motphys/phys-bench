#!/usr/bin/env python3
"""
Genesis Humanoid Benchmark

GPU-accelerated simulation of multiple humanoid robots using Genesis physics engine.
Supports 1-101 humanoids per environment with GPU batch processing.
Matches the implementation pattern of motrix_humanoid.py and mujoco_humanoid.py for consistency.
"""

import argparse
import json
import os
import sys
import time

import genesis as gs


def get_humanoid_positions(n_humanoids):
    """Calculate humanoid positions with 1.2m spacing along x-axis."""
    positions = []
    for i in range(n_humanoids):
        positions.append((i * 1.2, 0, 0))
    return positions


def main():
    parser = argparse.ArgumentParser(description="Genesis Humanoid Benchmark")
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
    parser.add_argument(
        "-v",
        action="store_true",
        default=False,
        help="Enable visualization (single env only)",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps")
    parser.add_argument("--benchmark", type=int, default=500, help="Benchmark steps")

    args = parser.parse_args()

    n_humanoids = args.N
    n_envs = args.B
    warmup_steps = args.warmup
    benchmark_steps = args.benchmark

    ########################## Init Genesis ##########################
    gs.init(backend=gs.gpu)

    ########################## Create Scene ##########################
    scene = gs.Scene(
        show_viewer=args.v,
        rigid_options=gs.options.RigidOptions(
            dt=0.005,  # From humanoid.xml
            max_collision_pairs=150 * n_humanoids,  # Support up to 100 humanoids
        ),
    )

    ########################## Add Ground Plane ##########################
    plane = scene.add_entity(gs.morphs.Plane())

    ########################## Add Humanoids ##########################
    humanoid_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../assets/humanoid.xml")
    )

    positions = get_humanoid_positions(n_humanoids)
    humanoids = []

    for i, pos in enumerate(positions):
        humanoid = scene.add_entity(
            gs.morphs.MJCF(
                file=humanoid_path,
                pos=pos,
                euler=(0, 0, 0),  # No rotation
            ),
        )
        humanoids.append(humanoid)

    ########################## Build Scene (Batch Processing) ##########################
    scene.build(n_envs=n_envs)

    ########################## Warmup Phase ##########################
    print(f"Configuration: B={n_envs} envs, N={n_humanoids} humanoids")
    print(f"Model: {humanoid_path}")
    print(
        f"Warmup: {n_humanoids} humanoids to standing position ({warmup_steps} steps)..."
    )

    # Simple warmup (no timeout check)
    for i in range(warmup_steps):
        scene.step()

    ########################## Benchmark Phase ##########################
    print(f"Benchmark: {benchmark_steps} steps with {n_humanoids} humanoids...")

    if args.v:
        # Visualization mode: run indefinitely
        while True:
            scene.step()
    else:
        # Benchmark mode: measure performance
        try:
            t0 = time.perf_counter()
            for i in range(benchmark_steps):
                scene.step()
                elapsed = time.perf_counter() - t0
                # Timeout check: if cumulative time exceeds (i+1) * 2 seconds
                if elapsed > (i + 1) * 2.0:
                    error_data = {
                        "status": "error",
                        "error_code": "TIMEOUT",
                        "error_message": f"Timeout at step {i+1}: {elapsed:.2f}s > {(i+1)*2.0:.2f}s",
                        "per_env_fps": 0.0,
                        "total_fps": 0.0,
                    }
                    print(json.dumps(error_data))
                    sys.exit(1)
            t1 = time.perf_counter()

            # Calculate and print results
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
                "total_fps": 0.0,
            }
            print(json.dumps(error_data))
            sys.exit(1)


if __name__ == "__main__":
    main()
