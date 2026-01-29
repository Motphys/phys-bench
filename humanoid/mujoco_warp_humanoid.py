#!/usr/bin/env python3
"""
MuJoCo-Warp Humanoid Benchmark

GPU-accelerated simulation of multiple humanoid robots using MuJoCo-Warp.
Supports 1-101 humanoids per environment with GPU batch processing.
Matches the implementation pattern of mujoco_humanoid.py for consistency.
"""

import argparse
import json
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

# Try to import mujoco_warp, provide helpful error if not available
try:
    import mujoco_warp as mjw
    import warp as wp
except ImportError as e:
    raise ImportError(
        "MuJoCo-Warp is not installed. Please install it with:\n"
        "  git clone https://github.com/google-deepmind/mujoco_warp.git\n"
        "  cd mujoco_warp\n"
        "  uv pip install -e .[dev,cuda]\n"
        "Or use: uv pip install -e '.[mujoco-warp]'"
    ) from e


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
    parser = argparse.ArgumentParser(description="MuJoCo-Warp Humanoid Benchmark")
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
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps")
    parser.add_argument("--benchmark", type=int, default=500, help="Benchmark steps")

    parser.add_argument(
        "-v",
        action="store_true",
        default=False,
        help="Enable visualization (single env only)",
    )
    parser.add_argument(
        "--use-capture",
        action="store_true",
        default=True,
        help="Enable CUDA graph capture for benchmark (CUDA only, incompatible with -v)",
    )

    args = parser.parse_args()

    n_humanoids = args.N
    n_envs = args.B
    warmup_steps = args.warmup
    benchmark_steps = args.benchmark

    ########################## Create Model ##########################
    positions = get_humanoid_positions(n_humanoids)
    mjm = create_multi_humanoid_scene(n_humanoids, positions)
    mjm.opt.timestep = 0.005
    mjm.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # Note: MuJoCo-Warp doesn't support mjINT_IMPLICIT

    ########################## Create GPU Data ##########################
    # Put model and data on GPU device using MuJoCo-Warp
    m = mjw.put_model(mjm)
    d = mjw.make_data(
        mjm, nworld=n_envs, nconmax=100 * n_humanoids, njmax=300 * n_humanoids
    )

    ########################## Validate Capture Mode ##########################
    # Validate capture mode constraints
    use_capture = args.use_capture
    if use_capture:
        if args.v:
            print(
                "Warning: Capture mode incompatible with visualization, disabling capture"
            )
            use_capture = False
        elif not wp.get_device().is_cuda:
            print(
                f"Warning: Capture mode requires CUDA device (current: {wp.get_device()}), disabling capture"
            )
            use_capture = False

    if use_capture:
        print(f"Mode: CUDA Graph Capture (device: {wp.get_device()})")
    else:
        print(f"Mode: Standard (device: {wp.get_device()})")

    ########################## Warmup Phase ##########################
    print(f"Configuration: B={n_envs} envs, N={n_humanoids} humanoids (GPU)")
    humanoid_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../assets/humanoid.xml")
    )
    print(f"Model: {humanoid_path}")
    print(
        f"Warmup: {n_humanoids} humanoids to standing position ({warmup_steps} steps)..."
    )

    # Simple warmup loop (no controls, just gravity)
    for i in range(warmup_steps):
        mjw.step(m, d)

    ########################## Benchmark Phase ##########################
    print(f"Benchmark: {benchmark_steps} steps with {n_humanoids} humanoids...")

    if args.v and n_envs == 1:
        # Visualization mode - need to copy GPU state to CPU
        cpu_data = mujoco.MjData(mjm)
        with mujoco.viewer.launch_passive(mjm, cpu_data) as viewer:
            t0 = time.perf_counter()
            for step in range(benchmark_steps):
                mjw.step(m, d)

                # Copy GPU state to CPU for visualization
                cpu_data.qpos[:] = d.qpos.numpy()[0]
                cpu_data.qvel[:] = d.qvel.numpy()[0]
                mujoco.mj_forward(mjm, cpu_data)
                viewer.sync()
                time.sleep(mjm.opt.timestep)
            t1 = time.perf_counter()

        elapsed = t1 - t0
        per_env_fps = benchmark_steps / elapsed
        total_fps = per_env_fps * n_envs

        print(f"per env: {per_env_fps:,.2f} FPS")
        print(f"total  : {total_fps:,.2f} FPS")
    else:
        # Benchmark mode - two paths: capture vs standard
        if use_capture:
            # CAPTURE PATH: Record once, replay many times
            print(f"Capturing graph for {benchmark_steps} steps...")

            try:
                # Step 1: Record the graph
                t0_capture = time.perf_counter()
                with wp.ScopedCapture() as capture:
                    # Record ONE step
                    mjw.step(m, d)
                t1_capture = time.perf_counter()

                capture_graph = capture.graph
                print(f"Graph captured in {t1_capture - t0_capture:.4f}s")

                # Step 2: Benchmark loop - replay captured graph
                print(f"Benchmark: Replaying graph {benchmark_steps} times...")
                t0 = time.perf_counter()

                for step in range(benchmark_steps):
                    wp.capture_launch(capture_graph)
                    wp.synchronize()  # Ensure GPU completion

                    # Move timeout check outside capture replay
                    # Check every 100 steps to reduce overhead
                    if (step + 1) % 100 == 0:
                        elapsed = time.perf_counter() - t0
                        if elapsed > (step + 1) * 2.0:
                            error_data = {
                                "status": "error",
                                "error_code": "TIMEOUT",
                                "error_message": f"Timeout at step {step + 1}: {elapsed:.2f}s > {(step + 1) * 2.0:.2f}s",
                                "per_env_fps": 0.0,
                                "total_fps": 0.0,
                                "mode": "capture",
                            }
                            print(json.dumps(error_data))
                            sys.exit(1)

                t1 = time.perf_counter()
                elapsed = t1 - t0
                per_env_fps = benchmark_steps / elapsed
                total_fps = per_env_fps * n_envs

                print(f"per env: {per_env_fps:,.2f} FPS (capture mode)")
                print(f"total  : {total_fps:,.2f} FPS")

            except Exception as e:
                error_data = {
                    "status": "error",
                    "error_code": "BENCHMARK_ERROR",
                    "error_message": f"{type(e).__name__}: {str(e)}",
                    "per_env_fps": 0.0,
                    "total_fps": 0.0,
                    "mode": "capture",
                }
                print(json.dumps(error_data))
                sys.exit(1)

        else:
            # STANDARD PATH: Original implementation for comparison
            print(f"Benchmark: {benchmark_steps} steps (standard mode)...")

            try:
                t0 = time.perf_counter()
                for step in range(benchmark_steps):
                    mjw.step(m, d)
                    elapsed = time.perf_counter() - t0
                    # Timeout check: if cumulative time exceeds (step+1) * 2 seconds
                    if elapsed > (step + 1) * 2.0:
                        error_data = {
                            "status": "error",
                            "error_code": "TIMEOUT",
                            "error_message": f"Timeout at step {step + 1}: {elapsed:.2f}s > {(step + 1) * 2.0:.2f}s",
                            "per_env_fps": 0.0,
                            "total_fps": 0.0,
                            "mode": "standard",
                        }
                        print(json.dumps(error_data))
                        sys.exit(1)
                t1 = time.perf_counter()

                elapsed = t1 - t0
                per_env_fps = benchmark_steps / elapsed
                total_fps = per_env_fps * n_envs

                print(f"per env: {per_env_fps:,.2f} FPS (standard mode)")
                print(f"total  : {total_fps:,.2f} FPS")

            except Exception as e:
                error_data = {
                    "status": "error",
                    "error_code": "BENCHMARK_ERROR",
                    "error_message": f"{type(e).__name__}: {str(e)}",
                    "per_env_fps": 0.0,
                    "total_fps": 0.0,
                    "mode": "standard",
                }
                print(json.dumps(error_data))
                sys.exit(1)


if __name__ == "__main__":
    main()
