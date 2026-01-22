#!/usr/bin/env python3
"""
MuJoCo CPU Rollout Benchmark

Tests CPU parallel simulation performance using mujoco.rollout module.
Simulates Franka Panda robot grasping a cube with batched environments.
"""

import argparse
import os
import time

import mujoco
from mujoco import rollout
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="MuJoCo CPU Rollout Benchmark")
    parser.add_argument("-B", type=int, default=1, help="Batch size (number of parallel environments)")
    parser.add_argument("-T", type=int, default=1, help="Number of CPU threads")
    parser.add_argument("-v", action="store_true", default=False, help="Visualize (single env only)")
    args = parser.parse_args()

    n_envs = args.B
    n_threads = args.T

    # Load model
    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../assets/grasp/mjx_pick_cube.xml")
    )
    model = mujoco.MjModel.from_xml_path(model_path)
    model.opt.timestep = 0.01  # Match other benchmarks (100 Hz)

    # Create data instances for threads
    data_list = [mujoco.MjData(model) for _ in range(n_threads)]
    
    # Reference positions
    grasp_qpos = np.array(
        [-1.0104, 1.5623, 1.3601, -1.6840, -1.5863, 1.7810, 1.4598, 0.04, 0.04],
        dtype=np.float64
    )
    lift_qpos = np.array(
        [-1.0426, 1.4028, 1.5634, -1.7114, -1.4055, 1.6015, 1.4510, 0.0, 0.0],
        dtype=np.float64
    )

    # Get state size
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    nctrl = model.nu

    # Create initial states for warmup phase 1 (grasp)
    data = mujoco.MjData(model)
    data.qpos[:9] = grasp_qpos
    data.ctrl[:7] = grasp_qpos[:7]
    data.ctrl[7] = 0.0  # Gripper closed
    mujoco.mj_forward(model, data)
    
    # Get initial state
    initial_state_grasp = np.zeros((n_envs, nstate), dtype=np.float64)
    state_buf = np.zeros(nstate, dtype=np.float64)
    mujoco.mj_getState(model, data, state_buf, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    initial_state_grasp[:] = state_buf

    # Create control sequence for warmup phase 1 (grasp - 100 steps)
    warmup1_steps = 100
    control_grasp = np.zeros((n_envs, warmup1_steps, nctrl), dtype=np.float64)
    ctrl_grasp = np.zeros(nctrl, dtype=np.float64)
    ctrl_grasp[:7] = grasp_qpos[:7]
    ctrl_grasp[7] = 0.0  # Gripper
    control_grasp[:] = ctrl_grasp

    print(f"Configuration: B={n_envs} envs, T={n_threads} threads")
    print(f"Model: {model_path}")
    print(f"State size: {nstate}, Control size: {nctrl}")

    ########################## Warmup Phase 1: Grasp (100 steps) ##########################
    print("Warmup Phase 1: Grasping (100 steps)...")
    
    # Pre-allocate state array to receive trajectory (nstep, not nstep+1)
    state_grasp = np.zeros((n_envs, warmup1_steps, nstate), dtype=np.float64)
    
    # Run warmup 1
    rollout.rollout(
        model,
        data_list,
        initial_state_grasp,
        control_grasp,
        state=state_grasp,
    )

    # Get final states from warmup 1 as initial states for warmup 2
    initial_state_lift = state_grasp[:, -1, :].copy()

    ########################## Warmup Phase 2: Lift (50 steps) ##########################
    print("Warmup Phase 2: Lifting (50 steps)...")
    
    warmup2_steps = 50
    control_lift = np.zeros((n_envs, warmup2_steps, nctrl), dtype=np.float64)
    ctrl_lift = np.zeros(nctrl, dtype=np.float64)
    ctrl_lift[:7] = lift_qpos[:7]
    ctrl_lift[7] = 0.0  # Gripper
    control_lift[:] = ctrl_lift

    # Pre-allocate state array (nstep, not nstep+1)
    state_lift = np.zeros((n_envs, warmup2_steps, nstate), dtype=np.float64)

    # Run warmup 2
    rollout.rollout(
        model,
        data_list,
        initial_state_lift,
        control_lift,
        state=state_lift,
    )

    # Get final states from warmup 2 as initial states for benchmark
    initial_state_bench = state_lift[:, -1, :].copy()

    ########################## Benchmark ##########################
    print("Benchmark: 500 steps...")
    
    benchmark_steps = 500
    
    # Fixed position control for pure physics benchmark
    control_bench = np.zeros((n_envs, benchmark_steps, nctrl), dtype=np.float64)
    control_bench[:, :, :7] = lift_qpos[:7]
    control_bench[:, :, 7] = 0.0

    # Pre-allocate state array for benchmark (only if visualization needed)
    if args.v and n_envs == 1:
        state_bench = np.zeros((n_envs, benchmark_steps, nstate), dtype=np.float64)
    else:
        state_bench = None

    # Run benchmark with timing
    t0 = time.perf_counter()
    
    rollout.rollout(
        model,
        data_list,
        initial_state_bench,
        control_bench,
        state=state_bench,
    )
    
    t1 = time.perf_counter()

    # Calculate and print results
    elapsed = t1 - t0
    per_env_fps = benchmark_steps / elapsed
    total_fps = per_env_fps * n_envs

    print(f"per env: {per_env_fps:,.2f} FPS")
    print(f"total  : {total_fps:,.2f} FPS")

    # Optional visualization (only for single env)
    if args.v and n_envs == 1 and state_bench is not None:
        print("\nStarting visualization...")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Replay the benchmark trajectory
            for step in range(benchmark_steps):
                mujoco.mj_setState(
                    model, data,
                    state_bench[0, step, :],
                    mujoco.mjtState.mjSTATE_FULLPHYSICS
                )
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
