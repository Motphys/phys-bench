#!/usr/bin/env python3
"""
Genesis Benchmark for CPU-bench comparison

Tests parallel simulation performance using Genesis.
Simulates Franka Panda robot grasping a cube with batched environments.
"""

import argparse
import time

import genesis as gs
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Genesis CPU Benchmark")
    parser.add_argument("-B", type=int, default=1, help="Batch size (number of parallel environments)")
    args = parser.parse_args()

    n_envs = args.B

    ########################## init ##########################
    # Use CPU backend for CPU benchmark
    gs.init(backend=gs.cpu)

    ########################## create a scene ##########################
    scene = gs.Scene(
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            constraint_solver=gs.constraint_solver.Newton,
            enable_self_collision=True,
        ),
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.65, 0.0, 0.02),
        ),
    )

    # Use mjx_panda.xml for consistency with other benchmarks
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="assets/franka_emika_panda/mjx_panda.xml"
        ),
    )

    ########################## build ##########################
    scene.build(n_envs=n_envs)

    # Set control gains
    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)

    grasp_qpos = np.array([-1.0104, 1.5623, 1.3601, -1.6840, -1.5863, 1.7810, 1.4598, 0.04, 0.04])
    lift_qpos = np.array([-1.0426, 1.4028, 1.5634, -1.7114, -1.4055, 1.6015, 1.4510, 0.0, 0.0])
    franka.set_dofs_position(grasp_qpos)

    print(f"Configuration: B={n_envs} envs (CPU)")

    ########################## Warmup Phase 1: Grasp (100 steps) ##########################
    print("Warmup Phase 1: Grasping (100 steps)...")
    franka.control_dofs_position(grasp_qpos[:-2], motors_dof)
    franka.control_dofs_force(np.array([-0.5, -0.5]), fingers_dof)
    for i in range(100):
        scene.step()

    ########################## Warmup Phase 2: Lift (50 steps) ##########################
    print("Warmup Phase 2: Lifting (50 steps)...")
    franka.control_dofs_position(lift_qpos[:-2], motors_dof)
    for i in range(50):
        scene.step()

    ########################## Benchmark ##########################
    print("Benchmark: 500 steps...")
    
    benchmark_steps = 500

    # Pure physics benchmark without rendering
    t0 = time.perf_counter()
    for i in range(benchmark_steps):
        scene.step()
    t1 = time.perf_counter()

    print(f"per env: {benchmark_steps / (t1 - t0):,.2f} FPS")
    print(f"total  : {benchmark_steps / (t1 - t0) * n_envs:,.2f} FPS")


if __name__ == "__main__":
    main()
