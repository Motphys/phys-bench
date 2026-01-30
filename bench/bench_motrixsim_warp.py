import argparse
import json
import os
import sys
import time

import numpy as np

import warp as wp
from motrixsim_warp import load_model, WorldData
from motrixsim_warp.msd import from_file
from motrixsim_warp.render import RenderApp, RenderSettings

parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, default=1, choices=[1, 5, 10], help="Number of robots")
parser.add_argument("-B", type=int, default=1, help="Batch size / parallel environments")
parser.add_argument("-v", action="store_true", default=False, help="Enable visualization")
parser.add_argument("--mode", type=str, default="franka_only", choices=["franka_only", "franka_grasp"], help="Scenario: franka_only or franka_grasp")
parser.add_argument("--object", type=str, default="ball", choices=["ball", "cube", "bottle"], help="Object for grasp mode")
parser.add_argument("-r", action="store_true", default=False, help="Random noise during grasp benchmark phase")

args = parser.parse_args()


# uv run --project envs/motrixsim_warp python -O bench/bench_motrixsim_warp.py -B 1024 -r --mode grasp

def get_robot_positions(n_robots):
    """Calculate robot positions - spread them out in a grid layout"""
    positions = []
    rotations = []

    if n_robots == 1:
        positions.append((0, 0, 0))
        rotations.append((0, 0, 0, 1))
    elif n_robots == 5:
        for i in range(5):
            x = (i - 2) * 2.0  # -4, -2, 0, 2, 4
            positions.append((x, 0, 0))
            rotations.append((0, 0, 0, 1))
    elif n_robots == 10:
        for i in range(10):
            row = i // 5
            col = i % 5
            x = (col - 2) * 2.0  # -4, -2, 0, 2, 4
            y = (row - 0.5) * 2.0  # -1, 1
            positions.append((x, y, 0))
            rotations.append((0, 0, 0, 1))

    return positions, rotations


# Object configurations for grasp mode
OBJECT_CONFIGS = {
    "ball": {
        "grasp_qpos": np.array([-1.0323, 1.7628, 1.4904, -1.6749, -1.7715, 1.6293, 1.4417, 0.04, 0.04]),
        "lift_steps": 100,
    },
    "cube": {
        "grasp_qpos": np.array([-1.0104, 1.5623, 1.3601, -1.6840, -1.5863, 1.7810, 1.4598, 0.04, 0.04]),
        "lift_steps": 50,
    },
    "bottle": {
        "grasp_qpos": np.array([-0.9937, 1.4588, 1.3058, -1.6924, -1.4882, 1.8461, 1.4577, 0.04, 0.04]),
        "lift_steps": 50,
    },
}

########################## load model ##########################
if args.mode == "franka_only":
    # Franka only mode: load base scene and attach N robots
    scene_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "xml/base_scene.xml")
    )
    scene = from_file(scene_path)

    robot_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../assets/franka_emika_panda/panda.xml")
    )
    robot = from_file(robot_path)

    positions, rotations = get_robot_positions(args.N)
    for i in range(args.N):
        scene.attach(
            robot,
            other_prefix=f"robot{i}_",
            other_translation=positions[i],
            other_rotation=rotations[i],
        )

    model = scene.build(150 * args.N)
    model.options.timestep = 0.01

elif args.mode == "franka_grasp":
    assert args.N == 1, "Franka grasp mode only supports N=1 robot currently"

    # Franka grasp mode with N=1: load pre-made grasp scene
    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), f"../assets/grasp/pick_{args.object}.xml")
    )
    model = load_model(model_path, 150)
    model.options.timestep = 0.01

########################## create batched data ##########################
n_envs = args.B
data = WorldData(model, n_envs)
    

########################## setup control ##########################
n_robots = args.N
actuators_per_robot = 8  # 7 joints + 1 gripper actuator
total_actuators = n_robots * actuators_per_robot

# Warmup position for random mode
warmup_qpos = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853, 0.04, 0.04])

########################## setup render (if needed) ##########################
if args.v:
    render = RenderApp()
    # render.__enter__()
    render.launch(model, num_worlds=n_envs, render_settings=RenderSettings.performance())
else:
    render = None

sim_dt = model.options.timestep

########################## mode-specific warmup and benchmark ##########################
if args.mode == "franka_only":
    print(f"Warmup: {n_robots} robots to initial position (200 steps)...")

    # Create warmup control for all robots
    if n_envs > 1:
        warmup_ctrl = np.zeros((n_envs, total_actuators), dtype=np.float32)
        for robot_idx in range(n_robots):
            offset = robot_idx * actuators_per_robot
            warmup_ctrl[:, offset : offset + 7] = warmup_qpos[:7]
            warmup_ctrl[:, offset + 7] = warmup_qpos[7]
    else:
        warmup_ctrl = np.zeros(total_actuators, dtype=np.float32)
        for robot_idx in range(n_robots):
            offset = robot_idx * actuators_per_robot
            warmup_ctrl[offset : offset + 7] = warmup_qpos[:7]
            warmup_ctrl[offset + 7] = warmup_qpos[7]

    init_qpos = warmup_qpos
    # TODO: delete this line on all engines
    data.set_dof_pos(np.tile(init_qpos, (n_envs, n_robots, 1)).tolist())

    for i in range(200):
        data.set_ctrls(warmup_ctrl)
        model.step(data)

    print(f"Benchmark: 1000 steps with {n_robots} robots (random motion)...")
    benchmark_steps = 1000
    ref_pos = warmup_qpos[:7].copy()
    
    ref_pos_wp = wp.array(ref_pos, dtype=float)
    step_counter_wp = wp.array([0], dtype=int)

    @wp.kernel
    def randomize_ctrl_kernel(ref_pos: wp.array(dtype=float), step_counter_wp: wp.array(dtype=int), ctrl_per_robot: int, ctrls: wp.array2d(dtype=float)):
        step = step_counter_wp[0]
        wid, robot_id, tid = wp.tid()
        noise = wp.randf(wp.uint32(wid * ctrls.shape[1] + tid + step), -0.2, 0.2)
        ctrls[wid, robot_id * ctrl_per_robot + tid] = ref_pos[tid] + noise

        if wid == 0 and tid == 0:
            step_counter_wp[0] = step + 1

    with wp.ScopedCapture(device="cuda") as capture:
        wp.launch(randomize_ctrl_kernel, dim=(n_envs, n_robots, 7), inputs=[ref_pos_wp, step_counter_wp, actuators_per_robot, data.ctrls], block_dim=32)
    randomize_graph = capture.graph

    if args.v:
        step_counter = [0]
        benchmark_complete = [False]
        t_start = [None]
        t_end = [None]

        def phys_step(loop_index, n_steps):
            if benchmark_complete[0]:
                return
            if t_start[0] is None:
                t_start[0] = time.perf_counter()

            i = step_counter[0]
            if i < benchmark_steps:
                
                wp.capture_launch(randomize_graph)

                model.step(data)

                # Check timeout: 2s per step cumulative
                elapsed = time.perf_counter() - t_start[0]
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

                step_counter[0] += 1
            else:
                t_end[0] = time.perf_counter()
                benchmark_complete[0] = True
                print(f"per env: {benchmark_steps / (t_end[0] - t_start[0]):,.2f} FPS")
                print(f"total  : {benchmark_steps / (t_end[0] - t_start[0]) * n_envs:,.2f} FPS")

            return data

        render.render_loop(phys_step, 60)
    else:
        t0 = time.perf_counter()
        for i in range(benchmark_steps):
    
            wp.capture_launch(randomize_graph)
            model.step(data)

            # Check timeout: 2s per step cumulative
            elapsed = time.perf_counter() - t0
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
        print(f"per env: {benchmark_steps / (t1 - t0):,.2f} FPS")
        print(f"total  : {benchmark_steps / (t1 - t0) * n_envs:,.2f} FPS")

else:  # franka_grasp mode
    config = OBJECT_CONFIGS[args.object]
    grasp_qpos = config["grasp_qpos"]
    lift_steps = config["lift_steps"]
    lift_qpos = np.array([-1.0426, 1.4028, 1.5634, -1.7114, -1.4055, 1.6015, 1.4510, 0.0, 0.0])

    # Initialize robot positions for grasp mode
    if args.N == 1:
        # Single robot: use standard initialization
        panda_index = model.get_body_index("link0")
        panda = model.get_body(panda_index)
        if n_envs > 1:
            init_pos = np.tile(grasp_qpos, (n_envs, 1))
            panda.set_dof_pos(data, init_pos, False)
        else:
            panda.set_dof_pos(data, grasp_qpos, False)
    else:
        # Multiple robots: initialize each separately
        for robot_idx in range(args.N):
            panda_index = model.get_body_index(f"robot{robot_idx}_link0")
            panda = model.get_body(panda_index)
            if n_envs > 1:
                init_pos = np.tile(grasp_qpos, (n_envs, 1))
                panda.set_dof_pos(data, init_pos, False)
            else:
                panda.set_dof_pos(data, grasp_qpos, False)

    def make_ctrl(joint_qpos, gripper_val, n_envs, n_robots):
        """Create actuator control array for multiple robots"""
        if n_envs > 1:
            ctrl = np.zeros((n_envs, n_robots * 8), dtype=np.float32)
            for robot_idx in range(n_robots):
                offset = robot_idx * 8
                ctrl[:, offset : offset + 7] = joint_qpos
                ctrl[:, offset + 7] = gripper_val
        else:
            ctrl = np.zeros(n_robots * 8, dtype=np.float32)
            for robot_idx in range(n_robots):
                offset = robot_idx * 8
                ctrl[offset : offset + 7] = joint_qpos
                ctrl[offset + 7] = gripper_val
        return ctrl

    print("Warmup Phase 1: Grasping (100 steps)...")
    for i in range(100):
        ctrl = make_ctrl(grasp_qpos[:7], 0.0, n_envs, n_robots)
        data.set_ctrls(ctrl)
        model.step(data)

    print(f"Warmup Phase 2: Lifting ({lift_steps} steps)...")
    for i in range(lift_steps):
        ctrl = make_ctrl(lift_qpos[:7], 0.0, n_envs, n_robots)
        data.set_ctrls(ctrl)
        model.step(data)

    print("Benchmark: 500 steps...")
    benchmark_steps = 500
    ref_ctrl = make_ctrl(lift_qpos[:7], 0.0, n_envs, n_robots)
    ref_pos = lift_qpos[:7].copy()

    
    ref_pos_wp = wp.array(ref_pos, dtype=float)
    step_counter_wp = wp.array([0], dtype=int)

    @wp.kernel
    def randomize_ctrl_kernel(ref_pos: wp.array(dtype=float), step_counter_wp: wp.array(dtype=int), ctrls: wp.array2d(dtype=float)):
        step = step_counter_wp[0]
        wid, tid = wp.tid()
        noise = wp.randf(wp.uint32(wid * ctrls.shape[1] + tid + step), -0.025, 0.025)
        ctrls[wid, tid] = ref_pos[tid] + noise

        if wid == 0 and tid == 0:
            step_counter_wp[0] = step + 1

    with wp.ScopedCapture(device="cuda") as capture:
        wp.launch(randomize_ctrl_kernel, dim=(n_envs, 7), inputs=[ref_pos_wp, step_counter_wp, data.ctrls], block_dim=32)
    randomize_graph = capture.graph

    if args.v:
        step_counter = [0]
        benchmark_complete = [False]
        t_start = [None]
        t_end = [None]

        def phys_step(loop_index, n_steps):
            if benchmark_complete[0]:
                return
            if t_start[0] is None:
                t_start[0] = time.perf_counter()

            i = step_counter[0]
            if i < benchmark_steps:
                if args.r and i % 2 == 0:
                    wp.capture_launch(randomize_graph)

                model.step(data)
                wp.synchronize()

                # Check timeout: 2s per step cumulative
                elapsed = time.perf_counter() - t_start[0]
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

                step_counter[0] += 1
            else:
                t_end[0] = time.perf_counter()
                benchmark_complete[0] = True
                print(f"per env: {benchmark_steps / (t_end[0] - t_start[0]):,.2f} FPS")
                print(f"total  : {benchmark_steps / (t_end[0] - t_start[0]) * n_envs:,.2f} FPS")

            return data

        render.render_loop(phys_step, 60)
    else:
        sum_t = 0.0
        t0 = time.perf_counter()
        for i in range(benchmark_steps):
            if args.r and i % 2 == 0:
                wp.capture_launch(randomize_graph)

            model.step(data)
            wp.synchronize()

            # Check timeout: 2s per step cumulative
            elapsed = time.perf_counter() - t0
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
        sum_t = (t1 - t0)

        print(f"per env: {benchmark_steps / sum_t:,.2f} FPS")
        print(f"total  : {benchmark_steps / sum_t * n_envs:,.2f} FPS")

# Cleanup render
if render:
    try:
        render.__exit__(None, None, None)
    except:
        pass
