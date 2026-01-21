import argparse
import math
import os
import time

import motrixsim as mx
import numpy as np
from motrixsim import run

parser = argparse.ArgumentParser()
parser.add_argument("-B", type=int, default=1)  # batch size
parser.add_argument("-N", type=int, default=5, choices=[1, 5, 10])  # number of robots
parser.add_argument("-v", action="store_true", default=False)  # visualize
args = parser.parse_args()


def get_robot_positions(n_robots):
    """Calculate robot positions - spread them out in a grid layout"""
    positions = []
    rotations = []

    if n_robots == 1:
        positions.append((0, 0, 0))
        rotations.append((0, 0, 0, 1))  # No rotation
    elif n_robots == 5:
        # 5 robots in a line along x-axis with 2m spacing
        for i in range(5):
            x = (i - 2) * 2.0  # -4, -2, 0, 2, 4
            positions.append((x, 0, 0))
            rotations.append((0, 0, 0, 1))  # No rotation
    elif n_robots == 10:
        # 10 robots in 2 rows of 5 with 2m spacing
        for i in range(10):
            row = i // 5
            col = i % 5
            x = (col - 2) * 2.0  # -4, -2, 0, 2, 4
            y = (row - 0.5) * 2.0  # -1, 1
            positions.append((x, y, 0))
            rotations.append((0, 0, 0, 1))  # No rotation

    return positions, rotations


########################## load model ##########################
# Load base scene
scene_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "xml/base_scene.xml")
)
scene = mx.msd.from_file(scene_path)

# Load robot template (with hand/fingers)
robot_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../assets/franka_emika_panda/panda.xml")
)
robot = mx.msd.from_file(robot_path)

# Attach multiple robots dynamically
positions, rotations = get_robot_positions(args.N)
for i in range(args.N):
    scene.attach(
        robot,
        other_prefix=f"robot{i}_",
        other_translation=positions[i],
        other_rotation=rotations[i],
    )

# Build the model
model = scene.build()
model.options.timestep = 0.01  # Match genesis benchmark (100 Hz)

########################## create batched data ##########################
n_envs = args.B
if n_envs > 1:
    data = mx.SceneData(model, batch=(n_envs,))
else:
    data = mx.SceneData(model)

########################## setup control ##########################
n_robots = args.N
actuators_per_robot = 8  # 7 joints + 1 gripper actuator
dofs_per_robot = 9  # 7 joints + 2 fingers
total_actuators = n_robots * actuators_per_robot

# Warmup position (stable initial pose with fingers)
warmup_qpos = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853, 0.04, 0.04])

########################## setup render (if needed) ##########################
if args.v:
    render = mx.render.RenderApp()
    render.__enter__()
    render.launch(model)
else:
    render = None


sim_dt = model.options.timestep

########################## Warmup: move to initial position ##########################
print(f"Warmup: {n_robots} robots to initial position (200 steps)...")

# Create warmup control for all robots
if n_envs > 1:
    warmup_ctrl = np.zeros((n_envs, total_actuators), dtype=np.float32)
    for robot_idx in range(n_robots):
        offset = robot_idx * actuators_per_robot
        warmup_ctrl[:, offset : offset + 7] = warmup_qpos[:7]
        warmup_ctrl[:, offset + 7] = warmup_qpos[7]  # gripper
else:
    warmup_ctrl = np.zeros(total_actuators, dtype=np.float32)
    for robot_idx in range(n_robots):
        offset = robot_idx * actuators_per_robot
        warmup_ctrl[offset : offset + 7] = warmup_qpos[:7]
        warmup_ctrl[offset + 7] = warmup_qpos[7]  # gripper

for i in range(200):
    data.actuator_ctrls = warmup_ctrl
    model.step(data)

########################## Benchmark ##########################
print(f"Benchmark: 1000 steps with {n_robots} robots (random motion)...")

benchmark_steps = 1000
ref_pos = warmup_qpos[:7].copy()  # Use warmup position as reference

if args.v:
    # Decoupled rendering mode using render_loop
    step_counter = [0]
    benchmark_complete = [False]
    t_start = [None]
    t_end = [None]

    def phys_step():
        if benchmark_complete[0]:
            return
        if t_start[0] is None:
            t_start[0] = time.perf_counter()

        i = step_counter[0]
        if i < benchmark_steps:
            # Create control with independent random noise for each robot
            if n_envs > 1:
                ctrl = np.zeros((n_envs, total_actuators), dtype=np.float32)
            else:
                ctrl = np.zeros(total_actuators, dtype=np.float32)

            for robot_idx in range(n_robots):
                offset = robot_idx * actuators_per_robot
                # Generate independent noise for each robot [-0.2, 0.2]
                if n_envs > 1:
                    noise = np.random.uniform(-0.2, 0.2, (n_envs, 7)).astype(np.float32)
                    ctrl[:, offset : offset + 7] = ref_pos + noise
                    ctrl[:, offset + 7] = warmup_qpos[7]
                else:
                    noise = np.random.uniform(-0.2, 0.2, 7).astype(np.float32)
                    ctrl[offset : offset + 7] = ref_pos + noise
                    ctrl[offset + 7] = warmup_qpos[7]

            data.actuator_ctrls = ctrl
            model.step(data)
            step_counter[0] += 1
        else:
            t_end[0] = time.perf_counter()
            benchmark_complete[0] = True
            print(f"per env: {benchmark_steps / (t_end[0] - t_start[0]):,.2f} FPS")
            print(
                f"total  : {benchmark_steps / (t_end[0] - t_start[0]) * n_envs:,.2f} FPS"
            )

    def render_func():
        render.sync(data)

    run.render_loop(sim_dt, 60, phys_step, render_func)
else:
    # Pure benchmark without rendering
    t0 = time.perf_counter()
    for i in range(benchmark_steps):
        # Create control with independent random noise for each robot
        if n_envs > 1:
            ctrl = np.zeros((n_envs, total_actuators), dtype=np.float32)
        else:
            ctrl = np.zeros(total_actuators, dtype=np.float32)

        for robot_idx in range(n_robots):
            offset = robot_idx * actuators_per_robot
            # Generate independent noise for each robot [-0.2, 0.2]
            if n_envs > 1:
                noise = np.random.uniform(-0.2, 0.2, (n_envs, 7)).astype(np.float32)
                ctrl[:, offset : offset + 7] = ref_pos + noise
                ctrl[:, offset + 7] = warmup_qpos[7]
            else:
                noise = np.random.uniform(-0.2, 0.2, 7).astype(np.float32)
                ctrl[offset : offset + 7] = ref_pos + noise
                ctrl[offset + 7] = warmup_qpos[7]

        data.actuator_ctrls = ctrl
        model.step(data)
    t1 = time.perf_counter()
    print(f"per env: {benchmark_steps / (t1 - t0):,.2f} FPS")
    print(f"total  : {benchmark_steps / (t1 - t0) * n_envs:,.2f} FPS")

# Cleanup render
if render:
    try:
        render.__exit__(None, None, None)
    except:
        pass
