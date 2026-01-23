import argparse
import os
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

parser = argparse.ArgumentParser()
parser.add_argument("-B", type=int, default=1)  # batch size
parser.add_argument("-N", type=int, default=5, choices=[1, 5, 10])  # number of robots
parser.add_argument("-v", action="store_true", default=False)  # visualize
args = parser.parse_args()


def get_robot_positions(n_robots):
    """Calculate robot positions - spread them out in a grid layout"""
    positions = []

    if n_robots == 1:
        positions.append((0, 0, 0))
    elif n_robots == 5:
        # 5 robots in a line along x-axis with 2m spacing
        for i in range(5):
            x = (i - 2) * 2.0  # -4, -2, 0, 2, 4
            positions.append((x, 0, 0))
    elif n_robots == 10:
        # 10 robots in 2 rows of 5 with 2m spacing
        for i in range(10):
            row = i // 5
            col = i % 5
            x = (col - 2) * 2.0  # -4, -2, 0, 2, 4
            y = (row - 0.5) * 2.0  # -1, 1
            positions.append((x, y, 0))

    return positions


def create_multi_robot_mjcf(n_robots, positions):
    """Generate MJCF XML with multiple robots at specified positions using MuJoCo spec API"""
    # Load the base robot model
    robot_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../assets/franka_emika_panda/mjx_panda.xml")
    )

    # Create a new spec for the multi-robot scene
    spec = mujoco.MjSpec()
    spec.worldbody.name = "world"

    # Add ground plane
    ground = spec.worldbody.add_geom()
    ground.type = mujoco.mjtGeom.mjGEOM_PLANE
    ground.size = [0, 0, 0.05]
    ground.rgba = [0.9, 0.9, 0.9, 1]

    # Add lighting
    light = spec.worldbody.add_light()
    light.name = "top"
    light.pos = [0, 0, 2]
    light.mode = mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM

    # Attach multiple robots
    for i, pos in enumerate(positions):
        # Create a frame for this robot at the desired position
        frame = spec.worldbody.add_frame()
        frame.name = f"robot{i}_frame"
        frame.pos = pos

        # Load robot spec
        robot_copy = mujoco.MjSpec.from_file(robot_path)

        # Attach to the frame with prefix
        spec.attach(robot_copy, prefix=f"robot{i}_", frame=frame)

    # Compile the spec to a model
    mjm = spec.compile()
    return mjm


########################## load model ##########################
n_robots = args.N
positions = get_robot_positions(n_robots)

# Create multi-robot model
mjm = create_multi_robot_mjcf(n_robots, positions)
mjm.opt.timestep = 0.01  # Match genesis benchmark (100 Hz)

########################## create batched data ##########################
n_envs = args.B

# Put model and data on GPU device using MuJoCo-Warp
m = mjw.put_model(mjm)
d = mjw.make_data(mjm, nworld=n_envs)

########################## setup control ##########################
actuators_per_robot = 8  # 7 joints + 1 gripper actuator
total_actuators = n_robots * actuators_per_robot

# Warmup position (stable initial pose with fingers)
warmup_qpos = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853, 0.04, 0.04])

########################## setup render (if needed) ##########################
if args.v:
    # Create CPU data for visualization
    mjd_cpu = mujoco.MjData(mjm)
    # Launch passive viewer for real-time visualization
    viewer = mujoco.viewer.launch_passive(mjm, mjd_cpu)
else:
    viewer = None
    mjd_cpu = None

sim_dt = mjm.opt.timestep

########################## Warmup: move to initial position ##########################
print(f"Warmup: {n_robots} robots to initial position (200 steps)...")

# Create warmup control for all robots
warmup_ctrl = np.zeros((n_envs, total_actuators), dtype=np.float32)
for robot_idx in range(n_robots):
    offset = robot_idx * actuators_per_robot
    warmup_ctrl[:, offset : offset + 7] = warmup_qpos[:7]
    warmup_ctrl[:, offset + 7] = warmup_qpos[7]  # gripper

for i in range(200):
    wp.copy(d.ctrl, wp.array(warmup_ctrl, dtype=wp.float32))
    mjw.step(m, d)

########################## Benchmark ##########################
print(f"Benchmark: 1000 steps with {n_robots} robots (random motion)...")

benchmark_steps = 1000
ref_pos = warmup_qpos[:7].copy()  # Use warmup position as reference

if args.v:
    # Visualization mode with viewer
    t0 = time.perf_counter()
    for i in range(benchmark_steps):
        # Create control with independent random noise for each robot
        ctrl = np.zeros((n_envs, total_actuators), dtype=np.float32)

        for robot_idx in range(n_robots):
            offset = robot_idx * actuators_per_robot
            # Generate independent noise for each robot [-0.2, 0.2]
            noise = np.random.uniform(-0.2, 0.2, (n_envs, 7)).astype(np.float32)
            ctrl[:, offset : offset + 7] = ref_pos + noise
            ctrl[:, offset + 7] = warmup_qpos[7]

        wp.copy(d.ctrl, wp.array(ctrl, dtype=wp.float32))
        mjw.step(m, d)

        # Update viewer with first environment
        mjd_cpu.qpos[:] = d.qpos.numpy()[0]
        mjd_cpu.qvel[:] = d.qvel.numpy()[0]
        mjd_cpu.ctrl[:] = ctrl[0]
        mujoco.mj_forward(mjm, mjd_cpu)
        viewer.sync()

    t1 = time.perf_counter()
    viewer.close()
else:
    # Pure benchmark without rendering
    t0 = time.perf_counter()
    for i in range(benchmark_steps):
        # Create control with independent random noise for each robot
        ctrl = np.zeros((n_envs, total_actuators), dtype=np.float32)

        for robot_idx in range(n_robots):
            offset = robot_idx * actuators_per_robot
            # Generate independent noise for each robot [-0.2, 0.2]
            noise = np.random.uniform(-0.2, 0.2, (n_envs, 7)).astype(np.float32)
            ctrl[:, offset : offset + 7] = ref_pos + noise
            ctrl[:, offset + 7] = warmup_qpos[7]

        wp.copy(d.ctrl, wp.array(ctrl, dtype=wp.float32))
        mjw.step(m, d)

    t1 = time.perf_counter()

print(f"per env: {benchmark_steps / (t1 - t0):,.2f} FPS")
print(f"total  : {benchmark_steps / (t1 - t0) * n_envs:,.2f} FPS")
