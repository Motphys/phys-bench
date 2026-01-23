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
parser.add_argument("-v", action="store_true", default=False)  # visualize
parser.add_argument("-r", action="store_true", default=False)  # random action
args = parser.parse_args()

########################## load model ##########################
model_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../assets/grasp/mjx_pick_cube.xml")
)

# Load MuJoCo model
mjm = mujoco.MjModel.from_xml_path(model_path)
mjm.opt.timestep = 0.002  # Match genesis benchmark (100 Hz)

########################## create batched data ##########################
n_envs = args.B

# Put model and data on GPU device using MuJoCo-Warp
m = mjw.put_model(mjm)
d = mjw.make_data(mjm, nworld=n_envs)

########################## setup control ##########################
# Reference positions
grasp_qpos = np.array(
    [-1.0104, 1.5623, 1.3601, -1.6840, -1.5863, 1.7810, 1.4598, 0.04, 0.04]
)
lift_qpos = np.array(
    [-1.0426, 1.4028, 1.5634, -1.7114, -1.4055, 1.6015, 1.4510, 0.0, 0.0]
)

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

########################## Initialize ctrl and qpos ##########################
# IMPORTANT: Set ctrl first, then qpos, to ensure consistency
# Create initial control - set to grasp position with gripper open
ctrl_array = np.zeros((n_envs, 8), dtype=np.float32)
ctrl_array[:, :7] = grasp_qpos[:7]
ctrl_array[:, 7] = grasp_qpos[7]  # Gripper open (0.04)
wp.copy(d.ctrl, wp.array(ctrl_array, dtype=wp.float32))

# Initialize qpos - set robot joints to grasp position
qpos_init = d.qpos.numpy()
for env_idx in range(n_envs):
    qpos_init[env_idx, :9] = grasp_qpos
wp.copy(d.qpos, wp.array(qpos_init, dtype=wp.float32))

# Initialize qvel - set all velocities to zero to prevent jittering
qvel_init = np.zeros_like(d.qvel.numpy())
wp.copy(d.qvel, wp.array(qvel_init, dtype=wp.float32))

########################## Warmup Phase 1: Grasp (100 steps) ##########################
print("Warmup Phase 1: Grasping (100 steps)...")

# Gradually close gripper during warmup
for i in range(100):
    # Linearly interpolate gripper from 0.04 to 0.0
    gripper_val = 0.04 * (1.0 - i / 100.0)
    ctrl_array[:, 7] = gripper_val
    wp.copy(d.ctrl, wp.array(ctrl_array, dtype=wp.float32))
    mjw.step(m, d)

########################## Warmup Phase 2: Lift (50 steps) ##########################
print("Warmup Phase 2: Lifting (50 steps)...")

# Create lift control for all envs
ctrl_array[:, :7] = lift_qpos[:7]
ctrl_array[:, 7] = 0.0  # Keep gripper closed

for i in range(50):
    wp.copy(d.ctrl, wp.array(ctrl_array, dtype=wp.float32))
    mjw.step(m, d)

########################## Benchmark ##########################
print("Benchmark: 500 steps...")

benchmark_steps = 500
ref_pos = lift_qpos[:7].copy()

t0 = time.perf_counter()
for i in range(benchmark_steps):
    if args.r and i % 2 == 0:
        noise = np.random.uniform(-0.025, 0.025, (n_envs, 7)).astype(np.float32)
        ctrl_array[:, :7] = ref_pos + noise
    else:
        ctrl_array[:, :7] = ref_pos

    wp.copy(d.ctrl, wp.array(ctrl_array, dtype=wp.float32))
    mjw.step(m, d)

    if args.v:
        mjd_cpu.qpos[:] = d.qpos.numpy()[0]
        mjd_cpu.qvel[:] = d.qvel.numpy()[0]
        mjd_cpu.ctrl[:] = ctrl_array[0]
        mujoco.mj_forward(mjm, mjd_cpu)
        viewer.sync()

t1 = time.perf_counter()

if args.v:
    viewer.close()

print(f"per env: {benchmark_steps / (t1 - t0):,.2f} FPS")
print(f"total  : {benchmark_steps / (t1 - t0) * n_envs:,.2f} FPS")
