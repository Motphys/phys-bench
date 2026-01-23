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
parser.add_argument("-N", type=int, default=1, choices=[1, 5, 10], help="Number of robots")
parser.add_argument("-B", type=int, default=1, help="Batch size / parallel environments")
parser.add_argument("-v", action="store_true", default=False, help="Enable visualization")
parser.add_argument("--mode", type=str, default="random", choices=["random", "grasp"], help="Scenario: random or grasp")
parser.add_argument("--object", type=str, default="ball", choices=["ball", "cube", "bottle"], help="Object for grasp mode")
parser.add_argument("-r", action="store_true", default=False, help="Random noise during grasp benchmark phase")

args = parser.parse_args()


def get_robot_positions(n_robots):
    """Calculate robot positions - spread them out in a grid layout"""
    positions = []

    if n_robots == 1:
        positions.append((0, 0, 0))
    elif n_robots == 5:
        for i in range(5):
            x = (i - 2) * 2.0  # -4, -2, 0, 2, 4
            positions.append((x, 0, 0))
    elif n_robots == 10:
        for i in range(10):
            row = i // 5
            col = i % 5
            x = (col - 2) * 2.0  # -4, -2, 0, 2, 4
            y = (row - 0.5) * 2.0  # -1, 1
            positions.append((x, y, 0))

    return positions


def create_multi_robot_scene(n_robots, positions, mode, object_type=None):
    """Generate MJCF XML with multiple robots using MuJoCo spec API"""
    robot_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../assets/franka_emika_panda/mjx_panda.xml")
    )
    share_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../assets/grasp/share.xml")
    )

    # Load share.xml as base (contains ground plane, lighting, visual settings)
    spec = mujoco.MjSpec.from_file(share_path)

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

        # Add objects for grasp mode
        if mode == "grasp" and object_type:
            obj_x = pos[0] + 0.65
            obj_y = pos[1]
            obj_z = 0.02

            if object_type == "ball":
                obj_body = spec.worldbody.add_body()
                obj_body.name = f"object{i}"
                obj_body.pos = [obj_x, obj_y, obj_z]
                obj_geom = obj_body.add_geom()
                obj_geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
                obj_geom.size = [0.02]
                obj_geom.rgba = [0.0, 1.0, 0.0, 1.0]
            elif object_type == "cube":
                obj_body = spec.worldbody.add_body()
                obj_body.name = f"object{i}"
                obj_body.pos = [obj_x, obj_y, obj_z]
                obj_geom = obj_body.add_geom()
                obj_geom.type = mujoco.mjtGeom.mjGEOM_BOX
                obj_geom.size = [0.02, 0.02, 0.02]
                obj_geom.rgba = [1.0, 0.0, 0.0, 1.0]
            elif object_type == "bottle":
                # Load and attach bottle model
                bottle_path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "../assets/objects/scene_bottle.xml")
                )
                bottle_frame = spec.worldbody.add_frame()
                bottle_frame.name = f"bottle{i}_frame"
                bottle_frame.pos = [obj_x, obj_y, 0]
                bottle_copy = mujoco.MjSpec.from_file(bottle_path)
                spec.attach(bottle_copy, prefix=f"bottle{i}_", frame=bottle_frame)

    # Compile the spec to a model
    mjm = spec.compile()
    return mjm


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
n_robots = args.N
positions = get_robot_positions(n_robots)

if args.mode == "random":
    # Random mode: create multi-robot scene
    mjm = create_multi_robot_scene(n_robots, positions, "random")
    mjm.opt.timestep = 0.01
elif args.mode == "grasp" and args.N == 1:
    # Grasp mode with N=1: load pre-made grasp scene
    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), f"../assets/grasp/mjx_pick_{args.object}.xml")
    )
    mjm = mujoco.MjModel.from_xml_path(model_path)
    mjm.opt.timestep = 0.002
else:
    # Grasp mode with N>1: create multi-robot scene with objects
    mjm = create_multi_robot_scene(n_robots, positions, "grasp", args.object)
    mjm.opt.timestep = 0.002

########################## create batched data ##########################
n_envs = args.B

# Put model and data on GPU device using MuJoCo-Warp
m = mjw.put_model(mjm)
d = mjw.make_data(mjm, nworld=n_envs)

########################## setup control ##########################
actuators_per_robot = 8  # 7 joints + 1 gripper actuator
total_actuators = n_robots * actuators_per_robot

# Warmup position for random mode
warmup_qpos = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853, 0.04, 0.04])

########################## setup render (if needed) ##########################
if args.v:
    mjd_cpu = mujoco.MjData(mjm)
    viewer = mujoco.viewer.launch_passive(mjm, mjd_cpu)
else:
    viewer = None
    mjd_cpu = None

sim_dt = mjm.opt.timestep

########################## mode-specific warmup and benchmark ##########################
if args.mode == "random":
    print(f"Warmup: {n_robots} robots to initial position (200 steps)...")

    # Create warmup control for all robots
    warmup_ctrl = np.zeros((n_envs, total_actuators), dtype=np.float32)
    for robot_idx in range(n_robots):
        offset = robot_idx * actuators_per_robot
        warmup_ctrl[:, offset : offset + 7] = warmup_qpos[:7]
        warmup_ctrl[:, offset + 7] = warmup_qpos[7]

    for i in range(200):
        wp.copy(d.ctrl, wp.array(warmup_ctrl, dtype=wp.float32))
        mjw.step(m, d)

    print(f"Benchmark: 1000 steps with {n_robots} robots (random motion)...")
    benchmark_steps = 1000
    ref_pos = warmup_qpos[:7].copy()

    if args.v:
        t0 = time.perf_counter()
        for i in range(benchmark_steps):
            ctrl = np.zeros((n_envs, total_actuators), dtype=np.float32)

            for robot_idx in range(n_robots):
                offset = robot_idx * actuators_per_robot
                noise = np.random.uniform(-0.2, 0.2, (n_envs, 7)).astype(np.float32)
                ctrl[:, offset : offset + 7] = ref_pos + noise
                ctrl[:, offset + 7] = warmup_qpos[7]

            wp.copy(d.ctrl, wp.array(ctrl, dtype=wp.float32))
            mjw.step(m, d)

            mjd_cpu.qpos[:] = d.qpos.numpy()[0]
            mjd_cpu.qvel[:] = d.qvel.numpy()[0]
            mjd_cpu.ctrl[:] = ctrl[0]
            mujoco.mj_forward(mjm, mjd_cpu)
            viewer.sync()

        t1 = time.perf_counter()
        viewer.close()
    else:
        t0 = time.perf_counter()
        for i in range(benchmark_steps):
            ctrl = np.zeros((n_envs, total_actuators), dtype=np.float32)

            for robot_idx in range(n_robots):
                offset = robot_idx * actuators_per_robot
                noise = np.random.uniform(-0.2, 0.2, (n_envs, 7)).astype(np.float32)
                ctrl[:, offset : offset + 7] = ref_pos + noise
                ctrl[:, offset + 7] = warmup_qpos[7]

            wp.copy(d.ctrl, wp.array(ctrl, dtype=wp.float32))
            mjw.step(m, d)

        t1 = time.perf_counter()

    print(f"per env: {benchmark_steps / (t1 - t0):,.2f} FPS")
    print(f"total  : {benchmark_steps / (t1 - t0) * n_envs:,.2f} FPS")

else:  # grasp mode
    config = OBJECT_CONFIGS[args.object]
    grasp_qpos = config["grasp_qpos"]
    lift_steps = config["lift_steps"]
    lift_qpos = np.array([-1.0426, 1.4028, 1.5634, -1.7114, -1.4055, 1.6015, 1.4510, 0.0, 0.0])

    # Initialize ctrl and qpos for grasp mode
    if args.N == 1:
        # Single robot initialization
        ctrl_array = np.zeros((n_envs, 8), dtype=np.float32)
        ctrl_array[:, :7] = grasp_qpos[:7]
        ctrl_array[:, 7] = grasp_qpos[7]
        wp.copy(d.ctrl, wp.array(ctrl_array, dtype=wp.float32))

        qpos_init = d.qpos.numpy()
        for env_idx in range(n_envs):
            qpos_init[env_idx, :9] = grasp_qpos
        wp.copy(d.qpos, wp.array(qpos_init, dtype=wp.float32))

        qvel_init = np.zeros_like(d.qvel.numpy())
        wp.copy(d.qvel, wp.array(qvel_init, dtype=wp.float32))
    else:
        # Multi-robot initialization
        ctrl_array = np.zeros((n_envs, total_actuators), dtype=np.float32)
        for robot_idx in range(n_robots):
            offset = robot_idx * actuators_per_robot
            ctrl_array[:, offset : offset + 7] = grasp_qpos[:7]
            ctrl_array[:, offset + 7] = grasp_qpos[7]
        wp.copy(d.ctrl, wp.array(ctrl_array, dtype=wp.float32))

        qpos_init = d.qpos.numpy()
        dofs_per_robot = 9  # 7 joints + 2 fingers (DOFs, not actuators)
        for env_idx in range(n_envs):
            for robot_idx in range(n_robots):
                offset = robot_idx * dofs_per_robot
                qpos_init[env_idx, offset : offset + 9] = grasp_qpos
        wp.copy(d.qpos, wp.array(qpos_init, dtype=wp.float32))

        qvel_init = np.zeros_like(d.qvel.numpy())
        wp.copy(d.qvel, wp.array(qvel_init, dtype=wp.float32))

    print("Warmup Phase 1: Grasping (100 steps)...")
    # Gradually close gripper during warmup
    for i in range(100):
        gripper_val = 0.04 * (1.0 - i / 100.0)
        if args.N == 1:
            ctrl_array[:, 7] = gripper_val
        else:
            for robot_idx in range(n_robots):
                offset = robot_idx * actuators_per_robot
                ctrl_array[:, offset + 7] = gripper_val
        wp.copy(d.ctrl, wp.array(ctrl_array, dtype=wp.float32))
        mjw.step(m, d)

    print(f"Warmup Phase 2: Lifting ({lift_steps} steps)...")
    # Create lift control
    if args.N == 1:
        ctrl_array[:, :7] = lift_qpos[:7]
        ctrl_array[:, 7] = 0.0
    else:
        for robot_idx in range(n_robots):
            offset = robot_idx * actuators_per_robot
            ctrl_array[:, offset : offset + 7] = lift_qpos[:7]
            ctrl_array[:, offset + 7] = 0.0

    for i in range(lift_steps):
        wp.copy(d.ctrl, wp.array(ctrl_array, dtype=wp.float32))
        mjw.step(m, d)

    print("Benchmark: 500 steps...")
    benchmark_steps = 500
    ref_pos = lift_qpos[:7].copy()

    t0 = time.perf_counter()
    for i in range(benchmark_steps):
        if args.r and i % 2 == 0:
            noise = np.random.uniform(-0.025, 0.025, (n_envs, 7)).astype(np.float32)
            if args.N == 1:
                ctrl_array[:, :7] = ref_pos + noise
            else:
                for robot_idx in range(n_robots):
                    offset = robot_idx * actuators_per_robot
                    ctrl_array[:, offset : offset + 7] = ref_pos + noise
        else:
            if args.N == 1:
                ctrl_array[:, :7] = ref_pos
            else:
                for robot_idx in range(n_robots):
                    offset = robot_idx * actuators_per_robot
                    ctrl_array[:, offset : offset + 7] = ref_pos

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
