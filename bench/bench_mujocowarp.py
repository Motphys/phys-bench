import argparse
import json
import os
import sys
import time

import warp as wp
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
parser.add_argument(
    "-N", type=int, default=1, choices=[1, 5, 10], help="Number of robots"
)
parser.add_argument(
    "-B", type=int, default=1, help="Batch size / parallel environments"
)
parser.add_argument(
    "-v", action="store_true", default=False, help="Enable visualization"
)
parser.add_argument(
    "--mode",
    type=str,
    default="franka_only",
    choices=["franka_only", "franka_grasp"],
    help="Scenario: franka_only or franka_grasp",
)
parser.add_argument(
    "--object",
    type=str,
    default="ball",
    choices=["ball", "cube", "bottle"],
    help="Object for grasp mode",
)
parser.add_argument(
    "-r",
    action="store_true",
    default=False,
    help="Random noise during grasp benchmark phase",
)
parser.add_argument(
    "--clutter",
    action="store_true",
    default=False,
    help="Fill scene with 200+ dynamic bottles (random mode only)",
)

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


def generate_clutter_positions(robot_pos, min_radius=0.3, spacing=0.15, min_count=40):
    """Generate positions for bottles in concentric circles around robot_pos.

    Note: scene_bottle.xml has bottle at pos="0.65 0 0.036" relative to worldbody,
    so we subtract 0.65 from x to compensate.
    """
    positions = []
    radius = min_radius
    z_offset = 0
    while len(positions) < min_count:
        circumference = 2 * np.pi * radius
        n_bottles = int(circumference / spacing)
        if n_bottles == 0:
            n_bottles = 1
        for j in range(n_bottles):
            if len(positions) >= min_count:
                break
            angle = 2 * np.pi * j / n_bottles
            x = robot_pos[0] + radius * np.cos(angle) - 0.65
            y = robot_pos[1] + radius * np.sin(angle)
            positions.append((x, y, z_offset))
        radius += spacing
        z_offset += 0.1  # Stack bottles vertically as we add more rings
    return positions


def create_multi_robot_scene(
    n_robots, positions, mode, object_type=None, clutter=False
):
    """Generate MJCF XML with multiple robots using MuJoCo spec API"""
    robot_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../assets/franka_emika_panda/panda.xml"
        )
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

        # Add objects for franka_grasp mode
        if mode == "franka_grasp" and object_type:
            obj_x = pos[0] + 0.65
            obj_y = pos[1]
            obj_z = 0.02

            if object_type == "ball":
                obj_body = spec.worldbody.add_body()
                obj_body.name = f"object{i}"
                obj_body.pos = [obj_x, obj_y, obj_z]
                obj_body.add_freejoint()
                obj_geom = obj_body.add_geom()
                obj_geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
                obj_geom.size = np.array([0.02, 0.0, 0.0])
                obj_geom.rgba = [0.0, 1.0, 0.0, 1.0]
            elif object_type == "cube":
                obj_body = spec.worldbody.add_body()
                obj_body.name = f"object{i}"
                obj_body.pos = [obj_x, obj_y, obj_z]
                obj_body.add_freejoint()
                obj_geom = obj_body.add_geom()
                obj_geom.type = mujoco.mjtGeom.mjGEOM_BOX
                obj_geom.size = [0.02, 0.02, 0.02]
                obj_geom.rgba = [1.0, 0.0, 0.0, 1.0]
            elif object_type == "bottle":
                # Load and attach bottle model
                bottle_path = os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__), "../assets/objects/scene_bottle.xml"
                    )
                )
                bottle_frame = spec.worldbody.add_frame()
                bottle_frame.name = f"bottle{i}_frame"
                bottle_frame.pos = [obj_x, obj_y, 0]
                bottle_copy = mujoco.MjSpec.from_file(bottle_path)
                spec.attach(bottle_copy, prefix=f"bottle{i}_", frame=bottle_frame)

    # Add clutter bottles for franka_only mode
    if mode == "franka_only" and clutter:
        bottle_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "../assets/objects/scene_bottle.xml"
            )
        )
        bottle_counter = 0
        for i, pos in enumerate(positions):
            clutter_positions = generate_clutter_positions(pos)
            for clutter_pos in clutter_positions:
                clutter_frame = spec.worldbody.add_frame()
                clutter_frame.name = f"clutter{bottle_counter}_frame"
                clutter_frame.pos = clutter_pos
                bottle_copy = mujoco.MjSpec.from_file(bottle_path)
                spec.attach(
                    bottle_copy, prefix=f"clutter{bottle_counter}_", frame=clutter_frame
                )
                bottle_counter += 1

    # Compile the spec to a model
    mjm = spec.compile()
    return mjm


# Object configurations for grasp mode
OBJECT_CONFIGS = {
    "ball": {
        "grasp_qpos": np.array(
            [-1.0323, 1.7628, 1.4904, -1.6749, -1.7715, 1.6293, 1.4417, 0.04, 0.04]
        ),
        "lift_steps": 100,
    },
    "cube": {
        "grasp_qpos": np.array(
            [-1.0104, 1.5623, 1.3601, -1.6840, -1.5863, 1.7810, 1.4598, 0.04, 0.04]
        ),
        "lift_steps": 50,
    },
    "bottle": {
        "grasp_qpos": np.array(
            [-0.9937, 1.4588, 1.3058, -1.6924, -1.4882, 1.8461, 1.4577, 0.04, 0.04]
        ),
        "lift_steps": 50,
    },
}

########################## load model ##########################
n_robots = args.N
positions = get_robot_positions(n_robots)

if args.mode == "franka_only":
    # Franka only mode: create multi-robot scene
    mjm = create_multi_robot_scene(n_robots, positions, "franka_only", clutter=args.clutter)
    mjm.opt.timestep = 0.01
else:
    # Franka grasp mode: create multi-robot scene with objects (for all N)
    mjm = create_multi_robot_scene(n_robots, positions, "franka_grasp", args.object)
    mjm.opt.timestep = 0.002

########################## create batched data ##########################
n_envs = args.B

# Put model and data on GPU device using MuJoCo-Warp
m = mjw.put_model(mjm)
# Increase contact and constraint limits for clutter mode
if args.mode == "franka_only" and args.clutter:
    d = mjw.make_data(mjm, nworld=n_envs, nconmax=20000, njmax=20000)
else:
    d = mjw.make_data(mjm, nworld=n_envs, nconmax=40 * n_robots, njmax=150 * n_robots)

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


with wp.ScopedCapture(device="cuda") as capture:
    mjw.step(m, d)
step_graph = capture.graph


########################## mode-specific warmup and benchmark ##########################
if args.mode == "franka_only":
    print(f"Warmup: {n_robots} robots to initial position (200 steps)...")

    # Create warmup control for all robots
    warmup_ctrl = np.zeros((n_envs, total_actuators), dtype=np.float32)
    for robot_idx in range(n_robots):
        offset = robot_idx * actuators_per_robot
        warmup_ctrl[:, offset : offset + 7] = warmup_qpos[:7]
        warmup_ctrl[:, offset + 7] = warmup_qpos[7]

    for i in range(200):
        wp.copy(d.ctrl, wp.array(warmup_ctrl, dtype=wp.float32))
        wp.capture_launch(step_graph)

    print(f"Benchmark: 1000 steps with {n_robots} robots (random motion)...")
    benchmark_steps = 1000
    ref_pos = warmup_qpos[:7].copy()
    
    ref_pos_wp = wp.array(ref_pos, dtype=float)
    step_counter_wp = wp.array([0], dtype=int)

    @wp.kernel
    def randomize_ctrl_kernel(ref_pos: wp.array(dtype=float), step_counter_wp: wp.array(dtype=int), ctrl_per_robot: int, ctrls: wp.array2d(dtype=float)):
        step = step_counter_wp[0]
        wid, robot_id, tid = wp.tid()
        noise = wp.randf(wp.uint32(wid * ctrls.shape[1] + tid + step), -0.025, 0.025)
        ctrls[wid, robot_id * ctrl_per_robot + tid] = ref_pos[tid] + noise

        if wid == 0 and tid == 0:
            step_counter_wp[0] = step + 1

    with wp.ScopedCapture(device="cuda") as capture:
        wp.launch(randomize_ctrl_kernel, dim=(n_envs, n_robots, 7), inputs=[ref_pos_wp, step_counter_wp, actuators_per_robot, d.ctrl], block_dim=32)
    randomize_graph = capture.graph


    if args.v:
        t0 = time.perf_counter()
        for i in range(benchmark_steps):
            wp.capture_launch(randomize_graph)
            wp.capture_launch(step_graph)
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

            mjd_cpu.qpos[:] = d.qpos.numpy()[0]
            mjd_cpu.qvel[:] = d.qvel.numpy()[0]
            # mjd_cpu.ctrl[:] = ctrl[0]
            mujoco.mj_forward(mjm, mjd_cpu)
            viewer.sync()

        t1 = time.perf_counter()
        viewer.close()
    else:
        t0 = time.perf_counter()
        for i in range(benchmark_steps):
            wp.capture_launch(randomize_graph)
            wp.capture_launch(step_graph)
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

    print(f"per env: {benchmark_steps / (t1 - t0):,.2f} FPS")
    print(f"total  : {benchmark_steps / (t1 - t0) * n_envs:,.2f} FPS")

else:  # franka_grasp mode
    config = OBJECT_CONFIGS[args.object]
    grasp_qpos = config["grasp_qpos"]
    lift_steps = config["lift_steps"]
    lift_qpos = np.array(
        [-1.0426, 1.4028, 1.5634, -1.7114, -1.4055, 1.6015, 1.4510, 0.0, 0.0]
    )

    # Initialize ctrl and qpos for grasp mode
    ctrl_array = np.zeros((n_envs, total_actuators), dtype=np.float32)
    for robot_idx in range(n_robots):
        offset = robot_idx * actuators_per_robot
        ctrl_array[:, offset : offset + 7] = grasp_qpos[:7]
        ctrl_array[:, offset + 7] = grasp_qpos[7]
    wp.copy(d.ctrl, wp.array(ctrl_array, dtype=wp.float32))

    qpos_init = d.qpos.numpy()
    dofs_per_robot = 9  # 7 joints + 2 fingers (DOFs, not actuators)
    dofs_per_object = 7  # freejoint (3 pos + 4 quat)
    dofs_per_pair = dofs_per_robot + dofs_per_object  # 16
    for env_idx in range(n_envs):
        for robot_idx in range(n_robots):
            robot_offset = robot_idx * dofs_per_pair
            qpos_init[env_idx, robot_offset : robot_offset + 9] = grasp_qpos
            # Initialize object freejoint position only, keep quaternion at default
            obj_offset = robot_offset + dofs_per_robot
            obj_x = positions[robot_idx][0] + 0.65
            obj_y = positions[robot_idx][1]
            obj_z = 0.02
            qpos_init[env_idx, obj_offset : obj_offset + 3] = [obj_x, obj_y, obj_z]
            # Don't override quaternion - keep default from model
    wp.copy(d.qpos, wp.array(qpos_init, dtype=wp.float32))

    qvel_init = np.zeros_like(d.qvel.numpy())
    wp.copy(d.qvel, wp.array(qvel_init, dtype=wp.float32))

    print("Warmup Phase 1: Grasping (100 steps)...")
    # Gradually close gripper during warmup
    for i in range(100):
        gripper_val = 0.04 * (1.0 - i / 100.0)
        for robot_idx in range(n_robots):
            offset = robot_idx * actuators_per_robot
            ctrl_array[:, offset + 7] = gripper_val
        wp.copy(d.ctrl, wp.array(ctrl_array, dtype=wp.float32))
        wp.capture_launch(step_graph)

    print(f"Warmup Phase 2: Lifting ({lift_steps} steps)...")
    # Create lift control
    for robot_idx in range(n_robots):
        offset = robot_idx * actuators_per_robot
        ctrl_array[:, offset : offset + 7] = lift_qpos[:7]
        ctrl_array[:, offset + 7] = 0.0

    for i in range(lift_steps):
        wp.copy(d.ctrl, wp.array(ctrl_array, dtype=wp.float32))
        wp.capture_launch(step_graph)

    print("Benchmark: 500 steps...")
    benchmark_steps = 500
    ref_pos = lift_qpos[:7].copy()

    ref_pos_wp = wp.array(ref_pos, dtype=float)
    step_counter_wp = wp.array([0], dtype=int)

    @wp.kernel
    def randomize_ctrl_kernel(ref_pos: wp.array(dtype=float), step_counter_wp: wp.array(dtype=int), ctrl_per_robot: int, ctrls: wp.array2d(dtype=float)):
        step = step_counter_wp[0]
        wid, robot_id, tid = wp.tid()
        noise = wp.randf(wp.uint32(wid * ctrls.shape[1] + tid + step), -0.025, 0.025)
        ctrls[wid, robot_id * ctrl_per_robot + tid] = ref_pos[tid] + noise

        if wid == 0 and tid == 0:
            step_counter_wp[0] = step + 1

    with wp.ScopedCapture(device="cuda") as capture:
        wp.launch(randomize_ctrl_kernel, dim=(n_envs, n_robots, 7), inputs=[ref_pos_wp, step_counter_wp, actuators_per_robot, d.ctrl], block_dim=32)
    randomize_graph = capture.graph


    t0 = time.perf_counter()
    for i in range(benchmark_steps):
        if args.r and i % 2 == 0:
            wp.capture_launch(randomize_graph)

        wp.capture_launch(step_graph)
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
