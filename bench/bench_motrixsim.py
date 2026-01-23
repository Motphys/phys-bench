import argparse
import os
import time

import motrixsim as mx
import numpy as np
from motrixsim import run

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
if args.mode == "random":
    # Random mode: load base scene and attach N robots
    scene_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "xml/base_scene.xml")
    )
    scene = mx.msd.from_file(scene_path)

    robot_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../assets/franka_emika_panda/panda.xml")
    )
    robot = mx.msd.from_file(robot_path)

    positions, rotations = get_robot_positions(args.N)
    for i in range(args.N):
        scene.attach(
            robot,
            other_prefix=f"robot{i}_",
            other_translation=positions[i],
            other_rotation=rotations[i],
        )

    model = scene.build()
    model.options.timestep = 0.01

elif args.mode == "grasp" and args.N == 1:
    # Grasp mode with N=1: load pre-made grasp scene
    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), f"../assets/grasp/pick_{args.object}.xml")
    )
    model = mx.load_model(model_path)
    model.options.timestep = 0.01

else:
    # Grasp mode with N>1: compose scene programmatically
    scene_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "xml/base_scene.xml")
    )
    scene = mx.msd.from_file(scene_path)

    robot_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../assets/franka_emika_panda/panda.xml")
    )

    positions, rotations = get_robot_positions(args.N)
    for i in range(args.N):
        # Attach robot
        robot = mx.msd.from_file(robot_path)
        scene.attach(
            robot,
            other_prefix=f"robot{i}_",
            other_translation=positions[i],
            other_rotation=rotations[i],
        )

        # Attach object for each robot
        obj_x = positions[i][0] + 0.65
        obj_y = positions[i][1]
        obj_z = 0.02

        if args.object == "ball":
            # Create sphere body programmatically
            body = mx.msd.Body(f"object{i}")
            body.translation = (obj_x, obj_y, obj_z)
            geom = mx.msd.Geom(shape=mx.msd.Sphere(radius=0.02))
            body.geoms.append(geom)
            scene.bodies.append(body)
        elif args.object == "cube":
            # Create box body programmatically
            body = mx.msd.Body(f"object{i}")
            body.translation = (obj_x, obj_y, obj_z)
            geom = mx.msd.Geom(shape=mx.msd.Box(size=(0.02, 0.02, 0.02)))
            body.geoms.append(geom)
            scene.bodies.append(body)
        elif args.object == "bottle":
            # For bottle, we need to load and attach the bottle model
            bottle_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../assets/objects/scene_bottle.xml")
            )
            bottle = mx.msd.from_file(bottle_path)
            scene.attach(
                bottle,
                other_prefix=f"bottle{i}_",
                other_translation=(obj_x, obj_y, 0),
                other_rotation=(0, 0, 0, 1),
            )

    model = scene.build()
    model.options.timestep = 0.01

########################## create batched data ##########################
n_envs = args.B
if n_envs > 1:
    data = mx.SceneData(model, batch=(n_envs,))
else:
    data = mx.SceneData(model)

########################## setup control ##########################
n_robots = args.N
actuators_per_robot = 8  # 7 joints + 1 gripper actuator
total_actuators = n_robots * actuators_per_robot

# Warmup position for random mode
warmup_qpos = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853, 0.04, 0.04])

########################## setup render (if needed) ##########################
if args.v:
    render = mx.render.RenderApp()
    render.__enter__()
    render.launch(model)
else:
    render = None

sim_dt = model.options.timestep

########################## mode-specific warmup and benchmark ##########################
if args.mode == "random":
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

    for i in range(200):
        data.actuator_ctrls = warmup_ctrl
        model.step(data)

    print(f"Benchmark: 1000 steps with {n_robots} robots (random motion)...")
    benchmark_steps = 1000
    ref_pos = warmup_qpos[:7].copy()

    if args.v:
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
                if n_envs > 1:
                    ctrl = np.zeros((n_envs, total_actuators), dtype=np.float32)
                else:
                    ctrl = np.zeros(total_actuators, dtype=np.float32)

                for robot_idx in range(n_robots):
                    offset = robot_idx * actuators_per_robot
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
                print(f"total  : {benchmark_steps / (t_end[0] - t_start[0]) * n_envs:,.2f} FPS")

        def render_func():
            render.sync(data)

        run.render_loop(sim_dt, 60, phys_step, render_func)
    else:
        t0 = time.perf_counter()
        for i in range(benchmark_steps):
            if n_envs > 1:
                ctrl = np.zeros((n_envs, total_actuators), dtype=np.float32)
            else:
                ctrl = np.zeros(total_actuators, dtype=np.float32)

            for robot_idx in range(n_robots):
                offset = robot_idx * actuators_per_robot
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

else:  # grasp mode
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
            panda.set_dof_pos(data, init_pos)
        else:
            panda.set_dof_pos(data, grasp_qpos)
    else:
        # Multiple robots: initialize each separately
        for robot_idx in range(args.N):
            panda_index = model.get_body_index(f"robot{robot_idx}_link0")
            panda = model.get_body(panda_index)
            if n_envs > 1:
                init_pos = np.tile(grasp_qpos, (n_envs, 1))
                panda.set_dof_pos(data, init_pos)
            else:
                panda.set_dof_pos(data, grasp_qpos)

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
        data.actuator_ctrls = ctrl
        model.step(data)

    print(f"Warmup Phase 2: Lifting ({lift_steps} steps)...")
    for i in range(lift_steps):
        ctrl = make_ctrl(lift_qpos[:7], 0.0, n_envs, n_robots)
        data.actuator_ctrls = ctrl
        model.step(data)

    print("Benchmark: 500 steps...")
    benchmark_steps = 500
    ref_ctrl = make_ctrl(lift_qpos[:7], 0.0, n_envs, n_robots)
    ref_pos = lift_qpos[:7].copy()

    if args.v:
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
                if args.r and i % 2 == 0:
                    if n_envs > 1:
                        noise = np.random.uniform(-0.025, 0.025, (n_envs, 7)).astype(np.float32)
                        ctrl = ref_ctrl.copy()
                        for robot_idx in range(n_robots):
                            offset = robot_idx * 8
                            ctrl[:, offset : offset + 7] = ref_pos + noise
                    else:
                        noise = np.random.uniform(-0.025, 0.025, 7).astype(np.float32)
                        ctrl = ref_ctrl.copy()
                        for robot_idx in range(n_robots):
                            offset = robot_idx * 8
                            ctrl[offset : offset + 7] = ref_pos + noise
                    data.actuator_ctrls = ctrl
                model.step(data)
                step_counter[0] += 1
            else:
                t_end[0] = time.perf_counter()
                benchmark_complete[0] = True
                print(f"per env: {benchmark_steps / (t_end[0] - t_start[0]):,.2f} FPS")
                print(f"total  : {benchmark_steps / (t_end[0] - t_start[0]) * n_envs:,.2f} FPS")

        def render_func():
            render.sync(data)

        run.render_loop(sim_dt, 60, phys_step, render_func)
    else:
        t0 = time.perf_counter()
        for i in range(benchmark_steps):
            if args.r and i % 2 == 0:
                if n_envs > 1:
                    noise = np.random.uniform(-0.025, 0.025, (n_envs, 7)).astype(np.float32)
                    ctrl = ref_ctrl.copy()
                    for robot_idx in range(n_robots):
                        offset = robot_idx * 8
                        ctrl[:, offset : offset + 7] = ref_pos + noise
                else:
                    noise = np.random.uniform(-0.025, 0.025, 7).astype(np.float32)
                    ctrl = ref_ctrl.copy()
                    for robot_idx in range(n_robots):
                        offset = robot_idx * 8
                        ctrl[offset : offset + 7] = ref_pos + noise
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
