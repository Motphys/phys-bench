import argparse
import json
import math
import sys
import time

import genesis as gs
import torch

parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, default=1, choices=[1, 5, 10], help="Number of robots")
parser.add_argument("-B", type=int, default=1, help="Batch size / parallel environments")
parser.add_argument("-v", action="store_true", default=False, help="Enable visualization")
parser.add_argument("--mode", type=str, default="franka_only", choices=["franka_only", "franka_grasp"], help="Scenario: franka_only or franka_grasp")
parser.add_argument("--object", type=str, default="ball", choices=["ball", "cube", "bottle"], help="Object for grasp mode")
parser.add_argument("-r", action="store_true", default=False, help="Random noise during grasp benchmark phase")
parser.add_argument("--clutter", action="store_true", default=False, help="Fill scene with 200+ dynamic bottles (random mode only)")

args = parser.parse_args()

########################## init ##########################
gs.init(backend=gs.gpu)


def get_robot_positions(n_robots):
    """Calculate robot positions - spread them out in a grid layout"""
    positions = []
    rotations = []

    if n_robots == 1:
        positions.append((0, 0, 0))
        rotations.append((0, 0, 0))
    elif n_robots == 5:
        # 5 robots in a line along x-axis with 2m spacing
        for i in range(5):
            x = (i - 2) * 2.0  # -4, -2, 0, 2, 4
            positions.append((x, 0, 0))
            rotations.append((0, 0, 0))
    elif n_robots == 10:
        # 10 robots in 2 rows of 5 with 2m spacing
        for i in range(10):
            row = i // 5
            col = i % 5
            x = (col - 2) * 2.0  # -4, -2, 0, 2, 4
            y = (row - 0.5) * 2.0  # -1, 1
            positions.append((x, y, 0))
            rotations.append((0, 0, 0))

    return positions, rotations


def generate_clutter_positions(robot_pos, min_radius=0.3, spacing=0.15, min_count=40):
    """Generate positions for bottles in concentric circles around robot_pos.

    Note: scene_bottle.xml has bottle at pos="0.65 0 0.036" relative to worldbody,
    so we subtract 0.65 from x to compensate.
    """
    positions = []
    radius = min_radius
    z_offset = 0.036
    while len(positions) < min_count:
        circumference = 2 * math.pi * radius
        n_bottles = int(circumference / spacing)
        if n_bottles == 0:
            n_bottles = 1
        for j in range(n_bottles):
            if len(positions) >= min_count:
                break
            angle = 2 * math.pi * j / n_bottles
            x = robot_pos[0] + radius * math.cos(angle) - 0.65
            y = robot_pos[1] + radius * math.sin(angle)
            positions.append((x, y, z_offset))
        radius += spacing
        z_offset += 0.1  # Stack bottles vertically as we add more rings
    return positions


# Object configurations for grasp mode (will be converted to torch tensors when used)
OBJECT_CONFIGS = {
    "ball": {
        "grasp_qpos": [-1.0323, 1.7628, 1.4904, -1.6749, -1.7715, 1.6293, 1.4417, 0.04, 0.04],
        "gripper_force": -0.5,
        "lift_steps": 100,
    },
    "cube": {
        "grasp_qpos": [-1.0104, 1.5623, 1.3601, -1.6840, -1.5863, 1.7810, 1.4598, 0.04, 0.04],
        "gripper_force": -0.5,
        "lift_steps": 50,
    },
    "bottle": {
        "grasp_qpos": [-0.9937, 1.4588, 1.3058, -1.6924, -1.4882, 1.8461, 1.4577, 0.04, 0.04],
        "gripper_force": -4.0,
        "lift_steps": 50,
    },
}

########################## create a scene ##########################
if args.mode == "franka_only":
    scene = gs.Scene(
        show_viewer=args.v,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            constraint_solver=gs.constraint_solver.CG,
            tolerance=1e-8,
        ),
    )
else:  # franka_grasp mode
    scene = gs.Scene(
        show_viewer=args.v,
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

# Add multiple Franka robots in grid layout
positions, rotations = get_robot_positions(args.N)
frankas = []
objects = []

for i in range(args.N):
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="assets/franka_emika_panda/panda.xml",
            pos=positions[i],
            euler=rotations[i],
        ),
    )
    frankas.append(franka)

# Add clutter bottles for franka_only mode
if args.mode == "franka_only" and args.clutter:
    for i in range(args.N):
        clutter_positions = generate_clutter_positions(positions[i])
        for clutter_pos in clutter_positions:
            scene.add_entity(
                gs.morphs.MJCF(
                    file="assets/objects/scene_bottle.xml",
                    pos=clutter_pos,
                ),
            )

# Add objects for franka_grasp mode
for i in range(args.N):
    if args.mode == "franka_grasp":
        obj_x = positions[i][0] + 0.65
        obj_y = positions[i][1]
        obj_z = 0.02

        if args.object == "ball":
            obj = scene.add_entity(
                gs.morphs.Sphere(
                    radius=0.02,
                    pos=(obj_x, obj_y, obj_z),
                ),
            )
        elif args.object == "cube":
            obj = scene.add_entity(
                gs.morphs.Box(
                    size=(0.04, 0.04, 0.04),
                    pos=(obj_x, obj_y, obj_z),
                ),
            )
        elif args.object == "bottle":
            # Note: scene_bottle.xml already has bottle at relative pos (0.65, 0, 0.036)
            # So we only need to set the base position to the robot's position
            obj = scene.add_entity(
                gs.morphs.MJCF(
                    file="assets/objects/scene_bottle.xml",
                    pos=(positions[i][0], positions[i][1], 0),
                ),
            )
        objects.append(obj)

########################## build ##########################
n_envs = args.B
scene.build(n_envs=n_envs)

########################## configure control gains ##########################
# Use same parameters as defined in panda.xml for consistency across all simulators
for franka in frankas:
    franka.set_dofs_kp(
        torch.tensor([4500, 4500, 3500, 3500, 2000, 2000, 2000, 350, 350], device="cuda"),
    )
    franka.set_dofs_kv(
        torch.tensor([450, 450, 350, 350, 200, 200, 200, 10, 10], device="cuda"),
    )
    franka.set_dofs_force_range(
        torch.tensor([-87, -87, -87, -87, -12, -12, -12, -200, -200], device="cuda"),
        torch.tensor([87, 87, 87, 87, 12, 12, 12, 200, 200], device="cuda"),
    )

motor_dofs = torch.arange(9, device="cuda")
motors_dof = torch.arange(7, device="cuda")
fingers_dof = torch.arange(7, 9, device="cuda")

########################## mode-specific warmup and benchmark ##########################
if args.mode == "franka_only":
    # Franka only mode warmup
    warmup_qpos = torch.tensor([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853, 0.04, 0.04], device="cuda")
    
    for franka in frankas:
        franka.set_dofs_position(warmup_qpos)

    for i in range(200):
        for franka in frankas:
            franka.control_dofs_position(warmup_qpos, motor_dofs)
        scene.step()

    # Franka only mode benchmark
    ref_pos = warmup_qpos[:7]
    gripper_pos = warmup_qpos[7]

    t0 = time.perf_counter()
    for i in range(1000):
        for franka in frankas:
            noise = torch.rand((n_envs, 7), device="cuda") * 0.4 - 0.2
            target_arm = ref_pos + noise
            target_pos = torch.cat(
                [target_arm, torch.full((n_envs, 2), gripper_pos, device="cuda")], dim=1
            )
            franka.control_dofs_position(target_pos, motor_dofs)
        scene.step()

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

    print(f"per env: {1000 / (t1 - t0):,.2f} FPS")
    print(f"total  : {1000 / (t1 - t0) * n_envs:,.2f} FPS")

else:  # franka_grasp mode
    config = OBJECT_CONFIGS[args.object]
    grasp_qpos = torch.tensor(config["grasp_qpos"], device="cuda")
    gripper_force = config["gripper_force"]
    lift_steps = config["lift_steps"]

    # Compute lift_qpos (referenced from ball test, reused for all objects)
    lift_qpos = torch.tensor([-1.0426, 1.4028, 1.5634, -1.7114, -1.4055, 1.6015, 1.4510, 0.0, 0.0], device="cuda")

    # Set initial position
    for franka in frankas:
        franka.set_dofs_position(grasp_qpos)
    scene.visualizer.update()

    # Grasp phase
    for franka in frankas:
        franka.control_dofs_position(grasp_qpos[:-2], motors_dof)
        franka.control_dofs_force(torch.tensor([gripper_force, gripper_force], device="cuda"), fingers_dof)

    for i in range(100):
        scene.step()

    # Lift phase
    for franka in frankas:
        franka.control_dofs_position(lift_qpos[:-2], motors_dof)

    for i in range(lift_steps):
        scene.step()

    # Benchmark phase
    ref_pos = torch.tile(lift_qpos[:7], [n_envs, 1])

    t0 = time.perf_counter()
    for i in range(500):
        if args.r and i % 2 == 0:
            for franka in frankas:
                franka.control_dofs_position(
                    ref_pos + torch.rand((n_envs, 7), device="cuda") * 0.05 - 0.025, motors_dof
                )
        scene.step()

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

    print(f"per env: {500 / (t1 - t0):,.2f} FPS")
    print(f"total  : {500 / (t1 - t0) * n_envs:,.2f} FPS")
