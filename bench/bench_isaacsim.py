import argparse
import time

import numpy as np

# Parse arguments BEFORE creating SimulationApp (critical!)
parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, default=1, choices=[1, 5, 10], help="Number of robots")
parser.add_argument("-B", type=int, default=1, help="Batch size / parallel environments")
parser.add_argument("-v", action="store_true", default=False, help="Enable visualization")
parser.add_argument("--mode", type=str, default="random", choices=["random", "grasp"], help="Scenario: random or grasp")
parser.add_argument("--object", type=str, default="ball", choices=["ball", "cube", "bottle"], help="Object for grasp mode")
parser.add_argument("-r", action="store_true", default=False, help="Random noise during grasp benchmark phase")

args = parser.parse_args()

if args.B > 1:
    print("Warning: IsaacSim doesn't support batch size > 1. Using B=1.")
    args.B = 1

# Create SimulationApp FIRST (must be before other Isaac Sim imports)
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": not args.v})

# Now import other Isaac Sim modules
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, XFormPrim
from isaacsim.core.api.objects import DynamicSphere, DynamicCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdGeom, Gf, UsdPhysics
from omni.isaac.core.utils.stage import get_current_stage


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


# Object configurations for grasp mode
OBJECT_CONFIGS = {
    "ball": {
        "grasp_qpos": np.array([-1.0323, 1.7628, 1.4904, -1.6749, -1.7715, 1.6293, 1.4417, 0.04, 0.04]),
        "lift_qpos": np.array([-1.0426, 1.4028, 1.5634, -1.7114, -1.4055, 1.6015, 1.4510, 0.0, 0.0]),
        "lift_steps": 100,
        "close_fingers": True,  # Close fingers to 0.0
    },
    "cube": {
        "grasp_qpos": np.array([-1.0104, 1.5623, 1.3601, -1.6840, -1.5863, 1.7810, 1.4598, 0.04, 0.04]),
        "lift_qpos": np.array([-1.0426, 1.4028, 1.5634, -1.7114, -1.4055, 1.6015, 1.4510, 0.0, 0.0]),
        "lift_steps": 50,
        "close_fingers": True,  # Close fingers to 0.0
    },
    "bottle": {
        "grasp_qpos": np.array([-0.9937, 1.4588, 1.3058, -1.6924, -1.4882, 1.8461, 1.4577, 0.04, 0.04]),
        "lift_qpos": np.array([-1.0411, 1.2861, 1.5200, -1.7065, -1.2946, 1.6572, 1.4315, 0.04, 0.04]),
        "lift_steps": 50,
        "close_fingers": False,  # Keep fingers partially open at 0.04
    },
}

########################## Setup World ##########################
sim_dt = 0.01
world = World(stage_units_in_meters=1.0, physics_dt=sim_dt, rendering_dt=sim_dt)
world.scene.add_default_ground_plane()

########################## Setup Robots and Objects ##########################
n_robots = args.N
n_envs = args.B
positions = get_robot_positions(n_robots)

# Get Franka asset path
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    raise RuntimeError("Could not find Isaac Sim assets folder")

franka_usd_path = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
stage = get_current_stage()

frankas = []
objects = []

for i in range(n_robots):
    prim_path = f"/World/Franka_{i}"
    add_reference_to_stage(usd_path=franka_usd_path, prim_path=prim_path)

    # Set position
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(positions[i][0], positions[i][1], positions[i][2]))

    # Set control gains (same as defined in panda.xml for consistency across all simulators)
    joint_configs = [
        ("panda_joint1", 4500, 450, "angular"),
        ("panda_joint2", 4500, 450, "angular"),
        ("panda_joint3", 3500, 350, "angular"),
        ("panda_joint4", 3500, 350, "angular"),
        ("panda_joint5", 2000, 200, "angular"),
        ("panda_joint6", 2000, 200, "angular"),
        ("panda_joint7", 2000, 200, "angular"),
        ("panda_finger_joint1", 350, 10, "linear"),
        ("panda_finger_joint2", 350, 10, "linear"),
    ]

    for joint_name, stiffness, damping, drive_type in joint_configs:
        joint_path = f"{prim_path}/{joint_name}"
        joint_prim = stage.GetPrimAtPath(joint_path)
        if joint_prim.IsValid():
            token = UsdPhysics.Tokens.angular if drive_type == "angular" else UsdPhysics.Tokens.linear
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, token)
            drive.GetStiffnessAttr().Set(float(stiffness))
            drive.GetDampingAttr().Set(float(damping))
            drive.GetTypeAttr().Set("force")

    franka = Articulation(prim_paths_expr=prim_path, name=f"franka_{i}")
    world.scene.add(franka)
    frankas.append(franka)

    # Add objects for grasp mode
    if args.mode == "grasp":
        obj_x = positions[i][0] + 0.65
        obj_y = positions[i][1]
        obj_z = 0.02

        if args.object == "ball":
            obj = DynamicSphere(
                prim_path=f"/World/ball_{i}",
                position=np.array([obj_x, obj_y, obj_z]),
                radius=0.02,
                color=np.array([0.0, 1.0, 0.0]),
            )
            world.scene.add(obj)
            objects.append(obj)
        elif args.object == "cube":
            obj = DynamicCuboid(
                prim_path=f"/World/cube_{i}",
                position=np.array([obj_x, obj_y, obj_z]),
                size=0.04,
                color=np.array([1.0, 0.0, 0.0]),
            )
            world.scene.add(obj)
            objects.append(obj)
        elif args.object == "bottle":
            import os
            bottle_usd_path = os.path.abspath("./assets/objects/scene_bottle.usd")
            add_reference_to_stage(usd_path=bottle_usd_path, prim_path=f"/World/Bottle_{i}")

            # Set bottle position to robot base position
            # Note: scene_bottle.usd already has bottle at relative pos (0.65, 0, 0.036)
            bottle_prim = stage.GetPrimAtPath(f"/World/Bottle_{i}")
            if bottle_prim.IsValid():
                xform = UsdGeom.Xformable(bottle_prim)
                xform.ClearXformOpOrder()
                translate_op = xform.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(positions[i][0], positions[i][1], 0))

            bottle = XFormPrim(f"/World/Bottle_{i}", name=f"bottle_{i}")
            world.scene.add(bottle)
            objects.append(bottle)

########################## Initialize World ##########################
world.reset()

########################## mode-specific warmup and benchmark ##########################
if args.mode == "random":
    # Random mode warmup
    warmup_qpos = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853, 0.04, 0.04])

    print(f"Warmup: {n_robots} robots to initial position (200 steps)...")
    for _ in range(200):
        for franka in frankas:
            franka.set_joint_position_targets(warmup_qpos)
        world.step(render=args.v)

    # Random mode benchmark
    print(f"Benchmark: 1000 steps with {n_robots} robots (random motion)...")
    benchmark_steps = 1000
    ref_pos = warmup_qpos[:7].copy()
    gripper_pos = warmup_qpos[7]

    t0 = time.perf_counter()
    for _ in range(benchmark_steps):
        for franka in frankas:
            noise = np.random.uniform(-0.2, 0.2, 7).astype(np.float32)
            target_arm = ref_pos + noise
            target_pos = np.concatenate([target_arm, [gripper_pos, gripper_pos]])
            franka.set_joint_position_targets(target_pos)
        world.step(render=args.v)
    t1 = time.perf_counter()

    print(f"per env: {benchmark_steps / (t1 - t0):,.2f} FPS")
    print(f"total  : {benchmark_steps / (t1 - t0) * n_envs:,.2f} FPS")

else:  # grasp mode
    config = OBJECT_CONFIGS[args.object]
    grasp_qpos = config["grasp_qpos"]
    lift_qpos = config["lift_qpos"]
    lift_steps = config["lift_steps"]
    close_fingers = config["close_fingers"]

    # Set initial positions
    for franka in frankas:
        franka.set_joint_positions(grasp_qpos)

    print("Warmup Phase 1: Grasping (100 steps)...")
    for _ in range(100):
        for franka in frankas:
            target_qpos = grasp_qpos.copy()
            if close_fingers:
                target_qpos[7:] = 0.0  # Close fingers
            else:
                target_qpos[7:] = 0.04  # Keep fingers partially open (bottle)
            franka.set_joint_position_targets(target_qpos)
        world.step(render=args.v)

    print(f"Warmup Phase 2: Lifting ({lift_steps} steps)...")
    for _ in range(lift_steps):
        for franka in frankas:
            franka.set_joint_position_targets(lift_qpos)
        world.step(render=args.v)

    print("Benchmark: 500 steps...")
    benchmark_steps = 500
    ref_pos = lift_qpos[:7].copy()

    t0 = time.perf_counter()
    for i in range(benchmark_steps):
        if args.r and i % 2 == 0:
            for franka in frankas:
                noise = np.random.uniform(-0.025, 0.025, 7).astype(np.float32)
                target_arm = ref_pos + noise
                target_pos = np.concatenate([target_arm, lift_qpos[7:]])
                franka.set_joint_position_targets(target_pos)
        world.step(render=args.v)
    t1 = time.perf_counter()

    print(f"per env: {benchmark_steps / (t1 - t0):,.2f} FPS")
    print(f"total  : {benchmark_steps / (t1 - t0) * n_envs:,.2f} FPS")

########################## Cleanup ##########################
simulation_app.close()
