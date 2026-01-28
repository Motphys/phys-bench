import argparse
import time

import numpy as np
import torch

# Parse arguments BEFORE creating SimulationApp (critical!)
parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, default=1, choices=[1, 5, 10], help="Number of robots")
parser.add_argument("-B", type=int, default=1, help="Batch size / parallel environments")
parser.add_argument("-v", action="store_true", default=False, help="Enable visualization")
parser.add_argument("--mode", type=str, default="random", choices=["random", "grasp"], help="Scenario: random or grasp")
parser.add_argument("--object", type=str, default="ball", choices=["ball", "cube", "bottle"], help="Object for grasp mode")
parser.add_argument("-r", action="store_true", default=False, help="Random noise during grasp benchmark phase")
parser.add_argument("--clutter", action="store_true", default=False, help="Fill scene with 200+ dynamic bottles (random mode only)")

args = parser.parse_args()

# Create SimulationApp FIRST (must be before other Isaac Sim imports)
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": not args.v})

# Now import other Isaac Sim modules
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, XFormPrim
from isaacsim.core.api.objects import DynamicSphere, DynamicCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.cloner import GridCloner
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


def generate_clutter_positions(robot_pos, min_radius=0.3, spacing=0.15, min_count=40):
    """Generate positions for bottles in concentric circles around robot_pos.

    Note: scene_bottle.usd has bottle at offset position,
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

# Create template environment (env_0) with all robots and objects
for i in range(n_robots):
    prim_path = f"/World/env_0/Franka_{i}"
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

    # Add objects for grasp mode
    if args.mode == "grasp":
        obj_x = positions[i][0] + 0.65
        obj_y = positions[i][1]
        obj_z = 0.02

        if args.object == "ball":
            obj = DynamicSphere(
                prim_path=f"/World/env_0/ball_{i}",
                name=f"dynamic_sphere_0_{i}",
                position=np.array([obj_x, obj_y, obj_z]),
                radius=0.02,
                color=np.array([0.0, 1.0, 0.0]),
            )
            world.scene.add(obj)
        elif args.object == "cube":
            obj = DynamicCuboid(
                prim_path=f"/World/env_0/cube_{i}",
                name=f"dynamic_cuboid_0_{i}",
                position=np.array([obj_x, obj_y, obj_z]),
                size=0.04,
                color=np.array([1.0, 0.0, 0.0]),
            )
            world.scene.add(obj)
        elif args.object == "bottle":
            import os
            bottle_usd_path = os.path.abspath("./assets/objects/scene_bottle.usd")
            add_reference_to_stage(usd_path=bottle_usd_path, prim_path=f"/World/env_0/Bottle_{i}")

            # Set bottle position to robot base position
            # Note: scene_bottle.usd already has bottle at relative pos (0.65, 0, 0.036)
            bottle_prim = stage.GetPrimAtPath(f"/World/env_0/Bottle_{i}")
            if bottle_prim.IsValid():
                xform = UsdGeom.Xformable(bottle_prim)
                xform.ClearXformOpOrder()
                translate_op = xform.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(positions[i][0], positions[i][1], 0))

            bottle = XFormPrim(f"/World/env_0/Bottle_{i}", name=f"bottle_0_{i}")
            world.scene.add(bottle)

# Add clutter bottles for random mode
if args.mode == "random" and args.clutter:
    import os
    bottle_usd_path = os.path.abspath("./assets/objects/scene_bottle.usd")
    bottle_counter = 0
    for i in range(n_robots):
        clutter_positions = generate_clutter_positions(positions[i])
        for clutter_pos in clutter_positions:
            prim_path = f"/World/env_0/ClutterBottle_{bottle_counter}"
            add_reference_to_stage(usd_path=bottle_usd_path, prim_path=prim_path)

            # Set bottle position
            bottle_prim = stage.GetPrimAtPath(prim_path)
            if bottle_prim.IsValid():
                xform = UsdGeom.Xformable(bottle_prim)
                xform.ClearXformOpOrder()
                translate_op = xform.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(clutter_pos[0], clutter_pos[1], 0))

            bottle_counter += 1

# Clone environments if B > 1
if n_envs > 1:
    print(f"Cloning {n_envs} environments using GridCloner...")
    # Calculate spacing based on robot layout
    max_extent = max([abs(p[0]) for p in positions] + [abs(p[1]) for p in positions]) * 2 + 3
    cloner = GridCloner(spacing=max_extent)
    target_paths = cloner.generate_paths("/World/env", n_envs)
    cloner.clone(
        source_prim_path="/World/env_0",
        prim_paths=target_paths,
        copy_from_source=True,
    )

# Create single Articulation object that manages all robots across all environments using regex
franka_view = Articulation(
    prim_paths_expr="/World/env_*/Franka_*",
    name="frankas"
)
world.scene.add(franka_view)
total_robots = n_envs * n_robots

########################## Initialize World ##########################
world.reset()

########################## mode-specific warmup and benchmark ##########################
if args.mode == "random":
    # Random mode warmup
    warmup_qpos = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853, 0.04, 0.04])
    warmup_qpos_tensor = torch.tensor(warmup_qpos, device="cuda", dtype=torch.float32).unsqueeze(0).repeat(total_robots, 1)

    print(f"Warmup: {n_robots} robots x {n_envs} envs to initial position (200 steps)...")
    for _ in range(200):
        franka_view.set_joint_position_targets(warmup_qpos_tensor)
        world.step(render=args.v)

    # Random mode benchmark
    print(f"Benchmark: 1000 steps with {n_robots} robots x {n_envs} envs (random motion)...")
    benchmark_steps = 1000
    ref_pos_tensor = torch.tensor(warmup_qpos[:7], device="cuda", dtype=torch.float32).unsqueeze(0).repeat(total_robots, 1)
    gripper_pos_tensor = torch.full((total_robots, 2), warmup_qpos[7], device="cuda", dtype=torch.float32)

    t0 = time.perf_counter()
    for _ in range(benchmark_steps):
        # Generate random noise on GPU
        noise = torch.rand((total_robots, 7), device="cuda", dtype=torch.float32) * 0.4 - 0.2
        target_arm = ref_pos_tensor + noise
        targets = torch.cat([target_arm, gripper_pos_tensor], dim=1)
        franka_view.set_joint_position_targets(targets)
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
    grasp_qpos_tensor = torch.tensor(grasp_qpos, device="cuda", dtype=torch.float32).unsqueeze(0).repeat(total_robots, 1)
    franka_view.set_joint_positions(grasp_qpos_tensor)

    print("Warmup Phase 1: Grasping (100 steps)...")
    grasp_target_qpos = grasp_qpos.copy()
    if close_fingers:
        grasp_target_qpos[7:] = 0.0  # Close fingers
    else:
        grasp_target_qpos[7:] = 0.04  # Keep fingers partially open (bottle)
    grasp_target_tensor = torch.tensor(grasp_target_qpos, device="cuda", dtype=torch.float32).unsqueeze(0).repeat(total_robots, 1)

    for _ in range(100):
        franka_view.set_joint_position_targets(grasp_target_tensor)
        world.step(render=args.v)

    print(f"Warmup Phase 2: Lifting ({lift_steps} steps)...")
    lift_qpos_tensor = torch.tensor(lift_qpos, device="cuda", dtype=torch.float32).unsqueeze(0).repeat(total_robots, 1)
    for _ in range(lift_steps):
        franka_view.set_joint_position_targets(lift_qpos_tensor)
        world.step(render=args.v)

    print("Benchmark: 500 steps...")
    benchmark_steps = 500
    ref_pos_tensor = torch.tensor(lift_qpos[:7], device="cuda", dtype=torch.float32).unsqueeze(0).repeat(total_robots, 1)
    gripper_tensor = torch.tensor(lift_qpos[7:], device="cuda", dtype=torch.float32).unsqueeze(0).repeat(total_robots, 1)

    t0 = time.perf_counter()
    for i in range(benchmark_steps):
        if args.r and i % 2 == 0:
            # Generate random noise on GPU
            noise = torch.rand((total_robots, 7), device="cuda", dtype=torch.float32) * 0.05 - 0.025
            target_arm = ref_pos_tensor + noise
            target_pos = torch.cat([target_arm, gripper_tensor], dim=1)
            franka_view.set_joint_position_targets(target_pos)
        world.step(render=args.v)
    t1 = time.perf_counter()

    print(f"per env: {benchmark_steps / (t1 - t0):,.2f} FPS")
    print(f"total  : {benchmark_steps / (t1 - t0) * n_envs:,.2f} FPS")

########################## Cleanup ##########################
simulation_app.close()
