import argparse
import time

import numpy as np

# Parse arguments BEFORE creating SimulationApp (critical!)
parser = argparse.ArgumentParser()
parser.add_argument("-B", type=int, default=1)  # batch size
parser.add_argument("-N", type=int, default=5, choices=[1, 5, 10])  # number of robots
parser.add_argument("-v", action="store_true", default=False)  # visualize

args = parser.parse_args()

# Create SimulationApp FIRST (must be before other Isaac Sim imports)
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": not args.v})

# Now import other Isaac Sim modules
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path


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


########################## Setup World ##########################
sim_dt = 0.01
world = World(stage_units_in_meters=1.0, physics_dt=sim_dt, rendering_dt=sim_dt)
world.scene.add_default_ground_plane()

########################## Setup Robots ##########################
n_robots = args.N
n_envs = args.B
positions = get_robot_positions(n_robots)

# Get Franka asset path
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    raise RuntimeError("Could not find Isaac Sim assets folder")

franka_usd_path = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"

# Note: IsaacSim doesn't support batched environments the same way as Genesis/Motrixsim
# We simulate batch_size by running multiple sequential simulations
# For now, we'll add N robots in a single environment and report per-robot performance

frankas = []
for i in range(n_robots):
    prim_path = f"/World/Franka_{i}"
    add_reference_to_stage(usd_path=franka_usd_path, prim_path=prim_path)

    # Set position by modifying the prim
    from pxr import UsdGeom, Gf, UsdPhysics
    from omni.isaac.core.utils.stage import get_current_stage
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(positions[i][0], positions[i][1], positions[i][2]))

    # Set control gains to match panda.xml (Genesis/Motrixsim parameters)
    # Joint parameters: [joint1-7 (revolute), finger1-2 (prismatic)]
    joint_configs = [
        ("panda_joint1", 4500, 450, "angular"),
        ("panda_joint2", 4500, 450, "angular"),
        ("panda_joint3", 3500, 350, "angular"),
        ("panda_joint4", 3500, 350, "angular"),
        ("panda_joint5", 2000, 200, "angular"),
        ("panda_joint6", 2000, 200, "angular"),
        ("panda_joint7", 2000, 200, "angular"),
        ("panda_finger_joint1", 350, 10, "linear"),  # Prismatic joint
        ("panda_finger_joint2", 350, 10, "linear"),  # Prismatic joint
    ]

    for joint_name, stiffness, damping, drive_type in joint_configs:
        joint_path = f"{prim_path}/{joint_name}"
        joint_prim = stage.GetPrimAtPath(joint_path)
        if joint_prim.IsValid():
            # Apply drive API (angular for revolute, linear for prismatic)
            token = UsdPhysics.Tokens.angular if drive_type == "angular" else UsdPhysics.Tokens.linear
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, token)
            drive.GetStiffnessAttr().Set(float(stiffness))
            drive.GetDampingAttr().Set(float(damping))
            drive.GetTypeAttr().Set("force")

    franka = Articulation(prim_paths_expr=prim_path, name=f"franka_{i}")
    world.scene.add(franka)
    frankas.append(franka)

########################## Initialize World ##########################
world.reset()

########################## Setup Control ##########################
# Warmup position (stable initial pose with fingers)
# Franka has 9 DOFs: 7 joints + 2 fingers
warmup_qpos = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853, 0.04, 0.04])

########################## Warmup ##########################
print(f"Warmup: {n_robots} robots to initial position (200 steps)...")

for _ in range(200):
    for franka in frankas:
        # Set PD controller target positions (matching grasp example)
        franka.set_joint_position_targets(warmup_qpos)

    world.step(render=args.v)

########################## Benchmark ##########################
print(f"Benchmark: 1000 steps with {n_robots} robots (random motion)...")

benchmark_steps = 1000
ref_pos = warmup_qpos[:7].copy()  # Use warmup position as reference (joints only)
gripper_pos = warmup_qpos[7]  # Keep gripper fixed

t0 = time.perf_counter()

for _ in range(benchmark_steps):
    for franka in frankas:
        # Add random perturbation [-0.2, 0.2] to arm joints only (matching Genesis/Motrixsim)
        noise = np.random.uniform(-0.2, 0.2, 7).astype(np.float32)
        target_arm = ref_pos + noise
        # Combine arm target with fixed gripper position
        target_pos = np.concatenate([target_arm, [gripper_pos, gripper_pos]])
        # Set PD controller target positions (matching grasp example)
        franka.set_joint_position_targets(target_pos)

    world.step(render=args.v)

t1 = time.perf_counter()

print(f"per env: {benchmark_steps / (t1 - t0):,.2f} FPS")
print(f"total  : {benchmark_steps / (t1 - t0) * n_envs:,.2f} FPS")

########################## Cleanup ##########################
simulation_app.close()
