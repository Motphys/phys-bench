import argparse
import time

import numpy as np

# Parse arguments BEFORE creating SimulationApp (critical!)
parser = argparse.ArgumentParser()
parser.add_argument("-B", type=int, default=1)  # batch size (IsaacSim doesn't support batching)
parser.add_argument("-v", action="store_true", default=False)  # visualize
parser.add_argument("-r", action="store_true", default=False)  # random action

args = parser.parse_args()

if args.B > 1:
    print("Warning: IsaacSim doesn't support batch size > 1. Using B=1.")
    args.B = 1

# Create SimulationApp FIRST (must be before other Isaac Sim imports)
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": not args.v})

# Now import other Isaac Sim modules
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdGeom, Gf, UsdPhysics
from omni.isaac.core.utils.stage import get_current_stage

########################## Setup World ##########################
sim_dt = 0.01  # Match Genesis/Motrixsim (100 Hz)
world = World(stage_units_in_meters=1.0, physics_dt=sim_dt, rendering_dt=sim_dt)
world.scene.add_default_ground_plane()

########################## Add Cube ##########################
cube = DynamicCuboid(
    prim_path="/World/cube",
    position=np.array([0.65, 0.0, 0.02]),
    size=0.04,
    color=np.array([1.0, 0.0, 0.0]),  # Red cube
)
world.scene.add(cube)

########################## Add Franka Robot ##########################
# Get Franka asset path
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    raise RuntimeError("Could not find Isaac Sim assets folder")

franka_usd_path = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
prim_path = "/World/Franka"
add_reference_to_stage(usd_path=franka_usd_path, prim_path=prim_path)

# Set control gains to match grasp-bench (Genesis/Motrixsim parameters)
# Note: grasp-bench uses kp=100 for fingers (different from franka-multi-bench's 350)
# Joint parameters: [joint1-7 (revolute), finger1-2 (prismatic)]
joint_configs = [
    ("panda_joint1", 4500, 450, "angular"),
    ("panda_joint2", 4500, 450, "angular"),
    ("panda_joint3", 3500, 350, "angular"),
    ("panda_joint4", 3500, 350, "angular"),
    ("panda_joint5", 2000, 200, "angular"),
    ("panda_joint6", 2000, 200, "angular"),
    ("panda_joint7", 2000, 200, "angular"),
    ("panda_finger_joint1", 100, 10, "linear"),  # Prismatic joint - grasp-bench uses 100
    ("panda_finger_joint2", 100, 10, "linear"),  # Prismatic joint - grasp-bench uses 100
]

stage = get_current_stage()
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

franka = Articulation(prim_paths_expr=prim_path, name="franka")
world.scene.add(franka)

########################## Initialize World ##########################
world.reset()

########################## Setup Control ##########################
# Franka Panda positions (from grasp-bench Genesis/Motrixsim)
# Joint positions: [7 DOF arm, 2 DOF gripper]
grasp_qpos = np.array([-1.0104, 1.5623, 1.3601, -1.6840, -1.5863, 1.7810, 1.4598, 0.04, 0.04])
lift_qpos = np.array([-1.0426, 1.4028, 1.5634, -1.7114, -1.4055, 1.6015, 1.4510, 0.0, 0.0])

# Set initial position to grasp_qpos (matching Genesis: franka.set_dofs_position(grasp_qpos))
franka.set_joint_positions(grasp_qpos)

########################## Warmup Phase 1: Grasp (100 steps) ##########################
print("Warmup Phase 1: Grasping (100 steps)...")

# Control arm to grasp position, apply force to close fingers
# Genesis: franka.control_dofs_position(grasp_qpos[:-2], motors_dof)
#          franka.control_dofs_force(np.array([-0.5, -0.5]), fingers_dof)
for _ in range(100):
    # Set arm target to grasp position, fingers close with position target
    target_qpos = grasp_qpos.copy()
    target_qpos[7:] = 0.0  # Close fingers
    franka.set_joint_position_targets(target_qpos)
    world.step(render=args.v)

########################## Warmup Phase 2: Lift (50 steps) ##########################
print("Warmup Phase 2: Lifting (50 steps)...")

# Control arm to lift position
# Genesis: franka.control_dofs_position(lift_qpos[:-2], motors_dof)
for _ in range(50):
    franka.set_joint_position_targets(lift_qpos)
    world.step(render=args.v)

########################## Benchmark ##########################
print("Benchmark: 500 steps...")

benchmark_steps = 500
ref_pos = lift_qpos[:7].copy()  # Use lift position as reference (joints only)

t0 = time.perf_counter()

for i in range(benchmark_steps):
    if args.r and i % 2 == 0:
        # Add random perturbation [-0.025, 0.025] to arm joints
        # Genesis: ref_pos + torch.rand((n_envs, 7), device='cuda')*0.05 - 0.025
        noise = np.random.uniform(-0.025, 0.025, 7).astype(np.float32)
        target_arm = ref_pos + noise
        # Combine arm target with fixed gripper position
        target_pos = np.concatenate([target_arm, lift_qpos[7:]])
        franka.set_joint_position_targets(target_pos)

    world.step(render=args.v)

t1 = time.perf_counter()

n_envs = args.B
print(f"per env: {benchmark_steps / (t1 - t0):,.2f} FPS")
print(f"total  : {benchmark_steps / (t1 - t0) * n_envs:,.2f} FPS")

########################## Cleanup ##########################
simulation_app.close()
