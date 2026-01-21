import argparse
import math
import time

import genesis as gs
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("-B", type=int, default=1)  # batch size
parser.add_argument("-N", type=int, default=5, choices=[1, 5, 10])  # number of robots
parser.add_argument("-v", action="store_true", default=False)  # visualize

args = parser.parse_args()

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    show_viewer=args.v,
    rigid_options=gs.options.RigidOptions(
        dt=0.01,
        constraint_solver=gs.constraint_solver.CG,  # to match mjx
        tolerance=1e-8,  # to match mjx
    ),
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)


def get_robot_positions(n_robots):
    """Calculate robot positions - spread them out in a grid layout"""
    positions = []
    rotations = []

    if n_robots == 1:
        positions.append((0, 0, 0))
        rotations.append((0, 0, 0))  # No rotation
    elif n_robots == 5:
        # 5 robots in a line along x-axis with 2m spacing
        for i in range(5):
            x = (i - 2) * 2.0  # -4, -2, 0, 2, 4
            positions.append((x, 0, 0))
            rotations.append((0, 0, 0))  # No rotation
    elif n_robots == 10:
        # 10 robots in 2 rows of 5 with 2m spacing
        for i in range(10):
            row = i // 5
            col = i % 5
            x = (col - 2) * 2.0  # -4, -2, 0, 2, 4
            y = (row - 0.5) * 2.0  # -1, 1
            positions.append((x, y, 0))
            rotations.append((0, 0, 0))  # No rotation

    return positions, rotations


# Add multiple Franka robots in grid layout
positions, rotations = get_robot_positions(args.N)
frankas = []
for i in range(args.N):
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="assets/franka_emika_panda/panda.xml",
            pos=positions[i],
            euler=rotations[i],
        ),
    )
    frankas.append(franka)

########################## build ##########################
n_envs = args.B
scene.build(n_envs=n_envs)

# Set control gains for all robots (matching panda.xml)
for franka in frankas:
    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 350, 350]),
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -200, -200]),
        np.array([87, 87, 87, 87, 12, 12, 12, 200, 200]),
    )

motor_dofs = np.arange(9)

# Warmup position (stable initial pose with fingers)
warmup_qpos = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853, 0.04, 0.04])

# Warmup - let robots stabilize at target position
for i in range(200):  # until stable
    for franka in frankas:
        franka.control_dofs_position(warmup_qpos, motor_dofs)
    scene.step()

########################## Benchmark ##########################
ref_pos = warmup_qpos[:7].copy()  # Use warmup position as reference (joints only)
gripper_pos = warmup_qpos[7]  # Keep gripper fixed

t0 = time.perf_counter()
for i in range(1000):
    for idx, franka in enumerate(frankas):
        # Add random perturbation [-0.2, 0.2] to arm joints only (not gripper)
        noise = torch.rand((n_envs, 7), device="cuda") * 0.4 - 0.2
        target_arm = torch.from_numpy(ref_pos).cuda() + noise
        # Combine arm target with fixed gripper position
        target_pos = torch.cat(
            [target_arm, torch.full((n_envs, 2), gripper_pos, device="cuda")], dim=1
        )
        franka.control_dofs_position(target_pos, motor_dofs)
    scene.step()
t1 = time.perf_counter()

print(f"per env: {1000 / (t1 - t0):,.2f} FPS")
print(f"total  : {1000 / (t1 - t0) * n_envs:,.2f} FPS")
