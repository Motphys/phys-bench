# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""IsaacSim 5.0 grasp shaking test.

This test implements a robotic pick-and-shake task using IsaacSim 5.0's API.
The test follows the same scenario as other physics engine tests for fair comparison.
"""

import sys

import numpy as np
from absl import app, flags

from test_output_utils import (
    ensure_output_directory,
    generate_video_path,
    save_test_result,
    save_video,
)

# IsaacSim 5.0 imports
from isaacsim import SimulationApp

# Check for --visual flag in command line args (before flags are parsed)
# Default to headless=True for server usage, use headless=False if --visual is set
headless_mode = "--visual" not in sys.argv and "-V" not in sys.argv
simulation_app = SimulationApp({"headless": headless_mode})  # start the simulation app


_Obj = flags.DEFINE_string(
    "object", "cube", "object to grasp, Choices: [cube, ball, bottle]"
)
_Shake = flags.DEFINE_boolean(
    "shake", True, "whether to shake the arm after grasping, Choices: [True, False]"
)
_Record = flags.DEFINE_boolean(
    "record", False, "whether to record the simulation, Choices: [True, False]"
)
_Dt = flags.DEFINE_float("dt", 0.002, "simulation timestep")

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.storage.native import get_assets_root_path
from isaacsim.sensors.camera import Camera
from isaacsim.core.api.objects import DynamicCuboid, DynamicSphere
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.prims import XFormPrim


def lerp(a, b, t):
    """Linear interpolation between a and b."""
    return a + t * (b - a)


def create_dynamic_cube(prim_path, position, size, color, physics_material=None):
    """Create a dynamic cube with optional physics material for friction configuration."""
    return DynamicCuboid(
        prim_path=prim_path,
        name="cube",
        position=position,
        size=size,
        color=color,
        physics_material=physics_material,
    )


def create_dynamic_sphere(prim_path, position, radius, color, physics_material=None):
    """Create a dynamic sphere with optional physics material for friction configuration."""
    return DynamicSphere(
        prim_path=prim_path,
        name="sphere",
        position=position,
        radius=radius,
        color=color,
        physics_material=physics_material,
    )


def create_xprim(usd_path, prim_path, name, position=None):
    """Create a XFormPrim from usd."""
    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    return XFormPrim(prim_path, name=name, positions=position)


def main(argv):
    # Initialize IsaacSim 5.0 simulation app
    # Note: SimulationApp must be initialized BEFORE importing other IsaacSim modules

    # Initialize IsaacSim world
    sim_dt = _Dt.value
    world = World(stage_units_in_meters=1.0, physics_dt=sim_dt)
    world.scene.add_default_ground_plane()

    # Initialize output and tracking
    output_dir = ensure_output_directory()
    video_path = generate_video_path(
        "isaacsim", _Obj.value, _Shake.value, True, _Dt.value, output_dir
    )
    test_passed = True
    drop_time = None

    # Initialize recording
    camera = None
    if _Record.value:
        frames = []
        recording_fps = 30

        # Headless模式警告
        if headless_mode:
            print("Warning: Recording in headless mode may produce empty frames.")
            print("Consider using --visual flag for proper rendering.")

        # Create recording camera
        try:
            camera_eye = np.array([2.155, 0.016, 1.526])
            camera_target = np.array([0.5, 0.0, 0.2])
            camera_orientation = rot_utils.euler_angles_to_quats(
                np.array([0.0, 45, 180]), degrees=True
            )
            # Create Camera object
            camera = Camera(
                prim_path="/World/recording_camera",
                position=camera_eye,
                frequency=recording_fps,
                resolution=(320, 240),
                orientation=camera_orientation,
            )

            # 初始化相机
            camera.initialize()

            # 添加RGB annotator
            camera.add_rgb_to_frame()

            print(f"Recording camera initialized at {camera_eye} -> {camera_target}")
        except Exception as e:
            print(f"Warning: Failed to initialize recording camera: {e}")
            print("Recording will be disabled.")
            _Record.value = False
            camera = None

    # Simulation loop parameters
    task = "shaking-grasp" if _Shake.value else "slip-grasp"
    step_cnt = 0

    # Franka Panda initial configuration (from MuJoCo keyframes)
    # Joint positions: [7 DOF arm, 2 DOF gripper]
    init_qpos = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853, 0.04, 0.04])
    grasp_qpos = np.array(
        [-1.0104, 1.5623, 1.3601, -1.6840, -1.5863, 1.7810, 1.4598, 0.04, 0.04]
    )
    lift_qpos = np.array(
        [-1.0426, 1.4028, 1.5634, -1.7114, -1.4055, 1.6015, 1.4510, 0.0, 0.0]
    )
    current_qpos = init_qpos.copy()

    # Load Franka Panda robot from IsaacSim assets
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        print("Error: Cannot find IsaacSim assets root path")
        simulation_app.close()
        return

    franka_usd_path = (
        assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    )
    add_reference_to_stage(usd_path=franka_usd_path, prim_path="/World/Franka")
    franka = Articulation(prim_paths_expr="/World/Franka", name="franka")

    # Add object to grasp (based on _Obj flag)
    obj_prim_path = f"/World/{_Obj.value}"

    # Create physics material with custom friction properties
    # Default IsaacSim values: static_friction=0.2, dynamic_friction=1.0, restitution=0.0
    obj_physics_material = PhysicsMaterial(
        prim_path="/World/ObjectPhysicsMaterial",
        static_friction=0.2,  # Higher static friction for better grasping
        dynamic_friction=0.8,  # Higher dynamic friction for better grasping
        restitution=0.0,  # No bounce
    )

    if _Obj.value == "cube":
        obj = create_dynamic_cube(
            prim_path=obj_prim_path,
            position=np.array([0.65, 0.0, 0.02]),
            size=0.04,
            color=np.array([1.0, 0.0, 0.0]),  # Red cube
            physics_material=obj_physics_material,
        )
        obj_z_threshold = 0.03
    elif _Obj.value == "ball":
        obj = create_dynamic_sphere(
            prim_path=obj_prim_path,
            position=np.array([0.65, 0.0, 0.02]),
            radius=0.02,
            color=np.array([0.0, 1.0, 0.0]),  # Green ball
            physics_material=obj_physics_material,
        )
        obj_z_threshold = 0.03
    elif _Obj.value == "bottle":
        obj = create_xprim(
            usd_path="./assets/objects/scene_bottle.usd",
            prim_path="/World/Bottle",
            name="bottle",
        )
        obj_z_threshold = 0.03
    else:
        print(f"Error: Unknown object type {_Obj.value}")
        simulation_app.close()
        return
    world.scene.add(obj)
    # Reset world to apply all changes
    world.reset()

    while True:
        step_cnt += 1
        elapsed_time = world.current_time  # Use IsaacSim's world time

        # Phase 1: Move from init to lift (0-1s)
        if 0 <= elapsed_time < 1:
            ctrl_arm = lerp(init_qpos[:7], lift_qpos[:7], elapsed_time)
            current_qpos[:7] = ctrl_arm
            current_qpos[7:] = init_qpos[7:]  # Keep gripper open
            # Apply joint positions (use list-of-lists format like test.py)
            franka.set_joint_position_targets(current_qpos)

        # Phase 2: Move from lift to grasp (1-2s)
        elif 1 <= elapsed_time < 2:
            ctrl_arm = lerp(lift_qpos[:7], grasp_qpos[:7], (elapsed_time - 1))
            current_qpos[:7] = ctrl_arm
            current_qpos[7:] = grasp_qpos[7:]  # Keep gripper at grasp position
            franka.set_joint_position_targets(current_qpos)

        # Phase 3: Close gripper (2-3s)
        elif 2 <= elapsed_time < 3:
            gripper_pos = lerp(0.04, 0.0, (elapsed_time - 2))
            current_qpos[:7] = grasp_qpos[:7]  # Keep arm at grasp position
            current_qpos[7] = gripper_pos
            current_qpos[8] = gripper_pos
            franka.set_joint_position_targets(current_qpos)

        # Phase 4: Lift object (3-4s)
        elif 3 <= elapsed_time < 4:
            ctrl_arm = lerp(grasp_qpos[:7], lift_qpos[:7], (elapsed_time - 3))
            current_qpos[:7] = ctrl_arm
            current_qpos[7:] = lift_qpos[7:]  # Keep gripper closed
            franka.set_joint_position_targets(current_qpos)

        # Phase 5: Shake and verify (4-20s)
        elif 4 <= elapsed_time < 20:
            if _Shake.value and step_cnt % 2 == 0:
                ctrl_arm = lift_qpos[:7] + np.random.normal(0, 0.025, size=7)
                current_qpos[:7] = ctrl_arm
                current_qpos[7:] = lift_qpos[7:]
                franka.set_joint_position_targets(current_qpos)

            # Check if object fell - get object position using IsaacSim high-level API
            if isinstance(obj, XFormPrim):
                obj_position, _ = obj.get_world_poses()
                obj_position = obj_position[0]
            else:
                obj_position, _ = obj.get_world_pose()
            obj_z = obj_position[2]
            if obj_z < obj_z_threshold:
                test_passed = False
                drop_time = elapsed_time
                print(
                    f"\n❌ The {task}-{_Obj.value} failed (object fell at {drop_time:.2f}s)"
                )
                break

        # Phase 6: Success (>= 20s)
        elif elapsed_time >= 20:
            print(f"\n✅ The {task}-{_Obj.value} passed.")
            break

        # Step simulation
        world.step(render=True)

        # Record frame if enabled
        if (
            _Record.value
            and camera is not None
            and len(frames) < int(world.current_time * recording_fps)
        ):
            try:
                rgba_image = camera.get_rgba()
                if rgba_image is not None and rgba_image.size > 0:
                    rgb_image = rgba_image[:, :, :3]
                    frames.append(rgb_image)
            except Exception as e:
                print(f"Warning: Failed to capture frame: {e}")

    # Save recording if enabled
    if _Record.value and len(frames) > 0:
        save_video(frames, video_path, fps=recording_fps, quality=8)
        save_test_result(
            video_path,
            "success" if test_passed else "failure",
            drop_time,
            output_dir,
            "isaacsim",
            _Obj.value,
            _Shake.value,
            False,
            _Dt.value,
        )

    # Clean up
    simulation_app.close()

    return


if __name__ == "__main__":
    app.run(main)
