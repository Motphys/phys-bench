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
headless_mode = True
simulation_app = SimulationApp({"headless": headless_mode})  # start the simulation app

_Velocity = flags.DEFINE_float("velocity", 20.0, "initial angular velocity")
_Record = flags.DEFINE_boolean(
    "record", False, "whether to record the simulation, Choices: [True, False]"
)
_Dt = flags.DEFINE_float("dt", 0.002, "simulation timestep")

from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.storage.native import get_assets_root_path
from isaacsim.sensors.camera import Camera


def main(argv):
    # Initialize IsaacSim 5.0 simulation app
    # Note: SimulationApp must be initialized BEFORE importing other IsaacSim modules

    # Initialize IsaacSim world
    sim_dt = _Dt.value
    world = World(stage_units_in_meters=1.0, physics_dt=sim_dt)
    world.scene.add_default_ground_plane()

    # Initialize output and tracking
    output_dir = ensure_output_directory()
    video_path = generate_video_path("isaacsim", _Velocity.value, _Dt.value, output_dir)
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
            camera_eye = np.array([5.155, 0.016, 1.526])
            camera_orientation = rot_utils.euler_angles_to_quats(
                np.array([0.0, 15, 180]), degrees=True
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

        except Exception as e:
            print(f"Warning: Failed to initialize recording camera: {e}")
            print("Recording will be disabled.")
            _Record.value = False
            camera = None

    # Simulation loop parameters
    task = "gyro_precession"
    step_cnt = 0

    # Load Franka Panda robot from IsaacSim assets
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        print("Error: Cannot find IsaacSim assets root path")
        simulation_app.close()
        return

    gyroscopic_usd_path = "./assets/objects/gyroscopic.usd"
    add_reference_to_stage(usd_path=gyroscopic_usd_path, prim_path="/World/Gyroscopic")
    gyroscopic = RigidPrim(prim_paths_expr="/World/Gyroscopic", name="gyroscopic")
    world.reset()

    velocities = np.zeros((1, 6))
    velocities[:, 5] = _Velocity.value
    gyroscopic.set_velocities(velocities)

    while True:
        step_cnt += 1
        elapsed_time = world.current_time  # Use IsaacSim's world time

        if elapsed_time >= 20:
            print(f"✅ The {task}-test finished.")
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
            _Velocity.value,
            _Dt.value,
        )

    # Clean up
    simulation_app.close()

    return


if __name__ == "__main__":
    app.run(main)
