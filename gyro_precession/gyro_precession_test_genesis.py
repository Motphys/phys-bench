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

import genesis as gs
import numpy as np
from absl import app, flags
from test_output_utils import (
    ensure_output_directory,
    generate_video_path,
    save_test_result,
    save_video,
)

_Velocity = flags.DEFINE_float("velocity", 20.0, "initial angular velocity")
_Record = flags.DEFINE_boolean(
    "record", False, "whether to record the simulation, Choices: [True, False]"
)
_Dt = flags.DEFINE_float("dt", 0.002, "simulation timestep")
_Visual = flags.DEFINE_boolean(
    "visual",
    False,
    "whether to visualize the simulation in a window, Choices: [True, False]",
    short_name="V",
)


def main(argv):
    # Initialize Genesis
    gs.init()

    sim_dt = _Dt.value  # Simulation timestep
    # Create scene with viewer
    scene = gs.Scene(
        show_viewer=_Visual.value,
        sim_options=gs.options.SimOptions(dt=sim_dt),
    )

    # Load Gyroscopic from MJCF
    gyroscopic = scene.add_entity(
        morph=gs.morphs.MJCF(file="gyro_precession/xml/gyroscopic.xml"),
    )

    # Add floor plane
    _ = scene.add_entity(morph=gs.morphs.Plane())

    # Add camera for recording (before build)
    if _Record.value:
        camera = scene.add_camera(
            res=(320, 240),
            pos=(2.155, 0.016, 1.026),
            lookat=(-0.2, 0.0, 0),
            fov=45,
            GUI=False,
        )

    # Build scene (CRITICAL STEP - must be before setting DOF properties)
    scene.build()

    gyroscopic.set_dofs_velocity([0, 0, 0, 0, 0, _Velocity.value])

    # Initialize recording
    if _Record.value:
        frames = []
        recording_fps = 30

    # Initialize output and tracking
    output_dir = ensure_output_directory()
    video_path = generate_video_path("genesis", _Velocity.value, _Dt.value, output_dir)
    test_passed = True
    drop_time = None

    # Simulation loop
    task = "gyro_precession"
    step_cnt = 0

    while True:
        step_cnt += 1
        elapsed_time = step_cnt * sim_dt
        print(f"Step: {step_cnt}", end="\r")

        if elapsed_time >= 20:
            print(f"✅ The {task}-test finished.")
            break

        # Step simulation
        interval = int(0.02 / sim_dt)  # render in 50 Hz
        update_visualizer = (
            interval == 0 or step_cnt % interval == 0 and not _Record.value
        ) and _Visual.value
        scene.step(
            update_visualizer=update_visualizer, refresh_visualizer=update_visualizer
        )

        # Record frame if enabled
        if _Record.value and len(frames) < int(scene.cur_t * recording_fps):
            (rgb, _, _, _) = camera.render()
            frames.append(rgb)

    # Save recording if enabled
    if _Record.value and len(frames) > 0:
        save_video(frames, video_path, fps=recording_fps, quality=8)
        save_test_result(
            video_path,
            "success" if test_passed else "failure",
            drop_time,
            output_dir,
            "genesis",
            _Velocity.value,
            _Dt.value,
        )


if __name__ == "__main__":
    app.run(main)
