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

import time

import mujoco
import mujoco.viewer
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
    path = "gyro_precession/xml/gyro_precession.xml"

    model = mujoco.MjModel.from_xml_path(path)
    model.opt.timestep = _Dt.value
    data = mujoco.MjData(model)

    data.qvel[5] = _Velocity.value

    # Initialize output and tracking
    output_dir = ensure_output_directory()
    video_path = generate_video_path("mujoco", _Velocity.value, _Dt.value, output_dir)
    test_passed = True
    drop_time = None

    if _Record.value:
        frames = []
        renderer = mujoco.Renderer(model)
        renderer.update_scene(data, 0)

    task = "sgyro_precession"

    # Run simulation with or without viewer based on -v/--visual flag
    viewer = None
    if _Visual.value:
        viewer = mujoco.viewer.launch_passive(model, data)

    step_cnt = 0
    while True:
        # Check if viewer is still running (only applicable if viewer exists)
        if viewer and not viewer.is_running():
            break
        step_cnt += 1
        step_start = time.time()
        elapsed_time = step_cnt * model.opt.timestep

        # Phase 6: Success (>= 20 seconds)
        if elapsed_time >= 20:
            print(f"✅ The {task}-test finished.")
            break

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(model, data)

        # Sync viewer if present
        if viewer:
            viewer.sync()
            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        # Recording (works in both modes)
        if _Record.value and len(frames) < data.time * 30:
            renderer.update_scene(data, 0)
            frames.append(renderer.render().copy())

    # Clean up viewer context manager
    if viewer:
        viewer.close()

    if _Record.value:
        save_video(frames, video_path, fps=30, quality=8)
        save_test_result(
            video_path,
            "success" if test_passed else "failure",
            drop_time,
            output_dir,
            "mujoco",
            _Velocity.value,
            _Dt.value,
        )


if __name__ == "__main__":
    app.run(main)
