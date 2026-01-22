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

from collections import deque

import numpy as np
from absl import app, flags

from motrixsim import SceneData, load_model, step
from motrixsim.render import CaptureTask, RenderApp
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


# Mouse controls:
# - Press and hold left button then drag to rotate the camera/view
# - Press and hold right button then drag to pan/translate the view
def main(argv):
    # Create render window for visualization
    show_visualizer = _Visual.value
    headless = _Record.value and not show_visualizer
    renderer = (
        None
        if not show_visualizer and not _Record.value
        else RenderApp(headless=headless)
    )
    # The scene description file
    path = "gyro_precession/xml/gyro_precession.xml"
    # Load the scene model
    model = load_model(path)
    # Set simulation timestep from command line argument
    model.options.timestep = _Dt.value
    cameras = model.cameras
    if _Record.value:
        cameras[0].set_render_target("image", 320, 240)
        frames = []
        capture_tasks = deque()
        capture_index = 0
    # Create the render instance of the model
    if renderer:
        renderer.launch(model)
    # Create the physics data of the model
    data = SceneData(model)

    body_fb = model.get_body(model.get_body_index("gyro")).floatingbase
    body_fb.set_local_angular_velocity(data, np.array([0, 0, _Velocity.value]))

    task = "gyro-precession"
    # Initialize output directory and video path
    output_dir = ensure_output_directory()
    video_path = generate_video_path("motrix", _Velocity.value, _Dt.value, output_dir)
    step_cnt = 0
    sim_dt = _Dt.value  # Simulation timestep
    render_dt = 1.0 / 60.0  # Render at 30 Hz
    phys_steps_per_render = int(render_dt / sim_dt)

    while True:
        for i in range(phys_steps_per_render):
            step_cnt += 1
            elapsed_time = step_cnt * sim_dt

            if elapsed_time >= 20:
                print(f"✅ The {task}-test finished.")
                if _Record.value:
                    save_video(frames, video_path, fps=30, quality=8)
                    save_test_result(
                        video_path,
                        "success",
                        None,
                        output_dir,
                        "motrix",
                        _Velocity.value,
                        _Dt.value,
                    )
                exit(0)

            # Physics world step
            step(model, data)

        if renderer:
            if _Record.value and capture_index < step_cnt * sim_dt * 30:
                rcam = renderer.get_camera(0)
                capture_tasks.append((capture_index, rcam.capture()))
                capture_index += 1
            renderer.sync(data)
            if _Record.value:
                while len(capture_tasks) > 0:
                    task: CaptureTask
                    idx, task = capture_tasks[0]
                    if task.state != "pending":
                        capture_tasks.popleft()
                        img = task.take_image()
                        # assert img is not None
                        if img is not None and img.pixels.max() > 0:
                            frames.append(img.pixels)
                    else:
                        break


if __name__ == "__main__":
    app.run(main)
