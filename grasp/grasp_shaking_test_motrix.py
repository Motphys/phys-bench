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
_UseMJX = flags.DEFINE_boolean(
    "mjx", False, "Use mjx_panda.xml or panda.xml for the Franka robot model"
)
_Visual = flags.DEFINE_boolean(
    "visual",
    False,
    "whether to visualize the simulation in a window, Choices: [True, False]",
    short_name="V",
)


def lerp(a, b, t):
    return a + t * (b - a)


init_qpos = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853, 0.04, 0.04])
grasp_qpos = np.array(
    [-1.0104, 1.5623, 1.3601, -1.6840, -1.5863, 1.7810, 1.4598, 0.04, 0.04]
)
lift_qpos = np.array(
    [-1.0426, 1.4028, 1.5634, -1.7114, -1.4055, 1.6015, 1.4510, 0.0, 0.0]
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
    prefix = _UseMJX.value and "mjx_" or ""
    path = f"grasp/xml/{prefix}pick_{_Obj.value}.xml"
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
    panda = model.get_body("link0")
    panda.set_dof_pos(data, init_qpos)
    obj = model.get_body(_Obj.value)
    gripper_actuator = model.get_actuator("actuator8")

    def set_arm_ctrl(target_qpos):
        ctrls = data.actuator_ctrls
        ctrls[:7] = target_qpos
        data.actuator_ctrls = ctrls

    def set_gripper_ctrl(target_gripper):
        gripper_actuator.set_ctrl(data, target_gripper)

    set_arm_ctrl(init_qpos[:7])
    set_gripper_ctrl(init_qpos[7])

    task = "shaking-grasp" if _Shake.value else "slip-grasp"
    # Initialize output directory and video path
    output_dir = ensure_output_directory()
    video_path = generate_video_path(
        "motrix", _Obj.value, _Shake.value, _UseMJX.value, _Dt.value, output_dir
    )
    drop_time = None  # Track when object drops
    step_cnt = 0
    sim_dt = _Dt.value  # Simulation timestep
    render_dt = 1.0 / 60.0  # Render at 30 Hz
    phys_steps_per_render = int(render_dt / sim_dt)

    while True:
        for i in range(phys_steps_per_render):
            step_cnt += 1
            elapsed_time = step_cnt * sim_dt

            # Phase 1: Move from init to lift (0-1 second)
            if 0 <= elapsed_time < 1:
                ctrl_arm = lerp(init_qpos[:7], lift_qpos[:7], elapsed_time)
                set_arm_ctrl(ctrl_arm)
            # Phase 2: Move from lift to grasp (1-2 seconds)
            elif 1 <= elapsed_time < 2:
                ctrl_arm = lerp(lift_qpos[:7], grasp_qpos[:7], (elapsed_time - 1))
                set_arm_ctrl(ctrl_arm)
            # Phase 3: Close gripper (2-3 seconds)
            elif 2 <= elapsed_time < 3:
                set_gripper_ctrl(lerp(0.04, 0, (elapsed_time - 2)))
            # Phase 4: Lift object (3-4 seconds)
            elif 3 <= elapsed_time < 4:
                ctrl_arm = lerp(grasp_qpos[:7], lift_qpos[:7], (elapsed_time - 3))
                set_arm_ctrl(ctrl_arm)
            # Phase 5: Shake and verify (4-20 seconds)
            elif 4 <= elapsed_time < 20:
                if _Shake.value and step_cnt % 2 == 0:
                    ctrl_arm = lift_qpos[:7] + np.random.normal(0, 0.025, size=7)
                    set_arm_ctrl(ctrl_arm)
                obj_pos = obj.get_pose(data)
                if obj_pos[2] < 0.03:
                    drop_time = elapsed_time
                    print(f"❌ The {task}-{_Obj.value}-test failed.")
                    if _Record.value:
                        save_video(frames, video_path, fps=30, quality=8)
                        save_test_result(
                            video_path,
                            "failure",
                            drop_time,
                            output_dir,
                            "motrix",
                            _Obj.value,
                            _Shake.value,
                            _UseMJX.value,
                            _Dt.value,
                        )
                    exit(0)
            # Phase 6: Success (>= 20 seconds)
            elif elapsed_time >= 20:
                print(f"✅ The {task}-{_Obj.value}-test passed.")
                if _Record.value:
                    save_video(frames, video_path, fps=30, quality=8)
                    save_test_result(
                        video_path,
                        "success",
                        None,
                        output_dir,
                        "motrix",
                        _Obj.value,
                        _Shake.value,
                        _UseMJX.value,
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
