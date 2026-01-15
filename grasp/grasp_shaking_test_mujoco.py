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


def lerp(a, b, t):
    return a + t * (b - a)


init_qpos = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853, 0.04, 0.04])
grasp_qpos = np.array(
    [-1.0104, 1.5623, 1.3601, -1.6840, -1.5863, 1.7810, 1.4598, 0.04, 0.04]
)
lift_qpos = np.array(
    [-1.0426, 1.4028, 1.5634, -1.7114, -1.4055, 1.6015, 1.4510, 0.0, 0.0]
)


def main(argv):
    prefix = _UseMJX.value and "mjx_" or ""
    path = f"grasp/xml/{prefix}pick_{_Obj.value}.xml"

    model = mujoco.MjModel.from_xml_path(path)
    model.opt.timestep = _Dt.value
    data = mujoco.MjData(model)

    data.qpos[0:9] = init_qpos
    data.ctrl[0:7] = init_qpos[0:7]
    data.ctrl[7] = init_qpos[7]

    if _Record.value:
        frames = []
        renderer = mujoco.Renderer(model)
        renderer.update_scene(data, 0)

    task = "shaking-grasp" if _Shake.value else "slip-grasp"
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_cnt = 0
        while viewer.is_running():
            step_cnt += 1
            step_start = time.time()
            elapsed_time = step_cnt * model.opt.timestep

            # Phase 1: Move from init to lift (0-1 second)
            if 0 <= elapsed_time < 1:
                ctrl_arm = lerp(init_qpos[:7], lift_qpos[:7], elapsed_time)
                data.ctrl[:7] = ctrl_arm
            # Phase 2: Move from lift to grasp (1-2 seconds)
            elif 1 <= elapsed_time < 2:
                ctrl_arm = lerp(lift_qpos[:7], grasp_qpos[:7], (elapsed_time - 1))
                data.ctrl[:7] = ctrl_arm
            # Phase 3: Close gripper (2-3 seconds)
            elif 2 <= elapsed_time < 3:
                data.ctrl[7] = lerp(0.04, 0, (elapsed_time - 2))
            # Phase 4: Lift object (3-4 seconds)
            elif 3 <= elapsed_time < 4:
                ctrl_arm = lerp(grasp_qpos[:7], lift_qpos[:7], (elapsed_time - 3))
                data.ctrl[:7] = ctrl_arm
            # Phase 5: Shake and verify (4-20 seconds)
            elif 4 <= elapsed_time < 20:
                if _Shake.value and step_cnt % 2 == 0:
                    ctrl_arm = lift_qpos[:7] + np.random.normal(0, 0.025, size=7)
                    data.ctrl[:7] = ctrl_arm
                obj_pos = data.xpos[model.body(_Obj.value).id]
                if obj_pos[2] < 0.04:
                    print(f"❌ The {task}-{_Obj.value} failed.")
                    break
            # Phase 6: Success (>= 20 seconds)
            elif elapsed_time >= 20:
                print(f"✅ The {task}-{_Obj.value} passed.")
                break

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(model, data)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            if _Record.value and len(frames) < data.time * 30:
                renderer.update_scene(data, 0)
                frames.append(renderer.render().copy())

    if _Record.value:
        import imageio

        imageio.mimwrite(
            f"mujoco_grasp_{'shake' if _Shake.value else 'slip'}_{_Obj.value}_noslip_iterations=1.mp4",
            frames,
            fps=30,
            quality=8,
        )


if __name__ == "__main__":
    app.run(main)
