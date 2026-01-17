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

import mujoco
import numpy as np
from absl import app, flags
from test_output_utils import (
    ensure_output_directory,
    generate_video_path,
    save_test_result,
    save_video,
)

# Try to import mujoco_warp, provide helpful error if not available
try:
    import mujoco_warp as mjw
    import warp as wp
except ImportError as e:
    raise ImportError(
        "MuJoCo-Warp is not installed. Please install it with:\n"
        "  git clone https://github.com/google-deepmind/mujoco_warp.git\n"
        "  cd mujoco_warp\n"
        "  uv pip install -e .[dev,cuda]\n"
        "Or use: uv pip install -e '.[mujoco-warp]'"
    ) from e

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

    # Load MuJoCo model
    mjm = mujoco.MjModel.from_xml_path(path)
    mjm.opt.timestep = _Dt.value

    # Put model and data on GPU device using MuJoCo-Warp
    # Note: mjw.make_data creates data with batch dimension (nworld, ...)
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm)

    # Initialize state - ctrl has shape (1, 8) where 1 is nworld, 8 is nu
    # qpos has shape (1, nq) where nq is 16 (includes object free joint)
    # We only set the first 9 elements, the rest are object DOFs
    ctrl_array = np.zeros((1, 8), dtype=np.float32)
    ctrl_array[0, :7] = init_qpos[:7]
    ctrl_array[0, 7] = init_qpos[7]
    wp.copy(d.ctrl, wp.array(ctrl_array, dtype=wp.float32))

    # Initialize qpos - set first 9 elements, keep rest at default (object position)
    qpos_init = d.qpos.numpy()[0]
    qpos_init[:9] = init_qpos
    # The rest of qpos (elements 9-15) contain the cube's free joint DOF
    # Leave them at default values from the model
    wp.copy(d.qpos, wp.array(qpos_init.reshape(1, -1), dtype=wp.float32))

    if _Record.value:
        frames = []
        renderer = mujoco.Renderer(mjm)
        # Get initial CPU data for rendering
        mjd_cpu = mujoco.MjData(mjm)
        mjd_cpu.qpos[:] = qpos_init  # Use full qpos_init, not just init_qpos
        mujoco.mj_forward(mjm, mjd_cpu)
        renderer.update_scene(mjd_cpu, 0)

    # Initialize output and tracking
    output_dir = ensure_output_directory()
    video_path = generate_video_path(
        "mujocowarp", _Obj.value, _Shake.value, _UseMJX.value, _Dt.value, output_dir
    )
    test_passed = True
    drop_time = None

    task = "shaking-grasp" if _Shake.value else "slip-grasp"

    # Get object body index for final verification
    obj_body_id = mjm.body(_Obj.value).id

    # Simulation loop
    step_cnt = 0
    # Track current gripper state
    current_gripper = init_qpos[7]

    while True:
        step_cnt += 1
        elapsed_time = step_cnt * mjm.opt.timestep

        # Phase 1: Move from init to lift (0-1 second)
        if 0 <= elapsed_time < 1:
            ctrl_arm = lerp(init_qpos[:7], lift_qpos[:7], elapsed_time)
            ctrl_array[0, :7] = ctrl_arm
            ctrl_array[0, 7] = current_gripper
            wp.copy(d.ctrl, wp.array(ctrl_array, dtype=wp.float32))
        # Phase 2: Move from lift to grasp (1-2 seconds)
        elif 1 <= elapsed_time < 2:
            ctrl_arm = lerp(lift_qpos[:7], grasp_qpos[:7], (elapsed_time - 1))
            ctrl_array[0, :7] = ctrl_arm
            ctrl_array[0, 7] = current_gripper
            wp.copy(d.ctrl, wp.array(ctrl_array, dtype=wp.float32))
        # Phase 3: Close gripper (2-3 seconds)
        elif 2 <= elapsed_time < 3:
            current_gripper = lerp(0.04, 0.0, (elapsed_time - 2))
            ctrl_array[0, :7] = grasp_qpos[:7]
            ctrl_array[0, 7] = current_gripper
            wp.copy(d.ctrl, wp.array(ctrl_array, dtype=wp.float32))
        # Phase 4: Lift object (3-4 seconds)
        elif 3 <= elapsed_time < 4:
            ctrl_arm = lerp(grasp_qpos[:7], lift_qpos[:7], (elapsed_time - 3))
            ctrl_array[0, :7] = ctrl_arm
            ctrl_array[0, 7] = current_gripper
            wp.copy(d.ctrl, wp.array(ctrl_array, dtype=wp.float32))
        # Phase 5: Shake and verify (4-20 seconds)
        elif 4 <= elapsed_time < 20:
            if _Shake.value and step_cnt % 2 == 0:
                ctrl_arm = lift_qpos[:7] + np.random.normal(0, 0.025, size=7)
                ctrl_array[0, :7] = ctrl_arm
                ctrl_array[0, 7] = current_gripper
                wp.copy(d.ctrl, wp.array(ctrl_array, dtype=wp.float32))
            # Check if object fell - copy final state to CPU MuJoCo and verify
            # We do this check periodically during the shake phase
            if step_cnt % 100 == 0:  # Check every 0.2 seconds
                obj_pos = d.xpos.numpy()[0, obj_body_id]
                if obj_pos[2] < 0.04:
                    test_passed = False
                    drop_time = elapsed_time
                    print(f"❌ The {task}-{_Obj.value} failed.")
                    break
        # Phase 6: Success (>= 20 seconds)
        elif elapsed_time >= 20:
            print(f"✅ The {task}-{_Obj.value} passed.")
            break

        # Step physics using MuJoCo-Warp
        mjw.step(m, d)

        # Record frame if enabled
        if _Record.value and len(frames) < int(elapsed_time * 30):
            # Copy GPU data to CPU for rendering
            mjd_cpu.qpos[:] = d.qpos.numpy()[0]
            mjd_cpu.qvel[:] = d.qvel.numpy()[0]
            mujoco.mj_forward(mjm, mjd_cpu)
            renderer.update_scene(mjd_cpu, 0)
            frames.append(renderer.render().copy())

    # Save recording if enabled
    if _Record.value:
        save_video(frames, video_path, fps=30, quality=8)
        save_test_result(
            video_path,
            "success" if test_passed else "failure",
            drop_time,
            output_dir,
            "mujocowarp",
            _Obj.value,
            _Shake.value,
            _UseMJX.value,
            _Dt.value,
        )


if __name__ == "__main__":
    app.run(main)
