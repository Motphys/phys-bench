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


def main(argv):
    # Initialize Genesis
    gs.init()

    sim_dt = _Dt.value  # Simulation timestep
    # Create scene with viewer
    scene = gs.Scene(
        show_viewer=True,
        sim_options=gs.options.SimOptions(dt=sim_dt),
    )

    # Load Franka Panda robot from MJCF
    franka = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="assets/franka_emika_panda/panda.xml"
            if not _UseMJX.value
            else "assets/franka_emika_panda/mjx_panda.xml",
        ),
    )

    # Add floor plane
    _ = scene.add_entity(morph=gs.morphs.Plane())

    # Add camera for recording (before build)
    if _Record.value:
        camera = scene.add_camera(
            res=(320, 240),
            pos=(1.155, 0.016, 0.526),
            lookat=(0.5, 0.0, 0),
            fov=45,
            GUI=False,
        )
    # Add object to grasp (based on _Obj flag)
    if _Obj.value == "cube":
        obj = scene.add_entity(
            morph=gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02))
        )
    elif _Obj.value == "ball":
        obj = scene.add_entity(
            morph=gs.morphs.Sphere(radius=0.02, pos=(0.65, 0.0, 0.02))
        )
    elif _Obj.value == "bottle":
        obj = scene.add_entity(
            morph=gs.morphs.MJCF(
                file="assets/objects/scene_bottle.xml",
            ),
        )

    # Build scene (CRITICAL STEP - must be before setting DOF properties)
    scene.build()

    # Set PD gains for position control (must be after build)
    # Match mjx_panda.xml actuator configuration:
    # - joint1-2: kp=4500, kv=450
    # - joint3-4: kp=3500, kv=350
    # - joint5-7: kp=2000, kv=200
    # - gripper (finger joints): gainprm=350 (general actuator)
    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 350, 350]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 0, 0]))

    # Set initial configuration (from MuJoCo keyframes)
    init_qpos = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853, 0.04, 0.04])
    grasp_qpos = np.array(
        [-1.0104, 1.5623, 1.3601, -1.6840, -1.5863, 1.7810, 1.4598, 0.04, 0.04]
    )
    lift_qpos = np.array(
        [-1.0426, 1.4028, 1.5634, -1.7114, -1.4055, 1.6015, 1.4510, 0.0, 0.0]
    )

    franka.set_dofs_position(init_qpos)

    # Initialize recording
    if _Record.value:
        frames = []
        recording_fps = 30

    # Simulation loop
    task = "shaking-grasp" if _Shake.value else "slip-grasp"
    step_cnt = 0
    current_qpos = init_qpos.copy()

    while True:
        step_cnt += 1
        elapsed_time = step_cnt * sim_dt
        print(f"Step: {step_cnt}", end="\r")

        # Phase 1: Move from init to lift (steps 0-500)
        if 0 <= elapsed_time < 1:
            ctrl_arm = lerp(init_qpos[:7], lift_qpos[:7], elapsed_time)
            current_qpos[:7] = ctrl_arm
            current_qpos[7:] = init_qpos[7:]  # Keep gripper open
            franka.control_dofs_position(current_qpos)

        # Phase 2: Move from lift to grasp (steps 500-1000)
        elif 1 <= elapsed_time < 2:
            ctrl_arm = lerp(lift_qpos[:7], grasp_qpos[:7], (elapsed_time - 1))
            current_qpos[:7] = ctrl_arm
            current_qpos[7:] = grasp_qpos[7:]  # Keep gripper at grasp position
            franka.control_dofs_position(current_qpos)

        # Phase 3: Close gripper (steps 1000-1500)
        elif 2 <= elapsed_time < 3:
            gripper_pos = lerp(0.04, 0.0, (elapsed_time - 2))
            current_qpos[:7] = grasp_qpos[:7]  # Keep arm at grasp position
            current_qpos[7] = gripper_pos
            current_qpos[8] = gripper_pos
            franka.control_dofs_position(current_qpos)

        # Phase 4: Lift object (steps 1500-2000)
        elif 3 <= elapsed_time < 4:
            ctrl_arm = lerp(grasp_qpos[:7], lift_qpos[:7], (elapsed_time - 3))
            current_qpos[:7] = ctrl_arm
            current_qpos[7:] = lift_qpos[7:]  # Keep gripper closed
            franka.control_dofs_position(current_qpos)

        # Phase 5: Shake and verify (steps 2000-10000)
        elif 4 <= elapsed_time < 20:
            if _Shake.value and step_cnt % 2 == 0:
                ctrl_arm = lift_qpos[:7] + np.random.normal(0, 0.025, size=7)
                current_qpos[:7] = ctrl_arm
                current_qpos[7:] = lift_qpos[7:]
                franka.control_dofs_position(current_qpos)

            # Check if object fell
            obj_pos = obj.get_pos()
            if obj_pos[2] < 0.03:
                print(f"❌ The {task}-{_Obj.value} failed.")
                break

        # Phase 6: Success (step > 10000)
        elif elapsed_time >= 20:
            print(f"✅ The {task}-{_Obj.value} passed.")
            break

        # Step simulation
        interval = int(0.02 / sim_dt)  # render in 50 Hz
        update_visualizer = (
            interval == 0 or step_cnt % interval == 0 and not _Record.value
        )
        scene.step(
            update_visualizer=update_visualizer, refresh_visualizer=update_visualizer
        )

        # Record frame if enabled
        if _Record.value and len(frames) < int(scene.cur_t * recording_fps):
            (rgb, _, _, _) = camera.render()
            frames.append(rgb)

    # Save recording if enabled
    if _Record.value and len(frames) > 0:
        import imageio

        imageio.mimwrite(
            f"genesis_grasp_{'shake' if _Shake.value else 'slip'}_{_Obj.value}.mp4",
            frames,
            fps=recording_fps,
            quality=8,
        )


if __name__ == "__main__":
    app.run(main)
