import argparse
import os
import time

import motrixsim as mx
from motrixsim import run
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-B", type=int, default=1)  # batch size
parser.add_argument("-v", action="store_true", default=False)  # visualize
parser.add_argument("-r", action="store_true", default=False)  # random action
args = parser.parse_args()

########################## load model ##########################
model_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../grasp/xml/mjx_pick_ball.xml"
    )
)

model = mx.load_model(model_path)
model.options.timestep = 0.01  # Match genesis benchmark (100 Hz)

########################## create batched data ##########################
n_envs = args.B
if n_envs > 1:
    data = mx.SceneData(model, batch=(n_envs,))
else:
    data = mx.SceneData(model)

########################## setup control ##########################
# Reference positions (ball has different grasp_qpos from cube)
grasp_qpos = np.array([-1.0323, 1.7628, 1.4904, -1.6749, -1.7715, 1.6293, 1.4417, 0.04, 0.04])
lift_qpos = np.array([-1.0426, 1.4028, 1.5634, -1.7114, -1.4055, 1.6015, 1.4510, 0.0, 0.0])

# Initialize robot to grasp position
panda_index = model.get_body_index("link0")
panda = model.get_body(panda_index)
# For batched data, need to tile grasp_qpos to match batch size
if n_envs > 1:
    init_pos = np.tile(grasp_qpos, (n_envs, 1))
    panda.set_dof_pos(data, init_pos)
else:
    panda.set_dof_pos(data, grasp_qpos)

########################## setup render (if needed) ##########################
if args.v:
    render = mx.render.RenderApp()
    render.__enter__()
    render.launch(model)
else:
    render = None


def make_ctrl(joint_qpos, gripper_val, n_envs):
    """Create actuator control array for mjx_panda (8 actuators)"""
    ctrl = np.zeros((n_envs, 8), dtype=np.float32)
    ctrl[:, :7] = joint_qpos
    ctrl[:, 7] = gripper_val  # gripper (actuator8 controls both fingers via equality constraint)
    return ctrl


sim_dt = model.options.timestep

########################## Warmup Phase 1: Grasp (100 steps) ##########################
print("Warmup Phase 1: Grasping (100 steps)...")
for i in range(100):
    ctrl = make_ctrl(grasp_qpos[:7], 0.0, n_envs)
    data.actuator_ctrls = ctrl
    model.step(data)

########################## Warmup Phase 2: Lift (100 steps) ##########################
print("Warmup Phase 2: Lifting (100 steps)...")
for i in range(100):
    ctrl = make_ctrl(lift_qpos[:7], 0.0, n_envs)
    data.actuator_ctrls = ctrl
    model.step(data)

########################## Benchmark ##########################
print("Benchmark: 500 steps...")

benchmark_steps = 500
ref_ctrl = make_ctrl(lift_qpos[:7], 0.0, n_envs)
ref_pos = lift_qpos[:7].copy()

if args.v:
    # Decoupled rendering mode using render_loop
    step_counter = [0]
    benchmark_complete = [False]
    t_start = [None]
    t_end = [None]

    def phys_step():
        if benchmark_complete[0]:
            return
        if t_start[0] is None:
            t_start[0] = time.perf_counter()

        i = step_counter[0]
        if i < benchmark_steps:
            if args.r and i % 2 == 0:
                noise = np.random.uniform(-0.025, 0.025, (n_envs, 7)).astype(np.float32)
                ctrl = ref_ctrl.copy()
                ctrl[:, :7] = ref_pos + noise
                data.actuator_ctrls = ctrl
            model.step(data)
            step_counter[0] += 1
        else:
            t_end[0] = time.perf_counter()
            benchmark_complete[0] = True
            print(f"per env: {benchmark_steps / (t_end[0] - t_start[0]):,.2f} FPS")
            print(f"total  : {benchmark_steps / (t_end[0] - t_start[0]) * n_envs:,.2f} FPS")

    def render_func():
        render.sync(data)

    run.render_loop(sim_dt, 60, phys_step, render_func)
else:
    # Pure benchmark without rendering
    t0 = time.perf_counter()
    for i in range(benchmark_steps):
        if args.r and i % 2 == 0:
            noise = np.random.uniform(-0.025, 0.025, (n_envs, 7)).astype(np.float32)
            ctrl = ref_ctrl.copy()
            ctrl[:, :7] = ref_pos + noise
            data.actuator_ctrls = ctrl
        model.step(data)
    t1 = time.perf_counter()
    print(f"per env: {benchmark_steps / (t1 - t0):,.2f} FPS")
    print(f"total  : {benchmark_steps / (t1 - t0) * n_envs:,.2f} FPS")

# Cleanup render
if render:
    try:
        render.__exit__(None, None, None)
    except:
        pass
