import argparse
import json
import os
import sys
import time

import motrixsim as mx


parser = argparse.ArgumentParser()
parser.add_argument(
    "-N", type=int, default=1, choices=range(1, 101), help="Number of humanoids (1-101)"
)
parser.add_argument("-B", type=int, default=1, help="Batch size")
parser.add_argument("-v", action="store_true", help="Enable visualization")
parser.add_argument("--warmup", type=int, default=10, help="Warmup steps")
parser.add_argument("--benchmark", type=int, default=500, help="Benchmark steps")

args = parser.parse_args()


def get_humanoid_positions(n_humanoids):
    """Calculate humanoid positions - spread them out in a grid layout with 3m spacing"""
    positions = []
    for i in range(n_humanoids):
        positions.append((i * 1.2, 0, 0))
    return positions


########################## load model ##########################
# Load first humanoid as base scene
humanoid_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../assets/humanoid.xml")
)
scene = mx.msd.from_file(humanoid_path)

# Attach additional humanoids if N > 1
if args.N > 1:
    positions = get_humanoid_positions(args.N)
    for i in range(1, args.N):
        humanoid = mx.msd.from_file(humanoid_path)
        scene.attach(
            humanoid,
            other_link_name="torso",
            other_prefix=f"robot{i}_",
            other_translation=positions[i],
            other_rotation=(0, 0, 0, 1),
        )

model = scene.build()
model.options.timestep = 0.005  # From humanoid.xml

########################## create batched data ##########################
n_envs = args.B
if n_envs > 1:
    data = mx.SceneData(model, batch=(n_envs,))
else:
    data = mx.SceneData(model)

########################## setup render (if needed) ##########################
if args.v:
    render = mx.render.RenderApp()
    render.__enter__()
    render.launch(model)
else:
    render = None

sim_dt = model.options.timestep

########################## warmup phase ##########################
print(f"Warmup: {args.N} humanoids to standing position ({args.warmup} steps)...")

# Simple warmup (no timeout check)
for i in range(args.warmup):
    model.step(data)


########################## benchmark phase ##########################
print(f"Benchmark: {args.benchmark} steps with {args.N} humanoids...")
benchmark_steps = args.benchmark

if args.v:
    t = 0
    while True:
        model.step(data)
        t += model.options.timestep
        if t > 0.16:
            render.sync(data)
            t -= 0.16
else:
    try:
        t0 = time.perf_counter()
        for i in range(benchmark_steps):
            model.step(data)
            elapsed = time.perf_counter() - t0
            # Timeout check: if cumulative time exceeds i * 2 seconds
            if elapsed > (i + 1) * 2.0:
                error_data = {
                    "status": "error",
                    "error_code": "TIMEOUT",
                    "error_message": f"Timeout at step {i+1}: {elapsed:.2f}s > {(i+1)*2.0:.2f}s",
                    "per_env_fps": 0.0,
                    "total_fps": 0.0
                }
                print(json.dumps(error_data))
                sys.exit(1)
        t1 = time.perf_counter()

        success_data = {
            "status": "success",
            "per_env_fps": benchmark_steps / (t1 - t0),
            "total_fps": benchmark_steps / (t1 - t0) * n_envs
        }
        print(f"per env: {success_data['per_env_fps']:,.2f} FPS")
        print(f"total  : {success_data['total_fps']:,.2f} FPS")
    except Exception as e:
        error_data = {
            "status": "error",
            "error_code": "BENCHMARK_ERROR",
            "error_message": f"{type(e).__name__}: {str(e)}",
            "per_env_fps": 0.0,
            "total_fps": 0.0
        }
        print(json.dumps(error_data))
        sys.exit(1)

# Cleanup render
if render:
    try:
        render.__exit__(None, None, None)
    except Exception as e:
        print(e)
