#!/usr/bin/env python3
"""
Comprehensive benchmark for unified bench/ scripts
Supports both random and grasp modes with multiple objects
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def run_test(cmd, description):
    """Run a test command and extract FPS results"""
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=300
        )

        output = result.stdout + result.stderr

        # Extract FPS from output
        per_env_match = re.search(r"per env:\s*([\d,]+\.?\d*)\s*FPS", output)
        total_match = re.search(r"total\s*:\s*([\d,]+\.?\d*)\s*FPS", output)

        if per_env_match and total_match:
            per_env_fps = per_env_match.group(1).replace(",", "")
            total_fps = total_match.group(1).replace(",", "")
            print(f"✓ per env: {per_env_fps} FPS")
            print(f"✓ total  : {total_fps} FPS")
            return float(per_env_fps), float(total_fps)
        else:
            print(f"✗ Failed to extract FPS")
            print(f"Output: {output[-500:]}")  # Print last 500 chars for debugging
            return None, None

    except subprocess.TimeoutExpired:
        print(f"✗ Timeout (300s)")
        return None, None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Comprehensive benchmark runner for unified bench/ scripts")
    parser.add_argument("--modes", nargs="+", default=["random", "grasp"], choices=["random", "grasp"], help="Modes to test")
    parser.add_argument("--simulators", nargs="+", default=["genesis", "motrixsim", "isaacsim", "mujocowarp"],
                        choices=["genesis", "motrixsim", "isaacsim", "mujocowarp"], help="Simulators to test")
    parser.add_argument("--robots", nargs="+", type=int, default=[1, 5, 10], help="Robot counts")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 64, 512, 1024], help="Batch sizes")
    parser.add_argument("--objects", nargs="+", default=["ball", "cube", "bottle"], choices=["ball", "cube", "bottle"],
                        help="Objects for grasp mode")

    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════╗
║            Unified Benchmark Suite (bench/)                  ║
║     Genesis / Motrixsim / IsaacSim / MuJoCo-Warp             ║
║         Random Mode + Grasp Mode (Ball/Cube/Bottle)          ║
╚══════════════════════════════════════════════════════════════╝
    """)

    results = {}
    simulator_configs = {
        "genesis": ("uv run bench/bench_genesis.py", "Genesis"),
        "motrixsim": ("uv run bench/bench_motrixsim.py", "Motrixsim"),
        "isaacsim": ("uv --project envs/isaacsim run python bench/bench_isaacsim.py", "IsaacSim"),
        "mujocowarp": ("uv run bench/bench_mujocowarp.py", "MuJoCo-Warp"),
    }

    # Run tests for all combinations
    for mode in args.modes:
        print(f"\n{'='*80}")
        print(f"MODE: {mode.upper()}")
        print(f"{'='*80}")

        if mode == "random":
            # Random mode: test all simulators x robots x batches
            for sim_key in args.simulators:
                cmd_prefix, sim_name = simulator_configs[sim_key]
                print(f"\n{'-'*60}")
                print(f"{sim_name} - Random Mode")
                print(f"{'-'*60}")

                for n in args.robots:
                    for b in args.batches:
                        cmd = f"{cmd_prefix} -N {n} -B {b} --mode random"
                        desc = f"{sim_name} Random - N={n}, B={b}"
                        per_env, total = run_test(cmd, desc)
                        key = f"{sim_key}_random_n{n}_b{b}"
                        results[key] = {"per_env": per_env, "total": total, "n": n, "b": b, "mode": "random", "sim": sim_key}

        elif mode == "grasp":
            # Grasp mode: test all simulators x robots x batches x objects
            for obj in args.objects:
                print(f"\n{'-'*60}")
                print(f"Object: {obj.upper()}")
                print(f"{'-'*60}")

                for sim_key in args.simulators:
                    cmd_prefix, sim_name = simulator_configs[sim_key]

                    for n in args.robots:
                        for b in args.batches:
                            cmd = f"{cmd_prefix} -N {n} -B {b} --mode grasp --object {obj}"
                            desc = f"{sim_name} Grasp ({obj}) - N={n}, B={b}"
                            per_env, total = run_test(cmd, desc)
                            key = f"{sim_key}_grasp_{obj}_n{n}_b{b}"
                            results[key] = {"per_env": per_env, "total": total, "n": n, "b": b, "mode": "grasp", "object": obj, "sim": sim_key}

    # Print summary tables
    print("\n\n" + "="*120)
    print("SUMMARY TABLES")
    print("="*120)

    # Summary for random mode
    if "random" in args.modes:
        for b in args.batches:
            print(f"\n{'='*120}")
            print(f"RANDOM MODE - Batch Size B={b}")
            print(f"{'='*120}")
            print(f"{'Simulator':<20} {'N=1 (FPS)':<25} {'N=5 (FPS)':<25} {'N=10 (FPS)':<25}")
            print("-"*120)

            for sim_key in args.simulators:
                sim_name = simulator_configs[sim_key][1]
                n1 = results.get(f"{sim_key}_random_n1_b{b}", {}).get("per_env")
                n5 = results.get(f"{sim_key}_random_n5_b{b}", {}).get("per_env")
                n10 = results.get(f"{sim_key}_random_n10_b{b}", {}).get("per_env")

                n1_str = f"{n1:,.0f}" if n1 else "FAILED"
                n5_str = f"{n5:,.0f}" if n5 else "FAILED"
                n10_str = f"{n10:,.0f}" if n10 else "FAILED"

                print(f"{sim_name:<20} {n1_str:<25} {n5_str:<25} {n10_str:<25}")

    # Summary for grasp mode
    if "grasp" in args.modes:
        for obj in args.objects:
            for b in args.batches:
                print(f"\n{'='*120}")
                print(f"GRASP MODE ({obj.upper()}) - Batch Size B={b}")
                print(f"{'='*120}")
                print(f"{'Simulator':<20} {'N=1 (FPS)':<25} {'N=5 (FPS)':<25} {'N=10 (FPS)':<25}")
                print("-"*120)

                for sim_key in args.simulators:
                    sim_name = simulator_configs[sim_key][1]
                    n1 = results.get(f"{sim_key}_grasp_{obj}_n1_b{b}", {}).get("per_env")
                    n5 = results.get(f"{sim_key}_grasp_{obj}_n5_b{b}", {}).get("per_env")
                    n10 = results.get(f"{sim_key}_grasp_{obj}_n10_b{b}", {}).get("per_env")

                    n1_str = f"{n1:,.0f}" if n1 else "FAILED"
                    n5_str = f"{n5:,.0f}" if n5 else "FAILED"
                    n10_str = f"{n10:,.0f}" if n10 else "FAILED"

                    print(f"{sim_name:<20} {n1_str:<25} {n5_str:<25} {n10_str:<25}")

    # Print throughput comparison (total FPS)
    print("\n\n" + "="*120)
    print("THROUGHPUT COMPARISON (Total FPS)")
    print("="*120)

    if "random" in args.modes:
        for b in args.batches:
            print(f"\n{'='*60}")
            print(f"RANDOM MODE - Batch Size B={b}")
            print(f"{'='*60}")

            for n in args.robots:
                print(f"\n  N={n} robots:")
                for sim_key in args.simulators:
                    sim_name = simulator_configs[sim_key][1]
                    total = results.get(f"{sim_key}_random_n{n}_b{b}", {}).get("total")
                    if total:
                        print(f"    {sim_name:<15}: {total:>12,.0f} FPS")

    if "grasp" in args.modes:
        for obj in args.objects:
            for b in args.batches:
                print(f"\n{'='*60}")
                print(f"GRASP MODE ({obj.upper()}) - Batch Size B={b}")
                print(f"{'='*60}")

                for n in args.robots:
                    print(f"\n  N={n} robots:")
                    for sim_key in args.simulators:
                        sim_name = simulator_configs[sim_key][1]
                        total = results.get(f"{sim_key}_grasp_{obj}_n{n}_b{b}", {}).get("total")
                        if total:
                            print(f"    {sim_name:<15}: {total:>12,.0f} FPS")

    print("\n" + "="*120)
    print("BENCHMARK COMPLETE")
    print("="*120)


if __name__ == "__main__":
    main()
