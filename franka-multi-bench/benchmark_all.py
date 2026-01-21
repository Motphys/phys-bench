#!/usr/bin/env python3
"""
Comprehensive benchmark comparing Genesis vs Motrixsim
9 DOF (with fingers)
N = 1, 5, 10 robots
"""

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
            cmd, shell=True, capture_output=True, text=True, timeout=120
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
        print(f"✗ Timeout (120s)")
        return None, None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None, None


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║         Franka Multi-Robot Performance Benchmark            ║
║                   Genesis vs Motrixsim                       ║
║                    9 DOF (with fingers)                      ║
║           Testing: N robots x B batch_size envs              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    results = {}

    robot_counts = [1, 5, 10]
    batch_sizes = [1, 64, 512, 1024]

    # Genesis 9 DOF (with fingers)
    print("\n" + "=" * 60)
    print("GENESIS 9 DOF (WITH FINGERS)")
    print("=" * 60)
    for n in robot_counts:
        for b in batch_sizes:
            cmd = f"uv run franka-multi-bench/test_genesis_9dof_with_fingers.py -N {n} -B {b}"
            desc = f"Genesis 9 DOF - N={n} robots, B={b} envs"
            per_env, total = run_test(cmd, desc)
            results[f"genesis_9dof_n{n}_b{b}"] = {"per_env": per_env, "total": total, "n": n, "b": b}

    # Motrixsim 9 DOF (with fingers)
    print("\n" + "=" * 60)
    print("MOTRIXSIM 9 DOF (WITH FINGERS)")
    print("=" * 60)
    for n in robot_counts:
        for b in batch_sizes:
            cmd = f"uv run franka-multi-bench/test_motrixsim_9dof.py -N {n} -B {b}"
            desc = f"Motrixsim 9 DOF - N={n} robots, B={b} envs"
            per_env, total = run_test(cmd, desc)
            results[f"motrixsim_9dof_n{n}_b{b}"] = {"per_env": per_env, "total": total, "n": n, "b": b}

    # Print summary tables for each batch size
    configs = [
        ("Genesis 9 DOF (with fingers)", "genesis_9dof"),
        ("Motrixsim 9 DOF (with fingers)", "motrixsim_9dof"),
    ]

    for b in batch_sizes:
        print("\n\n" + "=" * 100)
        print(f"SUMMARY TABLE - Batch Size B={b} envs")
        print("=" * 100)
        print(f"{'Configuration':<35} {'N=1 (FPS)':<20} {'N=5 (FPS)':<20} {'N=10 (FPS)':<20}")
        print("-" * 100)

        for config_name, config_key in configs:
            n1 = results.get(f"{config_key}_n1_b{b}", {}).get("per_env")
            n5 = results.get(f"{config_key}_n5_b{b}", {}).get("per_env")
            n10 = results.get(f"{config_key}_n10_b{b}", {}).get("per_env")

            n1_str = f"{n1:,.0f}" if n1 else "FAILED"
            n5_str = f"{n5:,.0f}" if n5 else "FAILED"
            n10_str = f"{n10:,.0f}" if n10 else "FAILED"

            print(f"{config_name:<35} {n1_str:<20} {n5_str:<20} {n10_str:<20}")

        print("=" * 100)

    # Print throughput comparison (total FPS)
    print("\n\n" + "=" * 100)
    print("THROUGHPUT COMPARISON (Total FPS = per_env_fps * batch_size)")
    print("=" * 100)

    for b in batch_sizes:
        print(f"\n{'='*50}")
        print(f"Batch Size B={b}")
        print(f"{'='*50}")

        for n in robot_counts:
            print(f"\n  N={n} robots:")

            g9 = results.get(f"genesis_9dof_n{n}_b{b}", {}).get("total")
            m9 = results.get(f"motrixsim_9dof_n{n}_b{b}", {}).get("total")

            if g9:
                print(f"    Genesis 9 DOF    : {g9:>10,.0f} FPS")
            if m9:
                print(f"    Motrixsim 9 DOF  : {m9:>10,.0f} FPS")

    # Print performance ratios
    print("\n\n" + "=" * 100)
    print("PERFORMANCE RATIOS")
    print("=" * 100)

    for b in batch_sizes:
        print(f"\n{'='*50}")
        print(f"Batch Size B={b}")
        print(f"{'='*50}")

        for n in robot_counts:
            print(f"\n  N={n} robots:")

            g9 = results.get(f"genesis_9dof_n{n}_b{b}", {}).get("per_env")
            m9 = results.get(f"motrixsim_9dof_n{n}_b{b}", {}).get("per_env")

            if g9 and m9:
                ratio = g9 / m9
                winner = "Genesis" if ratio > 1 else "Motrixsim"
                print(f"    9 DOF: Genesis vs Motrixsim  : {winner:<10} {abs(ratio):>6.2f}x ({g9:>8,.0f} vs {m9:>8,.0f} FPS)")

    print("\n" + "=" * 100)
    print("BENCHMARK COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
