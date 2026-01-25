#!/usr/bin/env python3
"""
Comprehensive benchmark runner for humanoid tests
Runs both motrix and mujoco humanoid benchmarks across various N and B configurations
"""

import argparse
import json
import os
import platform
import re
import subprocess
from datetime import datetime
from pathlib import Path


def get_cpu_model():
    """Get CPU model name, sanitized for directory names"""
    cpu_model = "Unknown"

    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
            model_match = re.search(r"model name\s*:\s*(.+)", cpuinfo)
            if model_match:
                cpu_model = model_match.group(1).strip()
        else:
            cpu_model = platform.processor() or "Unknown"
    except Exception:
        cpu_model = "Unknown"

    # Sanitize for directory name: replace special chars with underscores
    cpu_model = re.sub(r"[^\w\s-]", "_", cpu_model)  # Replace special chars
    cpu_model = re.sub(r"[\s]+", "_", cpu_model)  # Replace spaces with underscores
    cpu_model = cpu_model.strip("_")

    return cpu_model if cpu_model else "Unknown"


def run_test(cmd, description):
    """Run a test command and extract FPS results using regex"""
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=1000
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


def save_result(output_dir, engine, n, b, t, per_env_fps, total_fps):
    """Save result to JSON file"""
    os.makedirs(output_dir, exist_ok=True)

    result = {
        "engine": engine,
        "N": n,
        "B": b,
        "T": t,
        "per_env_fps": per_env_fps,
        "total_fps": total_fps,
        "timestamp": datetime.now().isoformat(),
    }

    filename = f"{output_dir}/{engine}_N{n}_B{b}.json"
    with open(filename, "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive benchmark runner for humanoid tests"
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        default=["motrix", "mujoco"],
        choices=["motrix", "mujoco"],
        help="Engines to test",
    )
    parser.add_argument(
        "--robots", nargs="+", type=int, default=[1, 5, 10], help="Humanoid counts (N)"
    )
    parser.add_argument(
        "--batches", nargs="+", type=int, default=[1, 512, 1024], help="Batch sizes (B)"
    )
    parser.add_argument(
        "--output-dir",
        default="output/humanoid",
        help="Output directory for JSON results",
    )

    args = parser.parse_args()

    # Get CPU core count
    cpu_cores = os.cpu_count() or 12

    # Get CPU model for subdirectory
    cpu_model = get_cpu_model()
    output_dir = os.path.join(args.output_dir, cpu_model)

    print("""
╔══════════════════════════════════════════════════════════════╗
║              Humanoid Benchmark Suite                       ║
║         Motrix / MuJoCo with Multi-threading                 ║
╚══════════════════════════════════════════════════════════════╝
    """)
    print(f"CPU Model: {cpu_model}")
    print(f"CPU cores: {cpu_cores} (for MuJoCo B>1)")
    print(f"Engines: {', '.join(args.engines)}")
    print(f"n_humanoids: {args.robots}")
    print(f"n_envs: {args.batches}")
    print(f"Output directory: {output_dir}")

    results = {}
    engine_configs = {
        "motrix": ("uv run humanoid/motrix_humanoid.py", "Motrix"),
        "mujoco": ("uv run humanoid/mujoco_humanoid.py", "MuJoCo"),
    }

    # Run tests for all combinations
    for engine in args.engines:
        cmd_prefix, engine_name = engine_configs[engine]
        print(f"\n{'=' * 80}")
        print(f"ENGINE: {engine_name.upper()}")
        print(f"{'=' * 80}")

        for b in args.batches:
            # Determine N values for this B
            n_values = args.robots.copy()
            if b == 1:
                n_values.append(50)  # Add N=50 for single environment

            for n in n_values:
                # For mujoco, add threading when B > 1
                if engine == "mujoco" and b > 1:
                    cmd = f"{cmd_prefix} -N {n} -B {b} -T {cpu_cores}"
                    t = cpu_cores
                else:
                    cmd = f"{cmd_prefix} -N {n} -B {b}"
                    t = 1

                desc = f"{engine_name} - n_humanoids={n}, n_envs={b}"
                per_env, total = run_test(cmd, desc)

                key = f"{engine}_n{n}_b{b}"
                results[key] = {
                    "engine": engine,
                    "engine_name": engine_name,
                    "per_env": per_env,
                    "total": total,
                    "n": n,
                    "b": b,
                    "t": t,
                }

                # Save result to JSON
                if per_env is not None and total is not None:
                    save_result(output_dir, engine, n, b, t, per_env, total)

    # Print summary tables
    print("\n\n" + "=" * 120)
    print("SUMMARY TABLES")
    print("=" * 120)

    # Get all unique N values from results, sorted
    all_n_values = sorted(set(results[key].get("n") for key in results if "n" in results[key]))

    for b in args.batches:
        print(f"\n{'=' * 120}")
        print(f"n_envs={b}")
        print(f"{'=' * 120}")

        # Build dynamic header
        headers = [f"{'Engine':<15}"] + [f"n_humanoids={n} (FPS)".ljust(22) for n in all_n_values]
        print("".join(headers))
        print("-" * 120)

        for engine in args.engines:
            engine_name = engine_configs[engine][1]
            row_values = [f"{engine_name:<15}"]

            for n in all_n_values:
                fps = results.get(f"{engine}_n{n}_b{b}", {}).get("per_env")
                fps_str = f"{fps:,.0f}" if fps else "FAILED"
                row_values.append(f"{fps_str:<22}")

            print("".join(row_values))

    # Print throughput comparison (total FPS)
    print("\n\n" + "=" * 120)
    print("THROUGHPUT COMPARISON (Total FPS)")
    print("=" * 120)

    for b in args.batches:
        print(f"\n{'=' * 60}")
        print(f"n_envs={b}")
        print(f"{'=' * 60}")

        for n in args.robots:
            print(f"\n  n_humanoids={n}:")
            for engine in args.engines:
                engine_name = engine_configs[engine][1]
                total = results.get(f"{engine}_n{n}_b{b}", {}).get("total")
                if total:
                    print(f"    {engine_name:<15}: {total:>12,.0f} FPS")

    print("\n" + "=" * 120)
    print("BENCHMARK COMPLETE")
    print("=" * 120)
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
