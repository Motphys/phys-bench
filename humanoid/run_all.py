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


def get_gpu_model():
    """Get GPU model name, sanitized for directory names"""
    gpu_model = "Unknown"

    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Get first GPU name if multiple GPUs
            gpu_model = result.stdout.strip().split("\n")[0].strip()
    except Exception:
        gpu_model = "Unknown"

    # Sanitize for directory name: replace special chars with underscores
    gpu_model = re.sub(r"[^\w\s-]", "_", gpu_model)  # Replace special chars
    gpu_model = re.sub(r"[\s]+", "_", gpu_model)  # Replace spaces with underscores
    gpu_model = gpu_model.strip("_")

    return gpu_model if gpu_model else "Unknown"


def detect_error_patterns(output):
    """Detect known error patterns in benchmark output.

    Args:
        output: Combined stdout+stderr from subprocess

    Returns:
        (error_code, error_message) tuple, or (None, None) if no errors detected
    """
    output_upper = output.upper()

    # CUDA memory errors
    if 'CUDA_ERROR_OUT_OF_MEMORY' in output_upper or 'OUT OF MEMORY' in output_upper:
        return 'CUDA_OOM', 'CUDA out of memory'

    # Genesis Jacobian errors
    if 'JACOBIAN SHAPE' in output_upper and 'TOO LARGE' in output_upper:
        return 'JACOBIAN_ERROR', 'Jacobian shape too large for this configuration'

    # General Genesis errors
    if 'GenesisException' in output or '[Genesis] [ERROR]' in output:
        return 'GENESIS_ERROR', 'Genesis engine error'

    # General CUDA errors
    if 'CUDA_ERROR' in output_upper:
        return 'CUDA_ERROR', 'CUDA error occurred'

    return None, None


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

        # Check if process exited with error (crash without JSON output)
        if result.returncode != 0:
            # Process crashed but didn't output JSON error
            # Try to detect error patterns
            error_code, error_message = detect_error_patterns(output)
            if not error_code:
                # Generic error if no pattern matched
                error_code = "CRASH_ERROR"
                error_message = f"Process exited with code {result.returncode}"

            print(f"✗ Error: {error_code}")
            print(f"  Message: {error_message}")
            return 0.0, 0.0, {"error_code": error_code, "error_message": error_message}

        # Check for JSON error output first
        json_match = re.search(r'\{[^{}]*"status"\s*:\s*"error"[^{}]*\}', output)
        if json_match:
            try:
                error_info = json.loads(json_match.group(0))
                print(f"✗ Error: {error_info.get('error_code', 'UNKNOWN')}")
                print(f"  Message: {error_info.get('error_message', '')}")
                return 0.0, 0.0, error_info
            except json.JSONDecodeError:
                pass

        # Check for known error patterns (crashes without JSON output)
        error_code, error_message = detect_error_patterns(output)
        if error_code:
            print(f"✗ Error: {error_code}")
            print(f"  Message: {error_message}")
            # Try to extract more details from output
            if 'out of memory' in output.lower():
                # Extract CUDA OOM details
                oom_match = re.search(r'CUDA_ERROR_OUT_OF_MEMORY:\s*([^\n]+)', output)
                if oom_match:
                    error_message = oom_match.group(1).strip()
            elif 'jacobian' in output.lower():
                # Extract Jacobian error details
                jacobian_match = re.search(r'Jacobian shape\s*\([^\)]+\)\s*is\s*([^\n]+)', output)
                if jacobian_match:
                    error_message = f"Jacobian shape {jacobian_match.group(1).strip()}"

            error_info = {"error_code": error_code, "error_message": error_message}
            return 0.0, 0.0, error_info

        # Extract FPS from output
        per_env_match = re.search(r"per env:\s*([\d,]+\.?\d*)\s*FPS", output)
        total_match = re.search(r"total\s*:\s*([\d,]+\.?\d*)\s*FPS", output)

        if per_env_match and total_match:
            per_env_fps = per_env_match.group(1).replace(",", "")
            total_fps = total_match.group(1).replace(",", "")
            print(f"✓ per env: {per_env_fps} FPS")
            print(f"✓ total  : {total_fps} FPS")
            return float(per_env_fps), float(total_fps), None
        else:
            print(f"✗ Failed to extract FPS")
            print(f"Output: {output[-500:]}")  # Print last 500 chars for debugging
            return None, None, None

    except subprocess.TimeoutExpired:
        print(f"✗ Timeout (1000s)")
        return None, None, {"error_code": "TIMEOUT", "error_message": "Test exceeded 1000s timeout"}
    except Exception as e:
        print(f"✗ Error: {e}")
        return None, None, {"error_code": "RUNNER_ERROR", "error_message": str(e)}


def save_result(output_dir, engine, n, b, t, per_env_fps, total_fps, error_info=None):
    """Save result to JSON file"""
    os.makedirs(output_dir, exist_ok=True)

    result = {
        "engine": engine,
        "N": n,
        "B": b,
        "T": t,
        "per_env_fps": per_env_fps if per_env_fps is not None else 0.0,
        "total_fps": total_fps if total_fps is not None else 0.0,
        "timestamp": datetime.now().isoformat(),
    }

    if error_info:
        result["status"] = "error"
        result["error_code"] = error_info.get("error_code", "UNKNOWN")
        result["error_message"] = error_info.get("error_message", "")
    else:
        result["status"] = "success"

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
        default=["motrix", "mujoco", "mujoco_warp", "genesis"],
        choices=["motrix", "mujoco", "mujoco_warp", "genesis"],
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

    # Get CPU and GPU models for subdirectories
    cpu_model = get_cpu_model()
    gpu_model = get_gpu_model()

    print("""
╔══════════════════════════════════════════════════════════════╗
║              Humanoid Benchmark Suite                       ║
║       Motrix / MuJoCo (CPU) / Genesis (GPU)                  ║
╚══════════════════════════════════════════════════════════════╝
    """)
    print(f"CPU Model: {cpu_model}")
    print(f"CPU cores: {cpu_cores} (for MuJoCo B>1)")
    print(f"GPU Model: {gpu_model} (for Genesis)")
    print(f"Engines: {', '.join(args.engines)}")
    print(f"n_humanoids: {args.robots}")
    print(f"n_envs: {args.batches}")
    cpu_output_dir = os.path.join(args.output_dir, cpu_model)
    gpu_output_dir = os.path.join(args.output_dir, gpu_model)
    print(f"Output directories:")
    print(f"  CPU engines: {cpu_output_dir}")
    print(f"  GPU engines: {gpu_output_dir}")

    results = {}
    engine_configs = {
        "motrix": ("uv run humanoid/motrix_humanoid.py", "Motrix"),
        "mujoco": ("uv run humanoid/mujoco_humanoid.py", "MuJoCo"),
        "mujoco_warp": ("uv run humanoid/mujoco_warp_humanoid.py", "MuJoCo-Warp"),
        "genesis": ("uv run humanoid/genesis_humanoid.py", "Genesis"),
    }

    # Define engine-specific batch size configurations
    engine_batch_configs = {
        "genesis": {
            1: [1, 512, 2048, 4096, 8192],  # N=1
            5: [1, 512, 1024],              # N=5
            10: [1, 128, 512],              # N=10
        },
        "mujoco_warp": {
            1: [1, 512, 2048, 4096, 8192],  # N=1
            5: [1, 512, 1024],              # N=5
            10: [1, 128, 512],              # N=10
        },
    }

    # Track all batch sizes used for summary tables
    all_batches_per_engine = {}

    # Run tests for all combinations
    for engine in args.engines:
        cmd_prefix, engine_name = engine_configs[engine]
        print(f"\n{'=' * 80}")
        print(f"ENGINE: {engine_name.upper()}")
        print(f"{'=' * 80}")

        # Determine batch sizes for this engine
        if engine in engine_batch_configs:
            # Use engine-specific batch sizes per N value
            for n in args.robots:
                batches_for_n = engine_batch_configs[engine].get(n, [1])
                for b in batches_for_n:
                    # Track batch sizes for summary tables
                    if engine not in all_batches_per_engine:
                        all_batches_per_engine[engine] = set()
                    all_batches_per_engine[engine].add(b)

                    # For mujoco, add threading when B > 1
                    if engine == "mujoco" and b > 1:
                        cmd = f"{cmd_prefix} -N {n} -B {b} -T {cpu_cores}"
                        t = cpu_cores
                    elif engine == "genesis":
                        # Genesis doesn't need thread parameter (GPU parallel)
                        cmd = f"{cmd_prefix} -N {n} -B {b}"
                        t = 1
                    else:
                        cmd = f"{cmd_prefix} -N {n} -B {b}"
                        t = 1

                    desc = f"{engine_name} - n_humanoids={n}, n_envs={b}"
                    per_env, total, error_info = run_test(cmd, desc)

                    key = f"{engine}_n{n}_b{b}"
                    results[key] = {
                        "engine": engine,
                        "engine_name": engine_name,
                        "per_env": per_env,
                        "total": total,
                        "n": n,
                        "b": b,
                        "t": t,
                        "error_info": error_info,
                    }

                    # Save result to JSON (always save, even on error)
                    # Use GPU model directory for Genesis and MuJoCo-Warp, CPU model for others
                    if engine in ["genesis", "mujoco_warp"]:
                        engine_output_dir = os.path.join(args.output_dir, gpu_model)
                    else:
                        engine_output_dir = os.path.join(args.output_dir, cpu_model)
                    save_result(engine_output_dir, engine, n, b, t, per_env, total, error_info)
        else:
            # Use default batch sizes for other engines (motrix, mujoco)
            for b in args.batches:
                # Determine N values for this B
                n_values = args.robots.copy()
                if b == 1:
                    n_values.append(50)  # Add N=50 for single environment

                # Track batch sizes for summary tables
                if engine not in all_batches_per_engine:
                    all_batches_per_engine[engine] = set()
                all_batches_per_engine[engine].add(b)

                for n in n_values:
                    # For mujoco, add threading when B > 1
                    if engine == "mujoco" and b > 1:
                        cmd = f"{cmd_prefix} -N {n} -B {b} -T {cpu_cores}"
                        t = cpu_cores
                    elif engine == "genesis":
                        # Genesis doesn't need thread parameter (GPU parallel)
                        cmd = f"{cmd_prefix} -N {n} -B {b}"
                        t = 1
                    else:
                        cmd = f"{cmd_prefix} -N {n} -B {b}"
                        t = 1

                    desc = f"{engine_name} - n_humanoids={n}, n_envs={b}"
                    per_env, total, error_info = run_test(cmd, desc)

                    key = f"{engine}_n{n}_b{b}"
                    results[key] = {
                        "engine": engine,
                        "engine_name": engine_name,
                        "per_env": per_env,
                        "total": total,
                        "n": n,
                        "b": b,
                        "t": t,
                        "error_info": error_info,
                    }

                    # Save result to JSON (always save, even on error)
                    # Use GPU model directory for Genesis and MuJoCo-Warp, CPU model for others
                    if engine in ["genesis", "mujoco_warp"]:
                        engine_output_dir = os.path.join(args.output_dir, gpu_model)
                    else:
                        engine_output_dir = os.path.join(args.output_dir, cpu_model)
                    save_result(engine_output_dir, engine, n, b, t, per_env, total, error_info)

    # Print summary tables
    print("\n\n" + "=" * 120)
    print("SUMMARY TABLES")
    print("=" * 120)

    # Get all unique N values from results, sorted
    all_n_values = sorted(set(results[key].get("n") for key in results if "n" in results[key]))

    # Collect all unique batch sizes across all engines
    all_batches = sorted(set(
        b for engine_batches in all_batches_per_engine.values() for b in engine_batches
    ))

    for b in all_batches:
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

    for b in all_batches:
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
    print(f"\nResults saved to:")
    print(f"  CPU engines: {cpu_output_dir}/")
    print(f"  GPU engines: {gpu_output_dir}/")

    # Generate HTML report
    print("\n" + "=" * 120)
    print("GENERATING HTML REPORT")
    print("=" * 120)

    try:
        from report_utils import generate_html_report
        generate_html_report()
        print(f"\n✓ HTML report generated: output/humanoid/comparison_report.html")
    except Exception as e:
        print(f"\n✗ Failed to generate HTML report: {e}")


if __name__ == "__main__":
    main()
