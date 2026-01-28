#!/usr/bin/env python3
"""
Comprehensive benchmark for unified bench/ scripts
Supports both random and grasp modes with multiple objects
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Import benchmark output utilities
try:
    from bench_output_utils import (
        ensure_output_directory,
        save_benchmark_result,
        generate_result_filename,
    )
    from bench_result_visualizer import generate_html_report
    REPORT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Report generation unavailable: {e}")
    REPORT_AVAILABLE = False


def run_test(cmd_list, description):
    """Run a test command and extract FPS results

    Args:
        cmd_list: List of command parts (e.g., ["uv", "run", "script.py", "-N", "1"])
        description: Human-readable description of the test
    """
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print(f"{'=' * 60}")
    print(f"Command: {' '.join(cmd_list)}")  # Debug: print actual command

    try:
        result = subprocess.run(
            cmd_list,
            capture_output=True,
            text=True,
            timeout=300,
        )

        output = result.stdout + result.stderr

        # Extract FPS from output (search in full output, not just last part)
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
            # Try to find FPS lines in output for debugging
            fps_lines = [line for line in output.split('\n') if 'FPS' in line]
            if fps_lines:
                print(f"Found FPS lines: {fps_lines}")
            else:
                print(f"Output (last 1000 chars): {output[-1000:]}")
            return None, None

    except subprocess.TimeoutExpired:
        print(f"✗ Timeout (300s)")
        return None, None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None, None


def check_and_load_existing_result(output_dir, sim_key, mode, n, b, object_name=None, clutter=False, release=False, force_fail=False):
    """Check if result JSON exists and is successful, load it if so.

    Args:
        force_fail: If True, rerun failed tests. If False, skip failed tests.

    Returns:
        (exists: bool, per_env_fps: float|None, total_fps: float|None)
    """
    if output_dir is None:
        return False, None, None

    filename = generate_result_filename(sim_key, mode, n, b, object_name, clutter, release)
    filepath = os.path.join(output_dir, filename)

    if not os.path.exists(filepath):
        return False, None, None

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Only skip if status is "success"
        if data.get("status") == "success":
            return True, data.get("per_env_fps"), data.get("total_fps")
        else:
            # JSON exists but failed
            # If force_fail is False, skip (don't rerun)
            # If force_fail is True, rerun
            if not force_fail:
                return True, None, None  # Skip failed test
            else:
                return False, None, None  # Rerun failed test
    except Exception:
        return False, None, None


def get_batch_sizes_for_robot_count(n_robots, mode, clutter=False):
    """
    Get appropriate batch sizes for a given robot count and mode.

    Rules:
    - grasp mode: N=1 [1,1024,4096,8192], N=5 [1,512,2048,4096], N=10 [1,256,1024,2048]
    - random mode (normal): N=1 [1,1024,4096,8192]
    - random mode (clutter): [1,16,32,64]
    """
    if mode == "random" and clutter:
        return [1, 16, 32, 64]
    elif mode == "random":
        if n_robots == 1:
            return [1, 1024, 4096, 8192]
        else:
            # For N>1 in random mode, use same as grasp
            if n_robots == 5:
                return [1, 512, 2048, 4096]
            elif n_robots == 10:
                return [1, 256, 1024, 2048]
    elif mode == "grasp":
        if n_robots == 1:
            return [1, 1024, 4096, 8192]
        elif n_robots == 5:
            return [1, 512, 2048, 4096]
        elif n_robots == 10:
            return [1, 256, 1024, 2048]

    # Fallback
    return [1, 64, 512, 1024]


def main():
    parser = argparse.ArgumentParser(description="Comprehensive benchmark runner for unified bench/ scripts")
    parser.add_argument("--modes", nargs="+", default=["random", "grasp"], choices=["random", "grasp"], help="Modes to test")
    parser.add_argument("--simulators", nargs="+", default=["genesis", "motrixsim", "isaacsim", "mujocowarp"],
                        choices=["genesis", "motrixsim", "isaacsim", "mujocowarp"], help="Simulators to test")
    parser.add_argument("--robots", nargs="+", type=int, default=[1, 5, 10], help="Robot counts")
    parser.add_argument("--batches", nargs="+", type=int, default=None, help="Batch sizes (overrides automatic scaling)")
    parser.add_argument("--objects", nargs="+", default=["ball", "cube", "bottle"], choices=["ball", "cube", "bottle"],
                        help="Objects for grasp mode")
    parser.add_argument("--clutter-only", action="store_true", default=False, help="Only run clutter tests for random mode")
    parser.add_argument("--no-clutter", action="store_true", default=False, help="Skip clutter tests for random mode")
    parser.add_argument("--release-only", action="store_true", default=False, help="Only run release/shake tests for grasp mode")
    parser.add_argument("--no-release", action="store_true", default=False, help="Skip release/shake tests for grasp mode")
    parser.add_argument("--no-report", action="store_true", help="Skip HTML report generation")
    parser.add_argument("--report-output", type=str, default="output/bench/comparison_report.html",
                        help="Output path for HTML report (default: output/bench/comparison_report.html)")
    parser.add_argument("--force", action="store_true",
                        help="Force run all benchmarks, ignore existing results")
    parser.add_argument("--force-fail", action="store_true",
                        help="Force rerun failed benchmarks (by default, failed results are skipped)")

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
        "genesis": {
            "name": "Genesis",
            "cmd_prefix": ["uv", "run", "bench/bench_genesis.py"],
        },
        "motrixsim": {
            "name": "Motrixsim",
            "cmd_prefix": ["uv", "run", "bench/bench_motrixsim.py"],
        },
        "isaacsim": {
            "name": "IsaacSim",
            "cmd_prefix": ["uv", "--project", "envs/isaacsim", "run", "python", "-u", "bench/bench_isaacsim.py"],
        },
        "mujocowarp": {
            "name": "MuJoCo-Warp",
            "cmd_prefix": ["uv", "run", "bench/bench_mujocowarp.py"],
        },
    }

    # Initialize output directory for saving results
    output_dir = ensure_output_directory() if REPORT_AVAILABLE else None

    # Run tests for all combinations
    for mode in args.modes:
        print(f"\n{'='*80}")
        print(f"MODE: {mode.upper()}")
        print(f"{'='*80}")

        if mode == "random":
            # Random mode: test all simulators x robots x batches x clutter variants
            # Determine which clutter variants to test
            clutter_variants = []
            if args.clutter_only:
                clutter_variants = [True]
            elif args.no_clutter:
                clutter_variants = [False]
            else:
                clutter_variants = [False, True]  # Default: test both

            for sim_key in args.simulators:
                config = simulator_configs[sim_key]
                sim_name = config["name"]
                cmd_prefix = config["cmd_prefix"]
                print(f"\n{'-'*60}")
                print(f"{sim_name} - Random Mode")
                print(f"{'-'*60}")

                for n in args.robots:
                    for clutter in clutter_variants:
                        # Get appropriate batch sizes for this robot count and clutter setting
                        if args.batches is not None:
                            batch_sizes = args.batches
                        else:
                            batch_sizes = get_batch_sizes_for_robot_count(n, "random", clutter)

                        print(f"  N={n}, clutter={clutter}, batch_sizes={batch_sizes}")

                        for b in batch_sizes:
                            # Check for existing result
                            if not args.force:
                                exists, cached_per_env, cached_total = check_and_load_existing_result(
                                    output_dir, sim_key, "random", n, b, clutter=clutter, force_fail=args.force_fail
                                )
                                if exists:
                                    status_label = "success" if cached_per_env else "failed"
                                    clutter_label = " (clutter)" if clutter else ""
                                    print(f"⏭️  Skipping {sim_name} Random N={n} B={b}{clutter_label} ({status_label})")
                                    key = f"{sim_key}_random_n{n}_b{b}"
                                    results[key] = {"per_env": cached_per_env, "total": cached_total, "n": n, "b": b, "mode": "random", "sim": sim_key}
                                    continue

                            # Build command as list
                            cmd = cmd_prefix + [f"-N", str(n), f"-B", str(b), "--mode", "random"]
                            if clutter:
                                cmd.append("--clutter")
                            desc = f"{sim_name} Random - N={n}, B={b}"
                            if clutter:
                                desc += " (clutter)"
                            per_env, total = run_test(cmd, desc)
                            key = f"{sim_key}_random_n{n}_b{b}"
                            results[key] = {"per_env": per_env, "total": total, "n": n, "b": b, "mode": "random", "sim": sim_key}

                            # Save result to JSON
                            if output_dir is not None:
                                save_benchmark_result(
                                    output_dir=output_dir,
                                    sim_key=sim_key,
                                    sim_name=sim_name,
                                    mode="random",
                                    n=n,
                                    b=b,
                                    per_env_fps=per_env,
                                    total_fps=total,
                                    clutter=clutter,
                                )

        elif mode == "grasp":
            # Grasp mode: test all simulators x robots x batches x objects x release variants
            # Determine which release variants to test
            release_variants = []
            if args.release_only:
                release_variants = [True]
            elif args.no_release:
                release_variants = [False]
            else:
                release_variants = [False, True]  # Default: test both

            for obj in args.objects:
                print(f"\n{'-'*60}")
                print(f"Object: {obj.upper()}")
                print(f"{'-'*60}")

                for sim_key in args.simulators:
                    config = simulator_configs[sim_key]
                    sim_name = config["name"]
                    cmd_prefix = config["cmd_prefix"]

                    for n in args.robots:
                        # Get appropriate batch sizes for this robot count
                        if args.batches is not None:
                            batch_sizes = args.batches
                        else:
                            batch_sizes = get_batch_sizes_for_robot_count(n, "grasp")

                        print(f"  N={n}, batch_sizes={batch_sizes}")

                        for b in batch_sizes:
                            for release in release_variants:
                                # Check for existing result
                                if not args.force:
                                    exists, cached_per_env, cached_total = check_and_load_existing_result(
                                        output_dir, sim_key, "grasp", n, b, object_name=obj, release=release, force_fail=args.force_fail
                                    )
                                    if exists:
                                        release_label = " (release)" if release else ""
                                        status_label = "success" if cached_per_env else "failed"
                                        print(f"⏭️  Skipping {sim_name} Grasp ({obj}) N={n} B={b}{release_label} ({status_label})")
                                        key = f"{sim_key}_grasp_{obj}_n{n}_b{b}"
                                        results[key] = {"per_env": cached_per_env, "total": cached_total, "n": n, "b": b, "mode": "grasp", "object": obj, "sim": sim_key}
                                        continue

                                # Build command as list
                                cmd = cmd_prefix + [f"-N", str(n), f"-B", str(b), "--mode", "grasp", "--object", obj]
                                if release:
                                    cmd.append("-r")
                                desc = f"{sim_name} Grasp ({obj}) - N={n}, B={b}"
                                if release:
                                    desc += " (release)"
                                per_env, total = run_test(cmd, desc)
                                key = f"{sim_key}_grasp_{obj}_n{n}_b{b}"
                                results[key] = {"per_env": per_env, "total": total, "n": n, "b": b, "mode": "grasp", "object": obj, "sim": sim_key}

                                # Save result to JSON
                                if output_dir is not None:
                                    save_benchmark_result(
                                        output_dir=output_dir,
                                        sim_key=sim_key,
                                        sim_name=sim_name,
                                        mode="grasp",
                                        n=n,
                                        b=b,
                                        per_env_fps=per_env,
                                        total_fps=total,
                                        object_name=obj,
                                        release=release,
                                    )

    # Print summary tables
    print("\n\n" + "="*120)
    print("SUMMARY TABLES")
    print("="*120)

    # Collect all batch sizes used
    all_batch_sizes = set()
    for key in results.keys():
        if results[key].get("b") is not None:
            all_batch_sizes.add(results[key]["b"])
    all_batch_sizes = sorted(all_batch_sizes)

    # Summary for random mode
    if "random" in args.modes:
        for b in all_batch_sizes:
            # Check if any results exist for this batch size in random mode
            has_results = any(f"_random_" in k and results[k].get("b") == b for k in results.keys())
            if not has_results:
                continue

            print(f"\n{'='*120}")
            print(f"RANDOM MODE - Batch Size B={b}")
            print(f"{'='*120}")
            print(f"{'Simulator':<20} {'N=1 (FPS)':<25} {'N=5 (FPS)':<25} {'N=10 (FPS)':<25}")
            print("-"*120)

            for sim_key in args.simulators:
                sim_name = simulator_configs[sim_key]["name"]
                n1_key = f"{sim_key}_random_n1_b{b}"
                n5_key = f"{sim_key}_random_n5_b{b}"
                n10_key = f"{sim_key}_random_n10_b{b}"

                n1 = results[n1_key].get("per_env") if n1_key in results else None
                n5 = results[n5_key].get("per_env") if n5_key in results else None
                n10 = results[n10_key].get("per_env") if n10_key in results else None

                # Distinguish between not tested (-) and failed (FAILED)
                n1_str = f"{n1:,.0f}" if n1 else ("FAILED" if n1_key in results else "-")
                n5_str = f"{n5:,.0f}" if n5 else ("FAILED" if n5_key in results else "-")
                n10_str = f"{n10:,.0f}" if n10 else ("FAILED" if n10_key in results else "-")

                print(f"{sim_name:<20} {n1_str:<25} {n5_str:<25} {n10_str:<25}")

    # Summary for grasp mode
    if "grasp" in args.modes:
        for obj in args.objects:
            for b in all_batch_sizes:
                # Check if any results exist for this batch size and object in grasp mode
                has_results = any(f"_grasp_{obj}_" in k and results[k].get("b") == b for k in results.keys())
                if not has_results:
                    continue

                print(f"\n{'='*120}")
                print(f"GRASP MODE ({obj.upper()}) - Batch Size B={b}")
                print(f"{'='*120}")
                print(f"{'Simulator':<20} {'N=1 (FPS)':<25} {'N=5 (FPS)':<25} {'N=10 (FPS)':<25}")
                print("-"*120)

                for sim_key in args.simulators:
                    sim_name = simulator_configs[sim_key]["name"]
                    n1_key = f"{sim_key}_grasp_{obj}_n1_b{b}"
                    n5_key = f"{sim_key}_grasp_{obj}_n5_b{b}"
                    n10_key = f"{sim_key}_grasp_{obj}_n10_b{b}"

                    n1 = results[n1_key].get("per_env") if n1_key in results else None
                    n5 = results[n5_key].get("per_env") if n5_key in results else None
                    n10 = results[n10_key].get("per_env") if n10_key in results else None

                    # Distinguish between not tested (-) and failed (FAILED)
                    n1_str = f"{n1:,.0f}" if n1 else ("FAILED" if n1_key in results else "-")
                    n5_str = f"{n5:,.0f}" if n5 else ("FAILED" if n5_key in results else "-")
                    n10_str = f"{n10:,.0f}" if n10 else ("FAILED" if n10_key in results else "-")

                    print(f"{sim_name:<20} {n1_str:<25} {n5_str:<25} {n10_str:<25}")

    # Print throughput comparison (total FPS)
    print("\n\n" + "="*120)
    print("THROUGHPUT COMPARISON (Total FPS)")
    print("="*120)

    if "random" in args.modes:
        for b in all_batch_sizes:
            # Check if any results exist for this batch size in random mode
            has_results = any(f"_random_" in k and results[k].get("b") == b for k in results.keys())
            if not has_results:
                continue

            print(f"\n{'='*60}")
            print(f"RANDOM MODE - Batch Size B={b}")
            print(f"{'='*60}")

            for n in args.robots:
                print(f"\n  N={n} robots:")
                for sim_key in args.simulators:
                    sim_name = simulator_configs[sim_key]["name"]
                    total = results.get(f"{sim_key}_random_n{n}_b{b}", {}).get("total")
                    if total:
                        print(f"    {sim_name:<15}: {total:>12,.0f} FPS")

    if "grasp" in args.modes:
        for obj in args.objects:
            for b in all_batch_sizes:
                # Check if any results exist for this batch size and object in grasp mode
                has_results = any(f"_grasp_{obj}_" in k and results[k].get("b") == b for k in results.keys())
                if not has_results:
                    continue

                print(f"\n{'='*60}")
                print(f"GRASP MODE ({obj.upper()}) - Batch Size B={b}")
                print(f"{'='*60}")

                for n in args.robots:
                    print(f"\n  N={n} robots:")
                    for sim_key in args.simulators:
                        sim_name = simulator_configs[sim_key]["name"]
                        total = results.get(f"{sim_key}_grasp_{obj}_n{n}_b{b}", {}).get("total")
                        if total:
                            print(f"    {sim_name:<15}: {total:>12,.0f} FPS")

    print("\n" + "="*120)
    print("BENCHMARK COMPLETE")
    print("="*120)

    # Generate HTML reports (4 separate reports)
    if REPORT_AVAILABLE and not args.no_report:
        print("\n" + "="*120)
        print("GENERATING HTML REPORTS")
        print("="*120)

        # Create html subdirectory
        html_dir = output_dir / "html"
        html_dir.mkdir(parents=True, exist_ok=True)

        # Delete old HTML files first
        print("🗑️  Removing old HTML reports...")
        html_files = list(html_dir.glob("*.html"))
        for html_file in html_files:
            try:
                html_file.unlink()
                print(f"   Deleted: {html_file.name}")
            except Exception as e:
                print(f"   Warning: Could not delete {html_file.name}: {e}")

        report_configs = [
            ("random_static", "Random Static", "random", False, False),
            ("random_clutter", "Random Clutter", "random", True, False),
            ("grasp_static", "Grasp Static", "grasp", False, False),
            ("grasp_shake", "Grasp Shake", "grasp", False, True),
        ]

        for report_name, title, mode, clutter, release in report_configs:
            try:
                report_path = html_dir / f"{report_name}.html"
                generate_html_report(
                    output_path=str(report_path),
                    results_dir=output_dir,
                    title=f"Benchmark Report - {title}",
                    filter_mode=mode,
                    filter_clutter=clutter,
                    filter_release=release,
                )
                print(f"✅ {title} report saved to: {report_path}")
            except Exception as e:
                print(f"⚠️  Failed to generate {title} report: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
