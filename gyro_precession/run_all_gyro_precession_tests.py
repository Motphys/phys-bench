#!/usr/bin/env python3
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

"""Batch runner for all gyro precession benchmark tests.

This script runs all gyro precession tests across multiple engines, velocities, and dt values,
then generates a comprehensive comparison report.

Example usage:
    # Run all tests
    python gyro_precession/run_all_gyro_precession_tests.py

    # Run specific engines
    python gyro_precession/run_all_gyro_precession_tests.py --engines mujoco,motrix
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Test configuration for each engine
TEST_CONFIGS = [
    {
        "engine": "mujoco",
        "script": "gyro_precession/gyro_precession_test_mujoco.py",
        "module_name": "grasp.gyro_precession_test_mujoco",
        "run_command": "uv",
    },
    # {
    #     "engine": "mujocowarp",
    #     "script": "grasp/gyro_precession_test_mujoco_warp.py",
    #     "module_name": "grasp.gyro_precession_test_mujoco_warp",
    #     "run_command": "uv",
    # },
    {
        "engine": "motrix",
        "script": "gyro_precession/gyro_precession_test_motrix.py",
        "module_name": "grasp.gyro_precession_test_motrix",
        "run_command": "uv",
    },
    {
        "engine": "genesis",
        "script": "gyro_precession/gyro_precession_test_genesis.py",
        "module_name": "grasp.gyro_precession_test_genesis",
        "run_command": "uv",
    },
    {
        "engine": "isaacsim",
        "script": "gyro_precession/gyro_precession_test_isaacsim.py",
        "module_name": "grasp.gyro_precession_test_isaacsim",
        "run_command": "uv --project envs/isaacsim run",
    },
]

# Default test parameters
DEFAULT_ENGINES = [cfg["engine"] for cfg in TEST_CONFIGS]
DEFAULT_VELOCITIES = [20, 50, 100]
DEFAULT_DT_VALUES = [0.0001, 0.002, 0.01]


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run all gyro precession benchmark tests and generate comparison report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              Run all tests
  %(prog)s --engines mujoco,motrix      Run specific engines
  %(prog)s --velocity 20,50             Run specific velocities
  %(prog)s --dt-values 0.002            Run specific dt values
  %(prog)s --no-report                  Skip report generation
        """,
    )
    parser.add_argument(
        "--engines",
        type=str,
        default=",".join(DEFAULT_ENGINES),
        help=f"Comma-separated list of engines to test (default: {','.join(DEFAULT_ENGINES)})",
    )
    parser.add_argument(
        "--velocity",
        type=str,
        default=",".join(map(str, DEFAULT_VELOCITIES)),
        help=f"Comma-separated list of velocities to test (default: {','.join(map(str, DEFAULT_VELOCITIES))})",
    )
    parser.add_argument(
        "--dt-values",
        type=str,
        default="0.002,0.01",
        help="Comma-separated list of dt values (default: 0.002,0.01)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating the comparison report after tests",
    )
    parser.add_argument(
        "--report-output",
        type=str,
        default="output/gyro_precession/comparison_report.html",
        help="Output path for the comparison report (default: output/gyro_precession/comparison_report.html)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel (experimental, may cause issues)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output from test scripts",
    )

    return parser.parse_args()


def get_engine_config(engine_name: str) -> Dict:
    """Get configuration for a specific engine."""
    for config in TEST_CONFIGS:
        if config["engine"] == engine_name:
            return config
    raise ValueError(f"Unknown engine: {engine_name}")


def run_single_test(
    engine: str, velocity: int, dt: float, verbose: bool = False
) -> Tuple[bool, str]:
    """Run a single test and return (success, output)."""
    config = get_engine_config(engine)
    script_path = Path(config["script"])

    if not script_path.exists():
        return False, f"Script not found: {script_path}"

    # Build command based on engine's run_command
    run_cmd = config.get("run_command", "uv run")

    if run_cmd == "uv":
        # For engines using main .venv: uv run script.py
        cmd = ["uv", "run", str(script_path)]
    elif "uv --project" in run_cmd:
        # For IsaacSim: uv --project envs/isaacsim/ run python -u script.py
        # Use -u for unbuffered output to ensure test results are captured
        parts = run_cmd.split()
        # Ensure project path ends with /
        if parts[1] == "--project" and not parts[2].endswith("/"):
            parts[2] += "/"
        cmd = parts + ["python", "-u", str(script_path)]
    else:
        # Fallback to direct python execution
        cmd = [sys.executable, str(script_path)]

    # Add test arguments
    cmd.extend(
        [
            f"--velocity={velocity}",
            f"--dt={dt}",
            "--record",  # Always record to generate output for report
        ]
    )

    if verbose:
        print(f"  Running: {' '.join(cmd)}")

    # Run test
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,  # 1 minute timeout per test (some tests may hang)
        )

        # Check for success indicators in output
        output = result.stdout + result.stderr
        # Check for "finished" in output OR if exit code is 0
        success = "finished" in output.lower() or (result.returncode == 0)

        return success, output
    except subprocess.TimeoutExpired as e:
        # Get partial output
        output = e.stdout.decode() if e.stdout else ""
        output += e.stderr.decode() if e.stderr else ""
        output += "\n[TIMEOUT] Test timed out after 60 seconds"
        return False, output
    except Exception as e:
        return False, f"Error running test: {e}"


def run_all_tests(
    engines: List[str],
    velocities: List[int],
    dt_values: List[float],
    verbose: bool = False,
    parallel: bool = False,
) -> Dict[str, Dict]:
    """Run all test combinations and collect results.

    Returns:
        Dict mapping test_key to {success, output, engine, velocity, dt}
    """
    results = {}
    # Calculate total tests
    total_tests = len(engines) * len(velocities) * len(dt_values)
    completed = 0

    print(f"\n{'=' * 70}")
    print(f"Running {total_tests} gyro precession benchmark tests")
    print(f"{'=' * 70}\n")

    for engine in engines:
        for velocity in velocities:
            for dt in dt_values:
                test_key = f"{engine}_velocity{velocity}_dt{dt:.3f}"
                completed += 1

                print(f"[{completed}/{total_tests}] {test_key}...", end=" ", flush=True)

                success, output = run_single_test(engine, velocity, dt, verbose)

                # Check if test timed out
                if "TIMEOUT" in output:
                    status = "TIMEOUT"
                else:
                    status = "PASSED" if success else "FAILED"
                print(f"{status}")

                if not success and verbose and "TIMEOUT" not in output:
                    # Print first few lines of error output
                    lines = output.strip().split("\n")[:5]
                    for line in lines:
                        if line.strip():
                            print(f"  {line}")

                results[test_key] = {
                    "success": success,
                    "output": output,
                    "engine": engine,
                    "velocity": velocity,
                    "dt": dt,
                }

    print(f"\n{'=' * 70}")
    print(f"Completed {completed} tests")
    print(f"{'=' * 70}\n")

    return results


def print_summary(results: Dict[str, Dict]):
    """Print summary statistics."""
    total = len(results)
    passed = sum(1 for r in results.values() if r["success"])
    timed_out = sum(1 for r in results.values() if "TIMEOUT" in r.get("output", ""))
    failed = total - passed - timed_out

    print(f"Summary:")
    print(f"  Total:    {total}")
    print(f"  Passed:   {passed} ({passed * 100 // total if total else 0}%)")
    print(f"  Failed:   {failed} ({failed * 100 // total if total else 0}%)")
    if timed_out > 0:
        print(f"  Timed out: {timed_out} ({timed_out * 100 // total if total else 0}%)")
        print(f"\n  Note: Some tests timed out - this may indicate a bug in the engine")
        print(f"        for specific velocities. See verbose output for details.")

    # Group by engine
    print(f"\nBy Engine:")
    for engine in set(r["engine"] for r in results.values()):
        engine_results = [r for r in results.values() if r["engine"] == engine]
        engine_passed = sum(1 for r in engine_results if r["success"])
        print(
            f"  {engine:12s}: {engine_passed}/{len(engine_results)} "
            f"({engine_passed * 100 // len(engine_results) if engine_results else 0}%)"
        )

    # Group by velocity
    print(f"\nBy Velocity:")
    for velocity in sorted(set(r["velocity"] for r in results.values())):
        velocity_results = [r for r in results.values() if r["velocity"] == velocity]
        velocity_passed = sum(1 for r in velocity_results if r["success"])
        print(
            f"  {velocity:3d}: {velocity_passed}/{len(velocity_results)} "
            f"({velocity_passed * 100 // len(velocity_results) if velocity_results else 0}%)"
        )

    # Group by dt
    print(f"\nBy DT:")
    for dt in sorted(set(r["dt"] for r in results.values())):
        dt_results = [r for r in results.values() if r["dt"] == dt]
        dt_passed = sum(1 for r in dt_results if r["success"])
        print(
            f"  {dt:.3f}: {dt_passed}/{len(dt_results)} "
            f"({dt_passed * 100 // len(dt_results) if dt_results else 0}%)"
        )


def generate_report(output_path: str):
    """Generate the comparison report."""
    # Import here to avoid issues if gyro_precession module is not in path
    sys.path.insert(0, str(Path(__file__).parent))
    from test_result_visualizer import generate_html_report

    print(f"\nGenerating comparison report...")
    generate_html_report(
        output_path=output_path,
        results_dir=Path("output/gyro_precession"),
        title="Gyro Precession Benchmark Comparison Report",
    )
    print(f"Report saved to: {output_path}")


def main():
    """Main entry point."""
    args = parse_arguments()

    # # Parse arguments
    # engines = [e.strip() for e in args.engines.split(",")]
    # velocities = [int(v.strip()) for v in args.velocity.split(",")]
    # dt_values = [float(dt.strip()) for dt in args.dt_values.split(",")]

    # # Validate engines
    # for engine in engines:
    #     if engine not in DEFAULT_ENGINES:
    #         print(f"Error: Unknown engine '{engine}'")
    #         print(f"Available engines: {', '.join(DEFAULT_ENGINES)}")
    #         sys.exit(1)

    # # Validate velocities
    # for velocity in velocities:
    #     if velocity <= 0:
    #         print(f"Error: Invalid velocity '{velocity}'. Must be positive.")
    #         sys.exit(1)

    # # Run all tests
    # results = run_all_tests(
    #     engines=engines,
    #     velocities=velocities,
    #     dt_values=dt_values,
    #     verbose=args.verbose,
    #     parallel=args.parallel,
    # )

    # # Print summary
    # print_summary(results)

    # Generate report
    if not args.no_report:
        generate_report(args.report_output)


if __name__ == "__main__":
    main()
