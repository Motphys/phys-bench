#!/usr/bin/env python3
"""
Comprehensive benchmark comparing MuJoCo CPU Rollout, Motrixsim, and Genesis

Tests CPU parallel simulation performance across three physics simulators
using Franka Panda grasping scenario.
"""

import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_system_info():
    """Collect system information for benchmark report"""
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Get CPU info
    try:
        if platform.system() == "Linux":
            # Get CPU model
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
            model_match = re.search(r"model name\s*:\s*(.+)", cpuinfo)
            info["cpu_model"] = model_match.group(1).strip() if model_match else "Unknown"
            
            # Count physical and logical cores
            physical_ids = set(re.findall(r"physical id\s*:\s*(\d+)", cpuinfo))
            cores_per_socket = re.search(r"cpu cores\s*:\s*(\d+)", cpuinfo)
            info["physical_cores"] = len(physical_ids) * int(cores_per_socket.group(1)) if cores_per_socket else os.cpu_count()
            info["logical_cores"] = os.cpu_count()
        else:
            info["cpu_model"] = platform.processor() or "Unknown"
            info["physical_cores"] = os.cpu_count()
            info["logical_cores"] = os.cpu_count()
    except Exception:
        info["cpu_model"] = "Unknown"
        info["physical_cores"] = os.cpu_count()
        info["logical_cores"] = os.cpu_count()
    
    return info


def run_test(cmd, description, timeout=300):
    """Run a test command and extract FPS results"""
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
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
        print(f"✗ Timeout ({timeout}s)")
        return None, None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None, None


def generate_markdown_report(results, batch_sizes, thread_counts, system_info):
    """Generate markdown report with benchmark results"""
    lines = []
    
    # Title
    lines.append("# CPU Parallel Physics Simulation Benchmark")
    lines.append("")
    lines.append("**Comparison of MuJoCo Rollout, Motrixsim, and Genesis for CPU batched physics simulation**")
    lines.append("")
    
    # System Info
    lines.append("## Experimental Setup")
    lines.append("")
    lines.append("### Hardware Configuration")
    lines.append("")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| CPU | {system_info['cpu_model']} |")
    lines.append(f"| Physical Cores | {system_info['physical_cores']} |")
    lines.append(f"| Logical Cores | {system_info['logical_cores']} |")
    lines.append(f"| Operating System | {system_info['os']} |")
    lines.append(f"| Python Version | {system_info['python']} |")
    lines.append(f"| Timestamp | {system_info['date']} |")
    lines.append("")
    
    lines.append("### Benchmark Parameters")
    lines.append("")
    lines.append("- **Scenario**: Franka Panda robot arm grasping task")
    lines.append("- **Simulation Steps**: 500 steps per benchmark run")
    lines.append("- **Timestep**: 0.01s (100 Hz)")
    lines.append(f"- **Batch Sizes**: {', '.join(map(str, batch_sizes))}")
    lines.append(f"- **Thread Counts (MuJoCo)**: {', '.join(map(str, thread_counts))}")
    lines.append("")
    
    # Collect comparison data
    comparison_data = []
    for b in batch_sizes:
        best_mujoco = None
        best_threads = None
        for t in thread_counts:
            fps = results.get(f"mujoco_b{b}_t{t}", {}).get("total")
            if fps and (best_mujoco is None or fps > best_mujoco):
                best_mujoco = fps
                best_threads = t
        
        mx_fps = results.get(f"motrixsim_b{b}", {}).get("total")
        gs_fps = results.get(f"genesis_b{b}", {}).get("total")
        
        comparison_data.append({
            "batch": b,
            "mujoco": best_mujoco,
            "mujoco_threads": best_threads,
            "motrixsim": mx_fps,
            "genesis": gs_fps,
        })
    
    # =============================================================
    # PERFORMANCE COMPARISON
    # =============================================================
    lines.append("## Performance Comparison")
    lines.append("")
    lines.append("### Throughput (Total FPS)")
    lines.append("")
    lines.append("| Batch Size | MuJoCo | Threads | Motrixsim | Genesis |")
    lines.append("|:----------:|-------:|:-------:|----------:|--------:|")
    
    for data in comparison_data:
        b = data["batch"]
        mj = data["mujoco"]
        mx = data["motrixsim"]
        gs = data["genesis"]
        t = data["mujoco_threads"]
        
        mj_str = f"{mj:,.0f}" if mj else "-"
        mx_str = f"{mx:,.0f}" if mx else "-"
        gs_str = f"{gs:,.0f}" if gs else "-"
        t_str = str(t) if t else "-"
        
        lines.append(f"| {b} | {mj_str} | {t_str} | {mx_str} | {gs_str} |")
    
    lines.append("")
    
    # Performance ratios
    lines.append("### Performance Ratios")
    lines.append("")
    lines.append("| Batch Size | MuJoCo/Motrixsim | MuJoCo/Genesis | Motrixsim/Genesis |")
    lines.append("|:----------:|-----------------:|---------------:|------------------:|")
    
    for data in comparison_data:
        b = data["batch"]
        mj = data["mujoco"]
        mx = data["motrixsim"]
        gs = data["genesis"]
        
        mj_mx = f"{mj/mx:.2f}x" if (mj and mx) else "-"
        mj_gs = f"{mj/gs:.2f}x" if (mj and gs) else "-"
        mx_gs = f"{mx/gs:.2f}x" if (mx and gs) else "-"
        
        lines.append(f"| {b} | {mj_mx} | {mj_gs} | {mx_gs} |")
    
    lines.append("")
    
    # =============================================================
    # MUJOCO DETAILED RESULTS
    # =============================================================
    lines.append("## MuJoCo Rollout: Detailed Results")
    lines.append("")
    lines.append("### Per-Environment FPS by Thread Count")
    lines.append("")
    
    # Header
    header = "| Batch Size |"
    separator = "|:----------:|"
    for t in thread_counts:
        header += f" T={t} |"
        separator += "------:|"
    lines.append(header)
    lines.append(separator)
    
    # Data rows
    for b in batch_sizes:
        row = f"| {b} |"
        for t in thread_counts:
            fps = results.get(f"mujoco_b{b}_t{t}", {}).get("per_env")
            row += f" {fps:,.0f} |" if fps else " - |"
        lines.append(row)
    lines.append("")
    
    lines.append("### Total FPS by Thread Count")
    lines.append("")
    
    # Header
    header = "| Batch Size |"
    separator = "|:----------:|"
    for t in thread_counts:
        header += f" T={t} |"
        separator += "------:|"
    lines.append(header)
    lines.append(separator)
    
    # Data rows
    for b in batch_sizes:
        row = f"| {b} |"
        for t in thread_counts:
            fps = results.get(f"mujoco_b{b}_t{t}", {}).get("total")
            row += f" {fps:,.0f} |" if fps else " - |"
        lines.append(row)
    lines.append("")
    
    # =============================================================
    # MOTRIXSIM RESULTS
    # =============================================================
    lines.append("## Motrixsim: Results")
    lines.append("")
    lines.append("| Batch Size | Per-Env FPS | Total FPS |")
    lines.append("|:----------:|------------:|----------:|")
    for b in batch_sizes:
        per_env = results.get(f"motrixsim_b{b}", {}).get("per_env")
        total = results.get(f"motrixsim_b{b}", {}).get("total")
        per_env_str = f"{per_env:,.0f}" if per_env else "-"
        total_str = f"{total:,.0f}" if total else "-"
        lines.append(f"| {b} | {per_env_str} | {total_str} |")
    lines.append("")
    
    # =============================================================
    # GENESIS RESULTS
    # =============================================================
    lines.append("## Genesis: Results")
    lines.append("")
    lines.append("| Batch Size | Per-Env FPS | Total FPS |")
    lines.append("|:----------:|------------:|----------:|")
    for b in batch_sizes:
        per_env = results.get(f"genesis_b{b}", {}).get("per_env")
        total = results.get(f"genesis_b{b}", {}).get("total")
        per_env_str = f"{per_env:,.0f}" if per_env else "-"
        total_str = f"{total:,.0f}" if total else "-"
        lines.append(f"| {b} | {per_env_str} | {total_str} |")
    lines.append("")
    
    # =============================================================
    # THREAD SCALING ANALYSIS
    # =============================================================
    lines.append("## Thread Scaling Analysis (MuJoCo)")
    lines.append("")
    lines.append("Speedup is calculated relative to single-threaded performance (T=1).")
    lines.append("Parallel efficiency is defined as: Efficiency = Speedup / Thread_Count × 100%")
    lines.append("")
    lines.append("| Batch Size | Threads | Total FPS | Speedup | Parallel Efficiency |")
    lines.append("|:----------:|--------:|----------:|--------:|--------------------:|")
    
    for b in batch_sizes:
        base_fps = results.get(f"mujoco_b{b}_t1", {}).get("total")
        if base_fps:
            for t in thread_counts:
                fps = results.get(f"mujoco_b{b}_t{t}", {}).get("total")
                if fps:
                    speedup = fps / base_fps
                    efficiency = speedup / t * 100
                    lines.append(f"| {b} | {t} | {fps:,.0f} | {speedup:.2f}x | {efficiency:.1f}% |")
    lines.append("")
    
    return "\n".join(lines)


def main():
    # Collect system info first
    system_info = get_system_info()
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║        CPU Parallel Physics Simulation Benchmark             ║
║          MuJoCo vs Motrixsim vs Genesis (CPU)                ║
║                 Franka Panda Grasp Scenario                  ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    print(f"CPU: {system_info['cpu_model']}")
    print(f"Cores: {system_info['physical_cores']} physical, {system_info['logical_cores']} logical")
    print("")

    results = {}

    batch_sizes = [1, 64, 512, 1024]
    thread_counts = [1, 4, 8, 16, 32]

    # MuJoCo CPU Rollout tests
    print("\n" + "=" * 60)
    print("MUJOCO CPU ROLLOUT")
    print("=" * 60)
    for b in batch_sizes:
        for t in thread_counts:
            cmd = f"uv run cpu-bench/test_mujoco_rollout.py -B {b} -T {t}"
            desc = f"MuJoCo Rollout - B={b} envs, T={t} threads"
            per_env, total = run_test(cmd, desc)
            results[f"mujoco_b{b}_t{t}"] = {
                "per_env": per_env,
                "total": total,
                "batch": b,
                "threads": t,
            }

    # Motrixsim CPU tests
    print("\n" + "=" * 60)
    print("MOTRIXSIM CPU")
    print("=" * 60)
    for b in batch_sizes:
        cmd = f"uv run cpu-bench/test_motrixsim.py -B {b}"
        desc = f"Motrixsim CPU - B={b} envs"
        per_env, total = run_test(cmd, desc)
        results[f"motrixsim_b{b}"] = {
            "per_env": per_env,
            "total": total,
            "batch": b,
        }

    # Genesis CPU tests
    print("\n" + "=" * 60)
    print("GENESIS CPU")
    print("=" * 60)
    for b in batch_sizes:
        cmd = f"uv run cpu-bench/test_genesis.py -B {b}"
        desc = f"Genesis CPU - B={b} envs"
        per_env, total = run_test(cmd, desc)
        results[f"genesis_b{b}"] = {
            "per_env": per_env,
            "total": total,
            "batch": b,
        }

    # Print summary tables (console output)
    print("\n\n" + "=" * 100)
    print("SUMMARY: MuJoCo CPU Rollout (per-env FPS)")
    print("=" * 100)
    
    # Header
    header = f"{'Batch Size':<15}"
    for t in thread_counts:
        header += f"{'T=' + str(t):<15}"
    print(header)
    print("-" * 100)

    # Data rows
    for b in batch_sizes:
        row = f"B={b:<12}"
        for t in thread_counts:
            fps = results.get(f"mujoco_b{b}_t{t}", {}).get("per_env")
            row += f"{fps:>12,.0f}   " if fps else f"{'FAILED':<15}"
        print(row)

    # Motrixsim summary
    print("\n" + "=" * 100)
    print("SUMMARY: Motrixsim CPU (per-env FPS)")
    print("=" * 100)
    print(f"{'Batch Size':<15}{'FPS':<15}")
    print("-" * 30)
    for b in batch_sizes:
        fps = results.get(f"motrixsim_b{b}", {}).get("per_env")
        fps_str = f"{fps:,.0f}" if fps else "FAILED"
        print(f"B={b:<12}{fps_str:<15}")

    # Genesis summary
    print("\n" + "=" * 100)
    print("SUMMARY: Genesis CPU (per-env FPS)")
    print("=" * 100)
    print(f"{'Batch Size':<15}{'FPS':<15}")
    print("-" * 30)
    for b in batch_sizes:
        fps = results.get(f"genesis_b{b}", {}).get("per_env")
        fps_str = f"{fps:,.0f}" if fps else "FAILED"
        print(f"B={b:<12}{fps_str:<15}")

    # Generate and save markdown report
    print("\n" + "=" * 100)
    print("GENERATING MARKDOWN REPORT")
    print("=" * 100)
    
    markdown = generate_markdown_report(results, batch_sizes, thread_counts, system_info)
    
    # Save to file
    output_dir = Path("output/cpu-bench")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / "benchmark_report.md"
    with open(report_path, "w") as f:
        f.write(markdown)
    
    print(f"\n✓ Markdown report saved to: {report_path}")
    
    # Also print markdown to console
    print("\n" + "=" * 100)
    print("MARKDOWN REPORT")
    print("=" * 100)
    print(markdown)

    print("\n" + "=" * 100)
    print("BENCHMARK COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
