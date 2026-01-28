"""Shared utilities for benchmark output and statistics."""

import json
from pathlib import Path
from typing import List, Dict, Optional


def ensure_output_directory() -> Path:
    """Create output directory if it doesn't exist."""
    output_dir = Path("output/bench")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def generate_result_filename(
    sim_key: str,
    mode: str,
    n: int,
    b: int,
    object_name: Optional[str] = None,
    clutter: bool = False,
    release: bool = False,
) -> str:
    """Generate standardized filename for benchmark result.

    Args:
        sim_key: Simulator identifier (genesis, motrixsim, etc.)
        mode: Test mode (random, grasp)
        n: Number of robots
        b: Batch size
        object_name: Object name for grasp mode (ball, cube, bottle)
        clutter: Whether clutter flag was enabled
        release: Whether release/shake flag was enabled

    Returns:
        Filename like: genesis_random_n5_b64_cluttertrue.json
                  or: motrixsim_grasp_cube_n10_b512.json
                  or: genesis_grasp_ball_n1_b64_release.json
    """
    if mode == "grasp" and object_name:
        release_str = "_release" if release else ""
        filename = f"{sim_key}_{mode}_{object_name}_n{n}_b{b}{release_str}.json"
    else:
        clutter_str = f"_clutter{str(clutter).lower()}" if clutter else ""
        filename = f"{sim_key}_{mode}_n{n}_b{b}{clutter_str}.json"
    return filename


def save_benchmark_result(
    output_dir: Path,
    sim_key: str,
    sim_name: str,
    mode: str,
    n: int,
    b: int,
    per_env_fps: Optional[float],
    total_fps: Optional[float],
    object_name: Optional[str] = None,
    clutter: bool = False,
    release: bool = False,
) -> None:
    """Save individual benchmark result to JSON.

    Args:
        output_dir: Directory to save results
        sim_key: Simulator key (genesis, motrixsim, etc.)
        sim_name: Simulator display name (Genesis, Motrixsim, etc.)
        mode: Test mode (random, grasp)
        n: Number of robots
        b: Batch size
        per_env_fps: Per-environment FPS (None if failed)
        total_fps: Total FPS (None if failed)
        object_name: Object for grasp mode
        clutter: Whether clutter was enabled
        release: Whether release/shake was enabled
    """
    from datetime import datetime

    filename = generate_result_filename(sim_key, mode, n, b, object_name, clutter, release)
    filepath = output_dir / filename

    result = {
        "simulator": sim_key,
        "simulator_name": sim_name,
        "mode": mode,
        "n_robots": n,
        "batch_size": b,
        "per_env_fps": per_env_fps,
        "total_fps": total_fps,
        "object": object_name,
        "clutter": clutter,
        "release": release,
        "status": "success" if per_env_fps is not None else "failed",
        "timestamp": datetime.now().isoformat(),
    }

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)


def load_benchmark_results(output_dir: Path = None) -> List[Dict]:
    """Load all JSON benchmark results from directory.

    Args:
        output_dir: Directory containing JSON files. Defaults to "output/bench".

    Returns:
        List of result dictionaries with benchmark data
    """
    if output_dir is None:
        output_dir = ensure_output_directory()

    if not output_dir.exists():
        return []

    results = []
    for json_file in sorted(output_dir.glob("*.json")):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            results.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
            continue

    return results


def generate_summary_stats(results: List[Dict]) -> Dict:
    """Calculate summary statistics grouped by various dimensions.

    Args:
        results: List of benchmark result dictionaries

    Returns:
        Dictionary with aggregated statistics
    """
    if not results:
        return {
            "total": 0,
            "success": 0,
            "failed": 0,
            "by_simulator": {},
            "by_mode": {},
            "by_config": {},
        }

    stats = {
        "total": len(results),
        "success": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "by_simulator": {},
        "by_mode": {},
        "by_config": {},
    }

    # Group by simulator
    for result in results:
        sim = result["simulator"]
        if sim not in stats["by_simulator"]:
            stats["by_simulator"][sim] = {
                "total": 0,
                "success": 0,
                "fps_values": [],
                "avg_fps": 0,
                "max_fps": 0,
                "min_fps": float('inf'),
            }

        stats["by_simulator"][sim]["total"] += 1
        if result["status"] == "success":
            stats["by_simulator"][sim]["success"] += 1
            fps = result["total_fps"]
            stats["by_simulator"][sim]["fps_values"].append(fps)

    # Calculate averages for simulators
    for sim, sim_stats in stats["by_simulator"].items():
        if sim_stats["fps_values"]:
            sim_stats["avg_fps"] = sum(sim_stats["fps_values"]) / len(sim_stats["fps_values"])
            sim_stats["max_fps"] = max(sim_stats["fps_values"])
            sim_stats["min_fps"] = min(sim_stats["fps_values"])
        del sim_stats["fps_values"]  # Remove raw values from output

    # Group by mode
    for result in results:
        mode = result["mode"]
        if mode not in stats["by_mode"]:
            stats["by_mode"][mode] = {
                "total": 0,
                "success": 0,
                "avg_fps": 0,
                "fps_values": [],
            }

        stats["by_mode"][mode]["total"] += 1
        if result["status"] == "success":
            stats["by_mode"][mode]["success"] += 1
            stats["by_mode"][mode]["fps_values"].append(result["total_fps"])

    # Calculate mode averages
    for mode, mode_stats in stats["by_mode"].items():
        if mode_stats["fps_values"]:
            mode_stats["avg_fps"] = sum(mode_stats["fps_values"]) / len(mode_stats["fps_values"])
        del mode_stats["fps_values"]

    # Group by configuration (N x B)
    for result in results:
        config_key = f"n{result['n_robots']}_b{result['batch_size']}"
        if config_key not in stats["by_config"]:
            stats["by_config"][config_key] = {
                "total": 0,
                "success": 0,
                "n_robots": result['n_robots'],
                "batch_size": result['batch_size'],
            }

        stats["by_config"][config_key]["total"] += 1
        if result["status"] == "success":
            stats["by_config"][config_key]["success"] += 1

    return stats


def group_results_by_mode(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Group results by mode (random, grasp).

    Args:
        results: List of benchmark results

    Returns:
        Dictionary mapping mode to list of results
    """
    grouped = {}
    for result in results:
        mode = result["mode"]
        if mode not in grouped:
            grouped[mode] = []
        grouped[mode].append(result)
    return grouped


def group_results_by_config(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Group results by configuration (N, B).

    Args:
        results: List of benchmark results

    Returns:
        Dictionary mapping config key (n{N}_b{B}) to list of results
    """
    grouped = {}
    for result in results:
        config_key = f"n{result['n_robots']}_b{result['batch_size']}"
        if config_key not in grouped:
            grouped[config_key] = []
        grouped[config_key].append(result)
    return grouped


def get_unique_values(results: List[Dict]) -> Dict[str, List]:
    """Extract unique values for various dimensions.

    Args:
        results: List of benchmark results

    Returns:
        Dictionary with unique values for simulators, modes, n_robots, batch_sizes, objects
    """
    return {
        "simulators": sorted(set(r["simulator"] for r in results)),
        "modes": sorted(set(r["mode"] for r in results)),
        "n_robots": sorted(set(r["n_robots"] for r in results)),
        "batch_sizes": sorted(set(r["batch_size"] for r in results)),
        "objects": sorted(set(r.get("object") for r in results if r.get("object"))),
    }
