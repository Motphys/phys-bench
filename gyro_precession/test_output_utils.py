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

"""Shared utilities for test output across all gyro precession test scripts."""

import json
from pathlib import Path
from typing import Optional, List, Dict


def ensure_output_directory() -> Path:
    """Create output directory if it doesn't exist."""
    output_dir = Path("output/gyro_precession")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def generate_video_path(engine: str, vel: float, dt: float, output_dir: Path) -> str:
    """Generate standardized video path in output directory."""
    vel_str = f"vel{vel:.3f}".replace(".", "_")  # Use underscore for decimal
    dt_str = f"dt{dt:.3f}".replace(".", "_")  # Use underscore for decimal
    filename = f"{engine}_gyro_precession_{vel_str}_{dt_str}.mp4"
    return str(output_dir / filename)


def save_video(frames: List, video_path: str, fps: int = 30, quality: int = 8) -> None:
    """Save video frames using imageio."""
    import imageio

    print(f"save video: {video_path}, frames = {len(frames)}")
    imageio.mimwrite(video_path, frames, fps=fps, quality=quality)


def save_test_result(
    video_path: str,
    status: str,
    drop_time: Optional[float],
    output_dir: Path,
    engine: str,
    vel: float,
    dt: float,
) -> None:
    """Save test result to JSON file matching video filename."""
    from datetime import datetime

    result = {
        "video_path": video_path,
        "status": status,
        "drop_time": drop_time,
        "engine": engine,
        "vel": vel,
        "dt": dt,
        "timestamp": datetime.now().isoformat(),
    }
    json_path = str(Path(video_path).with_suffix(".json"))
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)


def parse_result_filename(filename: str) -> Optional[Dict[str, any]]:
    """Extract engine, vel, dt from filename.

    Example: "mujoco_gyro_precession_vel20_0_dt0_002.json" ->
        {"engine": "mujoco", "vel": 20.0, "dt": 0.002}

    Args:
        filename: JSON filename (e.g., "mujoco_gyro_precession_vel20_0_dt0_002.json")

    Returns:
        Dict with keys: engine, vel, dt, or None if pattern doesn't match
    """
    stem = Path(filename).stem  # Remove .json
    parts = stem.split("_")

    # Pattern: {engine}_gyro_precession_vel{value}_dt{value}
    if len(parts) >= 6 and "gyro_precession" in stem:
        try:
            # Find vel and dt parts
            vel_idx = next(i for i, p in enumerate(parts) if p.startswith("vel"))
            dt_idx = next(i for i, p in enumerate(parts) if p.startswith("dt"))

            vel_val = float(parts[vel_idx].replace("vel", "").replace("_", "."))
            dt_val = float(parts[dt_idx].replace("dt", "").replace("_", "."))
            return {
                "engine": parts[0],
                "vel": vel_val,
                "dt": dt_val,
            }
        except (ValueError, IndexError, StopIteration):
            pass

    return None


def load_test_results(output_dir: Path = None) -> List[Dict]:
    """Scan output directory and load all JSON test results.

    Args:
        output_dir: Directory containing JSON files. Defaults to "output/gyro_precession/".

    Returns:
        List of dicts with keys: engine, vel, video_path, status, drop_time,
                                video_exists, json_file, dt
    """
    if output_dir is None:
        output_dir = ensure_output_directory()

    if not output_dir.exists():
        return []

    results = []
    for json_file in output_dir.glob("*.json"):
        parsed = parse_result_filename(json_file.name)
        if not parsed:
            continue

        with open(json_file, "r") as f:
            data = json.load(f)

        # Check if video exists
        video_path = Path(data["video_path"])
        video_exists = video_path.exists()

        # Use values from JSON if available (new format), otherwise use parsed values
        vel = data.get("vel", parsed.get("vel", 0.0))
        dt = data.get("dt", parsed.get("dt", 0.002))

        results.append(
            {
                **parsed,
                "video_path": data["video_path"],
                "status": data["status"],
                "drop_time": data["drop_time"],
                "vel": vel,
                "dt": dt,
                "video_exists": video_exists,
                "json_file": str(json_file),
            }
        )

    return results


def generate_summary_stats(results: List[Dict]) -> Dict:
    """Calculate success/failure statistics by engine/vel/dt.

    Args:
        results: List of test result dicts from load_test_results()

    Returns:
        Dict with total, success, failure counts and breakdowns by category
    """
    stats = {
        "total": len(results),
        "success": sum(1 for r in results if r["status"] == "success"),
        "failure": sum(1 for r in results if r["status"] == "failure"),
        "by_engine": {},
        "by_vel": {},
        "by_dt": {},
    }

    for key in ["engine", "vel", "dt"]:
        for result in results:
            value = result[key]
            # Format vel and dt to strings
            if key == "vel":
                value = f"{value:.1f}"
            elif key == "dt":
                value = f"{value:.3f}"
            if value not in stats[f"by_{key}"]:
                stats[f"by_{key}"][value] = {"total": 0, "success": 0, "failure": 0}
            stats[f"by_{key}"][value]["total"] += 1
            if result["status"] == "success":
                stats[f"by_{key}"][value]["success"] += 1
            else:
                stats[f"by_{key}"][value]["failure"] += 1

    return stats


def group_results_by_vel_and_dt(
    results: List[Dict],
) -> Dict[float, Dict[float, List[Dict]]]:
    """Group results by velocity, then by dt, preserving engine info for comparison.

    Args:
        results: List of test result dicts from load_test_results()

    Returns:
        Nested dict structure:
        {
            20.0: {
                0.002: [result_for_engine1, result_for_engine2, ...],
                0.01: [...]
            },
            50.0: {...},
            ...
        }
    """
    grouped = {}
    for result in results:
        vel = result["vel"]
        dt = result["dt"]

        if vel not in grouped:
            grouped[vel] = {}
        if dt not in grouped[vel]:
            grouped[vel][dt] = []

        grouped[vel][dt].append(result)

    return grouped


def get_config_combinations(results: List[Dict]) -> List[tuple]:
    """Get all unique (vel, dt) combinations sorted for display.

    Args:
        results: List of test result dicts from load_test_results()

    Returns:
        Sorted list of tuples: [(20.0, 0.002), (20.0, 0.01), (50.0, 0.002), ...]
    """
    # Get unique combinations
    combinations = set((r["vel"], r["dt"]) for r in results)

    # Sort by velocity value, then by dt value
    return sorted(combinations, key=lambda x: (x[0], x[1]))
