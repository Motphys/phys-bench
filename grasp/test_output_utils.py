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

"""Shared utilities for test output across all grasp test scripts."""

import json
from pathlib import Path
from typing import Optional, List, Dict


def ensure_output_directory() -> Path:
    """Create output directory if it doesn't exist."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def generate_video_path(
    engine: str, object_name: str, shake: bool, mjx: bool, dt: float, output_dir: Path
) -> str:
    """Generate standardized video path in output directory."""
    task = "shake" if shake else "slip"
    mjx_str = f"mjx{str(mjx).lower()}"
    dt_str = f"dt{dt:.3f}".replace(".", "_")  # Use underscore for decimal
    filename = f"{engine}_grasp_{task}_{object_name}_{mjx_str}_{dt_str}.mp4"
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
    object_name: str,
    shake: bool,
    mjx: bool,
    dt: float,
) -> None:
    """Save test result to JSON file matching video filename."""
    from datetime import datetime

    task = "shake" if shake else "slip"
    result = {
        "video_path": video_path,
        "status": status,
        "drop_time": drop_time,
        "engine": engine,
        "object": object_name,
        "task": task,
        "mjx": mjx,
        "dt": dt,
        "timestamp": datetime.now().isoformat(),
    }
    json_path = str(Path(video_path).with_suffix(".json"))
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)


def parse_result_filename(filename: str) -> Optional[Dict[str, any]]:
    """Extract engine, task, object, mjx, dt from filename.

    Example: "mujoco_grasp_shake_cube_mjxfalse_dt0_002.json" ->
        {"engine": "mujoco", "task": "shake", "object": "cube", "mjx": False, "dt": 0.002}

    Args:
        filename: JSON filename (e.g., "mujoco_grasp_shake_cube_mjxfalse_dt0_002.json")

    Returns:
        Dict with keys: engine, task, object, mjx, dt, or None if pattern doesn't match
    """
    stem = Path(filename).stem  # Remove .json
    parts = stem.split("_")

    # New pattern: {engine}_grasp_{task}_{object}_mjx{true|false}_dt{value}
    if len(parts) >= 8 and parts[1] == "grasp":
        try:
            mjx_val = parts[5].replace("mjx", "") == "true"
            dt_val = float(parts[6].replace("dt", "").replace("_", "."))
            return {
                "engine": parts[0],
                "task": parts[2],
                "object": parts[3],
                "mjx": mjx_val,
                "dt": dt_val,
            }
        except (ValueError, IndexError):
            pass

    # Fallback to old pattern for backward compatibility
    if len(parts) >= 4 and parts[1] == "grasp":
        return {
            "engine": parts[0],
            "task": parts[2],
            "object": parts[3],
            "mjx": False,
            "dt": 0.002,
        }

    return None


def load_test_results(output_dir: Path = None) -> List[Dict]:
    """Scan output directory and load all JSON test results.

    Args:
        output_dir: Directory containing JSON files. Defaults to "output/".

    Returns:
        List of dicts with keys: engine, object, task, video_path, status, drop_time,
                                video_exists, json_file, mjx, dt
    """
    if output_dir is None:
        output_dir = Path("output")

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
        mjx = data.get("mjx", parsed.get("mjx", False))
        dt = data.get("dt", parsed.get("dt", 0.002))

        results.append(
            {
                **parsed,
                "video_path": data["video_path"],
                "status": data["status"],
                "drop_time": data["drop_time"],
                "mjx": mjx,
                "dt": dt,
                "video_exists": video_exists,
                "json_file": str(json_file),
            }
        )

    return results


def generate_summary_stats(results: List[Dict]) -> Dict:
    """Calculate success/failure statistics by engine/object/task/mjx/dt.

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
        "by_object": {},
        "by_task": {},
        "by_mjx": {},  # NEW
        "by_dt": {},  # NEW
    }

    for key in ["engine", "object", "task", "mjx", "dt"]:
        for result in results:
            value = result[key]
            # Convert boolean to string for mjx
            if key == "mjx":
                value = str(value).lower()
            # Format dt to string with 3 decimal places
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


def group_results_by_object_and_dt(results: List[Dict]) -> Dict[str, Dict[float, List[Dict]]]:
    """Group results by object, then by dt, preserving engine info for comparison.

    Args:
        results: List of test result dicts from load_test_results()

    Returns:
        Nested dict structure:
        {
            "ball": {
                0.002: [result_for_engine1, result_for_engine2, ...],
                0.01: [...]
            },
            "cube": {...},
            ...
        }
    """
    grouped = {}
    for result in results:
        obj = result["object"]
        dt = result["dt"]

        if obj not in grouped:
            grouped[obj] = {}
        if dt not in grouped[obj]:
            grouped[obj][dt] = []

        grouped[obj][dt].append(result)

    return grouped


def get_config_combinations(results: List[Dict]) -> List[tuple]:
    """Get all unique (object, dt) combinations sorted for display.

    Args:
        results: List of test result dicts from load_test_results()

    Returns:
        Sorted list of tuples: [('ball', 0.002), ('ball', 0.01), ('cube', 0.002), ...]
    """
    # Get unique combinations
    combinations = set((r["object"], r["dt"]) for r in results)

    # Sort by object name, then by dt value
    return sorted(combinations, key=lambda x: (x[0], x[1]))
