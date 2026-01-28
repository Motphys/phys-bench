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

"""Generate HTML visualization of humanoid benchmark results."""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime


def detect_cpu_model_dir(results_dir: Path) -> Optional[str]:
    """Find CPU model subdirectory automatically.

    Args:
        results_dir: Base directory containing CPU model subdirectories

    Returns:
        CPU model directory name, or None if no valid subdirectory found
    """
    if not results_dir.exists():
        return None

    # Find all subdirectories
    subdirs = [d for d in results_dir.iterdir() if d.is_dir()]

    if not subdirs:
        return None

    if len(subdirs) == 1:
        return subdirs[0].name

    # If multiple subdirectories, use the most recently modified
    subdirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return subdirs[0].name


def detect_all_cpu_model_dirs(results_dir: Path) -> List[str]:
    """Find all CPU model subdirectories.

    Args:
        results_dir: Base directory containing CPU model subdirectories

    Returns:
        List of CPU model directory names, sorted by modification time (most recent first)
    """
    if not results_dir.exists():
        return []

    # Find all subdirectories
    subdirs = [d for d in results_dir.iterdir() if d.is_dir()]

    # Sort by modification time (most recent first)
    subdirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return [d.name for d in subdirs]


def format_cpu_name(cpu_dir_name: str) -> str:
    """Convert CPU directory name to friendly display name.

    Args:
        cpu_dir_name: CPU directory name (e.g., "Intel_R__Core_TM__i5-10600KF_CPU___4_10GHz")

    Returns:
        Formatted CPU name (e.g., "Intel R Core TM i5-10600KF CPU 4 10GHz")
    """
    # Replace underscores with spaces and clean up excess whitespace
    formatted = cpu_dir_name.replace('_', ' ').replace('  ', ' ').strip()
    return formatted


def parse_result_filename(filename: str) -> Optional[Dict[str, any]]:
    """Extract engine, N, B from filename.

    Example: "motrix_N1_B512.json" -> {"engine": "motrix", "N": 1, "B": 512}

    Args:
        filename: JSON filename (e.g., "motrix_N1_B512.json")

    Returns:
        Dict with keys: engine, N, B, or None if pattern doesn't match
    """
    stem = Path(filename).stem  # Remove .json

    # Pattern: {engine}_N{N}_B{B}
    pattern = r"^(\w+)_N(\d+)_B(\d+)$"
    match = re.match(pattern, stem)

    if match:
        return {
            "engine": match.group(1),
            "N": int(match.group(2)),
            "B": int(match.group(3)),
        }

    return None


def load_json_results(results_dir: Path, cpu_model: Optional[str] = None) -> List[Dict]:
    """Load all JSON benchmark results from directory.

    Args:
        results_dir: Base directory containing CPU model subdirectories
        cpu_model: Specific CPU model subdirectory (auto-detected if None)

    Returns:
        List of result dicts with keys: engine, N, B, T, per_env_fps, total_fps, timestamp
    """
    # Determine CPU model directory
    if cpu_model is None:
        cpu_model = detect_cpu_model_dir(results_dir)

    if cpu_model is None:
        return []

    data_dir = results_dir / cpu_model
    if not data_dir.exists():
        return []

    results = []
    for json_file in data_dir.glob("*.json"):
        parsed = parse_result_filename(json_file.name)
        if not parsed:
            continue

        with open(json_file, "r") as f:
            data = json.load(f)

        # Merge parsed filename data with JSON content
        results.append({
            **parsed,
            "T": data.get("T", 1),
            "per_env_fps": data.get("per_env_fps", 0.0),
            "total_fps": data.get("total_fps", 0.0),
            "timestamp": data.get("timestamp", ""),
        })

    return results


def load_all_cpu_results(results_dir: Path) -> Dict[str, List[Dict]]:
    """Load all JSON benchmark results from all CPU model directories.

    Args:
        results_dir: Base directory containing CPU model subdirectories

    Returns:
        Dict mapping CPU model directory names to lists of result dicts
        Example: {"Intel_i5": [...], "AMD_Ryzen": [...]}
    """
    cpu_dirs = detect_all_cpu_model_dirs(results_dir)

    all_results = {}
    for cpu_dir_name in cpu_dirs:
        cpu_data_dir = results_dir / cpu_dir_name
        if not cpu_data_dir.exists():
            continue

        results = []
        for json_file in cpu_data_dir.glob("*.json"):
            parsed = parse_result_filename(json_file.name)
            if not parsed:
                continue

            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                # Merge parsed filename data with JSON content
                results.append({
                    **parsed,
                    "T": data.get("T", 1),
                    "per_env_fps": data.get("per_env_fps", 0.0),
                    "total_fps": data.get("total_fps", 0.0),
                    "timestamp": data.get("timestamp", ""),
                })
            except (json.JSONDecodeError, IOError) as e:
                # Skip corrupted files and continue
                print(f"Warning: Skipping corrupted file {json_file}: {e}")
                continue

        if results:
            all_results[cpu_dir_name] = results

    return all_results


def group_by_dimensions(results: List[Dict]) -> Dict[str, Dict[int, Dict[int, Dict]]]:
    """Group data by dimensions for chart rendering.

    Returns:
        Nested dict structure:
        {
            "by_N": {N: {B: {"motrix": fps, "mujoco": fps}}},
            "by_B": {B: {N: {"motrix": fps, "mujoco": fps}}}
        }
    """
    grouped = {
        "by_N": {},
        "by_B": {},
    }

    for result in results:
        engine = result["engine"]
        n = result["N"]
        b = result["B"]
        per_env_fps = result["per_env_fps"]
        total_fps = result["total_fps"]

        # Group by N
        if n not in grouped["by_N"]:
            grouped["by_N"][n] = {}
        if b not in grouped["by_N"][n]:
            grouped["by_N"][n][b] = {}
        grouped["by_N"][n][b][engine] = {
            "per_env_fps": per_env_fps,
            "total_fps": total_fps,
        }

        # Group by B
        if b not in grouped["by_B"]:
            grouped["by_B"][b] = {}
        if n not in grouped["by_B"][b]:
            grouped["by_B"][b][n] = {}
        grouped["by_B"][b][n][engine] = {
            "per_env_fps": per_env_fps,
            "total_fps": total_fps,
        }

    return grouped


def calculate_statistics(results: List[Dict]) -> Dict:
    """Compute best FPS, averages, speedup ratios.

    Args:
        results: List of result dicts from load_json_results()

    Returns:
        Dict with summary statistics
    """
    if not results:
        return {
            "total": 0,
            "best_per_env_fps": 0,
            "best_total_fps": 0,
            "by_engine": {},
            "by_N": {},
            "by_B": {},
        }

    stats = {
        "total": len(results),
        "best_per_env_fps": max(r["per_env_fps"] for r in results),
        "best_total_fps": max(r["total_fps"] for r in results),
        "by_engine": {},
        "by_N": {},
        "by_B": {},
    }

    # Group by engine
    for result in results:
        engine = result["engine"]
        if engine not in stats["by_engine"]:
            stats["by_engine"][engine] = {
                "count": 0,
                "avg_per_env_fps": 0,
                "avg_total_fps": 0,
                "best_per_env_fps": 0,
                "best_total_fps": 0,
            }
        e_stats = stats["by_engine"][engine]
        e_stats["count"] += 1
        e_stats["avg_per_env_fps"] += result["per_env_fps"]
        e_stats["avg_total_fps"] += result["total_fps"]
        e_stats["best_per_env_fps"] = max(e_stats["best_per_env_fps"], result["per_env_fps"])
        e_stats["best_total_fps"] = max(e_stats["best_total_fps"], result["total_fps"])

    # Calculate averages
    for engine in stats["by_engine"]:
        e_stats = stats["by_engine"][engine]
        e_stats["avg_per_env_fps"] /= e_stats["count"]
        e_stats["avg_total_fps"] /= e_stats["count"]

    # Group by N and B
    for result in results:
        n = result["N"]
        b = result["B"]

        if n not in stats["by_N"]:
            stats["by_N"][n] = {"count": 0, "avg_per_env_fps": 0, "avg_total_fps": 0}
        stats["by_N"][n]["count"] += 1
        stats["by_N"][n]["avg_per_env_fps"] += result["per_env_fps"]
        stats["by_N"][n]["avg_total_fps"] += result["total_fps"]

        if b not in stats["by_B"]:
            stats["by_B"][b] = {"count": 0, "avg_per_env_fps": 0, "avg_total_fps": 0}
        stats["by_B"][b]["count"] += 1
        stats["by_B"][b]["avg_per_env_fps"] += result["per_env_fps"]
        stats["by_B"][b]["avg_total_fps"] += result["total_fps"]

    # Calculate averages for N and B
    for n in stats["by_N"]:
        count = stats["by_N"][n]["count"]
        stats["by_N"][n]["avg_per_env_fps"] /= count
        stats["by_N"][n]["avg_total_fps"] /= count

    for b in stats["by_B"]:
        count = stats["by_B"][b]["count"]
        stats["by_B"][b]["avg_per_env_fps"] /= count
        stats["by_B"][b]["avg_total_fps"] /= count

    return stats


def _get_chart_config(chart_id: str, title: str, labels: List[str],
                      datasets: List[Dict], y_label: str = "FPS") -> str:
    """Generate Chart.js configuration as JavaScript data object.

    Args:
        chart_id: Canvas element ID
        title: Chart title
        labels: X-axis labels
        datasets: List of dataset dicts with 'label', 'data', 'backgroundColor'
        y_label: Y-axis label

    Returns:
        JavaScript code string for chart config (not wrapped in new Chart())
    """
    datasets_json = json.dumps(datasets)

    return f"""{{
        id: '{chart_id}',
        config: {{
            type: 'bar',
            data: {{
                labels: {json.dumps(labels)},
                datasets: {datasets_json}
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: '{title}',
                        font: {{ size: 16 }}
                    }},
                    legend: {{
                        position: 'top'
                    }}
                }},
                scales: {{
                    y: {{
                        type: 'linear',
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: '{y_label}'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Engine'
                        }}
                    }}
                }}
            }}
        }}
    }}"""


def _get_by_n_charts_html(grouped: Dict, chart_configs: list) -> str:
    """Generate HTML for charts grouped by humanoid count (N)."""
    html = '<section class="charts-section" id="by-n"><h2>Performance by Humanoid Count (N)</h2>'

    for n in sorted(grouped["by_N"].keys()):
        b_data = grouped["by_N"][n]
        if not b_data:
            continue

        labels = sorted(b_data.keys())
        motrix_data = []
        mujoco_data = []

        for b in labels:
            engines = b_data[b]
            motrix_data.append(engines.get("motrix", {}).get("total_fps", 0))
            mujoco_data.append(engines.get("mujoco", {}).get("total_fps", 0))

        chart_id = f"chart-n{n}"
        chart_config = _get_chart_config(
            chart_id=chart_id,
            title=f"N={n} Humanoids - Total FPS",
            labels=[f"B={b}" for b in labels],
            datasets=[
                {
                    "label": "Motrix",
                    "data": motrix_data,
                    "backgroundColor": "#3b82f6",
                },
                {
                    "label": "MuJoCo",
                    "data": mujoco_data,
                    "backgroundColor": "#ef4444",
                },
            ],
            y_label="FPS (total)"
        )
        chart_configs.append(chart_config)

        html += f"""
        <div class="chart-container">
            <canvas id="{chart_id}"></canvas>
        </div>
        """

    html += "</section>"
    return html


def _get_by_b_charts_html(grouped: Dict, chart_configs: list) -> str:
    """Generate HTML for charts grouped by batch size (B)."""
    html = '<section class="charts-section" id="by-b"><h2>Performance by Batch Size (B)</h2>'

    for b in sorted(grouped["by_B"].keys()):
        n_data = grouped["by_B"][b]
        if not n_data:
            continue

        labels = sorted(n_data.keys())
        motrix_data = []
        mujoco_data = []

        for n in labels:
            engines = n_data[n]
            motrix_data.append(engines.get("motrix", {}).get("total_fps", 0))
            mujoco_data.append(engines.get("mujoco", {}).get("total_fps", 0))

        chart_id = f"chart-b{b}"
        chart_config = _get_chart_config(
            chart_id=chart_id,
            title=f"B={b} Batch Size - Total FPS",
            labels=[f"N={n}" for n in labels],
            datasets=[
                {
                    "label": "Motrix",
                    "data": motrix_data,
                    "backgroundColor": "#3b82f6",
                },
                {
                    "label": "MuJoCo",
                    "data": mujoco_data,
                    "backgroundColor": "#ef4444",
                },
            ],
            y_label="FPS (total)"
        )
        chart_configs.append(chart_config)

        html += f"""
        <div class="chart-container">
            <canvas id="{chart_id}"></canvas>
        </div>
        """

    html += "</section>"
    return html


def _get_overview_dashboard_html(stats: Dict, results: List[Dict]) -> str:
    """Generate overview dashboard with summary cards."""
    if not results:
        return ""

    # Get unique engines
    engines = sorted(stats["by_engine"].keys())

    cards = []

    # Best performance card
    cards.append(f"""
        <div class="summary-card">
            <div class="summary-label">Best Total FPS</div>
            <div class="summary-value">{stats["best_total_fps"]:,.0f}</div>
        </div>
    """)

    # Engine-specific cards
    for engine in engines:
        e_stats = stats["by_engine"][engine]
        engine_name = engine.capitalize()
        cards.append(f"""
            <div class="summary-card">
                <div class="summary-label">{engine_name} Average (Total)</div>
                <div class="summary-value">{e_stats["avg_total_fps"]:,.0f} FPS</div>
                <div class="summary-sub">Best: {e_stats["best_total_fps"]:,.0f} FPS</div>
            </div>
        """)

    return f"""
    <section class="overview-dashboard">
        <h2>Overview</h2>
        <div class="summary-cards">
            {"".join(cards)}
        </div>
    </section>
    """


def _get_comparison_matrix_html(grouped: Dict) -> str:
    """Generate engine comparison matrix table."""
    html = '<section class="matrix-section" id="matrix"><h2>Engine Comparison Matrix (Total FPS)</h2>'

    # Get all unique N and B values
    n_values = sorted(grouped["by_N"].keys())
    b_values = sorted(set(b for n_data in grouped["by_N"].values() for b in n_data.keys()))

    if not n_values or not b_values:
        html += "<p>No data available</p>"
        html += "</section>"
        return html

    html += """
    <div class="matrix-table-wrapper">
        <table class="matrix-table">
            <thead>
                <tr>
                    <th>Engine \\ Batch Size</th>
    """

    for b in b_values:
        html += f'<th>B={b}</th>'

    html += "</tr></thead><tbody>"

    # Find all engines
    engines = set()
    for n_data in grouped["by_N"].values():
        for b_data in n_data.values():
            engines.update(b_data.keys())
    engines = sorted(engines)

    for engine in engines:
        html += f'<tr><td class="engine-header">{engine.capitalize()}</td>'

        for b in b_values:
            # Find the first N value that has data for this (engine, B) combination
            cell_value = "—"
            cell_class = "matrix-cell-missing"

            for n in n_values:
                if b in grouped["by_N"][n] and engine in grouped["by_N"][n][b]:
                    fps = grouped["by_N"][n][b][engine]["total_fps"]
                    cell_value = f"{fps:,.0f}"
                    cell_class = "matrix-cell-value"
                    break

            html += f'<td class="{cell_class}">{cell_value}</td>'

        html += "</tr>"

    html += "</tbody></table></div></section>"
    return html


def _get_detailed_table_html(results: List[Dict]) -> str:
    """Generate detailed data table with filter controls."""
    html = '<section class="detailed-table-section" id="detailed"><h2>Detailed Data</h2>'

    if not results:
        html += "<p>No data available</p>"
        html += "</section>"
        return html

    html += """
    <div class="filter-controls">
        <label>Filter by N: <select id="filter-n">
            <option value="all">All</option>
        </select></label>
        <label>Filter by B: <select id="filter-b">
            <option value="all">All</option>
        </select></label>
    </div>
    <div class="table-wrapper">
        <table class="detailed-table" id="detailed-table">
            <thead>
                <tr>
                    <th>Engine</th>
                    <th>N (Humanoids)</th>
                    <th>B (Batch)</th>
                    <th>Total FPS</th>
                </tr>
            </thead>
            <tbody>
    """

    # Sort results: by engine, N, B
    sorted_results = sorted(results, key=lambda r: (r["engine"], r["N"], r["B"]))

    for r in sorted_results:
        html += f"""
                <tr data-n="{r["N"]}" data-b="{r["B"]}">
                    <td>{r["engine"]}</td>
                    <td>{r["N"]}</td>
                    <td>{r["B"]}</td>
                    <td>{r["total_fps"]:,.2f}</td>
                </tr>
        """

    html += """
            </tbody>
        </table>
    </div>
    </section>
    """

    return html


def _get_css_styles() -> str:
    """Return inline CSS styles."""
    return """<style>
        :root {
            --motrix-blue: #3b82f6;
            --mujoco-red: #ef4444;
            --success: #22c55e;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --border: #e2e8f0;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.5;
        }
        html { scroll-behavior: smooth; }

        /* Header */
        header {
            background: var(--card-bg);
            border-bottom: 1px solid var(--border);
            padding: 1.5rem 2rem;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        header h1 {
            font-size: 1.5rem;
            margin: 0;
        }
        header .subtitle {
            color: #64748b;
            font-size: 0.875rem;
            margin-top: 0.25rem;
        }

        /* CPU Tabs */
        .cpu-tabs {
            background: var(--card-bg);
            padding: 1rem 2rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            gap: 0.5rem;
            overflow-x: auto;
            position: sticky;
            top: 72px;
            z-index: 100;
        }
        .cpu-tab {
            padding: 0.5rem 1.5rem;
            border: 1px solid var(--border);
            border-radius: 20px;
            background: var(--bg);
            cursor: pointer;
            transition: all 0.2s;
            white-space: nowrap;
            font-weight: 500;
        }
        .cpu-tab:hover {
            background: #e2e8f0;
        }
        .cpu-tab.active {
            background: var(--success);
            color: white;
            border-color: var(--success);
        }

        /* Navigation */
        nav {
            background: var(--card-bg);
            padding: 1rem 2rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            gap: 0.5rem;
            overflow-x: auto;
            position: sticky;
            top: 130px;
            z-index: 99;
        }
        .nav-tab {
            padding: 0.5rem 1.5rem;
            border: 1px solid var(--border);
            border-radius: 20px;
            background: var(--bg);
            cursor: pointer;
            transition: all 0.2s;
            white-space: nowrap;
            font-weight: 500;
        }
        .nav-tab:hover {
            background: #e2e8f0;
        }
        .nav-tab.active {
            background: var(--motrix-blue);
            color: white;
            border-color: var(--motrix-blue);
        }

        /* Main content */
        main {
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }

        /* Sections */
        section {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        section h2 {
            font-size: 1.5rem;
            margin-bottom: 1rem;
            color: var(--text);
        }

        /* Overview Dashboard */
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }
        .summary-card {
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            border-radius: 8px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid var(--border);
        }
        .summary-label {
            font-size: 0.875rem;
            color: #64748b;
            margin-bottom: 0.5rem;
        }
        .summary-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--motrix-blue);
        }
        .summary-sub {
            font-size: 0.75rem;
            color: #64748b;
            margin-top: 0.5rem;
        }

        /* Charts */
        .charts-section {
            background: var(--card-bg);
        }
        .chart-container {
            position: relative;
            height: 400px;
            margin-bottom: 2rem;
        }
        .chart-container:last-child {
            margin-bottom: 0;
        }

        /* Matrix Table */
        .matrix-table-wrapper {
            overflow-x: auto;
        }
        .matrix-table {
            width: 100%;
            border-collapse: collapse;
        }
        .matrix-table th,
        .matrix-table td {
            padding: 0.75rem 1rem;
            text-align: center;
            border: 1px solid var(--border);
        }
        .matrix-table th {
            background: #f1f5f9;
            font-weight: 600;
        }
        .engine-header {
            background: #f1f5f9;
            font-weight: 600;
            text-align: left;
        }
        .matrix-cell-value {
            font-weight: 600;
        }
        .matrix-cell-missing {
            color: #94a3b8;
        }

        /* Filter Controls */
        .filter-controls {
            margin-bottom: 1rem;
            display: flex;
            gap: 1rem;
            align-items: center;
        }
        .filter-controls label {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-weight: 500;
        }
        .filter-controls select {
            padding: 0.5rem;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: var(--bg);
            font-size: 0.875rem;
        }

        /* Detailed Table */
        .table-wrapper {
            overflow-x: auto;
        }
        .detailed-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
        }
        .detailed-table th,
        .detailed-table td {
            padding: 0.75rem 1rem;
            text-align: left;
            border: 1px solid var(--border);
        }
        .detailed-table th {
            background: #f1f5f9;
            font-weight: 600;
            position: sticky;
            top: 0;
        }
        .detailed-table tr:nth-child(even) {
            background: #f8fafc;
        }
        .detailed-table tr:hover {
            background: #f1f5f9;
        }

        /* Responsive */
        @media (max-width: 768px) {
            header {
                padding: 1rem;
            }
            header h1 {
                font-size: 1.25rem;
            }
            main {
                padding: 1rem;
            }
            .summary-cards {
                grid-template-columns: 1fr;
            }
            nav {
                padding: 0.5rem 1rem;
            }
        }
    </style>"""


def _get_chartjs_inline() -> str:
    """Load Chart.js library as inline script for htmlpreview.github.io compatibility.

    Returns:
        HTML script tag with Chart.js library content embedded
    """
    chartjs_path = Path(__file__).parent / "chart.js"

    if not chartjs_path.exists():
        # Fallback to CDN if file doesn't exist
        return f'<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>'

    with open(chartjs_path, "r", encoding="utf-8") as f:
        chartjs_content = f.read()

    return f'<script>\n{chartjs_content}\n</script>'


def _get_html_template(title: str, cpu_model: str, results: List[Dict],
                       grouped: Dict, stats: Dict) -> str:
    """Generate complete HTML document."""

    # Collect chart configurations
    chart_configs = []

    by_n_html = _get_by_n_charts_html(grouped, chart_configs)
    by_b_html = _get_by_b_charts_html(grouped, chart_configs)

    # Build JavaScript array from configs
    chart_configs_js = "[\n" + ",\n".join(chart_configs) + "\n]"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    {_get_chartjs_inline()}
    {_get_css_styles()}
</head>
<body>
    <header>
        <h1>{title}</h1>
        <div class="subtitle">CPU: {cpu_model} | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
    </header>

    <nav>
        <button class="nav-tab active" data-target="by-n">By Humanoid Count</button>
        <button class="nav-tab" data-target="by-b">By Batch Size</button>
        <button class="nav-tab" data-target="detailed">Detailed Data</button>
    </nav>

    <main>
        {by_n_html}

        {by_b_html}

        {_get_detailed_table_html(results)}
    </main>

    <script>
        // Initialize all charts after DOM is ready
        document.addEventListener('DOMContentLoaded', function() {{
            const chartConfigs = {chart_configs_js};

            chartConfigs.forEach(function(chartConfig) {{
                const canvas = document.getElementById(chartConfig.id);
                if (canvas) {{
                    new Chart(canvas, chartConfig.config);
                }}
            }});

            // Setup filter controls for detailed table
            const nFilter = document.getElementById('filter-n');
            const bFilter = document.getElementById('filter-b');

            if (nFilter && bFilter) {{
                // Get unique N and B values from table
                const nValues = [...new Set([...document.querySelectorAll('#detailed-table tbody tr')].map(row => row.dataset.n))].sort((a, b) => parseInt(a) - parseInt(b));
                const bValues = [...new Set([...document.querySelectorAll('#detailed-table tbody tr')].map(row => row.dataset.b))].sort((a, b) => parseInt(a) - parseInt(b));

                // Populate filter dropdowns
                nValues.forEach(n => {{
                    const option = document.createElement('option');
                    option.value = n;
                    option.textContent = n;
                    nFilter.appendChild(option);
                }});

                bValues.forEach(b => {{
                    const option = document.createElement('option');
                    option.value = b;
                    option.textContent = b;
                    bFilter.appendChild(option);
                }});

                // Filter function
                function filterTable() {{
                    const selectedN = nFilter.value;
                    const selectedB = bFilter.value;

                    document.querySelectorAll('#detailed-table tbody tr').forEach(row => {{
                        const rowN = row.dataset.n;
                        const rowB = row.dataset.b;

                        const showN = selectedN === 'all' || rowN === selectedN;
                        const showB = selectedB === 'all' || rowB === selectedB;

                        row.style.display = (showN && showB) ? '' : 'none';
                    }});
                }}

                nFilter.addEventListener('change', filterTable);
                bFilter.addEventListener('change', filterTable);
            }}
        }});

        // Navigation tabs
        document.querySelectorAll('.nav-tab').forEach(tab => {{
            tab.addEventListener('click', () => {{
                const target = tab.dataset.target;
                const targetSection = document.getElementById(target) || document.querySelector('section');

                if (targetSection) {{
                    targetSection.scrollIntoView({{ behavior: 'smooth', block: 'start' }});

                    // Update active state
                    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
                    tab.classList.add('active');
                }}
            }});
        }});

        // Update active tab on scroll
        const observerOptions = {{
            root: null,
            rootMargin: '-20% 0px -60% 0px',
            threshold: 0
        }};

        const observer = new IntersectionObserver((entries) => {{
            entries.forEach(entry => {{
                if (entry.isIntersecting) {{
                    const sectionId = entry.target.id;
                    document.querySelectorAll('.nav-tab').forEach(tab => {{
                        tab.classList.toggle('active', tab.dataset.target === sectionId);
                    }});
                }}
            }});
        }}, observerOptions);

        // Observe all sections
        document.querySelectorAll('section').forEach(section => {{
            observer.observe(section);
        }});
    </script>
</body>
</html>"""


def _get_html_template_multi_cpu(title: str, all_cpu_data: Dict[str, Dict]) -> str:
    """Generate complete HTML document with multi-CPU support.

    Args:
        title: Report title
        all_cpu_data: Dict mapping CPU dir names to their data:
            {
                "cpu_dir_name": {
                    "results": [...],
                    "grouped": {...},
                    "stats": {...}
                }
            }

    Returns:
        Complete HTML document as string
    """
    if not all_cpu_data:
        # No data - return empty report
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    {_get_css_styles()}
</head>
<body>
    <header>
        <h1>{title}</h1>
        <div class="subtitle">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
    </header>
    <main>
        <section>
            <h2>No Data Available</h2>
            <p>No benchmark results found. Please run benchmarks first.</p>
        </section>
    </main>
</body>
</html>"""

    # Get the first CPU (most recent) for initial display
    first_cpu = list(all_cpu_data.keys())[0]

    # Generate CPU tabs HTML
    cpu_tabs_html = "        <div class=\"cpu-tabs\">\n"
    for i, cpu_dir_name in enumerate(all_cpu_data.keys()):
        formatted_name = format_cpu_name(cpu_dir_name)
        is_active = "active" if i == 0 else ""
        cpu_tabs_html += f"            <button class=\"cpu-tab {is_active}\" data-cpu=\"{cpu_dir_name}\">{formatted_name}</button>\n"
    cpu_tabs_html += "        </div>\n"

    # Build JavaScript data structure for all CPUs
    all_cpu_data_js = {}
    for cpu_dir_name, data in all_cpu_data.items():
        # Generate chart configs for this CPU
        chart_configs = []
        by_n_html = _get_by_n_charts_html(data["grouped"], chart_configs)
        by_b_html = _get_by_b_charts_html(data["grouped"], chart_configs)
        overview_html = ""
        table_html = _get_detailed_table_html(data["results"])

        chart_configs_js = "[\n" + ",\n".join(chart_configs) + "\n]"

        all_cpu_data_js[cpu_dir_name] = {
            "stats": data["stats"],
            "chartConfigs": chart_configs_js,
            "byNHtml": by_n_html,
            "byBHtml": by_b_html,
            "tableHtml": table_html,
        }

    # Convert to JavaScript
    all_cpu_data_js_json = json.dumps(all_cpu_data_js, ensure_ascii=False)

    # Get initial HTML sections
    initial_data = all_cpu_data_js[first_cpu]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    {_get_chartjs_inline()}
    {_get_css_styles()}
</head>
<body>
    <header>
        <h1>{title}</h1>
        <div class="subtitle">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
    </header>

    {cpu_tabs_html}

    <nav>
        <button class="nav-tab active" data-target="by-n">By Humanoid Count</button>
        <button class="nav-tab" data-target="by-b">By Batch Size</button>
        <button class="nav-tab" data-target="detailed">Detailed Data</button>
    </nav>

    <main id="main-content">
        <div id="charts-container">
            {initial_data['byNHtml']}
            {initial_data['byBHtml']}
        </div>
        {initial_data['tableHtml']}
    </main>

    <script>
        // All CPU data
        const allCpuData = {all_cpu_data_js_json};

        // Track current chart instances
        let currentCharts = [];

        // Initialize first CPU's charts
        document.addEventListener('DOMContentLoaded', function() {{
            const initialCpu = '{first_cpu}';
            initializeCharts(initialCpu);

            // Setup filter controls for detailed table
            setupTableFilters();
        }});

        function initializeCharts(cpuDirName) {{
            // Destroy existing charts
            currentCharts.forEach(chart => {{
                if (chart) chart.destroy();
            }});
            currentCharts = [];

            const chartConfigs = allCpuData[cpuDirName].chartConfigs;
            const configs = eval(chartConfigs);

            configs.forEach(function(chartConfig) {{
                const canvas = document.getElementById(chartConfig.id);
                if (canvas) {{
                    const chart = new Chart(canvas, chartConfig.config);
                    currentCharts.push(chart);
                }}
            }});
        }}

        function setupTableFilters() {{
            const nFilter = document.getElementById('filter-n');
            const bFilter = document.getElementById('filter-b');

            if (nFilter && bFilter) {{
                const nValues = [...new Set([...document.querySelectorAll('#detailed-table tbody tr')].map(row => row.dataset.n))].sort((a, b) => parseInt(a) - parseInt(b));
                const bValues = [...new Set([...document.querySelectorAll('#detailed-table tbody tr')].map(row => row.dataset.b))].sort((a, b) => parseInt(a) - parseInt(b));

                nValues.forEach(n => {{
                    const option = document.createElement('option');
                    option.value = n;
                    option.textContent = n;
                    nFilter.appendChild(option);
                }});

                bValues.forEach(b => {{
                    const option = document.createElement('option');
                    option.value = b;
                    option.textContent = b;
                    bFilter.appendChild(option);
                }});

                function filterTable() {{
                    const selectedN = nFilter.value;
                    const selectedB = bFilter.value;

                    document.querySelectorAll('#detailed-table tbody tr').forEach(row => {{
                        const rowN = row.dataset.n;
                        const rowB = row.dataset.b;

                        const showN = selectedN === 'all' || rowN === selectedN;
                        const showB = selectedB === 'all' || rowB === selectedB;

                        row.style.display = (showN && showB) ? '' : 'none';
                    }});
                }}

                nFilter.addEventListener('change', filterTable);
                bFilter.addEventListener('change', filterTable);
            }}
        }}

        // CPU tab handler
        document.querySelectorAll('.cpu-tab').forEach(tab => {{
            tab.addEventListener('click', () => {{
                const selectedCpu = tab.dataset.cpu;

                // Update active state
                document.querySelectorAll('.cpu-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');

                // Update report
                updateReport(selectedCpu);
            }});
        }});

        function updateReport(cpuDirName) {{
            const data = allCpuData[cpuDirName];

            // Update main content
            const mainContent = document.getElementById('main-content');
            mainContent.innerHTML = '<div id="charts-container">' + data.byNHtml + data.byBHtml + '</div>' +
                data.tableHtml;

            // Reinitialize charts
            initializeCharts(cpuDirName);

            // Reset up table filters
            setupTableFilters();
        }}

        // Navigation tabs
        document.querySelectorAll('.nav-tab').forEach(tab => {{
            tab.addEventListener('click', () => {{
                const target = tab.dataset.target;
                const targetSection = document.getElementById(target) || document.querySelector('section');

                if (targetSection) {{
                    targetSection.scrollIntoView({{ behavior: 'smooth', block: 'start' }});

                    // Update active state
                    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
                    tab.classList.add('active');
                }}
            }});
        }});

        // Update active tab on scroll
        const observerOptions = {{
            root: null,
            rootMargin: '-20% 0px -60% 0px',
            threshold: 0
        }};

        const observer = new IntersectionObserver((entries) => {{
            entries.forEach(entry => {{
                if (entry.isIntersecting) {{
                    const sectionId = entry.target.id;
                    document.querySelectorAll('.nav-tab').forEach(tab => {{
                        tab.classList.toggle('active', tab.dataset.target === sectionId);
                    }});
                }}
            }});
        }}, observerOptions);

        // Observe all sections
        document.querySelectorAll('section').forEach(section => {{
            observer.observe(section);
        }});
    </script>
</body>
</html>"""


def generate_html_report(
    output_path: str = "output/humanoid/comparison_report.html",
    results_dir: Path = None,
    cpu_model: Optional[str] = None,
    title: str = "Humanoid Benchmark Performance Report",
) -> None:
    """Generate self-contained HTML report with interactive charts.

    Args:
        output_path: Path where HTML file will be saved
        results_dir: Directory containing JSON results (defaults to "output/humanoid")
        cpu_model: Specific CPU model subdirectory (auto-detected if None) - deprecated,
                   now loads all CPU models automatically
        title: Title for the HTML page
    """
    if results_dir is None:
        results_dir = Path("output/humanoid")

    # Load all CPU results
    all_cpu_results = load_all_cpu_results(results_dir)

    if not all_cpu_results:
        print(f"Warning: No benchmark results found in {results_dir}")
        # Still generate the report with empty data
        all_cpu_results = {}

    # Prepare data for all CPUs
    all_cpu_data = {}
    for cpu_dir_name, results in all_cpu_results.items():
        grouped = group_by_dimensions(results)
        stats = calculate_statistics(results)
        all_cpu_data[cpu_dir_name] = {
            "results": results,
            "grouped": grouped,
            "stats": stats,
        }

    # Generate HTML
    html = _get_html_template_multi_cpu(title, all_cpu_data)

    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    total_results = sum(len(data["results"]) for data in all_cpu_data.values())
    print(f"Report generated: {output_path}")
    print(f"  - {len(all_cpu_data)} CPU models loaded")
    print(f"  - {total_results} total benchmark results loaded")
