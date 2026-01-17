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

"""Generate HTML visualization of grasp test results."""

from pathlib import Path
from typing import List, Dict
from test_output_utils import load_test_results, generate_summary_stats


def _make_relative_to_html(video_path: str, html_output_path: str) -> str:
    """Convert video path to be relative to HTML file location.

    Args:
        video_path: Original video path (e.g., "output/motrix_grasp_shake_cube.mp4")
        html_output_path: Path where HTML will be saved (e.g., "output/test_results.html")

    Returns:
        Relative path from HTML to video (e.g., "motrix_grasp_shake_cube.mp4")
    """
    video = Path(video_path)
    html_dir = Path(html_output_path).parent

    try:
        # Make path relative to HTML directory
        relative = video.resolve().relative_to(html_dir.resolve())
        return str(relative)
    except ValueError:
        # If can't make relative, use absolute path or return as-is
        return str(video)


def generate_html_report(
    output_path: str = "output/test_results.html",
    results_dir: Path = None,
    title: str = "Grasp Test Results",
) -> None:
    """Generate self-contained HTML report with embedded videos.

    Args:
        output_path: Path where HTML file will be saved
        results_dir: Directory containing JSON results (defaults to "output/")
        title: Title for the HTML page
    """
    results = load_test_results(results_dir)
    stats = generate_summary_stats(results)

    html = _create_html_template(title, results, stats, output_path)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Report generated: {output_path}")


def _create_html_template(
    title: str, results: List[Dict], stats: Dict, html_output_path: str
) -> str:
    """Generate HTML string with embedded videos and new grouped layout."""
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
    </header>

    <nav class="quick-nav">
        {_get_quick_nav_tabs(results)}
    </nav>

    <main>
        {_get_engine_overview_html(stats, results)}
        {_get_detailed_results_by_object_html(results, html_output_path)}
    </main>

    {_get_javascript()}
</body>
</html>"""


def _get_css_styles() -> str:
    """Return inline CSS styles."""
    return """<style>
        :root {
            --success: #22c55e;
            --failure: #ef4444;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --border: #e2e8f0;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
               background: var(--bg); color: var(--text); line-height: 1.5; }
        html { scroll-behavior: smooth; }

        /* Header styles - only title bar is sticky */
        header { background: var(--card-bg); border-bottom: 1px solid var(--border);
                 position: sticky; top: 0; z-index: 100; }
        header h1 { padding: 1rem 2rem; font-size: 1.5rem; margin: 0; }

        /* ===== Engine Overview Section ===== */
        .engine-overview-section {
            margin-bottom: 2rem;
        }

        /* Engine Success Cards */
        .engine-cards-container {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            justify-content: center;
            margin-bottom: 2rem;
        }

        .engine-card {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            min-width: 180px;
            text-align: center;
            border: 2px solid var(--border);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .engine-card.high-success {
            border-color: var(--success);
            background: linear-gradient(135deg, #ffffff 0%, #dcfce7 100%);
        }

        .engine-card.medium-success {
            border-color: #eab308;
            background: linear-gradient(135deg, #ffffff 0%, #fef9c3 100%);
        }

        .engine-card.low-success {
            border-color: var(--failure);
            background: linear-gradient(135deg, #ffffff 0%, #fee2e2 100%);
        }

        .engine-card .engine-name {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            color: var(--text);
        }

        .engine-card .success-rate {
            font-size: 2.5rem;
            font-weight: 800;
            margin: 0.5rem 0;
        }

        .engine-card .rate-high { color: var(--success); }
        .engine-card .rate-medium { color: #eab308; }
        .engine-card .rate-low { color: var(--failure); }

        .engine-card .stats-detail {
            font-size: 0.875rem;
            color: #64748b;
        }

        /* Engine Matrix Table */
        .engine-matrix-wrapper {
            overflow-x: auto;
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .engine-matrix-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 1rem;
            color: var(--text);
        }

        .engine-matrix-table {
            width: 100%;
            border-collapse: collapse;
        }

        .engine-matrix-table th,
        .engine-matrix-table td {
            padding: 0.75rem 1rem;
            text-align: center;
            border: 1px solid var(--border);
        }

        .engine-matrix-table thead tr.group-header th {
            border-bottom: none;
        }

        .engine-matrix-table thead tr.sub-header th {
            border-top: none;
            font-size: 0.875rem;
            color: #64748b;
        }

        .engine-matrix-table .engine-col-header {
            background: #f1f5f9;
            font-weight: 700;
            position: sticky;
            left: 0;
        }

        .matrix-status-cell {
            min-width: 80px;
        }

        .matrix-status-cell .status-icon {
            font-size: 1.5rem;
            display: block;
        }

        .matrix-status-cell .status-icon.success {
            color: var(--success);
        }

        .matrix-status-cell .status-icon.failure {
            color: var(--failure);
        }

        .matrix-status-cell .status-icon.missing {
            color: #94a3b8;
        }

        .matrix-status-cell .drop-time {
            font-size: 0.75rem;
            color: var(--failure);
            margin-top: 0.25rem;
        }

        /* ===== Quick Navigation ===== */
        .quick-nav {
            position: sticky;
            top: 0;
            z-index: 99;
            background: var(--card-bg);
            padding: 1rem 2rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            gap: 0.5rem;
            overflow-x: auto;
        }

        .object-tab {
            padding: 0.5rem 1.5rem;
            border: 1px solid var(--border);
            border-radius: 20px;
            background: var(--bg);
            cursor: pointer;
            transition: all 0.2s;
            white-space: nowrap;
            font-weight: 500;
        }

        .object-tab:hover {
            background: #e2e8f0;
        }

        .object-tab.active {
            background: #3b82f6;
            color: white;
            border-color: #3b82f6;
        }

        /* ===== Main Content Area ===== */
        main {
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }

        /* ===== Object Section (Collapsible) ===== */
        .detailed-results-section {
            margin-top: 2rem;
        }

        .object-section {
            margin: 3rem 0;
            background: var(--card-bg);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .object-section summary {
            cursor: pointer;
            user-select: none;
        }

        .object-section[open] summary {
            border-bottom: 1px solid var(--border);
        }

        .object-section-header {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            padding: 1.5rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .object-section-header h2 {
            margin: 0;
            font-size: 1.75rem;
        }

        .object-stats {
            display: flex;
            gap: 2rem;
            font-size: 0.875rem;
            opacity: 0.9;
        }

        /* ===== DT Subsection ===== */
        .dt-subsection {
            padding: 2rem;
            border-bottom: 1px solid var(--border);
        }

        .dt-subsection:last-child {
            border-bottom: none;
        }

        .dt-subsection-header {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            color: var(--text);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .dt-subsection-header::before {
            content: '';
            display: inline-block;
            width: 4px;
            height: 1.5rem;
            background: #3b82f6;
            border-radius: 2px;
        }

        /* ===== Engine Comparison Grid ===== */
        .engine-comparison-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
        }

        .engine-result-card {
            background: var(--bg);
            border-radius: 8px;
            overflow: hidden;
            border: 2px solid var(--border);
            transition: all 0.2s;
        }

        .engine-result-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }

        .engine-result-card.success {
            border-color: var(--success);
        }

        .engine-result-card.failure {
            border-color: var(--failure);
        }

        .engine-card-header {
            padding: 0.75rem 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            background: var(--card-bg);
        }

        .engine-card-header .engine-name {
            font-weight: 600;
            font-size: 1rem;
        }

        .engine-card-header .status-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .engine-card-header .status-badge.success {
            background: var(--success);
            color: white;
        }

        .engine-card-header .status-badge.failure {
            background: var(--failure);
            color: white;
        }

        .engine-card-video {
            width: 100%;
            aspect-ratio: 4/3;
            background: #000;
        }

        .engine-card-meta {
            padding: 1rem;
            font-size: 0.875rem;
            background: var(--bg);
        }

        .engine-card-meta .drop-time {
            color: var(--failure);
            font-weight: 600;
            margin-top: 0.5rem;
        }

        /* ===== Empty State ===== */
        .empty-section {
            text-align: center;
            padding: 3rem;
            color: #64748b;
            font-style: italic;
        }

        /* ===== Responsive Design ===== */
        @media (max-width: 768px) {
            header h1 {
                font-size: 1.25rem;
                padding: 0.75rem 1rem;
            }

            main {
                padding: 1rem;
            }

            .engine-cards-container {
                flex-direction: column;
            }

            .engine-card {
                min-width: 100%;
            }

            .engine-comparison-grid {
                grid-template-columns: 1fr;
            }

            .object-section-header {
                padding: 1rem;
                flex-direction: column;
                align-items: flex-start;
                gap: 0.5rem;
            }

            .dt-subsection {
                padding: 1rem;
            }

            .quick-nav {
                padding: 0.5rem 1rem;
            }

            .engine-matrix-wrapper {
                padding: 1rem;
            }
        }
    </style>"""


def _get_summary_dashboard_html(stats: Dict, results: List[Dict]) -> str:
    """Generate summary dashboard with key insights."""
    total = stats["total"]
    if total == 0:
        return ""

    success_rate = (stats["success"] / total) * 100

    # Find best performing engine
    best_engine = None
    best_engine_rate = 0
    for engine, engine_stats in stats["by_engine"].items():
        rate = (engine_stats["success"] / engine_stats["total"]) * 100
        if rate > best_engine_rate:
            best_engine_rate = rate
            best_engine = engine

    # Find most reliable object
    best_object = None
    best_object_rate = 0
    for obj, obj_stats in stats["by_object"].items():
        rate = (obj_stats["success"] / obj_stats["total"]) * 100
        if rate > best_object_rate:
            best_object_rate = rate
            best_object = obj

    # Effect of dt on success rate
    dt_effect = []
    for dt_val, dt_stats in sorted(stats["by_dt"].items(), key=lambda x: float(x[0])):
        rate = (dt_stats["success"] / dt_stats["total"]) * 100 if dt_stats["total"] > 0 else 0
        dt_effect.append(f"{dt_val}s: {rate:.0f}%")

    return f"""
    <div class="summary-dashboard">
        <div class="dashboard-card">
            <div class="label">Overall Success Rate</div>
            <div class="value">{success_rate:.1f}%</div>
        </div>
        <div class="dashboard-card highlight">
            <div class="label">Best Engine</div>
            <div class="value">{best_engine or 'N/A'}</div>
        </div>
        <div class="dashboard-card">
            <div class="label">Most Reliable Object</div>
            <div class="value">{(best_object or 'N/A').capitalize()}</div>
        </div>
        <div class="dashboard-card">
            <div class="label">DT Comparison</div>
            <div class="value" style="font-size: 1rem;">{" vs ".join(dt_effect)}</div>
        </div>
    </div>"""


def _get_success_rate_by_dimension_html(stats: Dict) -> str:
    """Generate success rate comparison bars for each dimension."""
    html = '<div class="success-rate-section">'

    # Success rate by engine
    if stats["by_engine"]:
        html += '<div class="success-rate-group">'
        html += '<div class="success-rate-title">Success Rate by Engine</div>'
        for engine, engine_stats in sorted(stats["by_engine"].items()):
            rate = (engine_stats["success"] / engine_stats["total"]) * 100 if engine_stats["total"] > 0 else 0
            level = "high" if rate >= 75 else "medium" if rate >= 50 else "low"
            html += f'''
                <div class="success-rate-bar-container">
                    <div class="success-rate-label">{engine.capitalize()}</div>
                    <div class="success-rate-bar">
                        <div class="success-rate-fill {level}" style="width: {rate}%">
                            {rate:.0f}%
                        </div>
                    </div>
                    <div class="success-rate-text">{engine_stats["success"]}/{engine_stats["total"]}</div>
                </div>'''
        html += '</div>'

    # Success rate by object
    if stats["by_object"]:
        html += '<div class="success-rate-group">'
        html += '<div class="success-rate-title">Success Rate by Object</div>'
        for obj, obj_stats in sorted(stats["by_object"].items()):
            rate = (obj_stats["success"] / obj_stats["total"]) * 100 if obj_stats["total"] > 0 else 0
            level = "high" if rate >= 75 else "medium" if rate >= 50 else "low"
            html += f'''
                <div class="success-rate-bar-container">
                    <div class="success-rate-label">{obj.capitalize()}</div>
                    <div class="success-rate-bar">
                        <div class="success-rate-fill {level}" style="width: {rate}%">
                            {rate:.0f}%
                        </div>
                    </div>
                    <div class="success-rate-text">{obj_stats["success"]}/{obj_stats["total"]}</div>
                </div>'''
        html += '</div>'

    # Success rate by dt
    if stats["by_dt"]:
        html += '<div class="success-rate-group">'
        html += '<div class="success-rate-title">Success Rate by Time Step (dt)</div>'
        for dt_val, dt_stats in sorted(stats["by_dt"].items(), key=lambda x: float(x[0])):
            rate = (dt_stats["success"] / dt_stats["total"]) * 100 if dt_stats["total"] > 0 else 0
            level = "high" if rate >= 75 else "medium" if rate >= 50 else "low"
            html += f'''
                <div class="success-rate-bar-container">
                    <div class="success-rate-label">{dt_val}s</div>
                    <div class="success-rate-bar">
                        <div class="success-rate-fill {level}" style="width: {rate}%">
                            {rate:.0f}%
                        </div>
                    </div>
                    <div class="success-rate-text">{dt_stats["success"]}/{dt_stats["total"]}</div>
                </div>'''
        html += '</div>'

    html += '</div>'
    return html


def _get_comparison_matrix_html(results: List[Dict]) -> str:
    """Generate pivot table matrix with engines vs (object, dt)."""
    # Get unique values
    engines = sorted(set(r["engine"] for r in results))
    objects = sorted(set(r["object"] for r in results))
    dt_values = sorted(set(r["dt"] for r in results))

    if not engines or not objects:
        return ""

    # Create column headers: (object, dt) pairs
    columns = [(obj, dt) for obj in objects for dt in dt_values]
    col_count = len(columns)

    html = f'''
    <div class="comparison-matrix-section">
        <div class="comparison-table-header">Engine vs Configuration Matrix</div>
        <div class="comparison-matrix" style="--var-col-count: {col_count};">
            <div class="matrix-header">
                <div class="matrix-cell matrix-engine-label">Engine</div>'''

    # Header row
    for obj, dt in columns:
        dt_str = f"{dt:.3f}".rstrip("0").rstrip(".") if "." in f"{dt:.3f}" else f"{dt:.3f}"
        html += f'''
                <div class="matrix-cell matrix-header-cell">
                    <span>{obj.capitalize()}</span>
                    <span style="font-weight: 400; color: #64748b;">dt={dt_str}</span>
                </div>'''

    html += '''
            </div>'''

    # Data rows
    for engine in engines:
        html += f'''
            <div class="matrix-row">
                <div class="matrix-cell matrix-engine-label">{engine.capitalize()}</div>'''

        for obj, dt in columns:
            # Find matching result
            matching = [r for r in results if r["engine"] == engine and r["object"] == obj and r["dt"] == dt]

            if matching:
                result = matching[0]
                if result["status"] == "success":
                    icon = "✓"
                    status_class = "success"
                    detail = ""
                else:
                    icon = "✗"
                    status_class = "failure"
                    drop_time = result.get("drop_time")
                    detail = f"@{drop_time:.1f}s" if drop_time else ""

                html += f'''
                <div class="matrix-cell">
                    <div class="matrix-result {status_class}">
                        <span class="matrix-result-icon">{icon}</span>
                        <span>{detail}</span>
                    </div>
                </div>'''
            else:
                html += '''
                <div class="matrix-cell">
                    <div class="matrix-result missing">
                        <span>—</span>
                    </div>
                </div>'''

        html += '''
            </div>'''

    html += '''
        </div>
    </div>'''

    return html


def _get_summary_html(stats: Dict, results: List[Dict]) -> str:
    """Generate summary statistics HTML."""
    return f"""
    {_get_summary_dashboard_html(stats, results)}
    {_get_success_rate_by_dimension_html(stats)}
    {_get_comparison_matrix_html(results)}"""


def _get_comparison_table_html(stats: Dict, results: List[Dict]) -> str:
    """Generate HTML comparison table for individual test results."""
    table_html = """
    <div class="comparison-table-header">Comparison Data</div>
    <div class="comparison-table-container">
        <table class="comparison-table">
            <thead>
                <tr>
                    <th>Engine</th>
                    <th>Object</th>
                    <th>UseMjx</th>
                    <th>Dt</th>
                    <th>Result</th>
                </tr>
            </thead>
            <tbody>
    """

    # Sort results for consistent display
    sorted_results = sorted(
        results, key=lambda r: (r["engine"], r["object"], r["mjx"], r["dt"], r["task"])
    )

    for r in sorted_results:
        mjx_display = "Yes" if r["mjx"] else "No"
        dt_display = (
            f"{r['dt']:.3f}".rstrip("0").rstrip(".")
            if "." in f"{r['dt']:.3f}"
            else f"{r['dt']:.3f}"
        )
        result_class = "success" if r["status"] == "success" else "failure"
        result_text = r["status"].capitalize()

        table_html += f'''
                <tr>
                    <td>{r["engine"]}</td>
                    <td>{r["object"]}</td>
                    <td>{mjx_display}</td>
                    <td>{dt_display}</td>
                    <td class="{result_class}">{result_text}</td>
                </tr>
            '''

    table_html += """
            </tbody>
        </table>
    </div>
    """

    return table_html


def _get_filter_buttons(stats: Dict, category: str) -> str:
    """Generate filter button HTML for a category."""
    buttons = []
    for key, values in stats[f"by_{category}"].items():
        buttons.append(
            f'<button class="filter-btn" data-filter="{category}:{key}">'
            f"{key.capitalize()} ({values['total']})</button>"
        )
    return "\n        ".join(buttons)


def _get_result_cards_html(results: List[Dict], html_output_path: str) -> str:
    """Generate HTML for all result cards."""
    if not results:
        return '<div class="empty-state">No test results found. Run tests with --record flag.</div>'

    cards = []
    for r in results:
        status_class = "success" if r["status"] == "success" else "failure"
        drop_time_html = (
            f'<div class="drop-time">Dropped at {r["drop_time"]:.2f}s</div>'
            if r["drop_time"]
            else ""
        )

        # Format mjx display name
        mjx_display = "MJX" if r["mjx"] else "No MJX"
        # Format dt display (use 3 decimal places, but strip trailing zeros)
        dt_display = (
            f"{r['dt']:.3f}".rstrip("0").rstrip(".")
            if "." in f"{r['dt']:.3f}"
            else f"{r['dt']:.3f}"
        )

        if r["video_exists"]:
            # Use relative path from HTML to video
            relative_video_path = _make_relative_to_html(
                r["video_path"], html_output_path
            )
            video_html = f'<video controls src="{relative_video_path}"></video>'
        else:
            video_html = f'<div style="width:100%;aspect-ratio:4/3;background:#eee;display:flex;align-items:center;justify-content:center;color:#666;">Video not found</div>'

        cards.append(f'''
        <div class="result-card" data-engine="{r["engine"]}" data-object="{r["object"]}" data-task="{r["task"]}" data-mjx="{r["mjx"]}" data-dt="{r["dt"]}">
            <div class="status-badge {status_class}">{r["status"].upper()}</div>
            {video_html}
            <div class="card-content">
                <div class="card-title">{r["engine"].capitalize()} - {r["object"].capitalize()} - {r["task"].capitalize()}</div>
                <div class="mjx-dt-labels">
                    <span class="label-tag mjx-{str(r["mjx"]).lower()}">{mjx_display}</span>
                    <span class="label-tag dt">DT: {dt_display}</span>
                </div>
                <div class="card-meta">
                    <span>{r["engine"]}</span>
                    <span>{r["object"]}</span>
                    <span>{r["task"]}</span>
                </div>
                {drop_time_html}
            </div>
        </div>''')

    return "\n".join(cards)


def _get_javascript() -> str:
    """Return inline JavaScript for object tab navigation and scroll tracking."""
    return """<script>
        // Object tab click navigation
        document.querySelectorAll('.object-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                const targetId = tab.dataset.target;
                const targetSection = document.getElementById(targetId);

                if (targetSection) {
                    targetSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

                    // Update active state
                    document.querySelectorAll('.object-tab').forEach(t => t.classList.remove('active'));
                    tab.classList.add('active');
                }
            });
        });

        // Update active tab on scroll using Intersection Observer
        const observerOptions = {
            root: null,
            rootMargin: '-20% 0px -60% 0px',
            threshold: 0
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const sectionId = entry.target.id;
                    document.querySelectorAll('.object-tab').forEach(tab => {
                        tab.classList.toggle('active', tab.dataset.target === sectionId);
                    });
                }
            });
        }, observerOptions);

        // Observe all object sections
        document.querySelectorAll('.object-section').forEach(section => {
            observer.observe(section);
        });
    </script>"""


def _get_engine_overview_html(stats: Dict, results: List[Dict]) -> str:
    """Generate engine dimension overview with success cards and matrix table."""
    return f"""
    <section class="engine-overview-section">
        {_get_engine_success_cards_html(stats)}
        {_get_engine_config_matrix_html(results)}
    </section>
    """


def _get_engine_success_cards_html(stats: Dict) -> str:
    """Generate horizontal cards showing each engine's success rate."""
    cards = []

    for engine, engine_stats in sorted(stats["by_engine"].items()):
        total = engine_stats["total"]
        success = engine_stats["success"]
        rate = (success / total * 100) if total > 0 else 0

        # Determine success level class
        if rate >= 75:
            level_class = "high-success"
            rate_class = "rate-high"
        elif rate >= 50:
            level_class = "medium-success"
            rate_class = "rate-medium"
        else:
            level_class = "low-success"
            rate_class = "rate-low"

        cards.append(f"""
        <div class="engine-card {level_class}">
            <div class="engine-name">{engine.capitalize()}</div>
            <div class="success-rate {rate_class}">{rate:.0f}%</div>
            <div class="stats-detail">{success}/{total} tests</div>
        </div>
        """)

    return f'<div class="engine-cards-container">{"".join(cards)}</div>'


def _get_engine_config_matrix_html(results: List[Dict]) -> str:
    """Generate matrix table with engines as rows, (object, dt) as grouped columns."""
    from test_output_utils import get_config_combinations

    # Get unique values
    engines = sorted(set(r["engine"] for r in results))
    objects = sorted(set(r["object"] for r in results))
    dt_values = sorted(set(r["dt"] for r in results))

    if not engines or not objects:
        return ""

    # Create result lookup: (engine, object, dt) -> result
    result_lookup = {}
    for r in results:
        result_lookup[(r["engine"], r["object"], r["dt"])] = r

    html = f"""
    <div class="engine-matrix-wrapper">
        <div class="engine-matrix-title">Engine vs Configuration Matrix</div>
        <table class="engine-matrix-table">
            <thead>
                <tr class="group-header">
                    <th rowspan="2">Engine</th>"""

    # Header row with object grouping
    for obj in objects:
        colspan = len(dt_values)
        html += f"""
                    <th colspan="{colspan}">{obj.capitalize()}</th>"""

    html += """
                </tr>
                <tr class="sub-header">"""

    # Sub-header row with dt values for each object
    for obj in objects:
        for dt in dt_values:
            dt_str = f"{dt:.3f}".rstrip("0").rstrip(".")
            html += f"""
                    <th>dt={dt_str}</th>"""

    html += """
                </tr>
            </thead>
            <tbody>"""

    # Data rows
    for engine in engines:
        html += f"""
                <tr>
                    <td class="engine-col-header">{engine.capitalize()}</td>"""

        for obj in objects:
            for dt in dt_values:
                result = result_lookup.get((engine, obj, dt))

                if result:
                    if result["status"] == "success":
                        icon = "✓"
                        icon_class = "success"
                        drop_time_html = ""
                    else:
                        icon = "✗"
                        icon_class = "failure"
                        drop_time = result.get("drop_time")
                        drop_time_html = f'<div class="drop-time">@{drop_time:.1f}s</div>' if drop_time else ""

                    html += f"""
                    <td class="matrix-status-cell">
                        <span class="status-icon {icon_class}">{icon}</span>
                        {drop_time_html}
                    </td>"""
                else:
                    html += """
                    <td class="matrix-status-cell">
                        <span class="status-icon missing">—</span>
                    </td>"""

        html += """
                </tr>"""

    html += """
            </tbody>
        </table>
    </div>"""

    return html


def _get_detailed_results_by_object_html(results: List[Dict], html_output_path: str) -> str:
    """Generate detailed results grouped by object (as sections) and dt (as subsections)."""
    from test_output_utils import group_results_by_object_and_dt

    grouped = group_results_by_object_and_dt(results)

    if not grouped:
        return '<div class="empty-section">No test results found</div>'

    html = '<div class="detailed-results-section">'

    # Sort objects alphabetically
    for object_name in sorted(grouped.keys()):
        dt_groups = grouped[object_name]
        html += _get_object_section_html(object_name, dt_groups, html_output_path)

    html += '</div>'

    return html


def _get_object_section_html(object_name: str, dt_groups: Dict[float, List[Dict]], html_output_path: str) -> str:
    """Generate HTML for one object section using <details> element."""
    # Calculate stats
    config_count = len(dt_groups)
    total_tests = sum(len(v) for v in dt_groups.values())

    html = f"""
    <details class="object-section" id="object-{object_name}" open>
        <summary class="object-section-header">
            <h2>{object_name.capitalize()}</h2>
            <div class="object-stats">
                <span>Configurations: {config_count}</span>
                <span>Total Tests: {total_tests}</span>
            </div>
        </summary>"""

    # Sort dt values
    for dt_value in sorted(dt_groups.keys()):
        engine_results = dt_groups[dt_value]
        html += _get_dt_subsection_html(dt_value, engine_results, html_output_path)

    html += """
    </details>"""

    return html


def _get_dt_subsection_html(dt_value: float, engine_results: List[Dict], html_output_path: str) -> str:
    """Generate HTML for one dt subsection with engine comparison grid."""
    dt_str = f"{dt_value:.3f}".rstrip("0").rstrip(".")

    html = f'''
    <div class="dt-subsection">
        <div class="dt-subsection-header">
            Time Step: dt={dt_str}
        </div>
        <div class="engine-comparison-grid">
    '''

    # Generate card for each engine
    for result in engine_results:
        status_class = "success" if result["status"] == "success" else "failure"

        # Video path
        if result["video_exists"]:
            relative_video_path = _make_relative_to_html(
                result["video_path"], html_output_path
            )
            video_html = f'<video class="engine-card-video" controls src="{relative_video_path}"></video>'
        else:
            video_html = f'<div class="engine-card-video" style="display:flex;align-items:center;justify-content:center;color:#666;">Video not found</div>'

        # Drop time
        drop_time_html = ""
        if result["status"] == "failure" and result.get("drop_time"):
            drop_time_html = f'<div class="drop-time">Dropped at {result["drop_time"]:.2f}s</div>'

        html += f'''
            <div class="engine-result-card {status_class}">
                <div class="engine-card-header">
                    <span class="engine-name">{result["engine"].capitalize()}</span>
                    <span class="status-badge {status_class}">{result["status"].upper()}</span>
                </div>
                {video_html}
                <div class="engine-card-meta">
                    {drop_time_html}
                </div>
            </div>
        '''

    html += '''
        </div>
    </div>
    '''

    return html


def _get_quick_nav_tabs(results: List[Dict]) -> str:
    """Generate quick navigation tabs to jump to each object section."""
    objects = sorted(set(r["object"] for r in results))
    tabs = []
    for i, obj in enumerate(objects):
        active_class = "active" if i == 0 else ""
        tabs.append(f'<button class="object-tab {active_class}" data-target="object-{obj}">{obj.capitalize()}</button>')
    return "\n".join(tabs)
