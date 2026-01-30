"""Generate HTML visualization of benchmark results."""

import json
from pathlib import Path
from typing import List, Dict, Optional
from bench_output_utils import (
    load_benchmark_results,
    load_all_hardware_results,
    generate_summary_stats,
    get_unique_values,
    detect_hardware_type,
    format_hardware_name,
    SIMULATOR_HARDWARE_MAPPING,
)


# Simulator color scheme for charts
SIMULATOR_COLORS = {
    "genesis": "#3b82f6",      # Blue
    "motrixsim": "#22c55e",    # Green
    "motrixsimwarp": "#a855f7",# Purple
    "isaacsim": "#ef4444",     # Red
    "mujocowarp": "#f59e0b",   # Amber
}

# Display names for simulators
SIM_DISPLAY_NAMES = {
    "genesis": "Genesis",
    "motrixsim": "MotrixSim",
    "motrixsimwarp": "Motrixsim-Warp",
    "isaacsim": "IsaacSim",
    "mujocowarp": "MuJoCo-Warp",
}


def _get_chartjs_inline() -> str:
    """Load Chart.js library as inline script for htmlpreview.github.io compatibility.

    Returns:
        HTML script tag with Chart.js library content embedded
    """
    chartjs_path = Path(__file__).parent / "chart.js"

    if not chartjs_path.exists():
        # Fallback to CDN if file doesn't exist
        return '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>'

    with open(chartjs_path, "r", encoding="utf-8") as f:
        chartjs_content = f.read()

    return f'<script>\n{chartjs_content}\n</script>'


def _get_chart_config(chart_id: str, title: str, labels: List[str],
                      datasets: List[Dict], y_label: str = "Total FPS") -> str:
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
                        font: {{ size: 16, weight: 'bold' }}
                    }},
                    legend: {{
                        position: 'top',
                        labels: {{
                            font: {{ size: 12 }}
                        }}
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                let label = context.dataset.label || '';
                                if (label) {{
                                    label += ': ';
                                }}
                                if (context.parsed.y !== null) {{
                                    label += context.parsed.y.toLocaleString() + ' FPS';
                                }}
                                return label;
                            }}
                        }}
                    }}
                }},
                scales: {{
                    y: {{
                        type: 'linear',
                        title: {{
                            display: true,
                            text: '{y_label}',
                            font: {{ size: 12 }}
                        }},
                        beginAtZero: true
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Simulator',
                            font: {{ size: 12 }}
                        }}
                    }}
                }}
            }}
        }}
    }}"""


def _group_results_for_charts(results: List[Dict]) -> Dict:
    """Group results by mode, object, n_robots, batch_size for chart rendering.

    Returns:
        Structure:
        {
            "franka_only": {n_robots: {batch_size: {simulator: {total_fps, per_env_fps}}}},
            "franka_grasp": {object: {n_robots: {batch_size: {simulator: {total_fps, per_env_fps}}}}}
        }
    """
    grouped = {
        "franka_only": {},
        "franka_grasp": {}
    }

    for r in results:
        if r["status"] != "success":
            continue

        mode = r["mode"]
        simulator = r["simulator"]
        n_robots = r["n_robots"]
        batch_size = r["batch_size"]

        fps_data = {
            "total_fps": r["total_fps"],
            "per_env_fps": r["per_env_fps"]
        }

        if mode == "franka_only":
            if n_robots not in grouped["franka_only"]:
                grouped["franka_only"][n_robots] = {}
            if batch_size not in grouped["franka_only"][n_robots]:
                grouped["franka_only"][n_robots][batch_size] = {}
            grouped["franka_only"][n_robots][batch_size][simulator] = fps_data

        elif mode == "franka_grasp":
            obj = r.get("object", "unknown")
            if obj not in grouped["franka_grasp"]:
                grouped["franka_grasp"][obj] = {}
            if n_robots not in grouped["franka_grasp"][obj]:
                grouped["franka_grasp"][obj][n_robots] = {}
            if batch_size not in grouped["franka_grasp"][obj][n_robots]:
                grouped["franka_grasp"][obj][n_robots][batch_size] = {}
            grouped["franka_grasp"][obj][n_robots][batch_size][simulator] = fps_data

    return grouped


def _group_by_n_and_b(results: List[Dict]) -> Dict:
    """Group data by n_robots and batch_size for chart rendering.

    Returns:
        {
            "by_N": {n_robots: {batch_size: {simulator: fps_data}}},
            "by_B": {batch_size: {n_robots: {simulator: fps_data}}}
        }
    """
    grouped = {
        "by_N": {},
        "by_B": {}
    }

    for r in results:
        if r["status"] != "success":
            continue

        simulator = r["simulator"]
        n_robots = r["n_robots"]
        batch_size = r["batch_size"]

        fps_data = {
            "total_fps": r["total_fps"],
            "per_env_fps": r["per_env_fps"]
        }

        # Group by N
        if n_robots not in grouped["by_N"]:
            grouped["by_N"][n_robots] = {}
        if batch_size not in grouped["by_N"][n_robots]:
            grouped["by_N"][n_robots][batch_size] = {}
        grouped["by_N"][n_robots][batch_size][simulator] = fps_data

        # Group by B
        if batch_size not in grouped["by_B"]:
            grouped["by_B"][batch_size] = {}
        if n_robots not in grouped["by_B"][batch_size]:
            grouped["by_B"][batch_size][n_robots] = {}
        grouped["by_B"][batch_size][n_robots][simulator] = fps_data

    return grouped


def generate_html_report(
    output_path: str = "output/bench/comparison_report.html",
    results_dir: Path = None,
    title: str = "Unified Benchmark Performance Report",
    filter_mode: str = None,
    filter_clutter: bool = None,
    filter_release: bool = None,
) -> None:
    """Generate self-contained HTML report with performance visualizations.

    Args:
        output_path: Path where HTML file will be saved
        results_dir: Directory containing JSON results (defaults to "output/bench")
        title: Title for the HTML page
        filter_mode: Filter by mode ("franka_only" or "franka_grasp", None for all)
        filter_clutter: Filter by clutter flag (True/False/None for all)
        filter_release: Filter by release flag (True/False/None for all)
    """
    if results_dir is None:
        results_dir = Path("output/bench")

    # Try to load hardware-grouped results first
    hardware_results = load_all_hardware_results(results_dir)

    if not hardware_results:
        # Fallback to flat load
        results = load_benchmark_results(results_dir)
        if not results:
            print("⚠️  No benchmark results found. Run benchmarks first.")
            return
        # Group under "legacy" key
        hardware_results = {"legacy": results}

    # Apply filters to each hardware group
    filtered_hw_data = {}
    for hw_name, results in hardware_results.items():
        filtered = results
        if filter_mode:
            filtered = [r for r in filtered if r["mode"] == filter_mode]
        if filter_clutter is not None:
            filtered = [r for r in filtered if r.get("clutter", False) == filter_clutter]
        if filter_release is not None:
            filtered = [r for r in filtered if r.get("release", False) == filter_release]

        if filtered:
            filtered_hw_data[hw_name] = filtered

    if not filtered_hw_data:
        print(f"⚠️  No results matching filters (mode={filter_mode}, clutter={filter_clutter}, release={filter_release})")
        return

    # Check if we have hardware subdirectories or just legacy
    if len(filtered_hw_data) == 1 and "legacy" in filtered_hw_data:
        # Legacy mode: render without hardware tabs
        results = filtered_hw_data["legacy"]
        stats = generate_summary_stats(results)
        unique_values = get_unique_values(results)
        html = _create_html_template(title, results, stats, unique_values)
    else:
        # Multi-hardware mode: render with hardware tabs
        html = _create_html_template_multi_hardware(title, filtered_hw_data)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Report generated: {output_path}")


def _create_html_template(
    title: str, results: List[Dict], stats: Dict, unique_values: Dict
) -> str:
    """Generate HTML string with embedded styles and scripts."""
    # Collect chart configurations
    chart_configs = []

    # Generate HTML components
    by_n_html = _get_by_n_charts_html(results, unique_values, chart_configs)
    by_b_html = _get_by_b_charts_html(results, unique_values, chart_configs)
    performance_charts = _get_performance_charts_html(results, unique_values, chart_configs)
    detailed_results = _get_detailed_results_html(results, unique_values)

    # Build JavaScript array from configs
    chart_configs_js = "[\n" + ",\n".join(chart_configs) + "\n]" if chart_configs else "[]"

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
    </header>

    <nav class="quick-nav">
        {_get_quick_nav_tabs(unique_values)}
    </nav>

    <main>
        {by_n_html}
        {by_b_html}
        {performance_charts}
        {detailed_results}
    </main>

    {_get_javascript(chart_configs_js)}
</body>
</html>"""


def _create_html_template_multi_hardware(
    title: str, filtered_hw_data: Dict[str, List[Dict]]
) -> str:
    """Generate HTML string with hardware tabs for multi-hardware results.

    Args:
        title: Page title
        filtered_hw_data: Dictionary mapping hardware names to result lists
    """
    # Pre-render all hardware sections
    all_hardware_data = {}

    for hw_name, results in filtered_hw_data.items():
        if not results:
            continue

        hw_type = detect_hardware_type(hw_name) if hw_name != "legacy" else "cpu"
        unique_values = get_unique_values(results)

        # Collect chart configurations for this hardware
        chart_configs = []

        # Generate HTML components filtered by hardware type
        by_n_html = _get_by_n_charts_html(results, unique_values, chart_configs, hardware_type=hw_type)
        by_b_html = _get_by_b_charts_html(results, unique_values, chart_configs, hardware_type=hw_type)
        performance_charts = _get_performance_charts_html(results, unique_values, chart_configs, hardware_type=hw_type)
        detailed_results = _get_detailed_results_html(results, unique_values, hardware_type=hw_type)

        # Build JavaScript array from configs
        chart_configs_js = "[\n" + ",\n".join(chart_configs) + "\n]" if chart_configs else "[]"

        all_hardware_data[hw_name] = {
            "hw_type": hw_type,
            "by_n_html": by_n_html,
            "by_b_html": by_b_html,
            "performance_charts": performance_charts,
            "detailed_results": detailed_results,
            "chart_configs_js": chart_configs_js,
        }

    # Generate hardware tab buttons
    hardware_tabs_html = '<div class="hardware-tabs">'
    first = True
    for hw_name in sorted(all_hardware_data.keys(), key=lambda x: (x == "legacy", x)):
        hw_type = all_hardware_data[hw_name]["hw_type"]
        display_name = format_hardware_name(hw_name, hw_type) if hw_name != "legacy" else "Legacy Results"
        tab_class = f"cpu-tab" if hw_type == "cpu" else "gpu-tab"
        active_class = " active" if first else ""
        badge = "CPU" if hw_type == "cpu" else "GPU"

        hardware_tabs_html += f'''
        <button class="hardware-tab {tab_class}{active_class}" data-hardware="{hw_name}">
            <span class="hardware-badge">{badge}</span>
            <span>{display_name.replace("[CPU] ", "").replace("[GPU] ", "")}</span>
        </button>'''
        first = False
    hardware_tabs_html += '\n    </div>'

    # Generate initial content (first hardware)
    first_hw = sorted(all_hardware_data.keys(), key=lambda x: (x == "legacy", x))[0]
    first_data = all_hardware_data[first_hw]
    initial_content = f"""
        {first_data['by_n_html']}
        {first_data['by_b_html']}
        {first_data['performance_charts']}
        {first_data['detailed_results']}
    """

    # Build JavaScript object with all hardware data
    hw_data_js = "{\n"
    for hw_name, data in all_hardware_data.items():
        hw_data_js += f'    "{hw_name}": {{\n'
        hw_data_js += f'        chartConfigs: {data["chart_configs_js"]},\n'
        hw_data_js += f'        contentHtml: `{_escape_for_js(data["by_n_html"] + data["by_b_html"] + data["performance_charts"] + data["detailed_results"])}`\n'
        hw_data_js += '    },\n'
    hw_data_js += "}"

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
    </header>

    {hardware_tabs_html}

    <main id="main-content">
        {initial_content}
    </main>

    {_get_javascript_multi_hardware(hw_data_js)}
</body>
</html>"""


def _escape_for_js(html: str) -> str:
    """Escape HTML for embedding in JavaScript template literal."""
    return html.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')


def _get_css_styles() -> str:
    """Return inline CSS styles."""
    return """<style>
        :root {
            --high-fps: #22c55e;
            --medium-fps: #eab308;
            --low-fps: #ef4444;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --border: #e2e8f0;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
               background: var(--bg); color: var(--text); line-height: 1.5; }
        html { scroll-behavior: smooth; }

        /* Header */
        header { background: var(--card-bg); border-bottom: 1px solid var(--border);
                 position: sticky; top: 0; z-index: 100; }
        header h1 { padding: 1rem 2rem; font-size: 1.5rem; margin: 0; }

        /* Quick Navigation */
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
            background: #3b82f6;
            color: white;
            border-color: #3b82f6;
        }

        /* Hardware Tabs */
        .hardware-tabs {
            position: sticky;
            top: 0;
            z-index: 98;
            background: var(--card-bg);
            padding: 1rem 2rem;
            border-bottom: 2px solid var(--border);
            display: flex;
            gap: 1rem;
            overflow-x: auto;
        }

        .hardware-tab {
            padding: 0.75rem 1.5rem;
            border: 2px solid var(--border);
            border-radius: 8px;
            background: white;
            cursor: pointer;
            transition: all 0.2s;
            white-space: nowrap;
            font-weight: 600;
            font-size: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .hardware-tab:hover {
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transform: translateY(-1px);
        }

        .hardware-tab.active {
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transform: translateY(-2px);
        }

        .hardware-tab.cpu-tab {
            border-color: #3b82f6;
        }

        .hardware-tab.cpu-tab.active {
            background: #3b82f6;
            color: white;
        }

        .hardware-tab.gpu-tab {
            border-color: #22c55e;
        }

        .hardware-tab.gpu-tab.active {
            background: #22c55e;
            color: white;
        }

        .hardware-badge {
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.05em;
        }

        .cpu-tab .hardware-badge {
            background: #dbeafe;
            color: #1e40af;
        }

        .cpu-tab.active .hardware-badge {
            background: rgba(255,255,255,0.3);
            color: white;
        }

        .gpu-tab .hardware-badge {
            background: #dcfce7;
            color: #15803d;
        }

        .gpu-tab.active .hardware-badge {
            background: rgba(255,255,255,0.3);
            color: white;
        }

        /* Main Content */
        main {
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }

        /* Section */
        .section {
            margin: 3rem 0;
            background: var(--card-bg);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .section-header {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            padding: 1.5rem 2rem;
        }

        .section-header h2 {
            margin: 0;
            font-size: 1.75rem;
        }

        .section-content {
            padding: 2rem;
        }

        /* Simulator Overview Cards */
        .simulator-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .simulator-card {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            border: 2px solid var(--border);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .simulator-card.high-fps {
            border-color: var(--high-fps);
            background: linear-gradient(135deg, #ffffff 0%, #dcfce7 100%);
        }

        .simulator-card.medium-fps {
            border-color: var(--medium-fps);
            background: linear-gradient(135deg, #ffffff 0%, #fef9c3 100%);
        }

        .simulator-card.low-fps {
            border-color: var(--low-fps);
            background: linear-gradient(135deg, #ffffff 0%, #fee2e2 100%);
        }

        .simulator-card .sim-name {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            color: var(--text);
        }

        .simulator-card .fps-value {
            font-size: 2.5rem;
            font-weight: 800;
            margin: 0.5rem 0;
        }

        .simulator-card .fps-value.high { color: var(--high-fps); }
        .simulator-card .fps-value.medium { color: var(--medium-fps); }
        .simulator-card .fps-value.low { color: var(--low-fps); }

        .simulator-card .fps-label {
            font-size: 0.75rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .simulator-card .stats-detail {
            font-size: 0.875rem;
            color: #64748b;
            margin-top: 0.5rem;
        }

        /* Performance Matrix Table */
        .matrix-wrapper {
            overflow-x: auto;
            background: var(--card-bg);
            border-radius: 8px;
            margin: 1.5rem 0;
        }

        .matrix-table {
            width: 100%;
            border-collapse: collapse;
            min-width: 600px;
        }

        .matrix-table th,
        .matrix-table td {
            padding: 0.75rem 1rem;
            text-align: center;
            border: 1px solid var(--border);
        }

        .matrix-table thead th {
            background: #f1f5f9;
            font-weight: 700;
            position: sticky;
            top: 0;
        }

        .matrix-table .sim-col {
            background: #f8fafc;
            font-weight: 600;
            text-align: left;
            position: sticky;
            left: 0;
        }

        .fps-cell {
            font-weight: 600;
            font-size: 0.95rem;
        }

        .fps-cell.high { color: var(--high-fps); }
        .fps-cell.medium { color: var(--medium-fps); }
        .fps-cell.low { color: var(--low-fps); }
        .fps-cell.failed { color: #94a3b8; }

        .fps-secondary {
            font-size: 0.75rem;
            color: #64748b;
            font-weight: 400;
            display: block;
            margin-top: 0.25rem;
        }

        /* Detailed Table */
        .detail-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }

        .detail-table th,
        .detail-table td {
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }

        .detail-table thead th {
            background: #f8fafc;
            font-weight: 600;
            color: #64748b;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .detail-table tbody tr:hover {
            background: #f8fafc;
        }

        /* Chart Container */
        .chart-container {
            position: relative;
            height: 400px;
            margin-bottom: 2rem;
            background: var(--card-bg);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        .chart-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 2rem;
        }

        /* Subsection */
        .subsection {
            margin: 2rem 0;
            padding: 1.5rem;
            background: #f8fafc;
            border-radius: 8px;
        }

        .subsection-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--text);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .subsection-title::before {
            content: '';
            display: inline-block;
            width: 4px;
            height: 1.5rem;
            background: #3b82f6;
            border-radius: 2px;
        }

        /* Empty State */
        .empty-state {
            text-align: center;
            padding: 3rem;
            color: #64748b;
            font-style: italic;
        }

        /* Responsive */
        @media (max-width: 768px) {
            header h1 { font-size: 1.25rem; padding: 0.75rem 1rem; }
            main { padding: 1rem; }
            .simulator-cards { grid-template-columns: 1fr; }
            .section-content { padding: 1rem; }
            .quick-nav { padding: 0.5rem 1rem; }
        }
    </style>"""


def _get_javascript(chart_configs_js: str) -> str:
    """Return inline JavaScript for navigation and chart rendering."""
    return f"""<script>
        // Initialize all charts after DOM is ready
        document.addEventListener('DOMContentLoaded', function() {{
            const chartConfigs = {chart_configs_js};

            chartConfigs.forEach(function(chartConfig) {{
                const canvas = document.getElementById(chartConfig.id);
                if (canvas) {{
                    new Chart(canvas, chartConfig.config);
                }}
            }});
        }});

        // Tab click navigation
        document.querySelectorAll('.nav-tab').forEach(tab => {{
            tab.addEventListener('click', () => {{
                const targetId = tab.dataset.target;
                const targetSection = document.getElementById(targetId);

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

        document.querySelectorAll('.section').forEach(section => {{
            observer.observe(section);
        }});
    </script>"""


def _get_javascript_multi_hardware(hw_data_js: str) -> str:
    """Return inline JavaScript for hardware tab switching and chart rendering.

    Args:
        hw_data_js: JavaScript object containing all hardware data
    """
    return f"""<script>
        // All hardware data
        const allHardwareData = {hw_data_js};

        // Track current charts for cleanup
        let currentCharts = [];

        // Initialize charts for initial hardware
        document.addEventListener('DOMContentLoaded', function() {{
            const firstHardware = Object.keys(allHardwareData)[0];
            if (firstHardware) {{
                initChartsForHardware(firstHardware);
            }}
        }});

        // Function to initialize charts for a specific hardware
        function initChartsForHardware(hardwareName) {{
            const hwData = allHardwareData[hardwareName];
            if (!hwData || !hwData.chartConfigs) return;

            hwData.chartConfigs.forEach(function(chartConfig) {{
                const canvas = document.getElementById(chartConfig.id);
                if (canvas) {{
                    const chart = new Chart(canvas, chartConfig.config);
                    currentCharts.push(chart);
                }}
            }});
        }}

        // Function to destroy all current charts
        function destroyCurrentCharts() {{
            currentCharts.forEach(chart => {{
                try {{
                    chart.destroy();
                }} catch(e) {{
                    console.warn('Failed to destroy chart:', e);
                }}
            }});
            currentCharts = [];
        }}

        // Hardware tab click handler
        document.querySelectorAll('.hardware-tab').forEach(tab => {{
            tab.addEventListener('click', () => {{
                const hardwareName = tab.dataset.hardware;
                const hwData = allHardwareData[hardwareName];

                if (!hwData) return;

                // Update active tab state
                document.querySelectorAll('.hardware-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');

                // Destroy existing charts
                destroyCurrentCharts();

                // Update content
                const mainContent = document.getElementById('main-content');
                if (mainContent && hwData.contentHtml) {{
                    mainContent.innerHTML = hwData.contentHtml;
                }}

                // Initialize new charts
                setTimeout(() => {{
                    initChartsForHardware(hardwareName);
                }}, 100);
            }});
        }});
    </script>"""


def _get_quick_nav_tabs(unique_values: Dict) -> str:
    """Generate quick navigation tabs."""
    tabs = [
        '<button class="nav-tab active" data-target="by-n">By Robot Count</button>',
        '<button class="nav-tab" data-target="by-b">By Batch Size</button>'
    ]

    # Mode display name mapping
    MODE_DISPLAY_NAMES = {"franka_only": "Franka Only", "franka_grasp": "Franka Grasp"}

    modes = unique_values.get("modes", [])
    for mode in modes:
        if mode == "franka_only":
            continue  # Already covered by "Performance by Robot Count (N)"
        mode_label = MODE_DISPLAY_NAMES.get(mode, mode.replace("_", " ").title())
        tabs.append(f'<button class="nav-tab" data-target="mode-{mode}">{mode_label}</button>')

    tabs.append('<button class="nav-tab" data-target="detailed">Detailed Results</button>')

    return "\n".join(tabs)


def _get_simulator_overview_html(stats: Dict, results: List[Dict]) -> str:
    """Generate simulator overview section with performance cards."""
    html = """
    <section class="section" id="overview">
        <div class="section-header">
            <h2>Simulator Performance Overview</h2>
        </div>
        <div class="section-content">
            <div class="simulator-cards">"""

    for sim, sim_stats in sorted(stats["by_simulator"].items()):
        avg_fps = sim_stats.get("avg_fps", 0)
        max_fps = sim_stats.get("max_fps", 0)
        success_count = sim_stats.get("success", 0)
        total_count = sim_stats.get("total", 0)

        # Determine FPS level (adjust thresholds as needed)
        if avg_fps >= 1000:
            level_class = "high-fps"
            fps_class = "high"
        elif avg_fps >= 100:
            level_class = "medium-fps"
            fps_class = "medium"
        else:
            level_class = "low-fps"
            fps_class = "low"

        # Get simulator display name
        sim_display = SIM_DISPLAY_NAMES.get(sim, sim.capitalize())

        html += f"""
                <div class="simulator-card {level_class}">
                    <div class="sim-name">{sim_display}</div>
                    <div class="fps-value {fps_class}">{avg_fps:,.0f}</div>
                    <div class="fps-label">Avg FPS</div>
                    <div class="stats-detail">Max: {max_fps:,.0f} FPS</div>
                    <div class="stats-detail">{success_count}/{total_count} tests</div>
                </div>"""

    html += """
            </div>
        </div>
    </section>"""

    return html


def _get_performance_charts_html(results: List[Dict], unique_values: Dict, chart_configs: list, hardware_type: str = None) -> str:
    """Generate performance charts and tables for each mode.

    Args:
        results: List of result dictionaries
        unique_values: Dictionary of unique values
        chart_configs: List to append chart configurations to
        hardware_type: Optional hardware type filter ("cpu" or "gpu")
    """
    html = ""

    # Mode display name mapping
    MODE_DISPLAY_NAMES = {"franka_only": "Franka Only", "franka_grasp": "Franka Grasp"}

    modes = unique_values.get("modes", [])
    for mode in modes:
        if mode == "franka_only":
            continue  # Already covered by "Performance by Robot Count (N)"
        mode_results = [r for r in results if r["mode"] == mode]
        if not mode_results:
            continue

        mode_label = MODE_DISPLAY_NAMES.get(mode, mode.replace("_", " ").title())
        html += f"""
    <section class="section" id="mode-{mode}">
        <div class="section-header">
            <h2>{mode_label} Mode Performance</h2>
        </div>
        <div class="section-content">"""

        if mode == "franka_only":
            html += _get_franka_only_charts_html(mode_results, unique_values, chart_configs, hardware_type)
            html += _get_franka_only_matrices(mode_results, unique_values, hardware_type)
        elif mode == "franka_grasp":
            html += _get_franka_grasp_charts_html(mode_results, unique_values, chart_configs, hardware_type)
            html += _get_franka_grasp_matrices(mode_results, unique_values, hardware_type)

        html += """
        </div>
    </section>"""

    return html


def _get_by_n_charts_html(results: List[Dict], unique_values: Dict, chart_configs: list, hardware_type: str = None) -> str:
    """Generate HTML for charts grouped by robot count (N).

    Shows franka_only mode data only (no clutter) for clear baseline performance.

    Args:
        results: List of result dictionaries
        unique_values: Dictionary of unique values
        chart_configs: List to append chart configurations to
        hardware_type: Optional hardware type filter ("cpu" or "gpu")
    """
    html = """
    <section class="section" id="by-n">
        <div class="section-header">
            <h2>Performance by Robot Count (N)</h2>
        </div>
        <div class="section-content">"""

    # Filter for franka_only mode only, exclude clutter tests
    franka_only_results = [r for r in results
                      if r.get("mode") == "franka_only"
                      and r.get("status") == "success"
                      and not r.get("clutter", False)]

    if not franka_only_results:
        html += '<div class="empty-state">No franka_only mode data available</div>'
        html += """
        </div>
    </section>"""
        return html

    grouped = _group_by_n_and_b(franka_only_results)["by_N"]
    simulators = unique_values.get("simulators", [])

    # Filter simulators by hardware type if specified
    if hardware_type:
        simulators = [s for s in simulators if SIMULATOR_HARDWARE_MAPPING.get(s) == hardware_type]

    for n in sorted(grouped.keys()):
        batch_data = grouped[n]
        if not batch_data:
            continue

        labels = [f"B={b}" for b in sorted(batch_data.keys())]
        datasets = []

        for sim in simulators:
            data = []
            for b in sorted(batch_data.keys()):
                fps = batch_data[b].get(sim, {}).get("total_fps", 0)
                data.append(fps)

            if any(d > 0 for d in data):  # Only add if there's data
                datasets.append({
                    "label": SIM_DISPLAY_NAMES.get(sim, sim.capitalize()),
                    "data": data,
                    "backgroundColor": SIMULATOR_COLORS.get(sim, "#94a3b8")
                })

        chart_id = f"chart-by-n{n}"
        chart_config = _get_chart_config(
            chart_id,
            f"N={n} Robot{'s' if n > 1 else ''} - Total FPS",
            labels,
            datasets
        )
        chart_configs.append(chart_config)

        html += f'<div class="chart-container"><canvas id="{chart_id}"></canvas></div>'

    html += """
        </div>
    </section>"""
    return html


def _get_by_b_charts_html(results: List[Dict], unique_values: Dict, chart_configs: list, hardware_type: str = None) -> str:
    """Generate HTML for charts grouped by batch size (B).

    Shows franka_only mode data only (no clutter) for clear baseline performance.

    Args:
        results: List of result dictionaries
        unique_values: Dictionary of unique values
        chart_configs: List to append chart configurations to
        hardware_type: Optional hardware type filter ("cpu" or "gpu")
    """
    html = """
    <section class="section" id="by-b">
        <div class="section-header">
            <h2>Performance by Batch Size (B)</h2>
        </div>
        <div class="section-content">"""

    # Filter for franka_only mode only, exclude clutter tests
    franka_only_results = [r for r in results
                      if r.get("mode") == "franka_only"
                      and r.get("status") == "success"
                      and not r.get("clutter", False)]

    if not franka_only_results:
        html += '<div class="empty-state">No franka_only mode data available</div>'
        html += """
        </div>
    </section>"""
        return html

    grouped = _group_by_n_and_b(franka_only_results)["by_B"]
    simulators = unique_values.get("simulators", [])

    # Filter simulators by hardware type if specified
    if hardware_type:
        simulators = [s for s in simulators if SIMULATOR_HARDWARE_MAPPING.get(s) == hardware_type]

    for b in sorted(grouped.keys()):
        n_data = grouped[b]
        if not n_data:
            continue

        labels = [f"N={n}" for n in sorted(n_data.keys())]
        datasets = []

        for sim in simulators:
            data = []
            for n in sorted(n_data.keys()):
                fps = n_data[n].get(sim, {}).get("total_fps", 0)
                data.append(fps)

            if any(d > 0 for d in data):  # Only add if there's data
                datasets.append({
                    "label": SIM_DISPLAY_NAMES.get(sim, sim.capitalize()),
                    "data": data,
                    "backgroundColor": SIMULATOR_COLORS.get(sim, "#94a3b8")
                })

        chart_id = f"chart-by-b{b}"
        chart_config = _get_chart_config(
            chart_id,
            f"B={b} Batch Size - Total FPS",
            labels,
            datasets
        )
        chart_configs.append(chart_config)

        html += f'<div class="chart-container"><canvas id="{chart_id}"></canvas></div>'

    html += """
        </div>
    </section>"""
    return html


def _get_franka_only_charts_html(results: List[Dict], unique_values: Dict, chart_configs: list, hardware_type: str = None) -> str:
    """Generate bar charts for franka_only mode - one chart per n_robots value.

    Args:
        results: List of result dictionaries
        unique_values: Dictionary of unique values
        chart_configs: List to append chart configurations to
        hardware_type: Optional hardware type filter ("cpu" or "gpu")
    """
    html = '<div class="subsection"><div class="subsection-title">Performance Charts</div>'
    grouped = _group_results_for_charts(results)["franka_only"]

    n_robots_list = sorted(set(r["n_robots"] for r in results))
    simulators = unique_values.get("simulators", [])

    # Filter simulators by hardware type if specified
    if hardware_type:
        simulators = [s for s in simulators if SIMULATOR_HARDWARE_MAPPING.get(s) == hardware_type]

    for n in n_robots_list:
        # Get batch sizes actually tested for this n_robots value
        batch_sizes_for_n = sorted(set(r["batch_size"] for r in results if r["n_robots"] == n))
        labels = [f"B={b}" for b in batch_sizes_for_n]
        datasets = []

        for sim in simulators:
            data = []
            for b in batch_sizes_for_n:
                fps = grouped.get(n, {}).get(b, {}).get(sim, {}).get("total_fps", 0)
                data.append(fps)

            datasets.append({
                "label": SIM_DISPLAY_NAMES.get(sim, sim.capitalize()),
                "data": data,
                "backgroundColor": SIMULATOR_COLORS.get(sim, "#94a3b8")
            })

        chart_id = f"chart-franka-only-n{n}"
        chart_config = _get_chart_config(
            chart_id,
            f"Franka Only - N={n} Robot{'s' if n > 1 else ''}",
            labels,
            datasets
        )
        chart_configs.append(chart_config)

        html += f'<div class="chart-container"><canvas id="{chart_id}"></canvas></div>'

    html += '</div>'
    return html


def _get_franka_only_matrices(results: List[Dict], unique_values: Dict, hardware_type: str = None) -> str:
    """Generate matrices for franka_only mode grouped by batch size.

    Args:
        results: List of result dictionaries
        unique_values: Dictionary of unique values
        hardware_type: Optional hardware type filter ("cpu" or "gpu")
    """
    html = '<div class="subsection"><div class="subsection-title">Detailed Data Tables</div>'

    batch_sizes = sorted(set(r["batch_size"] for r in results))
    n_robots_list = sorted(set(r["n_robots"] for r in results))
    simulators = unique_values.get("simulators", [])

    # Filter simulators by hardware type if specified
    if hardware_type:
        simulators = [s for s in simulators if SIMULATOR_HARDWARE_MAPPING.get(s) == hardware_type]

    # Check which clutter values exist in the data
    has_clutter_false = any(not r.get("clutter", False) for r in results)
    has_clutter_true = any(r.get("clutter", False) for r in results)

    for b in batch_sizes:
        html += f"""
            <div style="margin-top: 1.5rem;">
                <h4 style="margin-bottom: 0.75rem; color: #64748b;">Batch Size: {b}</h4>
                <div class="matrix-wrapper">
                    <table class="matrix-table">
                        <thead>
                            <tr>
                                <th>Simulator</th>"""

        for n in n_robots_list:
            html += f"""
                                <th>N={n} robots</th>"""

        html += """
                            </tr>
                        </thead>
                        <tbody>"""

        # Create result lookup for quick access
        result_lookup = {}
        for r in results:
            if r["batch_size"] == b:
                key = (r["simulator"], r["n_robots"], r.get("clutter", False))
                result_lookup[key] = r

        for sim in simulators:
            sim_display = SIM_DISPLAY_NAMES.get(sim, sim.capitalize())

            html += f"""
                            <tr>
                                <td class="sim-col">{sim_display}</td>"""

            for n in n_robots_list:
                result = result_lookup.get((sim, n, False))
                html += _format_fps_cell(result)

            html += """
                            </tr>"""

        html += """
                        </tbody>
                    </table>
                </div>"""

        # If clutter results exist, add comparison table
        if has_clutter_true:
            clutter_results = [r for r in results if r.get("clutter", False) and r["batch_size"] == b]
            if clutter_results:
                html += f"""
                <div style="margin-top: 1rem;">
                    <h5 style="margin-bottom: 0.5rem; color: #64748b; font-size: 0.9rem;">With Clutter (200+ bottles)</h5>
                    <div class="matrix-wrapper">
                        <table class="matrix-table">
                            <thead>
                                <tr>
                                    <th>Simulator</th>"""

                for n in n_robots_list:
                    html += f"""
                                    <th>N={n} robots</th>"""

                html += """
                                </tr>
                            </thead>
                            <tbody>"""

                for sim in simulators:
                    sim_display = SIM_DISPLAY_NAMES.get(sim, sim.capitalize())

                    html += f"""
                                <tr>
                                    <td class="sim-col">{sim_display}</td>"""

                    for n in n_robots_list:
                        result = result_lookup.get((sim, n, True))
                        html += _format_fps_cell(result)

                    html += """
                                </tr>"""

                html += """
                            </tbody>
                        </table>
                    </div>
                </div>"""

        html += """
            </div>"""

    html += '</div>'
    return html


def _get_franka_grasp_charts_html(results: List[Dict], unique_values: Dict, chart_configs: list, hardware_type: str = None) -> str:
    """Generate bar charts for franka_grasp mode - one chart per (object, n_robots) combination.

    Args:
        results: List of result dictionaries
        unique_values: Dictionary of unique values
        chart_configs: List to append chart configurations to
        hardware_type: Optional hardware type filter ("cpu" or "gpu")
    """
    html = '<div class="subsection"><div class="subsection-title">Performance Charts</div>'
    grouped = _group_results_for_charts(results)["franka_grasp"]

    objects = sorted(set(r.get("object") for r in results if r.get("object")))
    n_robots_list = sorted(set(r["n_robots"] for r in results))
    simulators = unique_values.get("simulators", [])

    # Filter simulators by hardware type if specified
    if hardware_type:
        simulators = [s for s in simulators if SIMULATOR_HARDWARE_MAPPING.get(s) == hardware_type]

    for obj in objects:
        html += f'<h4 style="margin: 1.5rem 0 1rem 0; color: #475569; font-size: 1.1rem;">Object: {obj.capitalize()}</h4>'

        for n in n_robots_list:
            # Get batch sizes actually tested for this (object, n_robots) combination
            batch_sizes_for_n = sorted(set(
                r["batch_size"] for r in results
                if r.get("object") == obj and r["n_robots"] == n
            ))
            labels = [f"B={b}" for b in batch_sizes_for_n]
            datasets = []

            for sim in simulators:
                data = []
                for b in batch_sizes_for_n:
                    fps = grouped.get(obj, {}).get(n, {}).get(b, {}).get(sim, {}).get("total_fps", 0)
                    data.append(fps)

                datasets.append({
                    "label": SIM_DISPLAY_NAMES.get(sim, sim.capitalize()),
                    "data": data,
                    "backgroundColor": SIMULATOR_COLORS.get(sim, "#94a3b8")
                })

            chart_id = f"chart-franka-grasp-{obj}-n{n}"
            chart_config = _get_chart_config(
                chart_id,
                f"Franka Grasp - {obj.capitalize()}, N={n} Robot{'s' if n > 1 else ''}",
                labels,
                datasets
            )
            chart_configs.append(chart_config)

            html += f'<div class="chart-container"><canvas id="{chart_id}"></canvas></div>'

    html += '</div>'
    return html


def _get_franka_grasp_matrices(results: List[Dict], unique_values: Dict, hardware_type: str = None) -> str:
    """Generate matrices for franka_grasp mode grouped by object.

    Args:
        results: List of result dictionaries
        unique_values: Dictionary of unique values
        hardware_type: Optional hardware type filter ("cpu" or "gpu")
    """
    html = '<div class="subsection"><div class="subsection-title">Detailed Data Tables</div>'

    objects = sorted(set(r.get("object") for r in results if r.get("object")))
    batch_sizes = sorted(set(r["batch_size"] for r in results))
    n_robots_list = sorted(set(r["n_robots"] for r in results))
    simulators = unique_values.get("simulators", [])

    # Filter simulators by hardware type if specified
    if hardware_type:
        simulators = [s for s in simulators if SIMULATOR_HARDWARE_MAPPING.get(s) == hardware_type]

    for obj in objects:
        html += f'<h4 style="margin: 1.5rem 0 1rem 0; color: #475569; font-size: 1.1rem;">Object: {obj.capitalize()}</h4>'

        for b in batch_sizes:
            html += f"""
                <div style="margin-top: 1rem;">
                    <h5 style="margin-bottom: 0.5rem; color: #64748b; font-size: 0.9rem;">Batch Size: {b}</h5>
                    <div class="matrix-wrapper">
                        <table class="matrix-table">
                            <thead>
                                <tr>
                                    <th>Simulator</th>"""

            for n in n_robots_list:
                html += f"""
                                    <th>N={n} robots</th>"""

            html += """
                                </tr>
                            </thead>
                            <tbody>"""

            # Create result lookup
            result_lookup = {}
            for r in results:
                if r.get("object") == obj and r["batch_size"] == b:
                    key = (r["simulator"], r["n_robots"])
                    result_lookup[key] = r

            for sim in simulators:
                sim_display = SIM_DISPLAY_NAMES.get(sim, sim.capitalize())

                html += f"""
                                <tr>
                                    <td class="sim-col">{sim_display}</td>"""

                for n in n_robots_list:
                    result = result_lookup.get((sim, n))
                    html += _format_fps_cell(result)

                html += """
                                </tr>"""

            html += """
                            </tbody>
                        </table>
                    </div>
                </div>"""

    html += '</div>'
    return html


def _format_fps_cell(result: Dict) -> str:
    """Format a table cell with FPS data."""
    # Distinguish between not tested and failed
    if not result:
        # Not tested (no JSON file exists for this configuration)
        return """
                                <td class="fps-cell" style="color: #cbd5e1;">—</td>"""

    if result["status"] == "failed":
        # Test was run but failed
        return """
                                <td class="fps-cell failed">FAILED</td>"""

    per_env = result["per_env_fps"]
    total = result["total_fps"]

    # Determine FPS level
    if per_env >= 1000:
        fps_class = "high"
    elif per_env >= 100:
        fps_class = "medium"
    else:
        fps_class = "low"

    return f"""
                                <td class="fps-cell {fps_class}">
                                    {per_env:,.0f}
                                    <span class="fps-secondary">({total:,.0f} total)</span>
                                </td>"""


def _get_detailed_results_html(results: List[Dict], unique_values: Dict, hardware_type: str = None) -> str:
    """Generate detailed results table.

    Args:
        results: List of result dictionaries
        unique_values: Dictionary of unique values
        hardware_type: Optional hardware type filter ("cpu" or "gpu")
    """
    # Mode display name mapping
    MODE_DISPLAY_NAMES = {"franka_only": "Franka Only", "franka_grasp": "Franka Grasp"}

    # Filter results by hardware type if specified
    if hardware_type:
        results = [r for r in results if SIMULATOR_HARDWARE_MAPPING.get(r["simulator"]) == hardware_type]

    html = """
    <section class="section" id="detailed">
        <div class="section-header">
            <h2>Detailed Results</h2>
        </div>
        <div class="section-content">
            <table class="detail-table">
                <thead>
                    <tr>
                        <th>Simulator</th>
                        <th>Mode</th>
                        <th>Object</th>
                        <th>N Robots</th>
                        <th>Batch Size</th>
                        <th>Clutter</th>
                        <th>Release</th>
                        <th>Per-Env FPS</th>
                        <th>Total FPS</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>"""

    # Sort results for consistent display
    sorted_results = sorted(
        results,
        key=lambda r: (r["mode"], r.get("object", ""), r["simulator"], r["n_robots"], r["batch_size"])
    )

    for r in sorted_results:
        sim_display = SIM_DISPLAY_NAMES.get(r["simulator"], r["simulator"].capitalize())

        obj_display = r.get("object", "—").capitalize() if r.get("object") else "—"
        clutter_display = "Yes" if r.get("clutter") else "No"
        release_display = "Yes" if r.get("release") else "No"

        if r["status"] == "success":
            per_env_fps = r["per_env_fps"]
            total_fps = r["total_fps"]
            fps_class = "high" if per_env_fps >= 1000 else "medium" if per_env_fps >= 100 else "low"
            per_env_display = f'{per_env_fps:,.0f}'
            total_display = f'{total_fps:,.0f}'
        else:
            fps_class = "failed"
            per_env_display = "—"
            total_display = "—"

        mode_display = MODE_DISPLAY_NAMES.get(r["mode"], r["mode"].replace("_", " ").title())
        html += f"""
                    <tr>
                        <td>{sim_display}</td>
                        <td>{mode_display}</td>
                        <td>{obj_display}</td>
                        <td>{r["n_robots"]}</td>
                        <td>{r["batch_size"]}</td>
                        <td>{clutter_display}</td>
                        <td>{release_display}</td>
                        <td class="fps-cell {fps_class}">{per_env_display}</td>
                        <td class="fps-cell {fps_class}">{total_display}</td>
                        <td>{r["status"].capitalize()}</td>
                    </tr>"""

    html += """
                </tbody>
            </table>
        </div>
    </section>"""

    return html
