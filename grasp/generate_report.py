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

"""Generate HTML visualization of grasp test results."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).parent))
from test_result_visualizer import generate_html_report


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML comparison report for grasp benchmark tests. "
                    "The report includes success rate analysis, engine vs configuration matrix, "
                    "and detailed test results with video playback.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              Generate report with default settings
  %(prog)s -o report.html              Save to custom output path
  %(prog)s -r results_dir -t "My Tests" Use custom results dir and title
        """,
    )
    parser.add_argument(
        "--output", "-o",
        default="output/comparison_report.html",
        help="Output HTML path (default: output/comparison_report.html)"
    )
    parser.add_argument(
        "--results-dir", "-r",
        default="output",
        help="Directory containing JSON results (default: output)"
    )
    parser.add_argument(
        "--title", "-t",
        default="Grasp Benchmark Comparison Report",
        help="Report title (default: 'Grasp Benchmark Comparison Report')"
    )

    args = parser.parse_args()

    generate_html_report(
        output_path=args.output,
        results_dir=Path(args.results_dir),
        title=args.title
    )


if __name__ == "__main__":
    main()
