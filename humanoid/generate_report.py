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

"""Generate HTML visualization of humanoid benchmark results."""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent))
from report_utils import generate_html_report


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML comparison report for humanoid benchmark tests. "
        "The report includes FPS performance analysis, engine comparison matrix, "
        "and detailed benchmark results with interactive bar charts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              Generate report with default settings
  %(prog)s -o report.html              Save to custom output path
  %(prog)s -r results_dir -t "My Tests" Use custom results dir and title
  %(prog)s --cpu-model Intel_R__Core_TM__i5-10600KF_CPU___4_10GHz  Use specific CPU model dir
        """,
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output/humanoid/comparison_report.html",
        help="Output HTML path (default: output/humanoid/comparison_report.html)",
    )
    parser.add_argument(
        "--results-dir",
        "-r",
        default="output/humanoid",
        help="Base directory containing CPU model subdirectories (default: output/humanoid)",
    )
    parser.add_argument(
        "--cpu-model",
        help="Specific CPU model subdirectory (auto-detected if not specified)",
    )
    parser.add_argument(
        "--title",
        "-t",
        default="Humanoid Benchmark Performance Report",
        help="Report title (default: 'Humanoid Benchmark Performance Report')",
    )

    args = parser.parse_args()

    generate_html_report(
        output_path=args.output,
        results_dir=Path(args.results_dir),
        cpu_model=args.cpu_model,
        title=args.title,
    )


if __name__ == "__main__":
    main()
