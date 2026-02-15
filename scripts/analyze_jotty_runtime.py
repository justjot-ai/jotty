#!/usr/bin/env python3
"""
Run python-code-analyzer on Jotty to trace executed code paths.

Usage:
    pip install python-code-analyzer   # or: pip install -e ".[dev]"
    python scripts/analyze_jotty_runtime.py [--scenario SCENARIO] [--out-dir DIR]

Scenarios:
    capabilities  - Import Jotty and call capabilities() (default)
    jotty-init    - Import Jotty and instantiate Jotty()
    explain       - Call capabilities() and explain("memory")
    minimal       - Only core.capabilities.explain("memory") (fewer deps)
    self-test     - No Jotty; runs a tiny loop (verifies analyzer works; use to confirm install)

Note: The analyzer uses sys.settrace and can crash with AttributeError (zipimporter) when
traced code or its dependencies are loaded from zip. If Jotty scenarios fail, run
--scenario self-test to confirm the tool works, then try a minimal venv with fewer deps.

Output:
    Writes analysis to --out-dir (default: scripts/analysis_output/) as:
    - <script_base>_code_analysis.txt        (plain text trace)
    - <script_base>_code_analysis_rich.html  (rich HTML trace)

See: https://pypi.org/project/python-code-analyzer/
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Jotty runtime with python-code-analyzer")
    parser.add_argument(
        "--scenario",
        choices=["capabilities", "jotty-init", "explain", "minimal", "self-test"],
        default="capabilities",
        help="Which Jotty code path to run under the tracer",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "analysis_output"),
        help="Directory for HTML and TXT output",
    )
    args = parser.parse_args()

    # Ensure project root is on path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        from code_analyzer import CodeAnalyzer
    except ImportError:
        print("Install python-code-analyzer: pip install python-code-analyzer", file=sys.stderr)
        sys.exit(1)

    code_analyzer = CodeAnalyzer()
    code_analyzer.start()

    if args.scenario == "capabilities":
        from Jotty import capabilities

        _ = capabilities()
    elif args.scenario == "jotty-init":
        from Jotty import Jotty

        _ = Jotty()
    elif args.scenario == "explain":
        from Jotty.core.capabilities import capabilities, explain

        _ = capabilities()
        _ = explain("memory")
    elif args.scenario == "minimal":
        from Jotty.core.capabilities import explain

        _ = explain("memory")
    elif args.scenario == "self-test":
        # No Jotty imports; proves the analyzer works (avoids zipimporter issues)
        def _dummy(n: int) -> int:
            return n + 1

        _ = [_dummy(i) for i in range(3)]

    code_analyzer.stop()

    os.makedirs(args.out_dir, exist_ok=True)
    # Library writes to cwd using sys.argv[0] for filenames (e.g. *_code_analysis.txt)
    cwd_before = os.getcwd()
    base = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    txt_path = os.path.join(args.out_dir, f"{base}_code_analysis.txt")
    html_path = os.path.join(args.out_dir, f"{base}_code_analysis_rich.html")
    try:
        os.chdir(args.out_dir)
        printer = code_analyzer.get_code_analyzer_printer()
        printer.export_to_txt()
        printer.export_rich_to_html()
        print(f"Analysis written to:\n  {txt_path}\n  {html_path}")
    except AttributeError as e:
        err = str(e)
        if "zipimporter" in err or "archive" in err:
            print(
                "python-code-analyzer hit a known bug when tracing code that uses zip-imported\n"
                "packages (e.g. some stdlib or installed deps). Use --scenario minimal, or run\n"
                "the analyzer on a small script that does not import Jotty. See:\n"
                "  https://github.com/josephedradan/code_analyzer/issues",
                file=sys.stderr,
            )
            sys.exit(2)
        raise
    finally:
        os.chdir(cwd_before)


if __name__ == "__main__":
    main()
