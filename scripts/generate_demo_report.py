from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent_eval.adversarial import adversarial_security_evaluators
from agent_eval.cli import load_suite_file
from agent_eval.evaluators import default_evaluators
from agent_eval.html_reporter import render_html_report
from agent_eval.runner import EvalRunner, ReplayAgentAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate live HTML demo report for GitHub Pages.")
    parser.add_argument("--suite", default=str(ROOT / "examples" / "mcp_tool_suite.json"))
    parser.add_argument("--out", default=str(ROOT / "site" / "index.html"))
    args = parser.parse_args()

    suite_name, cases, traces = load_suite_file(Path(args.suite))
    runner = EvalRunner(
        adapter=ReplayAgentAdapter(traces),
        evaluators=[*default_evaluators(), *adversarial_security_evaluators()],
    )
    report = asyncio.run(runner.run_suite(suite_name, cases))
    html_output = render_html_report(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_output, encoding="utf-8")
    print(f"Generated {out_path} ({report.passed_cases}/{report.total_cases} passed)")


if __name__ == "__main__":
    main()
