from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from .models import AgentTrace, EvalCase, ToolCall
from .reporter import render_json_report, render_markdown_report
from .runner import EvalRunner, ReplayAgentAdapter


def load_suite_file(path: Path) -> tuple[str, list[EvalCase], dict[str, AgentTrace]]:
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    suite_name = str(raw.get("suite_name", path.stem))
    cases: list[EvalCase] = []
    traces: dict[str, AgentTrace] = {}

    for item in raw["cases"]:
        case = EvalCase(
            case_id=item["case_id"],
            prompt=item["prompt"],
            expected_tools=tuple(item.get("expected_tools", ())),
            ordered_tools=bool(item.get("ordered_tools", False)),
            required_arg_keys={
                k: tuple(v) for k, v in item.get("required_arg_keys", {}).items()
            },
            max_latency_ms=float(item.get("max_latency_ms", 4000.0)),
            max_tokens=int(item.get("max_tokens", 4000)),
            max_repeated_calls=int(item.get("max_repeated_calls", 2)),
            forbid_pii=bool(item.get("forbid_pii", True)),
            expected_substrings=tuple(item.get("expected_substrings", ())),
        )
        cases.append(case)

        if "replay_trace" in item:
            t = item["replay_trace"]
            calls = tuple(
                ToolCall(
                    name=c["name"],
                    arguments=dict(c.get("arguments", {})),
                    latency_ms=float(c.get("latency_ms", 0.0)),
                    succeeded=bool(c.get("succeeded", True)),
                )
                for c in t.get("tool_calls", ())
            )
            traces[case.case_id] = AgentTrace(
                case_id=case.case_id,
                final_output=str(t.get("final_output", "")),
                tool_calls=calls,
                total_latency_ms=float(t.get("total_latency_ms", 0.0)),
                prompt_tokens=int(t.get("prompt_tokens", 0)),
                completion_tokens=int(t.get("completion_tokens", 0)),
            )

    return suite_name, cases, traces


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-eval",
        description="Deterministic evaluation & SLA harness for LLM tool-calling and MCP agents.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run", help="Run a benchmark suite and print an evaluation report.")
    run_cmd.add_argument("--suite", required=True, type=Path, help="Path to benchmark suite JSON.")
    run_cmd.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format (default: markdown).",
    )
    run_cmd.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.0,
        help="Minimum pass rate [0.0-1.0] required for zero exit code (useful for CI gates).",
    )
    run_cmd.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Maximum concurrent case evaluations.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        suite_name, cases, traces = load_suite_file(args.suite)
        runner = EvalRunner(
            adapter=ReplayAgentAdapter(traces),
            max_concurrency=args.concurrency,
        )
        report = asyncio.run(runner.run_suite(suite_name, cases))
        rendered = (
            render_json_report(report)
            if args.format == "json"
            else render_markdown_report(report)
        )
        sys.stdout.write(rendered)
        return 0 if report.pass_rate >= args.min_pass_rate else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
