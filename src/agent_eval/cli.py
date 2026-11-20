from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from .diff import compare_suite_reports, render_diff_markdown
from .html_reporter import render_html_report
from .models import AgentTrace, ConversationTurn, EvalCase, ToolCall
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
            allowed_tools=tuple(item.get("allowed_tools", ())),
            ordered_tools=bool(item.get("ordered_tools", False)),
            required_arg_keys={
                k: tuple(v) for k, v in item.get("required_arg_keys", {}).items()
            },
            max_latency_ms=float(item.get("max_latency_ms", 4000.0)),
            max_tokens=int(item.get("max_tokens", 4000)),
            max_cost_usd=(
                float(item["max_cost_usd"]) if item.get("max_cost_usd") is not None else None
            ),
            max_repeated_calls=int(item.get("max_repeated_calls", 2)),
            forbid_pii=bool(item.get("forbid_pii", True)),
            expected_substrings=tuple(item.get("expected_substrings", ())),
            expected_json_keys=tuple(item.get("expected_json_keys", ())),
            rubric_criteria=tuple(item.get("rubric_criteria", ())),
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
            turns = tuple(
                ConversationTurn(
                    role=str(turn.get("role", "assistant")),
                    content=str(turn.get("content", "")),
                    tool_calls=tuple(
                        ToolCall(
                            name=tc["name"],
                            arguments=dict(tc.get("arguments", {})),
                            latency_ms=float(tc.get("latency_ms", 0.0)),
                            succeeded=bool(tc.get("succeeded", True)),
                        )
                        for tc in turn.get("tool_calls", ())
                    ),
                )
                for turn in t.get("turns", ())
            )
            traces[case.case_id] = AgentTrace(
                case_id=case.case_id,
                final_output=str(t.get("final_output", "")),
                tool_calls=calls,
                turns=turns,
                model_name=str(t.get("model_name", "gemini-2.5-flash")),
                total_latency_ms=float(t.get("total_latency_ms", 0.0)),
                prompt_tokens=int(t.get("prompt_tokens", 0)),
                cached_prompt_tokens=int(t.get("cached_prompt_tokens", 0)),
                completion_tokens=int(t.get("completion_tokens", 0)),
            )

    return suite_name, cases, traces


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-eval",
        description="Deterministic evaluation, cost modeling & regression harness for LLM/MCP agents.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run", help="Run a benchmark suite and output a Markdown/JSON/HTML report.")
    run_cmd.add_argument("--suite", required=True, type=Path, help="Path to benchmark suite JSON.")
    run_cmd.add_argument(
        "--format",
        choices=("markdown", "json", "html"),
        default="markdown",
        help="Output format (default: markdown).",
    )
    run_cmd.add_argument("--out", type=Path, default=None, help="Optional output file path.")
    run_cmd.add_argument("--min-pass-rate", type=float, default=0.0, help="CI gate minimum pass rate [0-1].")
    run_cmd.add_argument("--concurrency", type=int, default=8, help="Max concurrent evaluations.")

    diff_cmd = sub.add_parser("diff", help="Compare baseline vs. candidate JSON suite reports.")
    diff_cmd.add_argument("--baseline", required=True, type=Path, help="Baseline JSON report path.")
    diff_cmd.add_argument("--candidate", required=True, type=Path, help="Candidate JSON report path.")
    diff_cmd.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero if any case regresses.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        suite_name, cases, traces = load_suite_file(args.suite)
        runner = EvalRunner(adapter=ReplayAgentAdapter(traces), max_concurrency=args.concurrency)
        report = asyncio.run(runner.run_suite(suite_name, cases))
        if args.format == "json":
            rendered = render_json_report(report)
        elif args.format == "html":
            rendered = render_html_report(report)
        else:
            rendered = render_markdown_report(report)

        if args.out is not None:
            args.out.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
        return 0 if report.pass_rate >= args.min_pass_rate else 1

    if args.command == "diff":
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
        suite_diff = compare_suite_reports(baseline, candidate)
        sys.stdout.write(render_diff_markdown(suite_diff))
        if args.fail_on_regression and suite_diff.has_regression:
            return 1
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
