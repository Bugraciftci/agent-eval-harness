from __future__ import annotations

import json
from dataclasses import asdict

from .models import SuiteReport


def render_json_report(report: SuiteReport) -> str:
    return json.dumps(asdict(report), indent=2, ensure_ascii=False)


def render_markdown_report(report: SuiteReport) -> str:
    lines = [
        f"# Agent Evaluation Report — `{report.suite_name}`",
        "",
        f"- **Pass Rate:** `{report.passed_cases}/{report.total_cases}` ({report.pass_rate * 100:.1f}%)",
        f"- **Average Weighted Score:** `{report.average_score * 100:.1f}%`",
        f"- **P95 Latency:** `{report.p95_latency_ms:.0f} ms`",
        f"- **Total Token Usage:** `{report.total_tokens:,}`",
        "",
        "| Case ID | Status | Score | Latency | Tokens | Notes |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |",
    ]

    for case in report.cases:
        status = "PASS" if case.passed else "FAIL"
        failures = [f"{m.evaluator}: {m.detail}" for m in case.metrics if not m.passed]
        notes = "; ".join(failures) if failures else "All checks passed"
        lines.append(
            f"| `{case.case_id}` | **{status}** | {case.weighted_score * 100:.1f}% | "
            f"{case.total_latency_ms:.0f} ms | {case.total_tokens:,} | {notes} |"
        )

    return "\n".join(lines) + "\n"
