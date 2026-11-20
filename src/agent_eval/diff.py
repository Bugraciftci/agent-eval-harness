from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CaseDelta:
    """Per-case regression/improvement comparison between baseline and candidate runs."""

    case_id: str
    baseline_passed: bool
    candidate_passed: bool
    score_delta: float
    latency_delta_ms: float
    cost_delta_usd: float
    status: str  # "regressed", "improved", "unchanged_pass", "unchanged_fail"


@dataclass(frozen=True)
class SuiteDiff:
    """Aggregated diff summary for CI regression gating across two suite JSON artifacts."""

    suite_name: str
    pass_rate_delta: float
    avg_score_delta: float
    p95_latency_delta_ms: float
    total_cost_delta_usd: float
    regressed_cases: tuple[str, ...]
    improved_cases: tuple[str, ...]
    case_deltas: tuple[CaseDelta, ...]

    @property
    def has_regression(self) -> bool:
        return bool(self.regressed_cases) or self.pass_rate_delta < -1e-9


def compare_suite_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    score_regression_threshold: float = 0.05,
) -> SuiteDiff:
    base_cases = {c["case_id"]: c for c in baseline.get("cases", [])}
    cand_cases = {c["case_id"]: c for c in candidate.get("cases", [])}
    all_ids = list(dict.fromkeys([*base_cases.keys(), *cand_cases.keys()]))

    deltas: list[CaseDelta] = []
    regressed: list[str] = []
    improved: list[str] = []

    for cid in all_ids:
        b = base_cases.get(cid)
        c = cand_cases.get(cid)
        if b is None or c is None:
            continue

        b_pass = bool(b["passed"])
        c_pass = bool(c["passed"])
        score_delta = round(float(c["weighted_score"]) - float(b["weighted_score"]), 4)
        lat_delta = round(float(c["total_latency_ms"]) - float(b["total_latency_ms"]), 2)
        cost_delta = round(float(c.get("estimated_cost_usd", 0.0)) - float(b.get("estimated_cost_usd", 0.0)), 6)

        if (b_pass and not c_pass) or score_delta <= -score_regression_threshold:
            status = "regressed"
            regressed.append(cid)
        elif (not b_pass and c_pass) or score_delta >= score_regression_threshold:
            status = "improved"
            improved.append(cid)
        else:
            status = "unchanged_pass" if c_pass else "unchanged_fail"

        deltas.append(
            CaseDelta(
                case_id=cid,
                baseline_passed=b_pass,
                candidate_passed=c_pass,
                score_delta=score_delta,
                latency_delta_ms=lat_delta,
                cost_delta_usd=cost_delta,
                status=status,
            )
        )

    return SuiteDiff(
        suite_name=str(candidate.get("suite_name", baseline.get("suite_name", "suite"))),
        pass_rate_delta=round(float(candidate["pass_rate"]) - float(baseline["pass_rate"]), 4),
        avg_score_delta=round(float(candidate["average_score"]) - float(baseline["average_score"]), 4),
        p95_latency_delta_ms=round(float(candidate["p95_latency_ms"]) - float(baseline["p95_latency_ms"]), 2),
        total_cost_delta_usd=round(
            float(candidate.get("total_cost_usd", 0.0)) - float(baseline.get("total_cost_usd", 0.0)),
            6,
        ),
        regressed_cases=tuple(regressed),
        improved_cases=tuple(improved),
        case_deltas=tuple(deltas),
    )


def render_diff_markdown(diff: SuiteDiff) -> str:
    verdict = "REGRESSION DETECTED" if diff.has_regression else "NO REGRESSIONS"
    lines = [
        f"# Regression Diff — `{diff.suite_name}` ({verdict})",
        "",
        f"- **Pass Rate Delta:** `{diff.pass_rate_delta * 100:+.1f}%`",
        f"- **Average Score Delta:** `{diff.avg_score_delta * 100:+.1f}%`",
        f"- **P95 Latency Delta:** `{diff.p95_latency_delta_ms:+.0f} ms`",
        f"- **Total Cost Delta:** `${diff.total_cost_delta_usd:+.6f}`",
        "",
        "| Case ID | Status | Score Delta | Latency Delta | Cost Delta |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for d in diff.case_deltas:
        lines.append(
            f"| `{d.case_id}` | **{d.status.upper()}** | `{d.score_delta * 100:+.1f}%` | "
            f"`{d.latency_delta_ms:+.0f} ms` | `${d.cost_delta_usd:+.6f}` |"
        )
    return "\n".join(lines) + "\n"
