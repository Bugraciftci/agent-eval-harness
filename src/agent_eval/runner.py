from __future__ import annotations

import asyncio
import math
from typing import Protocol, Sequence

from .evaluators import Evaluator, default_evaluators
from .models import AgentTrace, CaseReport, EvalCase, SuiteReport


class AgentAdapter(Protocol):
    """Async adapter executing an EvalCase against any LLM/MCP agent backend."""

    async def execute(self, case: EvalCase) -> AgentTrace:
        ...


class ReplayAgentAdapter:
    """Deterministic offline adapter replaying captured agent trajectories for CI regression testing."""

    def __init__(self, traces: dict[str, AgentTrace], simulated_delay_s: float = 0.0) -> None:
        self._traces = traces
        self._simulated_delay_s = simulated_delay_s

    async def execute(self, case: EvalCase) -> AgentTrace:
        if self._simulated_delay_s > 0:
            await asyncio.sleep(self._simulated_delay_s)
        if case.case_id not in self._traces:
            raise KeyError(f"No replay trace found for case '{case.case_id}'")
        return self._traces[case.case_id]


class EvalRunner:
    """Concurrent asyncio evaluation runner producing aggregated benchmark suite reports."""

    def __init__(
        self,
        adapter: AgentAdapter,
        evaluators: Sequence[Evaluator] | None = None,
        max_concurrency: int = 8,
    ) -> None:
        self._adapter = adapter
        self._evaluators = list(evaluators) if evaluators is not None else default_evaluators()
        self._max_concurrency = max(1, max_concurrency)

    async def run_case(self, case: EvalCase) -> CaseReport:
        trace = await self._adapter.execute(case)
        metrics = tuple(evaluator.evaluate(case, trace) for evaluator in self._evaluators)
        total_weight = sum(m.weight for m in metrics) or 1.0
        weighted_score = sum(m.score * m.weight for m in metrics) / total_weight
        passed = all(m.passed for m in metrics)
        return CaseReport(
            case_id=case.case_id,
            passed=passed,
            weighted_score=round(weighted_score, 4),
            total_latency_ms=trace.total_latency_ms,
            total_tokens=trace.total_tokens,
            metrics=metrics,
        )

    async def run_suite(self, suite_name: str, cases: Sequence[EvalCase]) -> SuiteReport:
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def _guarded(case: EvalCase) -> CaseReport:
            async with semaphore:
                return await self.run_case(case)

        reports = tuple(await asyncio.gather(*(_guarded(case) for case in cases)))
        total_cases = len(reports)
        passed_cases = sum(1 for r in reports if r.passed)
        pass_rate = round((passed_cases / total_cases) if total_cases else 0.0, 4)
        avg_score = round((sum(r.weighted_score for r in reports) / total_cases) if total_cases else 0.0, 4)
        latencies = sorted(r.total_latency_ms for r in reports)
        p95 = _percentile(latencies, 0.95)
        total_tokens = sum(r.total_tokens for r in reports)

        return SuiteReport(
            suite_name=suite_name,
            total_cases=total_cases,
            passed_cases=passed_cases,
            pass_rate=pass_rate,
            average_score=avg_score,
            p95_latency_ms=round(p95, 2),
            total_tokens=total_tokens,
            cases=reports,
        )


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = max(0, math.ceil(p * len(sorted_values)) - 1)
    return sorted_values[min(idx, len(sorted_values) - 1)]
