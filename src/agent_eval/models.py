from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """Single tool or MCP invocation emitted during an agent trajectory."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    succeeded: bool = True


@dataclass(frozen=True)
class AgentTrace:
    """Captured execution trace of an agent run."""

    case_id: str
    final_output: str
    tool_calls: tuple[ToolCall, ...] = ()
    total_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class EvalCase:
    """Benchmark specification for a single agentic task."""

    case_id: str
    prompt: str
    expected_tools: tuple[str, ...] = ()
    ordered_tools: bool = False
    required_arg_keys: dict[str, tuple[str, ...]] = field(default_factory=dict)
    max_latency_ms: float = 4000.0
    max_tokens: int = 4000
    max_repeated_calls: int = 2
    forbid_pii: bool = True
    expected_substrings: tuple[str, ...] = ()


@dataclass(frozen=True)
class MetricResult:
    """Outcome of a single deterministic evaluator."""

    evaluator: str
    passed: bool
    score: float
    weight: float = 1.0
    detail: str = ""


@dataclass(frozen=True)
class CaseReport:
    """Aggregated evaluation report for a single benchmark case."""

    case_id: str
    passed: bool
    weighted_score: float
    total_latency_ms: float
    total_tokens: int
    metrics: tuple[MetricResult, ...]


@dataclass(frozen=True)
class SuiteReport:
    """Aggregated benchmark summary across an evaluation suite."""

    suite_name: str
    total_cases: int
    passed_cases: int
    pass_rate: float
    average_score: float
    p95_latency_ms: float
    total_tokens: int
    cases: tuple[CaseReport, ...]
