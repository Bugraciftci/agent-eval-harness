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
class ConversationTurn:
    """Single turn inside a multi-turn agentic trajectory."""

    role: str  # "user" | "assistant" | "tool"
    content: str
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class AgentTrace:
    """Captured execution trace of a single or multi-turn agent run."""

    case_id: str
    final_output: str
    tool_calls: tuple[ToolCall, ...] = ()
    turns: tuple[ConversationTurn, ...] = ()
    model_name: str = "gemini-2.5-flash"
    total_latency_ms: float = 0.0
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def all_tool_calls(self) -> tuple[ToolCall, ...]:
        if self.tool_calls:
            return self.tool_calls
        collected: list[ToolCall] = []
        for turn in self.turns:
            collected.extend(turn.tool_calls)
        return tuple(collected)


@dataclass(frozen=True)
class EvalCase:
    """Benchmark specification for an agentic task."""

    case_id: str
    prompt: str
    expected_tools: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    ordered_tools: bool = False
    required_arg_keys: dict[str, tuple[str, ...]] = field(default_factory=dict)
    max_latency_ms: float = 4000.0
    max_tokens: int = 4000
    max_cost_usd: float | None = None
    max_repeated_calls: int = 2
    forbid_pii: bool = True
    expected_substrings: tuple[str, ...] = ()
    expected_json_keys: tuple[str, ...] = ()
    rubric_criteria: tuple[str, ...] = ()


@dataclass(frozen=True)
class MetricResult:
    """Outcome of a single deterministic or rubric evaluator."""

    evaluator: str
    passed: bool
    score: float
    weight: float = 1.0
    detail: str = ""


@dataclass(frozen=True)
class CaseReport:
    """Aggregated evaluation report for a single benchmark case."""

    case_id: str
    model_name: str
    passed: bool
    weighted_score: float
    total_latency_ms: float
    total_tokens: int
    estimated_cost_usd: float
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
    total_cost_usd: float
    cases: tuple[CaseReport, ...]
