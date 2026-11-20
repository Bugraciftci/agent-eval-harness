from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Protocol

from .models import AgentTrace, EvalCase, MetricResult
from .pricing import estimate_cost


class Evaluator(ABC):
    """Base contract for deterministic trajectory evaluators."""

    name: str = "base"
    weight: float = 1.0

    @abstractmethod
    def evaluate(self, case: EvalCase, trace: AgentTrace) -> MetricResult:
        raise NotImplementedError


class ToolSequenceEvaluator(Evaluator):
    """Verifies expected MCP/tool invocations were called (ordered or unordered)."""

    name = "tool_sequence"
    weight = 2.0

    def evaluate(self, case: EvalCase, trace: AgentTrace) -> MetricResult:
        if not case.expected_tools:
            return MetricResult(self.name, True, 1.0, self.weight, "No required tools specified")

        actual = [call.name for call in trace.all_tool_calls]
        expected = list(case.expected_tools)

        if case.ordered_tools:
            idx = 0
            for tool_name in actual:
                if idx < len(expected) and tool_name == expected[idx]:
                    idx += 1
            score = idx / len(expected)
            passed = idx == len(expected)
            detail = (
                "Ordered tool sequence matched"
                if passed
                else f"Matched {idx}/{len(expected)} ordered tools; actual={actual}"
            )
            return MetricResult(self.name, passed, round(score, 4), self.weight, detail)

        actual_set = set(actual)
        matched = [t for t in expected if t in actual_set]
        score = len(matched) / len(expected)
        passed = len(matched) == len(expected)
        missing = [t for t in expected if t not in actual_set]
        detail = "All expected tools invoked" if passed else f"Missing tools: {missing}"
        return MetricResult(self.name, passed, round(score, 4), self.weight, detail)


class HallucinatedToolEvaluator(Evaluator):
    """Flags invocations of tool names outside the declared MCP/tool manifest."""

    name = "hallucinated_tool_guard"
    weight = 2.0

    def evaluate(self, case: EvalCase, trace: AgentTrace) -> MetricResult:
        if not case.allowed_tools:
            return MetricResult(self.name, True, 1.0, self.weight, "No tool manifest restriction")

        allowed = set(case.allowed_tools)
        unknown = sorted({c.name for c in trace.all_tool_calls if c.name not in allowed})
        if not unknown:
            return MetricResult(self.name, True, 1.0, self.weight, "All invoked tools exist in manifest")
        return MetricResult(
            self.name,
            False,
            0.0,
            self.weight,
            f"Hallucinated / unauthorized tool(s): {', '.join(unknown)}",
        )


class ArgumentSchemaEvaluator(Evaluator):
    """Checks that required argument keys are present and non-empty in tool calls."""

    name = "argument_schema"
    weight = 1.5

    def evaluate(self, case: EvalCase, trace: AgentTrace) -> MetricResult:
        if not case.required_arg_keys:
            return MetricResult(self.name, True, 1.0, self.weight, "No argument constraints")

        total_checks = 0
        passed_checks = 0
        violations: list[str] = []

        calls = trace.all_tool_calls
        for tool_name, required_keys in case.required_arg_keys.items():
            matching_calls = [c for c in calls if c.name == tool_name]
            for key in required_keys:
                total_checks += 1
                if any(key in call.arguments and call.arguments[key] not in (None, "") for call in matching_calls):
                    passed_checks += 1
                else:
                    violations.append(f"{tool_name}.{key}")

        score = passed_checks / total_checks if total_checks else 1.0
        passed = not violations
        detail = "All required tool arguments valid" if passed else f"Missing args: {', '.join(violations)}"
        return MetricResult(self.name, passed, round(score, 4), self.weight, detail)


class LatencyBudgetEvaluator(Evaluator):
    """Asserts end-to-end agent latency stays within the case SLA budget."""

    name = "latency_sla"
    weight = 1.0

    def evaluate(self, case: EvalCase, trace: AgentTrace) -> MetricResult:
        if trace.total_latency_ms <= case.max_latency_ms:
            return MetricResult(
                self.name,
                True,
                1.0,
                self.weight,
                f"{trace.total_latency_ms:.0f}ms <= {case.max_latency_ms:.0f}ms SLA",
            )
        ratio = max(0.0, 1.0 - ((trace.total_latency_ms - case.max_latency_ms) / max(case.max_latency_ms, 1.0)))
        return MetricResult(
            self.name,
            False,
            round(ratio, 4),
            self.weight,
            f"Exceeded SLA: {trace.total_latency_ms:.0f}ms > {case.max_latency_ms:.0f}ms",
        )


class TokenBudgetEvaluator(Evaluator):
    """Asserts prompt + completion tokens stay within token ceiling."""

    name = "token_budget"
    weight = 1.0

    def evaluate(self, case: EvalCase, trace: AgentTrace) -> MetricResult:
        total = trace.total_tokens
        if total <= case.max_tokens:
            return MetricResult(self.name, True, 1.0, self.weight, f"{total} <= {case.max_tokens} tokens")
        ratio = max(0.0, 1.0 - ((total - case.max_tokens) / max(case.max_tokens, 1)))
        return MetricResult(
            self.name,
            False,
            round(ratio, 4),
            self.weight,
            f"Token budget exceeded: {total} > {case.max_tokens}",
        )


class CostBudgetEvaluator(Evaluator):
    """Enforces per-case estimated USD cost ceiling using the LLM pricing registry."""

    name = "cost_budget"
    weight = 1.0

    def evaluate(self, case: EvalCase, trace: AgentTrace) -> MetricResult:
        breakdown = estimate_cost(
            model_name=trace.model_name,
            prompt_tokens=trace.prompt_tokens,
            completion_tokens=trace.completion_tokens,
            cached_prompt_tokens=trace.cached_prompt_tokens,
        )
        if case.max_cost_usd is None:
            return MetricResult(
                self.name,
                True,
                1.0,
                self.weight,
                f"Estimated ${breakdown.total_cost_usd:.5f} (no ceiling)",
            )
        if breakdown.total_cost_usd <= case.max_cost_usd:
            return MetricResult(
                self.name,
                True,
                1.0,
                self.weight,
                f"${breakdown.total_cost_usd:.5f} <= ${case.max_cost_usd:.5f}",
            )
        ratio = max(0.0, 1.0 - ((breakdown.total_cost_usd - case.max_cost_usd) / max(case.max_cost_usd, 1e-6)))
        return MetricResult(
            self.name,
            False,
            round(ratio, 4),
            self.weight,
            f"Cost ceiling exceeded: ${breakdown.total_cost_usd:.5f} > ${case.max_cost_usd:.5f}",
        )


class LoopGuardEvaluator(Evaluator):
    """Detects infinite retry loops where the agent repeats identical tool calls."""

    name = "loop_guard"
    weight = 1.5

    def evaluate(self, case: EvalCase, trace: AgentTrace) -> MetricResult:
        signatures = [
            (call.name, tuple(sorted((k, str(v)) for k, v in call.arguments.items())))
            for call in trace.all_tool_calls
        ]
        counts = Counter(signatures)
        offenders = [(sig[0], cnt) for sig, cnt in counts.items() if cnt > case.max_repeated_calls]
        if not offenders:
            return MetricResult(self.name, True, 1.0, self.weight, "No tool retry loops detected")

        summary = ", ".join(f"{name} x{cnt}" for name, cnt in offenders)
        return MetricResult(
            self.name,
            False,
            0.0,
            self.weight,
            f"Loop detected (limit {case.max_repeated_calls}): {summary}",
        )


class SafetyPIIGuardEvaluator(Evaluator):
    """Scans final output for unredacted PII patterns (emails, card numbers, secrets)."""

    name = "safety_pii_guard"
    weight = 2.0

    _PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
        ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
        ("api_key", re.compile(r"\b(?:sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b")),
    )

    def evaluate(self, case: EvalCase, trace: AgentTrace) -> MetricResult:
        if not case.forbid_pii:
            return MetricResult(self.name, True, 1.0, self.weight, "PII guard disabled for case")

        findings = [label for label, pat in self._PATTERNS if pat.search(trace.final_output)]
        if not findings:
            return MetricResult(self.name, True, 1.0, self.weight, "No PII or secret leakage detected")
        return MetricResult(
            self.name,
            False,
            0.0,
            self.weight,
            f"Unredacted sensitive data detected: {', '.join(findings)}",
        )


class JSONSchemaOutputEvaluator(Evaluator):
    """Validates structured JSON outputs against required top-level schema keys."""

    name = "json_schema_output"
    weight = 1.5

    def evaluate(self, case: EvalCase, trace: AgentTrace) -> MetricResult:
        if not case.expected_json_keys:
            return MetricResult(self.name, True, 1.0, self.weight, "No JSON schema required")

        try:
            parsed = json.loads(trace.final_output)
        except json.JSONDecodeError as exc:
            return MetricResult(self.name, False, 0.0, self.weight, f"Invalid JSON output: {exc.msg}")

        if not isinstance(parsed, dict):
            return MetricResult(self.name, False, 0.0, self.weight, "Expected JSON object at root")

        missing = [k for k in case.expected_json_keys if k not in parsed or parsed[k] in (None, "")]
        matched = len(case.expected_json_keys) - len(missing)
        score = matched / len(case.expected_json_keys)
        passed = not missing
        detail = "JSON schema contract valid" if passed else f"Missing JSON keys: {missing}"
        return MetricResult(self.name, passed, round(score, 4), self.weight, detail)


class OutputContentEvaluator(Evaluator):
    """Checks required domain phrases or assertions in the agent's final response."""

    name = "output_assertions"
    weight = 1.0

    def evaluate(self, case: EvalCase, trace: AgentTrace) -> MetricResult:
        if not case.expected_substrings:
            return MetricResult(self.name, True, 1.0, self.weight, "No output substring assertions")

        lowered = trace.final_output.lower()
        missing = [s for s in case.expected_substrings if s.lower() not in lowered]
        matched = len(case.expected_substrings) - len(missing)
        score = matched / len(case.expected_substrings)
        passed = not missing
        detail = "All output assertions matched" if passed else f"Missing phrases: {missing}"
        return MetricResult(self.name, passed, round(score, 4), self.weight, detail)


class JudgeModelProtocol(Protocol):
    """Pluggable LLM-as-a-Judge protocol for semantic rubric scoring."""

    def score_criterion(self, prompt: str, output: str, criterion: str) -> tuple[float, str]:
        ...


class HeuristicRubricJudge:
    """Offline deterministic judge checking keyword grounding for CI-safe rubric verification."""

    def score_criterion(self, prompt: str, output: str, criterion: str) -> tuple[float, str]:
        tokens = [w for w in re.findall(r"[a-zA-Z0-9]{4,}", criterion.lower())]
        if not tokens:
            return 1.0, "Empty criterion"
        lowered = output.lower()
        hits = sum(1 for tok in tokens if tok in lowered)
        ratio = hits / len(tokens)
        return round(ratio, 4), f"Matched {hits}/{len(tokens)} rubric concepts"


class LLMJudgeEvaluator(Evaluator):
    """Evaluates semantic rubric criteria via an injected JudgeModelProtocol."""

    name = "rubric_judge"
    weight = 1.5

    def __init__(self, judge: JudgeModelProtocol | None = None, pass_threshold: float = 0.6) -> None:
        self._judge = judge or HeuristicRubricJudge()
        self._pass_threshold = pass_threshold

    def evaluate(self, case: EvalCase, trace: AgentTrace) -> MetricResult:
        if not case.rubric_criteria:
            return MetricResult(self.name, True, 1.0, self.weight, "No rubric criteria configured")

        scores: list[float] = []
        notes: list[str] = []
        for criterion in case.rubric_criteria:
            score, note = self._judge.score_criterion(case.prompt, trace.final_output, criterion)
            scores.append(score)
            notes.append(f"{criterion[:24]}={score:.2f}")

        avg = sum(scores) / len(scores)
        passed = avg >= self._pass_threshold
        return MetricResult(
            self.name,
            passed,
            round(avg, 4),
            self.weight,
            "; ".join(notes),
        )


def default_evaluators() -> list[Evaluator]:
    return [
        ToolSequenceEvaluator(),
        HallucinatedToolEvaluator(),
        ArgumentSchemaEvaluator(),
        LatencyBudgetEvaluator(),
        TokenBudgetEvaluator(),
        CostBudgetEvaluator(),
        LoopGuardEvaluator(),
        SafetyPIIGuardEvaluator(),
        JSONSchemaOutputEvaluator(),
        OutputContentEvaluator(),
        LLMJudgeEvaluator(),
    ]
