import asyncio
import json
import os
import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from agent_eval import (
    AgentTrace,
    ArgumentSchemaEvaluator,
    EvalCase,
    EvalRunner,
    LatencyBudgetEvaluator,
    LoopGuardEvaluator,
    ReplayAgentAdapter,
    SafetyPIIGuardEvaluator,
    TokenBudgetEvaluator,
    ToolCall,
    ToolSequenceEvaluator,
    render_json_report,
    render_markdown_report,
)
from agent_eval.cli import main as cli_main


class TestEvaluators(unittest.TestCase):
    def test_ordered_tool_sequence_pass_and_fail(self):
        ev = ToolSequenceEvaluator()
        case = EvalCase(
            case_id="seq",
            prompt="Run pipeline",
            expected_tools=("fetch", "transform", "publish"),
            ordered_tools=True,
        )
        ok_trace = AgentTrace(
            case_id="seq",
            final_output="done",
            tool_calls=(ToolCall("fetch"), ToolCall("log"), ToolCall("transform"), ToolCall("publish")),
        )
        bad_trace = AgentTrace(
            case_id="seq",
            final_output="done",
            tool_calls=(ToolCall("publish"), ToolCall("fetch")),
        )

        self.assertTrue(ev.evaluate(case, ok_trace).passed)
        res_bad = ev.evaluate(case, bad_trace)
        self.assertFalse(res_bad.passed)
        self.assertLess(res_bad.score, 1.0)

    def test_argument_schema_evaluator(self):
        ev = ArgumentSchemaEvaluator()
        case = EvalCase(
            case_id="args",
            prompt="Query DB",
            required_arg_keys={"sql_query": ("table", "limit")},
        )
        missing_trace = AgentTrace(
            case_id="args",
            final_output="ok",
            tool_calls=(ToolCall("sql_query", {"table": "orders"}),),
        )
        res = ev.evaluate(case, missing_trace)
        self.assertFalse(res.passed)
        self.assertEqual(res.score, 0.5)
        self.assertIn("sql_query.limit", res.detail)

    def test_latency_and_token_budgets(self):
        case = EvalCase(case_id="sla", prompt="p", max_latency_ms=1000, max_tokens=500)
        over_trace = AgentTrace(
            case_id="sla",
            final_output="ok",
            total_latency_ms=1500,
            prompt_tokens=400,
            completion_tokens=300,
        )
        self.assertFalse(LatencyBudgetEvaluator().evaluate(case, over_trace).passed)
        self.assertFalse(TokenBudgetEvaluator().evaluate(case, over_trace).passed)

    def test_loop_guard_detects_repeated_identical_calls(self):
        ev = LoopGuardEvaluator()
        case = EvalCase(case_id="loop", prompt="p", max_repeated_calls=2)
        loop_trace = AgentTrace(
            case_id="loop",
            final_output="stuck",
            tool_calls=(
                ToolCall("search", {"q": "status"}),
                ToolCall("search", {"q": "status"}),
                ToolCall("search", {"q": "status"}),
            ),
        )
        res = ev.evaluate(case, loop_trace)
        self.assertFalse(res.passed)
        self.assertEqual(res.score, 0.0)

    def test_safety_pii_guard_catches_email_and_api_keys(self):
        ev = SafetyPIIGuardEvaluator()
        case = EvalCase(case_id="pii", prompt="p", forbid_pii=True)
        leaky_trace = AgentTrace(
            case_id="pii",
            final_output="Contact user@domain.com with key sk-1234567890abcdef1234",
        )
        res = ev.evaluate(case, leaky_trace)
        self.assertFalse(res.passed)
        self.assertIn("email", res.detail)
        self.assertIn("api_key", res.detail)


class TestRunnerAndCLI(unittest.TestCase):
    def test_async_suite_runner_and_reporters(self):
        cases = [
            EvalCase(
                case_id="c1",
                prompt="p1",
                expected_tools=("search",),
                max_latency_ms=1000,
                expected_substrings=("summary",),
            ),
            EvalCase(
                case_id="c2",
                prompt="p2",
                expected_tools=("deploy",),
                max_latency_ms=500,
            ),
        ]
        traces = {
            "c1": AgentTrace(
                case_id="c1",
                final_output="Clean summary ready.",
                tool_calls=(ToolCall("search", {"q": "docs"}),),
                total_latency_ms=300,
                prompt_tokens=200,
                completion_tokens=100,
            ),
            "c2": AgentTrace(
                case_id="c2",
                final_output="Deployed slowly.",
                tool_calls=(ToolCall("deploy"),),
                total_latency_ms=900,
                prompt_tokens=250,
                completion_tokens=150,
            ),
        }

        runner = EvalRunner(adapter=ReplayAgentAdapter(traces), max_concurrency=2)
        report = asyncio.run(runner.run_suite("unit-suite", cases))

        self.assertEqual(report.total_cases, 2)
        self.assertEqual(report.passed_cases, 1)
        self.assertEqual(report.pass_rate, 0.5)
        self.assertEqual(report.total_tokens, 700)

        md = render_markdown_report(report)
        self.assertIn("# Agent Evaluation Report — `unit-suite`", md)
        self.assertIn("| `c1` | **PASS** |", md)

        parsed = json.loads(render_json_report(report))
        self.assertEqual(parsed["suite_name"], "unit-suite")

    def test_cli_runs_example_suite_with_gate(self):
        suite_path = Path(__file__).resolve().parent.parent / "examples" / "mcp_tool_suite.json"
        buf = StringIO()
        with patch("sys.stdout", buf):
            rc_pass = cli_main(["run", "--suite", str(suite_path), "--format", "json", "--min-pass-rate", "0.5"])
        self.assertEqual(rc_pass, 0)
        data = json.loads(buf.getvalue())
        self.assertEqual(data["total_cases"], 4)
        self.assertEqual(data["passed_cases"], 2)

        buf2 = StringIO()
        with patch("sys.stdout", buf2):
            rc_fail = cli_main(["run", "--suite", str(suite_path), "--format", "markdown", "--min-pass-rate", "0.9"])
        self.assertEqual(rc_fail, 1)


if __name__ == "__main__":
    unittest.main()
