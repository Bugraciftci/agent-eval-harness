<div align="center">
  <img src="./assets/banner.svg" width="100%" alt="agent-eval-harness v2.0 Banner" />
</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python_3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Core_Deps-Zero_Stdlib_Only-22C55E?style=for-the-badge" alt="Zero Deps" />
  <img src="https://img.shields.io/badge/Evaluators-14_Built--In-38BDF8?style=for-the-badge" alt="Evaluators" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
</p>

Deterministic **async Python evaluation, cost modeling, and regression-diff studio** for LLM tool-calling and Model Context Protocol (MCP) agents — with **zero external runtime dependencies**.

## Architecture

```mermaid
flowchart LR
    S["JSON Benchmark Suite<br/>Single & Multi-Turn Cases"] --> R["Async EvalRunner<br/>asyncio.Semaphore Concurrency"]
    R --> A["AgentAdapter Protocol<br/>Live LLM · MCP · Offline Replay"]
    A --> T["AgentTrace + Cost Engine<br/>Tools · Turns · Latency · Tokens · USD"]
    T --> E["14 Evaluators<br/>Sequence · Hallucination · Args · SLA · Cost · Loop · PII · JSON · Judge · Prompt-Injection · Path/Cmd · Exfiltration"]
    E --> REP["Reporters & CI Gate<br/>Markdown · JSON · Self-Contained HTML SVG"]
    REP --> DIFF["Regression Diff Engine<br/>agent-eval diff --fail-on-regression"]
```

## 14 Built-In Evaluators

| Evaluator | Weight | What It Verifies |
| :--- | :---: | :--- |
| `PathTraversalAndCommandInjectionEvaluator` | `2.5` | Blocks `../`, `/etc/passwd`, `~/.ssh/`, and injected shell commands in MCP arguments (`adversarial.py`) |
| `PromptInjectionGuardEvaluator` | `2.0` | Detects indirect prompt-injection compliance markers in tool arguments and outputs (`adversarial.py`) |
| `DataExfiltrationGuardEvaluator` | `2.0` | Flags unauthorized outbound HTTP/webhook hosts in tool call payloads (`adversarial.py`) |
| `ToolSequenceEvaluator` | `2.0` | Ordered or unordered expected tool/MCP call trajectories |
| `HallucinatedToolEvaluator` | `2.0` | Rejects unauthorized or hallucinated tools outside the MCP manifest |
| `SafetyPIIGuardEvaluator` | `2.0` | Flags unredacted emails, credit cards, and API keys (`sk-`, `ghp_`, `AKIA`) |
| `ArgumentSchemaEvaluator` | `1.5` | Non-empty required argument keys passed into tool invocations |
| `LoopGuardEvaluator` | `1.5` | Detects infinite retry loops with identical tool + argument signatures |
| `JSONSchemaOutputEvaluator` | `1.5` | Validates structured JSON root objects and required keys |
| `LLMJudgeEvaluator` | `1.5` | Pluggable `JudgeModelProtocol` for rubric scoring (offline heuristic judge included) |
| `LatencyBudgetEvaluator` | `1.0` | End-to-end latency within SLA (`max_latency_ms`) |
| `TokenBudgetEvaluator` | `1.0` | Prompt + completion token usage within ceiling (`max_tokens`) |
| `CostBudgetEvaluator` | `1.0` | Estimated USD cost (including prompt-cache discounts) within `max_cost_usd` |
| `OutputContentEvaluator` | `1.0` | Required domain assertions in the agent's final output |

## Quick Start

```bash
git clone https://github.com/Bugraciftci/agent-eval-harness.git
cd agent-eval-harness

# 1. Run 8-case MCP benchmark suite -> Markdown report
PYTHONPATH=src python3 -m agent_eval.cli run \
  --suite examples/mcp_tool_suite.json \
  --format markdown

# 2. Export a self-contained dark-mode HTML dashboard with adversarial MCP security guards enabled
PYTHONPATH=src python3 -m agent_eval.cli run \
  --suite examples/mcp_tool_suite.json \
  --security \
  --format html \
  --out eval_report.html

# 3. Compare baseline vs. candidate runs in CI
PYTHONPATH=src python3 -m agent_eval.cli diff \
  --baseline baseline.json \
  --candidate candidate.json \
  --fail-on-regression
```

### Run Tests

```bash
python3 -m unittest discover -s tests -v
```

### Run with Docker

```bash
docker build -t agent-eval-harness .
docker run --rm agent-eval-harness
```

## License

MIT — see [LICENSE](./LICENSE).
