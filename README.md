# agent-eval-harness

[![Python](https://img.shields.io/badge/Python_3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Zero Dependencies](https://img.shields.io/badge/Core_Deps-Zero_Stdlib_Only-22C55E?style=for-the-badge)](./pyproject.toml)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](./Dockerfile)
[![License: MIT](https://img.shields.io/badge/License-MIT-38BDF8?style=for-the-badge)](./LICENSE)

Lightweight, deterministic **async Python evaluation & reliability harness** for LLM tool-calling and Model Context Protocol (MCP) agents. Validates tool sequences, argument schemas, latency/token SLAs, retry loops, and PII/secret leakage — with zero external runtime dependencies.

## Architecture

```mermaid
flowchart LR
    S[JSON Benchmark Suite<br/>EvalCase Specs] --> R[Async EvalRunner<br/>asyncio.Semaphore Concurrency]
    R --> A[AgentAdapter Protocol<br/>Live LLM / MCP / Replay]
    A --> T[AgentTrace<br/>Tools · Latency · Tokens · Output]
    T --> E[Deterministic Evaluators<br/>Sequence · Args · SLA · Loop · PII]
    E --> REP[Reporter & CI Gate<br/>Markdown / JSON / --min-pass-rate]
```

## Built-In Deterministic Evaluators

| Evaluator | Weight | What It Verifies |
| :--- | :---: | :--- |
| `ToolSequenceEvaluator` | `2.0` | Required tool/MCP invocations (ordered subsequence or set match) |
| `ArgumentSchemaEvaluator` | `1.5` | Non-empty required argument keys passed into each tool call |
| `LatencyBudgetEvaluator` | `1.0` | End-to-end execution latency within case SLA (`max_latency_ms`) |
| `TokenBudgetEvaluator` | `1.0` | Prompt + completion token cost within ceiling (`max_tokens`) |
| `LoopGuardEvaluator` | `1.5` | Detects infinite retry loops with identical tool + argument signatures |
| `SafetyPIIGuardEvaluator` | `2.0` | Flags unredacted emails, card numbers, and API keys (`sk-`, `ghp_`, `AKIA`) |
| `OutputContentEvaluator` | `1.0` | Verifies required domain phrases in the final agent output |

## Quick Start

```bash
git clone https://github.com/Bugraciftci/agent-eval-harness.git
cd agent-eval-harness

# Run benchmark suite (Markdown report)
PYTHONPATH=src python3 -m agent_eval.cli run \
  --suite examples/mcp_tool_suite.json \
  --format markdown

# Enforce a CI pass-rate gate (exits non-zero if pass_rate < 0.80)
PYTHONPATH=src python3 -m agent_eval.cli run \
  --suite examples/mcp_tool_suite.json \
  --format json \
  --min-pass-rate 0.80
```

### Run with Docker

```bash
docker build -t agent-eval-harness .
docker run --rm agent-eval-harness
```

### Run Tests

```bash
python3 -m unittest discover -s tests -v
```

## License

MIT — see [LICENSE](./LICENSE).
