# Changelog

## [2.0.0] - 2026-11-27

### Added
- 14 built-in evaluators including adversarial MCP security guards (`PromptInjectionGuardEvaluator`, `PathTraversalAndCommandInjectionEvaluator`, `DataExfiltrationGuardEvaluator`).
- Zero-dependency async `EvalRunner` with prompt-cache cost modeling and self-contained dark SVG HTML dashboard reports.
- Adversarial edge-case unit tests and 250-case concurrent throughput benchmark (`tests/test_edge_and_benchmark.py`).
- GitHub Actions CI (Python 3.11/3.12 + Docker build) and live GitHub Pages evaluation report deployment.
