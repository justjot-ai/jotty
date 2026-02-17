# Jotty Execution — Optimization Scope

**Full flow and optimization details:** see **`JOTTY_EXECUTION_FLOWCHART.md`** (flowchart + section 11 “Optimization”).

---

## Implemented

| Item | What | Where |
|------|------|--------|
| **Pass gate decision to AgentRunner** | One `ValidationGate.decide()` per run; result passed in `kwargs["gate_decision"]`; runner reuses it. | `swarm_manager.py` (set `kwargs["gate_decision"]`), `agent_runner.py` (_setup_context: pop and use or call decide) |
| **Cache ValidationGate.decide()** | Cache by `(goal[:300], agent_name)` with TTL (default 60s); max 500 entries. | `validation_gate.py` (`cache_ttl_seconds`, `enable_cache`, `_cache`) |
| **Optional / shorter learning wait** | `learning_wait_timeout_seconds` in config; 0 = skip wait. | `SwarmLearningConfig.learning_wait_timeout_seconds`, `swarm_manager.py` ExecutionEngine.run() |

The runner uses the **best part** of ValidationGate (single LLM classification, safety rails, drift/sampling) by reusing the engine’s decision instead of calling `decide()` again.

---

## Remaining (future)

- Unify intent/tier (TierExecutor) with ValidationGate (one classification for both paths).
- Async LM on ExecutionEngine fast path.
- Parallelize independent tool calls in Tier 1 tool loop.
