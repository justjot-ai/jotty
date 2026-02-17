# ADR: Facade Singletons vs SwarmResources

**Status:** Accepted
**Date:** 2026-02-17
**Context:** Deep integration analysis flagged "facades not used by orchestration" as a potential gap.

## Decision

Two resource access patterns coexist intentionally:

| Pattern | Purpose | Consumers | Lifecycle |
|---------|---------|-----------|-----------|
| **Facades** (`memory/facade.py`, `learning/facade.py`, etc.) | External API for SDK and applications | SDK, CLI, API server, standalone scripts | Process-wide singletons |
| **SwarmResources** (`planners/swarm_resources_stub.py`) | Internal singleton for orchestration subsystem | SwarmLearning, SwarmTemplate, domain swarms | Per-SwarmConfig singleton |

## Why Two Patterns?

### Facades (External API)
- Provide zero-config entry points: `get_memory_system()`, `get_td_lambda()`
- Used by SDK (`from Jotty.core.memory import get_memory_system`)
- Create standalone instances with default configuration
- Thread-safe via `_lock + _singletons` double-checked locking
- Suitable for apps that don't need swarm coordination

### SwarmResources (Internal)
- Ensures all agents within a swarm execution share the **same** memory, context, bus, and learner instances
- Keyed by `SwarmConfig` — different configs produce different resource sets
- Initialized by `SwarmLearning._init_shared_resources()`
- Injected into agents via `SwarmTemplate._create_agent()` (lines 304-306)
- Guarantees single-instance within a swarm execution graph

## When to Use Which

| Scenario | Use |
|----------|-----|
| Standalone agent (no swarm) | Facade — `get_memory_system()` |
| SDK client application | Facade — via `Jotty()` client |
| Agent inside a swarm | SwarmResources — injected automatically |
| Custom orchestration code | SwarmResources — via `SwarmResources.get_instance(config)` |
| Unit tests (isolated) | Direct construction or mocks |

## Consequences

- Agents created standalone get their own independent resources (via BaseAgent lazy properties)
- Agents created inside swarms share injected resources (via sentinel pattern, Step 2)
- The two patterns never conflict because swarm injection replaces the sentinel before lazy creation triggers
- If a future refactoring unifies these, the facade layer would become a thin wrapper over SwarmResources
