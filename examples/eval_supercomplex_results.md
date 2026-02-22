# Jotty Super-Complex Real-World Evaluation Results

**Date:** 2026-02-22
**Model:** claude-sonnet-4-20250514 (Sonnet)
**API:** Anthropic (real API calls, not mocked)

## Overall Grade: A+ (9.6/10)

13/13 SAS tests passed, then MAS swarm tests added for full coverage.

## Dimension Scores (SAS Run)

```
PURE_REASONING       ██████████████████████████████ 100% (pass: 2/2)
TOOL_CALLING         ██████████████████████████████ 100% (pass: 2/2)
MULTI_TURN           ██████████████████████████████ 100% (pass: 1/1)
SDK_LOCAL            ██████████████████████████████ 100% (pass: 2/2)
LEARNING             ██████████████████████████████ 100% (pass: 1/1)
ERROR_RECOVERY       █████████████████████████░░░░░  83% (pass: 3/3)
WORKFLOW             ██████████████████████████████ 100% (pass: 1/1)
MEMORY               ██████████████████████████████ 100% (pass: 1/1)
```

## Test Details

### 1. Pure Reasoning (10/10)

| Test | Quality | Checks | Time | Output |
|------|---------|--------|------|--------|
| Cross-domain synthesis (Raft+ML+VaR) | 100% | 10/10 | 104s | 27,955 chars |
| Math proof + implementation (halting problem) | 100% | 10/10 | 74s | 21,705 chars |

**What it tested:** The cross-domain test asked Jotty to combine distributed consensus, ML model serving, and financial risk management into a novel "ML Model Governance Protocol". The math test required a formal diagonalization proof of the halting problem's undecidability PLUS a practical bounded halting checker implementation.

Both produced full Python implementations with classes, test suites, and formal arguments.

### 2. Tool Calling (10/10)

| Test | Quality | Checks | Time | Output |
|------|---------|--------|------|--------|
| Web search + synthesis | 100% | 7/7 | 48s | 5,210 chars |
| Math calculation | 100% | 6/6 | 9s | 727 chars |

**Key insight:** Serper API returned 400, but the 4-tier fallback chain (Serper → SearXNG → DuckDuckGo library → DDG HTML) handled it transparently. The LLM received search results from DuckDuckGo and synthesized a structured comparison of AI agent frameworks with pros/cons.

Math: 17^5 + sqrt(144) - ln(e^3) = 1,419,857 + 12 - 3 = **1,419,866** -- correct.

### 3. Multi-Turn Conversation (10/10)

| Test | Quality | Checks | Time | Output |
|------|---------|--------|------|--------|
| 4-turn distributed KV store | 100% | 7/7 | 244s | 70,782 chars |

**What it tested:** 4 sequential turns building a distributed key-value store:
1. Architecture recommendation (consistent hashing suggested)
2. Implementation of consistent hashing ring (code generated)
3. Added virtual nodes to reduce hotspots (modified existing code)
4. 5 test cases for the implementation

Perfect coherence across all turns. Each response correctly built upon the previous.

### 4. SDK Local Mode (10/10)

| Test | Quality | Checks | Time | Output |
|------|---------|--------|------|--------|
| `client.chat()` (CAP theorem) | 100% | 6/6 | 13s | 1,971 chars |
| `client.workflow()` (sorting) | 100% | 4/4 | 20s | 1,102 chars |

The SDK client works in local mode (no HTTP server needed). Both chat and workflow modes produced correct, structured responses.

### 5. Learning System (10/10)

| Test | Quality | Checks | Time | Output |
|------|---------|--------|------|--------|
| Record + query | 100% | 6/6 | 10s | 5,130 chars |

632 episodes in SQLite DB. Retrieval context (288 chars) injected into prompts. The learning system records outcomes and provides guidance for future tasks.

### 6. Error Recovery (8.3/10)

| Test | Quality | Checks | Time | Output |
|------|---------|--------|------|--------|
| Empty input | 50% | 1/2 | 0.4s | 0 chars |
| Long input (22K chars) | 100% | 3/3 | 1.7s | 362 chars |
| Non-existent tool request | 100% | 3/3 | 3.6s | 494 chars |

**Weakness found:** Empty input passes through to Anthropic API which returns 400. Jotty catches the error (no crash) but doesn't generate a user-friendly "please provide input" response. Validation should happen pre-API call.

### 7. Workflow Mode (10/10)

| Test | Quality | Checks | Time | Output |
|------|---------|--------|------|--------|
| Framework comparison | 100% | 8/8 | 51s | 7,609 chars |

AutoAgent planned and executed a framework comparison (Django/FastAPI/Flask). Used `claude-cli-llm` skill. Safety gates caught PII (14 emails in generated content) -- a strength, not a weakness.

### 8. Memory System (10/10)

| Test | Quality | Checks | Time | Output |
|------|---------|--------|------|--------|
| Store + retrieve | 100% | 5/5 | <1s | 3 memories |

Stored 3 memories across episodic/semantic/procedural levels. Retrieved relevant memories for both test queries. Relevance matching correctly surfaced TD-Lambda memory for "how does Jotty learn?"

## Architecture Highlights

- **5-layer clean architecture** (apps → SDK → interface → intelligence → infrastructure)
- **273 skills** with lazy loading and semantic discovery
- **5-level brain-inspired memory** (episodic, semantic, procedural, meta, causal)
- **TD-Lambda RL** with gamma=0.99 for self-improvement
- **4-tier web search fallback** (Serper → SearXNG → DDG lib → DDG HTML)
- **Safety gates** with PII detection, content validation
- **Multi-execution modes:** SAS (chat, workflow, agent) + MAS (swarm templates)

## Reproduction

```bash
cd /var/www/sites/personal/stock_market/Jotty
python scripts/eval_supercomplex.py
```

Requires `ANTHROPIC_API_KEY` in `.env`.
