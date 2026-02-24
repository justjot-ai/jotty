# Learning Module Status

Last audited: 2026-02-24

## Module Status (by actual usage)

### ACTIVE — Core learning pipeline (reported via benchmarks; verify locally)

| Module | Lines | Status | Used by |
|--------|-------|--------|---------|
| `td_lambda.py` | ~2,400 | **Reported** | TDLambdaLearner, SkillQTable, StepQTable — central learning engine with convergence tracking |
| `crystallization.py` | ~520 | **Reported** | should_crystallize, crystallize, run_probation — graduation pipeline |
| `advanced_learning.py` | ~1,100 | **Reported** | DomainDSPyOptimizer (MIPROv2 + BootstrapRS), _gold_metric, Reflexion, VoyagerSkillLib |
| `learning_service.py` | ~3,000 | **Active** | Central service: episodes, Q-tables, distillation, DSPy integration |
| `learning_store.py` | ~800 | **Active** | SQLite persistence for episodes, lessons, Q-tables |
| `facade.py` | ~200 | **Active** | get_td_lambda, get_learning_service, get_reward_manager |

### SUPPORT — Internal utilities for active modules

| Module | Lines | Used by |
|--------|-------|---------|
| `adaptive_components.py` | 279 | td_lambda.py (AdaptiveLearningRate, IntermediateRewardCalculator) |
| `health_budget.py` | 340 | LearningHealthMonitor, DynamicBudgetManager |
| `base_classes.py` | 246 | Base dataclasses for learning types |
| `utils.py` | 513 | Shared utilities |

### INTEGRATED — Wired into production execution paths

Modules that were previously "aspirational" but are now reported to be called in
real execution paths. Confirmed by code tracing (grep for call sites).

| Module | Lines | Integration Point | How |
|--------|-------|-------------------|-----|
| `shaped_rewards.py` | 401 | `agent_runner.py:351` | `ShapedRewardManager` instantiated when learning enabled; `.check_rewards()` on every step; `.get_total_reward()` for final episode reward |
| `q_learning.py` | 1,791 | `swarm_manager.py:193` | `LLMQPredictor` lazy-created; `.predict_q_value()` in `swarm_roadmap.py:714` for ε-greedy task selection; buffer persisted via `vault.py` |
| `algorithmic_credit.py` | 869 | `learning_service.py:_update_skill_credits()` | Per-skill credit-weighted Q-table updates using cheap heuristic (step success). Full Shapley available for offline analysis. |

### DEAD CODE — Duplicates of reported modules

These modules contain valid algorithms but every capability they provide is
already covered by an active module. Marked with `STATUS: DEAD CODE` docstrings
pointing to the replacement module.

| Module | Lines | Replaced By |
|--------|-------|-------------|
| `predictive_cooperation.py` | 523 | `algorithmic_credit.py` (Shapley) + `shaped_rewards.py` (intermediate rewards) |
| `transfer_learning.py` | ~1,000 | `td_lambda.py` hierarchical domain keys (domain-specific → base fallback) |
| `predictive_marl.py` | ~734 | `algorithmic_credit.py` + `td_lambda.py` |
| `learning_coordinator.py` | ~819 | `LearningService` (central service coordinates everything) |
| `reasoning_credit.py` | 233 | `algorithmic_credit.py` (strict superset) |
| `rl_components.py` | 423 | `td_lambda.py` + `q_learning.py` + `shaped_rewards.py` |
| `base_learning_manager.py` | 245 | No external implementations; abstract interfaces unused |
| `offline_learning.py` | 645 | `LearningService._extract_patterns()` + `algorithmic_credit.DifferenceRewardEstimator` |

### Benchmark Results (2026-02-24)

Last audit reported all 4 core tests passing across 4 domains (16/16).
Re-run locally to confirm:

| Domain | Skill Ranking | Convergence | Gold Metric | Crystallization |
|--------|:---:|:---:|:---:|:---:|
| coding | PASS | PASS | PASS | PASS |
| research | PASS | PASS | PASS | PASS |
| writing | PASS | PASS | PASS | PASS |
| data_analysis | PASS | PASS | PASS | PASS |

### Summary

- **Active+Support+Integrated**: ~11,500 lines (reported active/integrated; see audit above)
- **Dead Code**: ~4,600 lines (valid algorithms, fully replaced by active modules)
- **Total**: ~16,100 lines across 20+ files
- **Effective code**: 71% active, 29% dead (all dead code has replacement pointers)
