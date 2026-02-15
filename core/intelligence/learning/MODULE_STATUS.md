# Learning Module Status

Last audited: 2026-02-08

## Module Status (by actual usage)

### ACTIVE — Instantiated in production code paths

| Module | Lines | Used by |
|--------|-------|---------|
| `learning.py` | 28 | Re-exports from td_lambda, adaptive_components, health_budget, reasoning_credit |
| `td_lambda.py` | 722 | TDLambdaLearner used in agent_runner, jotty.py |
| `q_learning.py` | 1,634 | LLMQPredictor used in learning_coordinator, jotty.py |
| `shaped_rewards.py` | 378 | ShapedRewardManager used in agent_runner |
| `algorithmic_credit.py` | 530 | ShapleyValueEstimator, DifferenceRewardEstimator, AlgorithmicCreditAssigner used in jotty.py, algorithmic_foundations.py |
| `transfer_learning.py` | 1,026 | TransferableLearningStore used in learning_pipeline |
| `learning_coordinator.py` | 819 | LearningCoordinator used in learning_pipeline |
| `predictive_marl.py` | 734 | CooperativeCreditAssigner, LLMTrajectoryPredictor used in learning_pipeline |

### SUPPORT — Used internally by active modules only

| Module | Lines | Used by |
|--------|-------|---------|
| `adaptive_components.py` | 279 | Used by td_lambda.py, learning.py (AdaptiveLearningRate, IntermediateRewardCalculator) |
| `health_budget.py` | 340 | Used by learning.py (LearningHealthMonitor, DynamicBudgetManager) |
| `base_classes.py` | 246 | Base dataclasses for learning types |
| `utils.py` | 513 | Shared utilities |

### UNUSED — Never instantiated outside learning/

| Module | Lines | Notes |
|--------|-------|-------|
| `reasoning_credit.py` | 233 | ReasoningCreditAssigner imported but never instantiated. Credit overlap with algorithmic_credit. |
| `predictive_cooperation.py` | 537 | CooperationReasoner, NashBargainingSolver, PredictiveCooperativeAgent — 0 instantiations |
| `rl_components.py` | 423 | RLComponents — 0 external imports |
| `base_learning_manager.py` | 267 | BaseLearningManager interfaces — 0 external instantiations |
| `offline_learning.py` | 645 | CounterfactualLearner, OfflineLearner, PatternDiscovery — only used through utils.py |

### CONSOLIDATION OPPORTUNITIES

1. **reasoning_credit → algorithmic_credit**: Both do credit assignment. ReasoningCreditAssigner can merge into algorithmic_credit.
2. **predictive_cooperation → predictive_marl**: Both do multi-agent prediction. CooperationReasoner overlaps conceptually.
3. **rl_components + base_learning_manager**: Pure interfaces with no external users. Could merge or archive.
4. **offline_learning**: Large but only internally referenced. Could be lazy-loaded.

### Total: 9,534 lines across 18 files
### Active: ~5,571 lines (8 files) — 58% of total
### Unused: ~2,105 lines (5 files) — 22% of total
### Support: ~1,378 lines (4 files) — 14% of total
