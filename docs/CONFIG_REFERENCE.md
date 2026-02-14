# Jotty Configuration Reference

Complete reference for all configuration classes, organized by importance with wiring status.

**Source files:**
- `core/foundation/data_structures.py` — SwarmConfig (120+ fields), ConfigViews, DEFAULT_PARAM_ALIASES
- `core/execution/types.py` — ExecutionConfig (25 fields)
- `core/foundation/agent_config.py` — AgentConfig (23 fields)

**Wiring status legend:**
- Fields without annotation are **wired** — read by production code
- Fields marked **[NOT WIRED]** are defined but never read by any production code
- Fields marked **[DUPLICATE]** exist in another config class that IS wired

---

## Dead Config Summary

32 SwarmConfig fields are defined but never consumed by production code. They are persisted via `to_flat_dict()` but never read back. Grouped by category:

| Category | Dead Fields | Notes |
|----------|-------------|-------|
| Execution (6) | `actor_timeout`, `max_episode_iterations`, `allow_partial_execution`, `stream_message_timeout`, `parallel_architect`, `parallel_auditor` | |
| Budget (2) | `enable_budget_enforcement`, `budget_warning_threshold` | Duplicated in `BudgetConfig` with different names |
| Validation (8) | `advisory_confidence_threshold`, `max_validation_retries`, `enable_confidence_override`, `confidence_override_threshold`, `confidence_moving_average_alpha`, `min_confidence_for_override`, `max_validator_confidence_to_override`, `max_refinement_rounds` | Entire confidence override block is dead |
| RL (2) | `baseline`, `n_step` | Set in `universal_wrapper.py` but never read |
| Protection (4) | `ood_entropy_threshold`, `approval_reward_bonus`, `rejection_penalty`, `task_memory_ratio` | |
| Discovery Budget (4) | `preview_token_budget`, `max_description_tokens`, `compression_trigger_ratio`, `chunking_threshold_tokens` | Derived fields `preview_char_limit`, `max_description_chars` also dead |
| Agent Registry (2) | `enable_agent_registry`, `auto_infer_capabilities` | |
| Other (4) | `enable_llm_planning`, `enable_beautified_logs`, `profiling_verbosity`, `learning_boost_factor` | |

**Config Views** (`config.execution.`, `config.learning.`, etc.) are also never used in production — no view property access patterns found.

---

## Quick Start

```python
from Jotty.core.foundation.data_structures import SwarmConfig

# Minimal (defaults work)
config = SwarmConfig()

# Fast iteration (no learning overhead)
config = SwarmConfig(enable_rl=False, enable_validation=False, enable_multi_round=False, verbose=0)

# Production (full learning + validation)
config = SwarmConfig(
    enable_rl=True,
    enable_validation=True,
    validation_mode='full',
    enable_monitoring=True,
    output_base_dir="./outputs",
)
```

---

## SwarmConfig

Primary configuration dataclass. All fields are flat with `_ConfigView` proxies for organized access (views exist but are not yet used in production).

```python
config = SwarmConfig(gamma=0.99, enable_rl=True)
config.gamma  # 0.99 (flat access — this is what production code uses)
```

---

## Essential — Configure These First

### Execution

Runtime limits and timeouts. Consumers: `inspector.py`, `swarm_manager.py`, `universal_wrapper.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_actor_iters` | int | `50` | Max iterations per actor agent |
| `max_eval_iters` | int | `1` | Architect/Auditor ReAct iterations. 1=minimal, 2-3=balanced, 5-10=thorough |
| `max_eval_retries` | int | `3` | Retry attempts for validation |
| `llm_timeout_seconds` | float | `180.0` | LLM API call timeout (3 minutes) |
| `async_timeout` | float | `60.0` | General async operation timeout (seconds) |
| `max_concurrent_agents` | int | `10` | Max agents running in parallel |

**Not wired:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_episode_iterations` | int | `12` | [NOT WIRED] Max task iterations per episode |
| `actor_timeout` | float | `900.0` | [NOT WIRED] Actor execution timeout |
| `allow_partial_execution` | bool | `False` | [NOT WIRED] Allow agents to run with missing params |
| `stream_message_timeout` | float | `0.15` | [NOT WIRED] Streaming message timeout |
| `parallel_architect` | bool | `True` | [NOT WIRED] Run architect validation in parallel |
| `parallel_auditor` | bool | `True` | [NOT WIRED] Run auditor validation in parallel |

### Budget Controls

LLM cost limits. **Note:** SwarmConfig budget fields are NOT wired. Production uses `BudgetConfig` in `budget_tracker.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_llm_calls_per_episode` | int | `100` | [DUPLICATE] Mirrors `BudgetConfig` — SwarmConfig copy not read |
| `max_llm_calls_per_agent` | int | `50` | [DUPLICATE] Mirrors `BudgetConfig` — SwarmConfig copy not read |
| `max_total_tokens_per_episode` | int | `500000` | [DUPLICATE] Mirrors `BudgetConfig` — SwarmConfig copy not read |
| `enable_budget_enforcement` | bool | `True` | [NOT WIRED] `BudgetConfig.enable_enforcement` is the active version |
| `budget_warning_threshold` | float | `0.8` | [NOT WIRED] `BudgetConfig.warning_threshold` is the active version |
| `enable_monitoring` | bool | `False` | [NOT WIRED] Comprehensive monitoring (opt-in) |
| `baseline_cost_per_success` | Optional[float] | `None` | [NOT WIRED] Baseline cost for efficiency comparison |

### Validation

Quality loop. Consumers: `agent_runner.py`, `jotty.py`, `executor.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_validation` | bool | `True` | Master switch for all validation |
| `validation_mode` | str | `"full"` | `"full"`, `"architect_only"`, `"auditor_only"`, `"none"` |
| `max_validation_rounds` | int | `3` | Max rounds of validation per step |
| `refinement_timeout` | float | `30.0` | Timeout for each refinement round (seconds) |
| `require_all_architect` | bool | `True` | Require all architects to pass |
| `require_all_auditor` | bool | `False` | Require all auditors to pass (False = majority) |
| `enable_per_actor_swarm_auditor` | bool | `False` | Run swarm auditor after each actor (slow) |
| `enable_final_swarm_auditor` | bool | `True` | Run swarm auditor once at end (recommended) |
| `swarm_validation_confidence_threshold` | float | `0.6` | Only retry if confidence below this |
| `min_confidence` | float | `0.5` | Minimum confidence to accept a result |

**Default confidence values** (previously hardcoded, now configurable):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `default_confidence_on_error` | float | `0.3` | Confidence assigned when validation errors |
| `default_confidence_no_validation` | float | `0.5` | Confidence when no validation is run |
| `default_confidence_insight_share` | float | `0.7` | Confidence for shared insights |
| `default_estimated_reward` | float | `0.6` | Estimated reward when no Auditor result yet |

**Not wired:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `advisory_confidence_threshold` | float | `0.85` | [NOT WIRED] Advisory feedback retry threshold |
| `max_validation_retries` | int | `5` | [NOT WIRED] Max retries on validation failure |
| `enable_confidence_override` | bool | `True` | [NOT WIRED] Confident-actor override mechanism |
| `confidence_override_threshold` | float | `0.30` | [NOT WIRED] Min gap for override |
| `confidence_moving_average_alpha` | float | `0.7` | [NOT WIRED] EMA weight for confidence |
| `min_confidence_for_override` | float | `0.70` | [NOT WIRED] Actor min confidence to override |
| `max_validator_confidence_to_override` | float | `0.95` | [NOT WIRED] Don't override above this |
| `max_refinement_rounds` | int | `2` | [NOT WIRED] Max refinement iterations |
| `enable_llm_planning` | bool | `False` | [NOT WIRED] Use LLM to create initial plan |

---

## Core Features — Most Users Tune These

### Memory (Hierarchical)

5-level memory hierarchy. Consumer: `cortex.py`, `fallback_memory.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `episodic_capacity` | int | `1000` | Max episodic memory entries (recent experiences) |
| `semantic_capacity` | int | `500` | Max semantic memory entries (facts/knowledge) |
| `procedural_capacity` | int | `200` | Max procedural memory entries (how-to patterns) |
| `meta_capacity` | int | `100` | Max meta-memory entries (learning about learning) |
| `causal_capacity` | int | `150` | Max causal knowledge entries (cause-effect links) |
| `max_entry_tokens` | int | `2000` | Max tokens per memory entry |

**Computed:** `config.total_memory_capacity` — sum of all 5 capacities.

### Context Budget

Token allocation. Consumer: `_inference_mixin.py`, context management code.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_context_tokens` | int | `100000` | Total context window size |
| `system_prompt_budget` | int | `5000` | Tokens reserved for system prompt |
| `current_input_budget` | int | `15000` | Tokens reserved for current user input |
| `trajectory_budget` | int | `20000` | Tokens reserved for agent trajectory/history |
| `tool_output_budget` | int | `15000` | Tokens reserved for tool outputs |
| `enable_dynamic_budget` | bool | `True` | Dynamically reallocate unused budget |
| `min_memory_budget` | int | `10000` | Floor for memory token allocation |
| `max_memory_budget` | int | `60000` | Ceiling for memory token allocation |
| `token_model_name` | Optional[str] | `None` | Override model name for token counting |

**Computed:** `config.memory_budget` — `max(min_memory_budget, max_context_tokens - reserved)`.

**Not wired (discovery budget):**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `preview_token_budget` | int | `20000` | [NOT WIRED] Token budget for artifact analysis |
| `max_description_tokens` | int | `5000` | [NOT WIRED] Per-artifact description limit |
| `compression_trigger_ratio` | float | `0.8` | [NOT WIRED] Compress above 80% of limit |
| `chunking_threshold_tokens` | int | `15000` | [NOT WIRED] Chunk artifacts above this |

### RL Parameters (TD-Lambda)

The learning engine. Consumer: `td_lambda.py`, `q_learning.py`, `adaptive_components.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_rl` | bool | `True` | Master switch for all RL features |
| `rl_verbosity` | str | `"quiet"` | `"quiet"`, `"normal"`, `"verbose"` |
| `gamma` | float | `0.99` | TD discount factor |
| `lambda_trace` | float | `0.95` | Eligibility trace decay for TD(lambda) |
| `alpha` | float | `0.01` | TD learning rate |
| `q_value_mode` | str | `"simple"` | `"simple"` (average reward) or `"llm"` (LLM Q-value prediction) |
| `enable_adaptive_alpha` | bool | `True` | Dynamically adjust learning rate |
| `alpha_min` | float | `0.001` | Learning rate floor |
| `alpha_max` | float | `0.1` | Learning rate ceiling |
| `alpha_adaptation_rate` | float | `0.1` | Speed of learning rate adaptation |

**Intermediate rewards** (consumer: `td_lambda.py`):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_intermediate_rewards` | bool | `True` | Reward agents during execution, not just at end |
| `architect_proceed_reward` | float | `0.1` | Reward when architect approves a step |
| `tool_success_reward` | float | `0.05` | Reward per successful tool call |

**Cooperative rewards** (consumer: `td_lambda.py`, `credit_assignment.py`):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `base_reward_weight` | float | `0.3` | Weight for agent's own success |
| `cooperation_bonus` | float | `0.4` | Bonus for helping other agents |
| `predictability_bonus` | float | `0.3` | Bonus for predictable behavior |

**Not wired:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `baseline` | float | `0.5` | [NOT WIRED] Reward baseline for advantage calc. Set in universal_wrapper but never read |
| `n_step` | int | `3` | [NOT WIRED] N-step returns lookahead. Set in universal_wrapper but never read |

### Persistence

Where and how state is saved. Consumer: `session_manager.py`, `base_learning_manager.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `schema_version` | str | `"2.0"` | Schema version for persisted JSON |
| `output_base_dir` | str | `"./outputs"` | Base directory for all outputs |
| `create_run_folder` | bool | `True` | Create timestamped `run_YYYYMMDD_HHMMSS/` folders |
| `auto_save_interval` | int | `3` | Save state every N steps |
| `auto_load_on_start` | bool | `True` | Auto-load from `outputs/latest/` |
| `save_interval` | int | `1` | Episodes between saves |
| `persist_memories` | bool | `True` | Persist memory entries to disk |
| `persist_q_tables` | bool | `True` | Persist Q-tables to disk |
| `persist_brain_state` | bool | `True` | Persist brain/swarm intelligence state |
| `persist_todos` | bool | `True` | Save session TODOs to markdown |
| `persist_agent_outputs` | bool | `True` | Save IOManager outputs |
| `storage_format` | str | `"json"` | `"json"` or `"sqlite"` |
| `compress_large_files` | bool | `True` | Gzip files larger than 1MB |
| `max_runs_to_keep` | int | `10` | Auto-cleanup old run folders |
| `enable_backups` | bool | `True` | Enable periodic backups |
| `backup_interval` | int | `100` | Episodes between backups |
| `max_backups` | int | `10` | Maximum backup copies |
| `base_path` | str | `"~/.jotty"` | Legacy persistence location |
| `auto_load` | bool | `True` | Legacy auto-load flag |
| `auto_save` | bool | `True` | Legacy auto-save flag |

**Learning persistence** (consumer: `td_lambda.py`, `q_learning.py`):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `auto_load_learning` | bool | `True` | Auto-load previous learning on startup |
| `per_agent_learning` | bool | `True` | Each agent has its own Q-table |
| `shared_learning` | bool | `True` | Also maintain shared cross-agent learning |
| `learning_alpha` | float | `0.3` | Q-learning rate for persistence layer |
| `learning_gamma` | float | `0.9` | Discount factor for persistence layer |
| `learning_epsilon` | float | `0.1` | Exploration rate for persistence layer |
| `max_q_table_size` | int | `10000` | Max Q-table entries before pruning |
| `q_prune_percentage` | float | `0.2` | Fraction pruned when limit hit (20%) |
| `enable_domain_transfer` | bool | `True` | Load learning from similar domains |

---

## Advanced Tuning — Power Users

### Swarm Intelligence

Trust adaptation and routing. Consumer: `swarm_intelligence.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `trust_decrease_on_struggle` | float | `0.1` | Trust penalty per adaptation window |
| `trust_increase_on_excel` | float | `0.05` | Trust bonus per adaptation window |
| `trust_min` | float | `0.1` | Minimum trust floor |
| `adaptation_interval` | int | `5` | Adapt trust every N experiences |
| `adaptation_struggle_threshold` | float | `0.3` | Success rate below this = struggling |
| `adaptation_excel_threshold` | float | `0.8` | Success rate above this = excelling |
| `stigmergy_routing_threshold` | float | `0.5` | Min signal strength for stigmergy routing |
| `morph_min_rcs` | float | `0.3` | Min Role Clarity Score for TRAS routing |
| `judge_intervention_confidence` | float | `0.6` | Auditor confidence below this triggers retry |
| `memory_retrieval_budget` | int | `3000` | Tokens for memory retrieval in AgentRunner |
| `collective_memory_limit` | int | `200` | Max items in swarm collective memory |

### Exploration

Epsilon-greedy and UCB controls. Consumer: `td_lambda.py`, `adaptive_components.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `epsilon_start` | float | `0.3` | Initial exploration rate |
| `epsilon_end` | float | `0.05` | Final exploration rate after decay |
| `epsilon_decay_episodes` | int | `500` | Episodes over which epsilon decays |
| `ucb_coefficient` | float | `2.0` | UCB coefficient (higher = more exploration) |
| `enable_adaptive_exploration` | bool | `True` | Boost exploration when learning stalls |
| `exploration_boost_on_stall` | float | `0.1` | Epsilon boost on stall detection |
| `max_exploration_iterations` | int | `10` | Max iterations for policy exploration |
| `policy_update_threshold` | int | `3` | Episodes before updating policy |

### Credit Assignment

Multi-agent contribution attribution. Consumer: `credit_assignment.py`, `td_lambda.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `credit_decay` | float | `0.9` | Temporal decay for credit assignment |
| `min_contribution` | float | `0.1` | Minimum contribution floor per agent |
| `enable_reasoning_credit` | bool | `True` | Use reasoning quality for credit |
| `reasoning_weight` | float | `0.3` | Weight for reasoning-based credit |
| `evidence_weight` | float | `0.2` | Weight for evidence-based credit |

### Inter-Agent Communication

Consumer: `swarm_manager.py`, `agent_runner.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_agent_communication` | bool | `True` | Enable inter-agent messaging |
| `share_tool_results` | bool | `True` | Cache and share tool results across agents |
| `share_insights` | bool | `True` | Share discovered insights between agents |
| `max_messages_per_episode` | int | `20` | Cap on inter-agent messages per episode |

### Multi-Round Validation

Consumer: `swarm_manager.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_multi_round` | bool | `True` | Enable multi-round validation |
| `refinement_on_low_confidence` | float | `0.6` | Trigger refinement when confidence below this |
| `refinement_on_disagreement` | bool | `True` | Trigger refinement when agents disagree |

**Not wired:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_refinement_rounds` | int | `2` | [NOT WIRED] Max refinement iterations |

---

## Specialized

### LLM-Based RAG

Semantic retrieval without embeddings. Consumer: `llm_rag.py`, `cortex.py`, `_retrieval_mixin.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_llm_rag` | bool | `True` | Enable LLM-based retrieval |
| `rag_window_size` | int | `5` | Memories per LLM call for relevance scoring |
| `rag_max_candidates` | int | `50` | Pre-filter before LLM scoring |
| `rag_relevance_threshold` | float | `0.6` | Min relevance score |
| `rag_use_cot` | bool | `True` | Chain-of-thought for scoring |
| `retrieval_mode` | str | `"synthesize"` | `"synthesize"` or `"discrete"` |
| `synthesis_fetch_size` | int | `200` | Memories to fetch for synthesis |
| `synthesis_max_tokens` | int | `800` | Max tokens for synthesized output |
| `chunk_size` | int | `500` | Tokens per chunk |
| `chunk_overlap` | int | `50` | Overlap between chunks |

### Consolidation

Pattern extraction. Consumer: `_consolidation_mixin.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `consolidation_threshold` | int | `100` | Min memories before consolidation |
| `consolidation_interval` | int | `3` | Consolidate every N episodes |
| `min_cluster_size` | int | `5` | Min cluster size to extract a pattern |
| `pattern_confidence_threshold` | float | `0.7` | Min confidence to keep a pattern |

### Offline Learning

Experience replay. Consumer: `offline_learning.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `episode_buffer_size` | int | `1000` | Max episodes in replay buffer |
| `offline_update_interval` | int | `50` | Run offline update every N episodes |
| `replay_batch_size` | int | `20` | Batch size for replay updates |
| `counterfactual_samples` | int | `5` | Counterfactual scenarios per update |

### Protection Mechanisms

Forgetting prevention. Consumer: `cortex.py`, `swarm_intelligence.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `protected_memory_threshold` | float | `0.8` | Memories above this importance are protected |
| `suspicion_threshold` | float | `0.95` | Flags suspiciously certain agent |
| `min_rejection_rate` | float | `0.05` | Min rejection rate for calibration |

**Not wired:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `task_memory_ratio` | float | `0.3` | [NOT WIRED] Memory budget fraction for current task |
| `ood_entropy_threshold` | float | `0.8` | [NOT WIRED] Out-of-distribution detection |
| `approval_reward_bonus` | float | `0.1` | [NOT WIRED] Human approval reward |
| `rejection_penalty` | float | `0.05` | [NOT WIRED] Human rejection penalty |

### Adaptive Learning

Stall detection. Consumer: `adaptive_components.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_adaptive_learning` | bool | `True` | Enable adaptive parameter tuning |
| `stall_detection_window` | int | `100` | Episodes to check for stall |
| `stall_threshold` | float | `0.001` | Improvement below this = stalled |
| `adaptive_window_size` | int | `50` | Window for learning rate adaptation |
| `instability_threshold_multiplier` | float | `1.5` | `std_dev > mean * this` = unstable |
| `slow_learning_threshold` | float | `0.01` | `mean_error < this` = too slow |
| `goal_transfer_discount` | float | `0.5` | Discount for value transfer to related goals |

**Not wired:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `learning_boost_factor` | float | `2.0` | [NOT WIRED] LR multiplier on stall |

### Goal Hierarchy & Causal Learning

Consumer: `td_lambda.py`, `cortex.py`, `_retrieval_mixin.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_goal_hierarchy` | bool | `True` | Enable hierarchical goal structure |
| `goal_transfer_weight` | float | `0.3` | Weight for transferred knowledge |
| `enable_causal_learning` | bool | `True` | Enable causal relationship tracking |
| `causal_confidence_threshold` | float | `0.7` | Min confidence for causal link |
| `causal_min_support` | int | `3` | Episodes before link confirmed |

### Deduplication

Consumer: `cortex.py`, `llm_rag.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_deduplication` | bool | `True` | Enable memory deduplication |
| `similarity_threshold` | float | `0.85` | Similarity above this = duplicate |

### Learning Pipeline

Consumer: `learning_coordinator.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `learning_components` | Optional[List[str]] | `None` | Components to run in `post_episode()`. `None` = all |

Valid names: `td_lambda`, `swarm_learner`, `brain_consolidation`, `neurochunk_tiering`, `agent_abstractor`, `transfer_learning`, `swarm_intelligence`, `stigmergy`, `effectiveness`, `mas_learning`, `byzantine`, `credit_assignment`, `auditor_fixes`, `adaptive_learning`, `effectiveness_intervention`, `credit_pruning`, `curriculum`.

### Local-First Mode

Consumer: widely used (33 references across codebase).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `local_mode` | bool | `False` | Master switch for local-only inference |
| `local_model` | str | `"ollama/llama3"` | Local model identifier (Ollama format) |

### Reproducibility

Consumer: `__post_init__` calls `set_reproducible_seeds()`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `random_seed` | Optional[int] | `None` | Fixed random seed. `None` = non-deterministic |
| `numpy_seed` | Optional[int] | `None` | NumPy seed. Falls back to `random_seed` |
| `torch_seed` | Optional[int] | `None` | PyTorch seed. Falls back to `random_seed` |
| `python_hash_seed` | Optional[int] | `None` | Hash seed. Falls back to `random_seed` |
| `enable_deterministic` | bool | `True` | Enable deterministic operations |

### Logging & Profiling

Consumer: `swarm_manager.py`, various.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `verbose` | int | `1` | Verbosity level (0=silent, 1=normal, 2+=debug) |
| `log_file` | Optional[str] | `None` | Log file path. `None` = stdout only |
| `log_level` | str | `"INFO"` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `enable_debug_logging` | bool | `False` | Debug logging (default OFF for production) |
| `enable_debug_logs` | bool | `True` | Keep raw debug logs |
| `enable_metrics` | bool | `True` | Enable metrics collection |
| `enable_profiling` | bool | `False` | Track execution times |

**Not wired:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_beautified_logs` | bool | `True` | [NOT WIRED] Human-readable logs |
| `profiling_verbosity` | str | `"summary"` | [NOT WIRED] `"summary"` or `"detailed"` |

### Agent Registry & Parameter Mappings

Consumer: `custom_param_mappings` used by `ParameterResolver`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `custom_param_mappings` | Dict[str, List[str]] | `{}` | Custom parameter name mappings |

**Not wired:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_agent_registry` | bool | `True` | [NOT WIRED] Track actor capabilities |
| `auto_infer_capabilities` | bool | `True` | [NOT WIRED] LLM infers capabilities |

---

## Config Views

Eight read/write proxy views exist but are **not yet used in production code**. They work and are tested, but no production module accesses config through views.

| View Property | Class | Description |
|---------------|-------|-------------|
| `config.execution` | `ExecutionView` | Timeouts, parallelism, seeds |
| `config.monitoring` | `MonitoringView` | Logging, profiling, budgets |
| `config.validation_settings` | `ValidationView` | Validation, confidence, multi-round |
| `config.memory_settings` | `MemoryView` | Capacities, RAG, chunking |
| `config.context_budget` | `ContextBudgetView` | Token allocation |
| `config.learning` | `LearningView` | RL, exploration, credit, consolidation, protection |
| `config.persistence` | `PersistenceView` | Storage paths, save/load, retention |
| `config.intelligence` | `SwarmIntelligenceView` | Trust, routing, communication, local mode |

```python
# Views work — just not used by production code yet
config.learning.gamma  # reads config.gamma
config.learning.gamma = 0.95  # writes config.gamma
config.learning.to_dict()  # {'alpha': 0.01, 'gamma': 0.99, ...}
```

---

## ExecutionConfig

Separate config class for the tiered execution system. Not to be confused with SwarmConfig execution fields.

```python
from Jotty.core.execution.types import ExecutionConfig, ExecutionTier
config = ExecutionConfig(tier=ExecutionTier.AGENTIC)
```

### Tiers

| Tier | Name | Value | Description |
|------|------|-------|-------------|
| 1 | `DIRECT` | 1 | Single LLM call with tools |
| 2 | `AGENTIC` | 2 | Planning + multi-step orchestration |
| 3 | `LEARNING` | 3 | Memory + validation |
| 4 | `RESEARCH` | 4 | Domain swarm execution |
| 5 | `AUTONOMOUS` | 5 | Sandbox, coalition, curriculum, full features |

### Fields

| Field | Type | Default | Tiers | Description |
|-------|------|---------|-------|-------------|
| `tier` | Optional[ExecutionTier] | `None` | All | Execution tier. `None` = auto-detect |
| `max_planning_depth` | int | `3` | 2+ | Max recursive planning depth |
| `enable_parallel_execution` | bool | `True` | 2+ | Parallel step execution |
| `max_concurrent_steps` | int | `3` | 2+ | Max concurrent steps |
| `memory_backend` | str | `"json"` | 3+ | `"json"`, `"redis"`, `"postgres"` |
| `memory_ttl_hours` | int | `24` | 3+ | Memory time-to-live |
| `enable_validation` | bool | `True` | 3+ | Enable output validation |
| `validation_retries` | int | `1` | 3+ | Retries on validation failure |
| `track_success_rate` | bool | `True` | 3+ | Track per-skill success rates |
| `enable_td_lambda` | bool | `False` | 4+ | Enable TD(lambda) RL |
| `enable_hierarchical_memory` | bool | `False` | 4+ | Enable 5-level memory |
| `enable_multi_round_validation` | bool | `False` | 4+ | Enable multi-round validation |
| `memory_levels` | int | `2` | 4+ | Memory levels (2 = episodic + semantic) |
| `enable_swarm_intelligence` | bool | `False` | 4+ | Enable swarm intelligence |
| `swarm_name` | Optional[str] | `None` | 4-5 | `"coding"`, `"research"`, `"testing"` |
| `paradigm` | Optional[str] | `None` | 4-5 | `"relay"`, `"debate"`, `"refinement"` |
| `enable_sandbox` | bool | `False` | 5 | Sandboxed execution |
| `enable_coalition` | bool | `False` | 5 | Coalition formation |
| `trust_level` | str | `"standard"` | 5 | `"standard"`, `"elevated"`, `"restricted"` |
| `timeout_seconds` | int | `300` | All | Execution timeout (seconds) |
| `max_retries` | int | `3` | All | Max retries on failure |
| `provider` | Optional[str] | `None` | All | LLM provider. `None` = auto-detect |
| `model` | Optional[str] | `None` | All | Model name. `None` = provider default |
| `temperature` | float | `0.7` | All | LLM temperature |
| `max_tokens` | int | `4000` | All | Max output tokens |

### `to_swarm_config()` Mapping

| ExecutionConfig field | Maps to SwarmConfig field |
|-----------------------|--------------------------|
| `enable_validation` | `enable_validation` |
| `enable_multi_round_validation` | `enable_multi_round` |
| `enable_td_lambda` | `enable_rl`, `enable_causal_learning` |
| `timeout_seconds` | `llm_timeout_seconds` |
| `validation_retries` | `max_eval_retries` |
| `enable_multi_round_validation` | `validation_mode` (`"full"` or `"none"`) |

Note: `provider`, `model`, `temperature` are NOT mapped — set via DSPy LM or `LLMProvider`.

---

## AgentConfig

Per-agent configuration. All fields are wired.

```python
from Jotty.core.foundation.agent_config import AgentConfig
config = AgentConfig(name="MyAgent", agent=my_dspy_module)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | (required) | Agent name identifier |
| `agent` | Any | (required) | DSPy Module instance |
| `architect_prompts` | Optional[List[str]] | `None` | Pre-execution planning prompts (initialized to `[]`) |
| `auditor_prompts` | Optional[List[str]] | `None` | Post-execution validation prompts (initialized to `[]`) |
| `parameter_mappings` | Optional[Dict[str, str]] | `None` | Input parameter name remapping |
| `outputs` | Optional[List[str]] | `None` | Output field names this agent produces |
| `provides` | Optional[List[str]] | `None` | Parameters this agent can provide to others |
| `context_requirements` | Optional[Any] | `None` | `ContextRequirements` instance |
| `architect_tools` | List[Any] | `[]` | Tools for Architect phase |
| `auditor_tools` | List[Any] | `[]` | Tools for Auditor phase |
| `feedback_rules` | Optional[List[Dict]] | `None` | Feedback routing rules |
| `capabilities` | Optional[List[str]] | `None` | Capability tags for dynamic orchestration |
| `dependencies` | Optional[List[str]] | `None` | Agent dependencies |
| `metadata` | Optional[Dict[str, Any]] | `None` | Arbitrary metadata |
| `enable_architect` | bool | `True` | Enable Architect for this agent |
| `enable_auditor` | bool | `True` | Enable Auditor for this agent |
| `validation_mode` | str | `"standard"` | `"quick"`, `"standard"`, `"thorough"` |
| `is_critical` | bool | `False` | Failure of this agent fails the swarm |
| `max_retries` | int | `0` | `0` = use `config_defaults.MAX_RETRIES` |
| `retry_strategy` | str | `"with_hints"` | Provides feedback from previous failure |
| `is_executor` | bool | `False` | Agent executes actions (queries, API calls) |
| `enabled` | bool | `True` | `False` to disable without removing |

---

## DEFAULT_PARAM_ALIASES

Used by `ParameterResolver` for auto-wiring tool parameters.

| Canonical Name | Aliases |
|---------------|---------|
| `tables` | `relevant_tables`, `selected_tables`, `table_list`, `available_tables`, `get_all_tables` |
| `table_names` | `available_tables`, `all_tables`, `tables`, `relevant_tables`, `selected_tables`, `table_list`, `get_all_tables` |
| `columns` | `selected_columns`, `column_list`, `relevant_columns`, `available_columns` |
| `columns_metadata` | `column_metadata`, `columns_info` |
| `resolved_terms` | `terms`, `business_terms`, `term_mapping`, `get_business_terms` |
| `business_terms` | `get_business_terms`, `business_context`, `get_business_context` |
| `filters` | `filter_conditions`, `where_conditions` |
| `tables_metadata` | `get_all_table_metadata`, `table_metadata`, `schema_info` |
| `content` | `content`, `text`, `body` |
| `data` | `data`, `output_data`, `results` |
| `file` | `file`, `filepath`, `path`, `file_path` |
| `url` | `url`, `uri`, `link`, `href` |

Extend with `SwarmConfig.custom_param_mappings`:

```python
config = SwarmConfig(custom_param_mappings={'user_id': ['customer_id', 'uid', 'account_id']})
```

---

## Serialization

```python
# SwarmConfig → flat dict (includes dead fields)
flat = config.to_flat_dict()

# View → dict (subset)
learning_dict = config.learning.to_dict()

# ExecutionConfig → SwarmConfig-compatible dict
swarm_kwargs = exec_config.to_swarm_config()
config = SwarmConfig(**swarm_kwargs)
```
