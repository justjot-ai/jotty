# Jotty Configuration Reference

Complete reference for all configuration classes in the Jotty AI Agent Framework, organized by importance.

**Source files:**
- `core/foundation/data_structures.py` — SwarmConfig (120+ fields), ConfigViews, DEFAULT_PARAM_ALIASES
- `core/execution/types.py` — ExecutionConfig (25 fields)
- `core/foundation/agent_config.py` — AgentConfig (23 fields)

---

## Quick Start

### Minimal config (defaults work out of the box)

```python
from Jotty.core.foundation.data_structures import SwarmConfig
config = SwarmConfig()
```

### Fast iteration (no learning overhead)

```python
config = SwarmConfig(
    enable_rl=False,
    enable_validation=False,
    enable_multi_round=False,
    verbose=0,
)
```

### Production (full learning + validation + budget limits)

```python
config = SwarmConfig(
    enable_rl=True,
    enable_validation=True,
    validation_mode='full',
    enable_budget_enforcement=True,
    max_llm_calls_per_episode=100,
    max_total_tokens_per_episode=500000,
    enable_monitoring=True,
    output_base_dir="./outputs",
    storage_format="json",
)
```

---

## SwarmConfig

Primary configuration dataclass for swarm orchestration. All 120+ fields live in a single flat dataclass with `_ConfigView` proxies for organized access.

```python
from Jotty.core.foundation.data_structures import SwarmConfig
config = SwarmConfig(gamma=0.99, enable_rl=True)

# Flat access
config.gamma  # 0.99

# View access (same fields, grouped)
config.learning.gamma  # 0.99
config.learning.gamma = 0.95  # writes back to config.gamma
```

---

## Essential — Configure These First

These control runtime behavior, cost, and quality. Misconfiguration here causes hangs, runaway bills, or silent bad outputs.

### Execution

Runtime limits, timeouts, and parallelism. Controls how long agents run and how many run concurrently.

**View:** `config.execution`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_actor_iters` | int | `50` | Max iterations per actor agent |
| `max_eval_iters` | int | `1` | Architect/Auditor ReAct iterations. 1=minimal, 2-3=balanced, 5-10=thorough |
| `max_episode_iterations` | int | `12` | Max task iterations per episode in `swarm.run()` |
| `async_timeout` | float | `60.0` | General async operation timeout (seconds) |
| `actor_timeout` | float | `900.0` | Actor execution timeout (15 minutes) |
| `max_concurrent_agents` | int | `10` | Max agents running in parallel |
| `allow_partial_execution` | bool | `False` | Allow agents to execute with missing required params. Set `True` for RL with natural dependencies |
| `max_eval_retries` | int | `3` | Retry attempts for validation |
| `stream_message_timeout` | float | `0.15` | Streaming message timeout (seconds) |
| `llm_timeout_seconds` | float | `180.0` | LLM API call timeout (3 minutes) |
| `parallel_architect` | bool | `True` | Run architect validation in parallel |
| `parallel_auditor` | bool | `True` | Run auditor validation in parallel |

### Budget Controls

LLM call and token budget enforcement. Prevents runaway cost.

**View:** `config.monitoring`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_llm_calls_per_episode` | int | `100` | Max LLM calls per episode |
| `max_llm_calls_per_agent` | int | `50` | Max LLM calls per individual agent |
| `max_total_tokens_per_episode` | int | `500000` | Max tokens consumed per episode |
| `enable_budget_enforcement` | bool | `True` | Enforce budget limits (hard stop) |
| `budget_warning_threshold` | float | `0.8` | Warn at 80% of budget |
| `enable_monitoring` | bool | `False` | Enable comprehensive monitoring (opt-in) |
| `baseline_cost_per_success` | Optional[float] | `None` | Baseline cost for efficiency comparison |

### Validation

Quality loop: architect pre-checks, auditor post-checks, confidence overrides, and iterative refinement.

**View:** `config.validation_settings`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_validation` | bool | `True` | Master switch for all validation |
| `validation_mode` | str | `"full"` | `"full"`, `"architect_only"`, `"auditor_only"`, or `"none"` |
| `max_validation_rounds` | int | `3` | Max rounds of validation per step |
| `refinement_timeout` | float | `30.0` | Timeout for each refinement round (seconds) |
| `advisory_confidence_threshold` | float | `0.85` | Below this, advisory feedback triggers retry |
| `max_validation_retries` | int | `5` | Max retries on validation failure |
| `require_all_architect` | bool | `True` | Require all architects to pass |
| `require_all_auditor` | bool | `False` | Require all auditors to pass (False = majority) |
| `enable_per_actor_swarm_auditor` | bool | `False` | Run swarm auditor after each actor (slow) |
| `enable_final_swarm_auditor` | bool | `True` | Run swarm auditor once at end (recommended) |
| `swarm_validation_confidence_threshold` | float | `0.6` | Only retry if confidence below this |
| `enable_llm_planning` | bool | `False` | Use LLM to create initial TODO plan |
| `min_confidence` | float | `0.5` | Minimum confidence to accept a result |

#### Confidence Override

Allows confident actors to override uncertain validators.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_confidence_override` | bool | `True` | Enable confident-actor override mechanism |
| `confidence_override_threshold` | float | `0.30` | Min gap (actor - validator confidence) to allow override |
| `confidence_moving_average_alpha` | float | `0.7` | Weight for exponential moving average of confidence |
| `min_confidence_for_override` | float | `0.70` | Actor must be at least this confident to override |
| `max_validator_confidence_to_override` | float | `0.95` | Don't override if validator is above 95% confident |

#### Default Confidence Values

Previously hardcoded, now configurable:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `default_confidence_on_error` | float | `0.3` | Confidence assigned when validation errors |
| `default_confidence_no_validation` | float | `0.5` | Confidence when no validation is run |
| `default_confidence_insight_share` | float | `0.7` | Confidence for shared insights |
| `default_estimated_reward` | float | `0.6` | Estimated reward when no Auditor result yet |

---

## Core Features — Most Users Tune These

### Memory (Hierarchical)

Capacity limits for the 5-level memory hierarchy.

**View:** `config.memory_settings`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `episodic_capacity` | int | `1000` | Max episodic memory entries (recent experiences) |
| `semantic_capacity` | int | `500` | Max semantic memory entries (facts/knowledge) |
| `procedural_capacity` | int | `200` | Max procedural memory entries (how-to patterns) |
| `meta_capacity` | int | `100` | Max meta-memory entries (learning about learning) |
| `causal_capacity` | int | `150` | Max causal knowledge entries (cause-effect links) |
| `max_entry_tokens` | int | `2000` | Max tokens per memory entry to prevent oversized entries |

**Computed property:** `config.total_memory_capacity` — sum of all 5 capacities.

### Context Budget

Token allocation across the context window. Controls how the model's context is divided.

**View:** `config.context_budget`

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

**Computed property:** `config.memory_budget` — `max(min_memory_budget, max_context_tokens - reserved)` where reserved = system + input + trajectory + tool budgets.

### RL Parameters (TD-Lambda)

The learning engine. Set `enable_rl=False` to disable entirely for one-off tasks.

**View:** `config.learning`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_rl` | bool | `True` | Master switch for all RL features |
| `rl_verbosity` | str | `"quiet"` | `"quiet"` (minimal), `"normal"` (info), `"verbose"` (debug) |
| `gamma` | float | `0.99` | TD discount factor |
| `lambda_trace` | float | `0.95` | Eligibility trace decay for TD(lambda) |
| `alpha` | float | `0.01` | TD learning rate |
| `baseline` | float | `0.5` | Reward baseline for advantage calculation |
| `n_step` | int | `3` | N-step returns lookahead |
| `q_value_mode` | str | `"simple"` | `"simple"` (average reward per actor) or `"llm"` (LLM-based Q-value prediction) |

#### Adaptive Learning Rate

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_adaptive_alpha` | bool | `True` | Dynamically adjust learning rate |
| `alpha_min` | float | `0.001` | Learning rate floor |
| `alpha_max` | float | `0.1` | Learning rate ceiling |
| `alpha_adaptation_rate` | float | `0.1` | Speed of learning rate adaptation |

#### Intermediate Rewards

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_intermediate_rewards` | bool | `True` | Reward agents during execution, not just at end |
| `architect_proceed_reward` | float | `0.1` | Reward when architect approves a step |
| `tool_success_reward` | float | `0.05` | Reward per successful tool call |

#### Cooperative Rewards (Multi-Agent)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `base_reward_weight` | float | `0.3` | Weight for agent's own success contribution |
| `cooperation_bonus` | float | `0.4` | Bonus for helping other agents |
| `predictability_bonus` | float | `0.3` | Bonus for predictable behavior |

### Persistence

Where and how state is saved. Controls output paths, backup strategy, and learning storage.

**View:** `config.persistence`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `schema_version` | str | `"2.0"` | Schema version for persisted JSON. Major version mismatch triggers migration warning |
| `output_base_dir` | str | `"./outputs"` | Base directory for all outputs |
| `create_run_folder` | bool | `True` | Create timestamped `run_YYYYMMDD_HHMMSS/` folders per run |
| `auto_save_interval` | int | `3` | Save state every N steps |
| `auto_load_on_start` | bool | `True` | Auto-load from `outputs/latest/` if exists |
| `save_interval` | int | `1` | Legacy: episodes between saves |
| `persist_memories` | bool | `True` | Persist memory entries to disk |
| `persist_q_tables` | bool | `True` | Persist Q-tables to disk |
| `persist_brain_state` | bool | `True` | Persist brain/swarm intelligence state |
| `persist_todos` | bool | `True` | Save session TODOs to markdown |
| `persist_agent_outputs` | bool | `True` | Save IOManager outputs |
| `storage_format` | str | `"json"` | `"json"` or `"sqlite"` (not pickle) |
| `compress_large_files` | bool | `True` | Gzip files larger than 1MB |
| `max_runs_to_keep` | int | `10` | Auto-cleanup old run folders beyond this count |
| `enable_backups` | bool | `True` | Enable periodic backups |
| `backup_interval` | int | `100` | Episodes between backups |
| `max_backups` | int | `10` | Maximum backup copies to retain |
| `base_path` | str | `"~/.jotty"` | Legacy persistence location |
| `auto_load` | bool | `True` | Legacy auto-load flag |
| `auto_save` | bool | `True` | Legacy auto-save flag |

#### Learning Persistence

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `auto_load_learning` | bool | `True` | Auto-load previous learning state on startup |
| `per_agent_learning` | bool | `True` | Each agent maintains its own Q-table |
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

Trust adaptation, routing thresholds, and collective memory. Controls how the swarm evaluates and routes to agents.

**View:** `config.intelligence`

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

Epsilon-greedy and UCB exploration controls.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `epsilon_start` | float | `0.3` | Initial exploration rate |
| `epsilon_end` | float | `0.05` | Final exploration rate after decay |
| `epsilon_decay_episodes` | int | `500` | Episodes over which epsilon decays |
| `ucb_coefficient` | float | `2.0` | UCB exploration coefficient (higher = more exploration) |
| `enable_adaptive_exploration` | bool | `True` | Boost exploration when learning stalls |
| `exploration_boost_on_stall` | float | `0.1` | Epsilon boost amount on stall detection |
| `max_exploration_iterations` | int | `10` | Max iterations for policy exploration |
| `policy_update_threshold` | int | `3` | Episodes before updating policy |

### Credit Assignment

How contribution is attributed across agents in a multi-agent episode.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `credit_decay` | float | `0.9` | Temporal decay for credit assignment |
| `min_contribution` | float | `0.1` | Minimum contribution floor per agent |
| `enable_reasoning_credit` | bool | `True` | Use reasoning quality for credit |
| `reasoning_weight` | float | `0.3` | Weight for reasoning-based credit |
| `evidence_weight` | float | `0.2` | Weight for evidence-based credit |

### Inter-Agent Communication

Agent-to-agent message passing and result sharing.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_agent_communication` | bool | `True` | Enable inter-agent messaging |
| `share_tool_results` | bool | `True` | Cache and share tool results across agents |
| `share_insights` | bool | `True` | Share discovered insights between agents |
| `max_messages_per_episode` | int | `20` | Cap on inter-agent messages per episode |

### Multi-Round Validation

Iterative refinement on low-confidence or disagreement.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_multi_round` | bool | `True` | Enable multi-round validation |
| `refinement_on_low_confidence` | float | `0.6` | Trigger refinement when confidence below this |
| `refinement_on_disagreement` | bool | `True` | Trigger refinement when agents disagree |
| `max_refinement_rounds` | int | `2` | Max refinement iterations |

---

## Specialized — Domain-Specific or Rare Use

### LLM-Based RAG

Semantic retrieval without embedding models — uses LLM with sliding window.

**View:** `config.memory_settings`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_llm_rag` | bool | `True` | Enable LLM-based retrieval |
| `rag_window_size` | int | `5` | Memories per LLM call for relevance scoring |
| `rag_max_candidates` | int | `50` | Pre-filter candidates before LLM scoring (discrete mode) |
| `rag_relevance_threshold` | float | `0.6` | Minimum relevance score to include a memory |
| `rag_use_cot` | bool | `True` | Use chain-of-thought for scoring |
| `retrieval_mode` | str | `"synthesize"` | `"synthesize"` (LLM synthesizes wisdom from broad fetch) or `"discrete"` (return individual memories) |
| `synthesis_fetch_size` | int | `200` | Memories to fetch for synthesis mode |
| `synthesis_max_tokens` | int | `800` | Max tokens for synthesized wisdom output |
| `chunk_size` | int | `500` | Tokens per chunk for sliding window |
| `chunk_overlap` | int | `50` | Overlap between chunks |

### Consolidation

Pattern extraction from accumulated memories.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `consolidation_threshold` | int | `100` | Min memories before consolidation triggers |
| `consolidation_interval` | int | `3` | Consolidate every N episodes |
| `min_cluster_size` | int | `5` | Min memories in a cluster to extract a pattern |
| `pattern_confidence_threshold` | float | `0.7` | Min confidence to keep an extracted pattern |

### Offline Learning

Batch updates from experience replay.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `episode_buffer_size` | int | `1000` | Max episodes stored in replay buffer |
| `offline_update_interval` | int | `50` | Run offline update every N episodes |
| `replay_batch_size` | int | `20` | Batch size for replay updates |
| `counterfactual_samples` | int | `5` | Counterfactual scenarios per offline update |

### Protection Mechanisms

Prevent catastrophic forgetting and detect anomalies.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `protected_memory_threshold` | float | `0.8` | Memories above this importance are protected from eviction |
| `task_memory_ratio` | float | `0.3` | Fraction of memory budget reserved for current task |
| `suspicion_threshold` | float | `0.95` | Confidence above this flags a suspiciously certain agent |
| `ood_entropy_threshold` | float | `0.8` | Entropy above this flags out-of-distribution input |
| `min_rejection_rate` | float | `0.05` | Minimum rejection rate to maintain calibration |
| `approval_reward_bonus` | float | `0.1` | Reward bonus for human-approved outputs |
| `rejection_penalty` | float | `0.05` | Penalty for human-rejected outputs |

### Adaptive Learning

Dynamic parameter adjustment when learning stalls.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_adaptive_learning` | bool | `True` | Enable adaptive parameter tuning |
| `stall_detection_window` | int | `100` | Episodes to check for stall |
| `stall_threshold` | float | `0.001` | Improvement below this = stalled |
| `learning_boost_factor` | float | `2.0` | Multiply learning rate by this on stall |

#### Adaptive Learning Thresholds

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `adaptive_window_size` | int | `50` | Window size for learning rate adaptation |
| `instability_threshold_multiplier` | float | `1.5` | `std_dev > mean * this` means unstable |
| `slow_learning_threshold` | float | `0.01` | `mean_error < this` means learning too slowly |
| `goal_transfer_discount` | float | `0.5` | Discount for value transfer to related goals |

### Goal Hierarchy

Knowledge transfer between related goals.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_goal_hierarchy` | bool | `True` | Enable hierarchical goal structure |
| `goal_transfer_weight` | float | `0.3` | Weight for transferred knowledge between goals |

### Causal Learning

Cause-effect relationship discovery.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_causal_learning` | bool | `True` | Enable causal relationship tracking |
| `causal_confidence_threshold` | float | `0.7` | Min confidence to confirm a causal link |
| `causal_min_support` | int | `3` | Episodes needed before causal link is confirmed |

### Deduplication

Redundant memory removal.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_deduplication` | bool | `True` | Enable memory deduplication |
| `similarity_threshold` | float | `0.85` | LLM-judged similarity above this = duplicate |

### Learning Pipeline

Control which learning components run in `post_episode()`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `learning_components` | Optional[List[str]] | `None` | Components to run. `None` = all |

Valid component names: `td_lambda`, `swarm_learner`, `brain_consolidation`, `neurochunk_tiering`, `agent_abstractor`, `transfer_learning`, `swarm_intelligence`, `stigmergy`, `effectiveness`, `mas_learning`, `byzantine`, `credit_assignment`, `auditor_fixes`, `adaptive_learning`, `effectiveness_intervention`, `credit_pruning`, `curriculum`.

### Agentic Discovery Budget

Token budgets for LLM-based artifact analysis.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `preview_token_budget` | int | `20000` | Token budget for LLM artifact analysis |
| `max_description_tokens` | int | `5000` | Per-artifact description token limit |
| `compression_trigger_ratio` | float | `0.8` | Compress when total context exceeds 80% of limit |
| `chunking_threshold_tokens` | int | `15000` | Chunk artifacts larger than this |

**Derived fields** (set in `__post_init__`, 1 token ~ 4 chars):

| Field | Type | Derived From | Description |
|-------|------|-------------|-------------|
| `preview_char_limit` | int | `preview_token_budget * 4` | Char limit for previews (default: 80000) |
| `max_description_chars` | int | `max_description_tokens * 4` | Char limit per description (default: 20000) |

### Token Counting

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `token_model_name` | Optional[str] | `None` | Override model name for token counting (e.g. `"gpt-4o"`). If `None`, uses main model name with automatic mapping |

### Local-First Mode

Privacy-first operation without external API calls.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `local_mode` | bool | `False` | Master switch for local-only inference |
| `local_model` | str | `"ollama/llama3"` | Local model identifier (Ollama format) |

### Agent Registry & Parameter Mappings

Dynamic agent capability tracking and custom parameter aliases.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_agent_registry` | bool | `True` | Track actor capabilities and performance |
| `auto_infer_capabilities` | bool | `True` | LLM infers capabilities if not provided |
| `custom_param_mappings` | Dict[str, List[str]] | `{}` | Custom parameter name mappings for your domain |

### Reproducibility

Deterministic execution via fixed seeds.

**View:** `config.execution`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `random_seed` | Optional[int] | `None` | Fixed random seed. `None` = non-deterministic |
| `numpy_seed` | Optional[int] | `None` | NumPy random seed. Falls back to `random_seed` |
| `torch_seed` | Optional[int] | `None` | PyTorch random seed. Falls back to `random_seed` |
| `python_hash_seed` | Optional[int] | `None` | Python hash randomization seed. Falls back to `random_seed` |
| `enable_deterministic` | bool | `True` | Enable deterministic operations where supported |

When `random_seed` is set, `__post_init__` calls `set_reproducible_seeds()` automatically.

### Logging & Profiling

**View:** `config.monitoring`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `verbose` | int | `1` | Verbosity level (0=silent, 1=normal, 2+=debug) |
| `log_file` | Optional[str] | `None` | Path to log file. `None` = stdout only |
| `enable_debug_logging` | bool | `False` | Enable debug-level logging. Default OFF for production |
| `enable_metrics` | bool | `True` | Enable metrics collection |
| `enable_beautified_logs` | bool | `True` | Generate human-readable logs |
| `enable_debug_logs` | bool | `True` | Keep raw debug logs |
| `log_level` | str | `"INFO"` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `enable_profiling` | bool | `False` | Track execution times for performance analysis |
| `profiling_verbosity` | str | `"summary"` | `"summary"` (end only) or `"detailed"` (per operation) |

---

## Config Views

Eight read/write proxy views group related fields for organized access. All views are properties on `SwarmConfig` — changes propagate bidirectionally.

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
# Read via view
timeout = config.execution.llm_timeout_seconds  # 180.0

# Write via view (updates the parent SwarmConfig)
config.execution.llm_timeout_seconds = 300.0
assert config.llm_timeout_seconds == 300.0  # Same object

# Serialize a view
config.learning.to_dict()  # {'alpha': 0.01, 'gamma': 0.99, ...}
```

---

## ExecutionConfig

Configuration for the tiered execution system. Each tier progressively enables more features. This is a separate config class from SwarmConfig, used by the execution engine.

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
| `max_planning_depth` | int | `3` | 2+ | Max depth for recursive planning |
| `enable_parallel_execution` | bool | `True` | 2+ | Enable parallel step execution |
| `max_concurrent_steps` | int | `3` | 2+ | Max steps running concurrently |
| `memory_backend` | str | `"json"` | 3+ | `"json"`, `"redis"`, or `"postgres"` |
| `memory_ttl_hours` | int | `24` | 3+ | Memory time-to-live in hours |
| `enable_validation` | bool | `True` | 3+ | Enable output validation |
| `validation_retries` | int | `1` | 3+ | Retries on validation failure |
| `track_success_rate` | bool | `True` | 3+ | Track per-skill success rates |
| `enable_td_lambda` | bool | `False` | 4+ | Enable TD(lambda) reinforcement learning |
| `enable_hierarchical_memory` | bool | `False` | 4+ | Enable 5-level memory hierarchy |
| `enable_multi_round_validation` | bool | `False` | 4+ | Enable multi-round validation |
| `memory_levels` | int | `2` | 4+ | Memory levels (2 = episodic + semantic) |
| `enable_swarm_intelligence` | bool | `False` | 4+ | Enable swarm intelligence layer |
| `swarm_name` | Optional[str] | `None` | 4-5 | Swarm type: `"coding"`, `"research"`, `"testing"` |
| `paradigm` | Optional[str] | `None` | 4-5 | Orchestration paradigm: `"relay"`, `"debate"`, `"refinement"` |
| `enable_sandbox` | bool | `False` | 5 | Enable sandboxed execution |
| `enable_coalition` | bool | `False` | 5 | Enable coalition formation |
| `trust_level` | str | `"standard"` | 5 | `"standard"`, `"elevated"`, `"restricted"` |
| `timeout_seconds` | int | `300` | All | Execution timeout (seconds) |
| `max_retries` | int | `3` | All | Max retries on failure |
| `provider` | Optional[str] | `None` | All | LLM provider. `None` = auto-detect |
| `model` | Optional[str] | `None` | All | Model name. `None` = provider default |
| `temperature` | float | `0.7` | All | LLM temperature |
| `max_tokens` | int | `4000` | All | Max output tokens |

### `to_swarm_config()` Mapping

`ExecutionConfig.to_swarm_config()` converts to a `SwarmConfig`-compatible dict:

| ExecutionConfig field | Maps to SwarmConfig field |
|-----------------------|--------------------------|
| `enable_validation` | `enable_validation` |
| `enable_multi_round_validation` | `enable_multi_round` |
| `enable_td_lambda` | `enable_rl`, `enable_causal_learning` |
| `timeout_seconds` | `llm_timeout_seconds` |
| `validation_retries` | `max_eval_retries` |
| `enable_multi_round_validation` | `validation_mode` (`"full"` or `"none"`) |

Note: `provider`, `model`, `temperature` are NOT mapped — they're set via DSPy LM configuration or passed directly to `LLMProvider`.

---

## AgentConfig

Per-agent configuration for individual agents in a swarm.

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
| `provides` | Optional[List[str]] | `None` | Parameter names this agent can provide to others |
| `context_requirements` | Optional[Any] | `None` | `ContextRequirements` instance for context management |
| `architect_tools` | List[Any] | `[]` | Tools available to the Architect phase |
| `auditor_tools` | List[Any] | `[]` | Tools available to the Auditor phase |
| `feedback_rules` | Optional[List[Dict]] | `None` | Rules for routing feedback between agents |
| `capabilities` | Optional[List[str]] | `None` | Agent capability tags for dynamic orchestration |
| `dependencies` | Optional[List[str]] | `None` | Names of agents this agent depends on |
| `metadata` | Optional[Dict[str, Any]] | `None` | Arbitrary metadata |
| `enable_architect` | bool | `True` | Enable Architect validation for this agent |
| `enable_auditor` | bool | `True` | Enable Auditor validation for this agent |
| `validation_mode` | str | `"standard"` | `"quick"`, `"standard"`, or `"thorough"` |
| `is_critical` | bool | `False` | If `True`, failure of this agent fails the whole swarm |
| `max_retries` | int | `0` | Max retries on failure. `0` = use `config_defaults.MAX_RETRIES` |
| `retry_strategy` | str | `"with_hints"` | Retry approach: `"with_hints"` provides feedback from previous failure |
| `is_executor` | bool | `False` | `True` for agents that execute actions (query runners, API callers, file writers) |
| `enabled` | bool | `True` | `False` to disable this agent without removing it |

---

## DEFAULT_PARAM_ALIASES

Centralized parameter name aliases defined in `data_structures.py`. Used by `ParameterResolver` for auto-wiring tool parameters across different naming conventions.

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
config = SwarmConfig(
    custom_param_mappings={
        'user_id': ['customer_id', 'uid', 'account_id'],
    }
)
```

---

## Serialization

```python
# SwarmConfig → flat dict
flat = config.to_flat_dict()

# View → dict (subset)
learning_dict = config.learning.to_dict()

# ExecutionConfig → SwarmConfig-compatible dict
swarm_kwargs = exec_config.to_swarm_config()
config = SwarmConfig(**swarm_kwargs)
```
