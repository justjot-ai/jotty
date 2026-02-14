"""
Q-Learning components for ReVal.

Provides LLM-based Q-value prediction and experience management with
NATURAL LANGUAGE Q-table for semantic generalization.
"""
import dspy
import json
import time
import random
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field

from ..foundation.robust_parsing import AdaptiveWeightGroup

logger = logging.getLogger(__name__)


class LLMQPredictorSignature(dspy.Signature):
    """Predict Q-value (expected reward) for a state-action pair using LLM reasoning."""
    
    state_description = dspy.InputField(desc="Current state of the swarm (TODO state, actor states, memory summary)")
    proposed_action = dspy.InputField(desc="The action being considered (which task, which actor)")
    historical_outcomes = dspy.InputField(desc="JSON of similar past state-action-reward tuples")
    goal_context = dspy.InputField(desc="The root goal we're trying to achieve")
    
    reasoning = dspy.OutputField(desc="Step-by-step reasoning about expected outcome")
    q_value = dspy.OutputField(desc="Predicted Q-value between 0.0 and 1.0")
    confidence = dspy.OutputField(desc="Confidence in this prediction (0.0-1.0)")
    alternative_suggestion = dspy.OutputField(desc="If confidence is low, suggest better action")


class LLMQPredictor:
    """
    LLM-based Q-value predictor using natural language Q-table.
    
    KEY INNOVATION: Instead of Q[s,a] = float, we use:
    Q[(verbose_state_description, verbose_action_description)] = {
        'value': float,
        'context': [experiences],
        'learned_lessons': [natural language lessons],
        'visit_count': int
    }
    
    LLMs can generalize across similar states/actions because they
    understand semantic similarity in natural language!
    
     GENERIC: No domain-specific logic, works for any swarm.
    """
    
    def __init__(self, config: Any) -> None:
        self.config = config
        self.predictor = dspy.ChainOfThought(LLMQPredictorSignature)
        
        # REAL Q-TABLE: (state_desc, action_desc) -> Q-value + context
        self.Q = {}  # Natural language Q-table!
        
        # ===== NEUROCHUNK TIERED MEMORY =====
        # Tier 1: Working Memory (always in context)
        self.tier1_working = []  # Hot path, high Q-value or recent
        self.tier1_max_size = getattr(config, 'tier1_max_size', 50)
        self.tier1_threshold = 0.8  # Adaptive threshold for promotion to Tier 1
        
        # Tier 2: Semantic Clusters (compressed, retrieval-based)
        self.tier2_clusters = {}  # {cluster_id: {'centroid': str, 'members': [keys]}}
        self.tier2_max_clusters = getattr(config, 'tier2_max_clusters', 10)
        
        # Tier 3: Long-term Archive (causal impact pruning)
        self.tier3_archive = []  # Low Q-value but novel/high-variance
        self.tier3_max_size = getattr(config, 'tier3_max_size', 500)
        
        # Adaptive threshold management
        self.last_episode_reward = 0.0
        self.episodes_without_improvement = 0
        
        # Chunker for semantic clustering (lazy-init)
        self._chunker = None
        # ====================================
        
        # Experience buffer for replay
        # A-TEAM ENHANCEMENT: Prioritized replay buffer
        self.experience_buffer = []
        self.max_buffer_size = getattr(config, 'max_experience_buffer', 1000)
        self.priority_alpha = 0.6  # Priority exponent (0=uniform, 1=full prioritization)
        self.priority_beta = 0.4   # Importance sampling (0=no correction, 1=full correction)
        self.priority_epsilon = 0.01  # Small constant to ensure non-zero priority
    
        # Learning parameters
        self.alpha = getattr(config, 'alpha', 0.1)
        self.gamma = getattr(config, 'gamma', 0.99)
        self.epsilon = getattr(config, 'epsilon', 0.1)

        # A-Team v8.0: Q-table size limits (prevents unbounded growth)
        self.max_q_table_size = getattr(config, 'max_q_table_size', 10000)
        self.q_prune_percentage = getattr(config, 'q_prune_percentage', 0.2)  # Remove 20% when limit hit

        # A-Team v8.0: Adaptive retention weights (replaces hardcoded 0.4/0.3/0.2/0.1)
        self.retention_weights = AdaptiveWeightGroup({
            'q_value': 0.4,       # Reward salience
            'novelty': 0.3,       # Rare = important
            'causal_impact': 0.2, # Influence on future
            'staleness': 0.1      # Penalty for old memories
        })
    
    def add_experience(self, state: Dict, action: Dict, reward: float, next_state: Dict = None, done: bool = False) -> None:
        """
        Add experience to buffer AND update Q-table.
        
        This is REAL Q-learning - not just storage!
        
        Args:
            state: Current state (will be converted to natural language description)
            action: Action taken (will be converted to natural language description)
            reward: Reward received
            next_state: Next state (for bootstrapping)
            done: Whether episode terminated
        """
        # Convert to natural language descriptions
        state_desc = self._state_to_natural_language(state)
        action_desc = self._action_to_natural_language(action)
        next_state_desc = self._state_to_natural_language(next_state) if next_state else None
        
        # Add to experience buffer
        experience = {
            'state': state,
            'state_desc': state_desc,
            'action': action,
            'action_desc': action_desc,
            'reward': reward,
            'next_state': next_state,
            'next_state_desc': next_state_desc,
            'done': done,
            'timestamp': time.time()
        }
        # A-TEAM: Prioritized Experience Replay
        # Priority = |TD_error| + ε (per GRF MARL paper)
        td_error = abs(reward - self._get_q_value(state_desc, action_desc))
        priority = (td_error + self.priority_epsilon) ** self.priority_alpha
        experience['priority'] = priority
        experience['td_error'] = td_error
        
        self.experience_buffer.append(experience)
        
        # Keep buffer bounded - evict LOW priority experiences (not FIFO)
        if len(self.experience_buffer) > self.max_buffer_size:
            # Sort by priority (ascending) and remove lowest
            self.experience_buffer.sort(key=lambda e: e.get('priority', 0))
            self.experience_buffer.pop(0)
    
        # REAL Q-LEARNING UPDATE
        self._update_q_value(state_desc, action_desc, reward, next_state_desc, done)
    
    def record_outcome(self, state: Dict, action: Any, reward: float, next_state: Dict = None, done: bool = False) -> None:
        """Record outcome - alias for add_experience() for backwards compat."""
        action_dict = action if isinstance(action, dict) else {'actor': str(action)}
        self.add_experience(state, action_dict, reward, next_state, done)
    
    def _state_to_natural_language(self, state: Dict) -> str:
        """
        Convert state dict to RICH natural language description.
        
         A-TEAM CRITICAL FIX: State must capture SEMANTIC CONTEXT!
        
        This enables:
        1. Similar query matching (Q-value transfer)
        2. Error pattern recognition (avoid repeated mistakes)
        3. Metadata-aware learning (partition columns, date formats)
        4. Tool usage patterns (which tools work for which queries)
        
        Example:
        Instead of: "TODO: 2 completed, 1 pending"
        We get: "QUERY: Count P2P transactions yesterday | DOMAIN: UPI/transactions | 
                 TABLES: fact_upi_transactions | PARTITION: dl_last_updated | 
                 DATE_FILTER: yesterday | AGGREGATION: COUNT_DISTINCT | 
                 ERRORS: [COLUMN_NOT_FOUND:txn_date,date,dt] | RESOLUTION: use partition column"
        """
        if not state:
            return "Initial state (no history)"
        
        parts = []
        
        # ====== 1. QUERY SEMANTICS (CRITICAL!) ======
        if 'query' in state:
            query = state['query']
            parts.append(f"QUERY: {query[:100]}")
            
            # Extract query intent
            query_lower = query.lower()
            intents = []
            if 'count' in query_lower:
                intents.append('COUNT')
            if 'sum' in query_lower or 'total' in query_lower or 'gmv' in query_lower:
                intents.append('SUM')
            if 'average' in query_lower or 'avg' in query_lower:
                intents.append('AVG')
            if intents:
                parts.append(f"INTENT: {'+'.join(intents)}")
        
        # ====== 2. GENERIC KEY-VALUE CONTEXT ======
        # Extract all non-complex key-value pairs from state dict
        # (replaces hardcoded temporal/domain sections)
        skip_keys = {'query', 'tables', 'relevant_tables', 'partition_column',
                      'date_column', 'columns', 'filters', 'aggregation', 'errors',
                      'columns_tried', 'working_column', 'error_resolution',
                      'tool_calls', 'successful_tools', 'failed_tools',
                      'tool_io_schemas', 'df_schema', 'attempts', 'success',
                      'execution_time_ms', 'todo', 'current_actor', 'actor_output',
                      'actor_role', 'actor_strengths', 'cooperation_score',
                      'help_received', 'help_given', 'architect_confidence',
                      'auditor_result', 'validation_passed', 'trajectory_length',
                      'recent_outcomes'}
        generic_kvs = []
        for k, v in state.items():
            if k in skip_keys:
                continue
            if isinstance(v, (str, int, float, bool)):
                generic_kvs.append(f"{k.upper()}={v}")
            elif isinstance(v, list) and len(v) <= 5 and all(isinstance(i, str) for i in v):
                generic_kvs.append(f"{k.upper()}={','.join(v)}")
        if generic_kvs:
            parts.append(f"CONTEXT: {' | '.join(generic_kvs[:10])}")
        
        # ====== 4. METADATA CONTEXT (Tables, Columns, Partitions) ======
        if 'tables' in state or 'relevant_tables' in state:
            tables = state.get('tables') or state.get('relevant_tables', [])
            if isinstance(tables, list) and tables:
                # Extract short table names
                short_names = [t.split('.')[-1] if '.' in t else t for t in tables[:3]]
                parts.append(f"TABLES: {','.join(short_names)}")
        
        if 'partition_column' in state:
            parts.append(f"PARTITION_COL: {state['partition_column']}")
        
        if 'date_column' in state:
            parts.append(f"DATE_COL: {state['date_column']}")
        
        if 'columns' in state:
            cols = state['columns']
            if isinstance(cols, list) and cols:
                parts.append(f"COLUMNS: {','.join(cols[:5])}")
        
        # ====== 5. FILTER CONTEXT ======
        if 'filters' in state:
            filters = state['filters']
            if isinstance(filters, dict):
                filter_keys = list(filters.keys())[:3]
                parts.append(f"FILTERS: {','.join(filter_keys)}")
            elif isinstance(filters, str) and filters:
                parts.append(f"FILTER_EXPR: {filters[:50]}")
        
        # ====== 6. AGGREGATION CONTEXT ======
        if 'aggregation' in state:
            parts.append(f"AGGREGATION: {state['aggregation']}")
        
        # ====== 7. ERROR PATTERNS (CRITICAL FOR LEARNING!) ======
        if 'errors' in state:
            errors = state['errors']
            if isinstance(errors, list) and errors:
                # Group by error type
                error_types = []
                for err in errors[:5]:
                    if isinstance(err, dict):
                        error_types.append(f"{err.get('type','ERR')}:{err.get('column','')}")
                    else:
                        error_types.append(str(err)[:30])
                parts.append(f"ERRORS: [{','.join(error_types)}]")
        
        if 'columns_tried' in state:
            cols_tried = state['columns_tried']
            if isinstance(cols_tried, list) and cols_tried:
                parts.append(f"COLS_TRIED: {','.join(cols_tried[:5])}")
        
        if 'working_column' in state:
            parts.append(f"WORKING_COL: {state['working_column']}")
        
        if 'error_resolution' in state:
            parts.append(f"RESOLUTION: {state['error_resolution']}")
        
        # ====== 8. TOOL USAGE PATTERNS ======
        if 'tool_calls' in state:
            tools = state['tool_calls']
            if isinstance(tools, list) and tools:
                tool_names = [t.get('tool', t) if isinstance(t, dict) else str(t) for t in tools[:5]]
                parts.append(f"TOOLS_USED: {','.join(tool_names)}")
        
        if 'successful_tools' in state:
            succ = state['successful_tools']
            if isinstance(succ, list) and succ:
                parts.append(f"TOOLS_SUCCESS: {','.join(succ[:3])}")
        
        if 'failed_tools' in state:
            failed = state['failed_tools']
            if isinstance(failed, list) and failed:
                parts.append(f"TOOLS_FAILED: {','.join(failed[:3])}")
        
        # A-TEAM FIX: Tool I/O schemas (esp DataFrame)
        if 'tool_io_schemas' in state:
            schemas = state['tool_io_schemas']
            if isinstance(schemas, dict):
                schema_parts = []
                for tool_name, schema in list(schemas.items())[:3]:
                    if isinstance(schema, dict):
                        input_keys = schema.get('input', [])
                        output_keys = schema.get('output', [])
                        schema_parts.append(f"{tool_name}({','.join(input_keys[:2])})->{','.join(output_keys[:2])}")
                    else:
                        schema_parts.append(f"{tool_name}:{str(schema)[:20]}")
                if schema_parts:
                    parts.append(f"TOOL_SCHEMAS: {';'.join(schema_parts)}")
        
        # A-TEAM FIX: DataFrame schema hints
        if 'df_schema' in state:
            df_schema = state['df_schema']
            if isinstance(df_schema, dict):
                cols = df_schema.get('columns', [])
                if cols:
                    parts.append(f"DF_COLS: {','.join(cols[:5])}")
                if 'row_count' in df_schema:
                    parts.append(f"DF_ROWS: {df_schema['row_count']}")
        
        # ====== 9. EXECUTION STATS ======
        if 'attempts' in state:
            parts.append(f"ATTEMPTS: {state['attempts']}")
        
        if 'success' in state:
            parts.append(f"SUCCESS: {state['success']}")
        
        if 'execution_time_ms' in state:
            parts.append(f"TIME_MS: {state['execution_time_ms']}")
        
        # ====== 10. TASK PROGRESS ======
        if 'todo' in state:
            todo = state['todo']
            if isinstance(todo, dict):
                pending = todo.get('pending', 0)
                completed = todo.get('completed', 0)
                failed = todo.get('failed', 0)
                parts.append(f"TODO: {completed}/{pending}⏳/{failed}")
        
        # ====== 11. ACTOR CONTEXT ======
        if 'current_actor' in state:
            parts.append(f"ACTOR: {state['current_actor']}")
        
        if 'actor_output' in state:
            output = state['actor_output']
            if isinstance(output, dict):
                output_keys = list(output.keys())[:3]
                parts.append(f"OUTPUT_KEYS: {','.join(output_keys)}")
        
        # A-TEAM: Role hints and specialization (per GRF MARL paper)
        if 'actor_role' in state:
            parts.append(f"ROLE: {state['actor_role']}")
        
        if 'actor_strengths' in state:
            strengths = state['actor_strengths']
            if isinstance(strengths, list):
                parts.append(f"STRENGTHS: {','.join(strengths[:3])}")
        
        if 'cooperation_score' in state:
            parts.append(f"COOP_SCORE: {state['cooperation_score']:.2f}")
        
        if 'help_received' in state:
            helpers = state['help_received']
            if isinstance(helpers, dict):
                top_helpers = sorted(helpers.items(), key=lambda x: x[1], reverse=True)[:2]
                if top_helpers:
                    parts.append(f"HELPED_BY: {','.join([h[0] for h in top_helpers])}")
        
        if 'help_given' in state:
            helped = state['help_given']
            if isinstance(helped, dict):
                top_helped = sorted(helped.items(), key=lambda x: x[1], reverse=True)[:2]
                if top_helped:
                    parts.append(f"HELPED: {','.join([h[0] for h in top_helped])}")
        
        # ====== 12. VALIDATION CONTEXT ======
        if 'architect_confidence' in state:
            parts.append(f"ARCHITECT_CONF: {state['architect_confidence']:.2f}")
        
        if 'auditor_result' in state:
            parts.append(f"AUDITOR: {state['auditor_result']}")
        
        if 'validation_passed' in state:
            parts.append(f"VAL_PASSED: {state['validation_passed']}")
        
        # ====== 13. TRAJECTORY/HISTORY ======
        if 'trajectory_length' in state:
            parts.append(f"TRAJ_LEN: {state['trajectory_length']}")
        
        if 'recent_outcomes' in state:
            outcomes = state['recent_outcomes']
            if isinstance(outcomes, list) and outcomes:
                outcome_str = ''.join(['' if o else '' for o in outcomes[-5:]])
                parts.append(f"RECENT: {outcome_str}")
        
        # ====== FALLBACK ======
        if not parts:
            # Include at least some structure from state
            state_keys = list(state.keys())[:5]
            return f"STATE_KEYS: {','.join(state_keys)}"
        
        return " | ".join(parts)
    
    def _action_to_natural_language(self, action: Dict) -> str:
        """
        Convert action dict to RICH natural language description.
        
         A-TEAM CRITICAL FIX: Action must capture SEMANTIC INTENT!
        
        Example:
        Instead of: "Actor: SQLGenerator; Task: SQLGenerator_main"
        We get: "ACTOR: SQLGenerator | TASK: Generate SQL | TARGET_TABLES: fact_upi_transactions |
                 OPERATION: COUNT_DISTINCT | FILTERS: P2P+yesterday | TOOL: execute_query |
                 PARTITION_STRATEGY: use dl_last_updated"
        """
        if not action:
            return "No action"
        
        parts = []
        
        # ====== 1. ACTOR IDENTITY ======
        if 'actor' in action:
            parts.append(f"ACTOR: {action['actor']}")
        
        # ====== 2. TASK/GOAL ======
        if 'task' in action:
            parts.append(f"TASK: {action['task']}")
        
        if 'goal' in action:
            parts.append(f"GOAL: {action['goal'][:80]}")
        
        # ====== 3. QUERY CONTEXT (if generating SQL) ======
        if 'query' in action:
            parts.append(f"QUERY: {action['query'][:50]}")
        
        if 'sql' in action:
            sql = action['sql']
            # Extract key SQL components
            sql_lower = sql.lower() if isinstance(sql, str) else ''
            if 'count' in sql_lower:
                parts.append("OP: COUNT")
            if 'sum' in sql_lower:
                parts.append("OP: SUM")
            if 'distinct' in sql_lower:
                parts.append("DISTINCT: True")
        
        # ====== 4. TARGET TABLES ======
        if 'tables' in action:
            tables = action['tables']
            if isinstance(tables, list) and tables:
                short_names = [t.split('.')[-1] for t in tables[:2]]
                parts.append(f"TABLES: {','.join(short_names)}")
        
        if 'table' in action:
            parts.append(f"TABLE: {action['table'].split('.')[-1]}")
        
        # ====== 5. COLUMNS/FIELDS ======
        if 'columns' in action:
            cols = action['columns']
            if isinstance(cols, list) and cols:
                parts.append(f"COLS: {','.join(cols[:3])}")
        
        if 'select_columns' in action:
            cols = action['select_columns']
            if isinstance(cols, list):
                parts.append(f"SELECT: {','.join(cols[:3])}")
        
        # ====== 6. FILTERS ======
        if 'filters' in action:
            filters = action['filters']
            if isinstance(filters, dict):
                filter_desc = '+'.join(list(filters.keys())[:3])
                parts.append(f"FILTERS: {filter_desc}")
            elif isinstance(filters, str):
                parts.append(f"FILTER: {filters[:30]}")
        
        # ====== 7. DATE HANDLING (CRITICAL!) ======
        if 'date_column' in action:
            parts.append(f"DATE_COL: {action['date_column']}")
        
        if 'partition_column' in action:
            parts.append(f"PARTITION: {action['partition_column']}")
        
        if 'date_filter' in action:
            parts.append(f"DATE_FILTER: {action['date_filter']}")
        
        # ====== 8. TOOL USAGE ======
        if 'tool' in action:
            parts.append(f"TOOL: {action['tool']}")
        
        if 'tool_args' in action:
            args = action['tool_args']
            if isinstance(args, dict):
                arg_keys = list(args.keys())[:3]
                parts.append(f"ARGS: {','.join(arg_keys)}")
        
        # ====== 9. STRATEGY/APPROACH ======
        if 'strategy' in action:
            parts.append(f"STRATEGY: {action['strategy']}")
        
        if 'approach' in action:
            parts.append(f"APPROACH: {action['approach']}")
        
        # ====== 10. ERROR HANDLING ======
        if 'retry_reason' in action:
            parts.append(f"RETRY: {action['retry_reason'][:30]}")
        
        if 'fallback' in action:
            parts.append(f"FALLBACK: {action['fallback']}")
        
        # ====== 11. PARAMETERS ======
        if 'params' in action:
            params = action['params']
            if isinstance(params, dict) and params:
                # Extract key params only
                key_params = []
                for k, v in list(params.items())[:3]:
                    v_str = str(v)[:20] if len(str(v)) > 20 else str(v)
                    key_params.append(f"{k}={v_str}")
                parts.append(f"PARAMS: {','.join(key_params)}")
        
        # ====== FALLBACK ======
        if not parts:
            action_keys = list(action.keys())[:5]
            return f"ACTION_KEYS: {','.join(action_keys)}"
        
        return " | ".join(parts)
    
    def _update_q_value(self, state_desc: str, action_desc: str, reward: float, next_state_desc: str = None, done: bool = False) -> Any:
        """
        REAL Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        
        KEY: State and action are NATURAL LANGUAGE, so LLM can generalize!
        """
        key = (state_desc, action_desc)
        
        # Get current Q-value
        current_q = self._get_q_value_from_table(state_desc, action_desc)
        
        # Get max Q-value for next state (if not terminal)
        if done or not next_state_desc:
            max_next_q = 0.0
        else:
            # Find all actions we've tried from similar states
            # (LLM will generalize based on semantic similarity!)
            max_next_q = self._get_max_next_q(next_state_desc)
        
        # Q-learning update
        target = reward + self.gamma * max_next_q
        td_error = target - current_q
        new_q = current_q + self.alpha * td_error
        
        # Clamp to [0, 1]
        new_q = max(0.0, min(1.0, new_q))
        
        # Store/update in Q-table
        if key not in self.Q:
            self.Q[key] = {
                'value': new_q,
                'context': [],
                'learned_lessons': [],
                'visit_count': 1,
                'avg_reward': reward,
                'td_errors': [td_error],
                'created_at': time.time(),
                'last_updated': time.time()
            }
        else:
            self.Q[key]['value'] = new_q
            self.Q[key]['visit_count'] += 1
            # Running average of reward
            n = self.Q[key]['visit_count']
            self.Q[key]['avg_reward'] = ((n-1) * self.Q[key]['avg_reward'] + reward) / n
            self.Q[key]['td_errors'].append(td_error)
            self.Q[key]['last_updated'] = time.time()
            
            # Keep TD errors bounded
            if len(self.Q[key]['td_errors']) > 20:
                self.Q[key]['td_errors'] = self.Q[key]['td_errors'][-20:]
        
        # Add learned lesson (natural language!)
        lesson = self._extract_lesson(state_desc, action_desc, reward, td_error)
        if lesson:
            self.Q[key]['learned_lessons'].append(lesson)
            # Keep bounded
            if len(self.Q[key]['learned_lessons']) > 5:
                self.Q[key]['learned_lessons'] = self.Q[key]['learned_lessons'][-5:]

        # A-Team v8.0: Enforce Q-table size limits
        self._enforce_q_table_limits()

        return td_error

    def _enforce_q_table_limits(self) -> int:
        """
        Enforce Q-table size limits with smart pruning.

        A-Team Critical Fix: Prevents unbounded Q-table growth.

        Pruning strategy:
        1. Compute retention score for each entry
        2. Sort by retention score (ascending)
        3. Remove bottom N% lowest scores

        Returns:
            Number of entries pruned
        """
        if len(self.Q) <= self.max_q_table_size:
            return 0

        # Calculate how many to remove (20% by default)
        num_to_remove = int(len(self.Q) * self.q_prune_percentage)
        num_to_remove = max(num_to_remove, len(self.Q) - self.max_q_table_size)

        logger.info(
            f" Q-table limit exceeded ({len(self.Q)} > {self.max_q_table_size}). "
            f"Pruning {num_to_remove} entries..."
        )

        # Calculate retention scores for all entries
        scored_keys = []
        current_time = time.time()

        for key, q_data in self.Q.items():
            # 1. Q-value component (higher = more valuable)
            q_value_score = q_data.get('value', 0.5)

            # 2. Visit count component (more visits = more valuable)
            visit_count = q_data.get('visit_count', 1)
            visit_score = min(1.0, visit_count / 10.0)  # Cap at 10 visits

            # 3. Recency component (more recent = more valuable)
            last_updated = q_data.get('last_updated', q_data.get('created_at', current_time))
            age_hours = (current_time - last_updated) / 3600
            recency_score = max(0.0, 1.0 - (age_hours / 168))  # Decay over 1 week

            # 4. TD error variance (high variance = interesting, keep)
            td_errors = q_data.get('td_errors', [])
            if len(td_errors) > 1:
                import statistics
                td_variance = statistics.variance(td_errors)
                variance_score = min(1.0, td_variance * 5)  # Scale up
            else:
                variance_score = 0.5

            # 5. Lesson count (more lessons = more valuable)
            lesson_count = len(q_data.get('learned_lessons', []))
            lesson_score = min(1.0, lesson_count / 3.0)

            # Compute weighted retention score using adaptive weights
            weights = self.retention_weights.get_all()
            retention_score = (
                weights.get('q_value', 0.4) * q_value_score +
                weights.get('novelty', 0.3) * variance_score +
                weights.get('causal_impact', 0.2) * (visit_score + lesson_score) / 2 +
                weights.get('staleness', 0.1) * recency_score
            )

            scored_keys.append((key, retention_score, q_data))

        # Sort by retention score (ascending - lowest first for removal)
        scored_keys.sort(key=lambda x: x[1])

        # Remove entries with lowest retention scores
        keys_to_remove = [k for k, _, _ in scored_keys[:num_to_remove]]
        for key in keys_to_remove:
            del self.Q[key]

        logger.info(
            f" Q-table pruned: {len(self.Q)} entries remaining "
            f"(removed {num_to_remove} with lowest retention scores)"
        )

        return num_to_remove

    def get_q_table_stats(self) -> Dict[str, Any]:
        """
        Get Q-table statistics.

        Returns:
            Dictionary with Q-table size, average Q-value, etc.
        """
        if not self.Q:
            return {
                'size': 0,
                'max_size': self.max_q_table_size,
                'utilization': 0.0,
            }

        q_values = [q_data.get('value', 0.5) for q_data in self.Q.values()]
        visit_counts = [q_data.get('visit_count', 1) for q_data in self.Q.values()]

        return {
            'size': len(self.Q),
            'max_size': self.max_q_table_size,
            'utilization': len(self.Q) / self.max_q_table_size,
            'avg_q_value': sum(q_values) / len(q_values),
            'min_q_value': min(q_values),
            'max_q_value': max(q_values),
            'total_visits': sum(visit_counts),
            'avg_visits': sum(visit_counts) / len(visit_counts),
        }
    
    def _get_q_value(self, state_desc: str, action_desc: str) -> float:
        """Alias for _get_q_value_from_table (for backward compatibility)."""
        return self._get_q_value_from_table(state_desc, action_desc)
    
    def _get_q_value_from_table(self, state_desc: str, action_desc: str) -> float:
        """Get Q-value from table, with semantic fallback."""
        key = (state_desc, action_desc)
        
        if key in self.Q:
            return self.Q[key]['value']
        
        # Check for semantically similar states (simple heuristic)
        # In a more advanced version, use embedding similarity
        for (s, a), q_data in self.Q.items():
            if self._are_similar(state_desc, s) and self._are_similar(action_desc, a):
                # Use Q-value from similar (state, action) pair
                return q_data['value'] * 0.9  # Slight discount for not exact match
        
        return 0.5  # Neutral default for truly novel (state, action)
    
    def _get_max_next_q(self, next_state_desc: str) -> float:
        """Get max Q-value for next state across all actions."""
        max_q = 0.0
        
        for (s, a), q_data in self.Q.items():
            if self._are_similar(next_state_desc, s):
                max_q = max(max_q, q_data['value'])
        
        return max_q if max_q > 0 else 0.5  # Default if no similar states
    
    def _are_similar(self, desc1: str, desc2: str) -> bool:
        """
        Check if two natural language descriptions are similar.

        Uses structured-field-aware similarity when descriptions contain
        KEY: value | KEY: value segments (from _state_to_natural_language).
        Falls back to Jaccard word-overlap for unstructured strings.
        """
        if desc1 == desc2:
            return True

        # Try structured comparison first
        fields1 = self._parse_structured_fields(desc1)
        fields2 = self._parse_structured_fields(desc2)

        if fields1 and fields2:
            # Structured comparison: compare by field-key overlap
            keys1 = set(fields1.keys())
            keys2 = set(fields2.keys())
            shared_keys = keys1 & keys2
            all_keys = keys1 | keys2

            if not all_keys:
                return False

            # Must share at least 50% of keys
            key_overlap = len(shared_keys) / len(all_keys)
            if key_overlap < 0.5:
                return False

            # For shared keys, compute word-overlap on values
            value_scores = []
            for key in shared_keys:
                v1_words = set(fields1[key].lower().split())
                v2_words = set(fields2[key].lower().split())
                if v1_words and v2_words:
                    v_overlap = len(v1_words & v2_words) / len(v1_words | v2_words)
                    value_scores.append(v_overlap)
                else:
                    value_scores.append(0.0)

            avg_value_sim = sum(value_scores) / len(value_scores) if value_scores else 0.0

            # Combined: 40% key overlap + 60% value similarity
            combined = 0.4 * key_overlap + 0.6 * avg_value_sim
            return combined > 0.4

        # Fallback: Jaccard word-overlap for unstructured strings
        # Filter out stopwords to focus on meaningful content words
        _stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                      'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                      'and', 'or', 'but', 'not', 'no', 'if', 'then', 'than',
                      'this', 'that', 'these', 'those', 'it', 'its', 'my', 'your',
                      'query', 'agent', 'auto', 'task', 'type'}
        words1 = set(desc1.lower().split()) - _stopwords
        words2 = set(desc2.lower().split()) - _stopwords

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2)
        union = len(words1 | words2)

        similarity = overlap / union if union > 0 else 0
        return similarity > 0.6  # Raised from 0.5 to reduce false matches

    def _parse_structured_fields(self, desc: str) -> Dict[str, str]:
        """Parse KEY: value segments from structured descriptions."""
        fields = {}
        segments = desc.split(' | ')
        for segment in segments:
            segment = segment.strip()
            if ':' in segment:
                key, _, value = segment.partition(':')
                key = key.strip().upper()
                value = value.strip()
                if key and value:
                    fields[key] = value
        return fields
    
    def _extract_lesson(self, state_desc: str, action_desc: str, reward: float, td_error: float) -> Optional[str]:
        """
        Extract ACTIONABLE natural language lesson from experience.
        
         A-TEAM: Lessons must be SPECIFIC and ACTIONABLE!
        
        Example (BAD):  "SUCCESS: In state 'TODO: 2 completed', action 'Actor: SQL' worked"
        Example (GOOD): "SUCCESS: For P2P transaction counts, use dl_last_updated as partition column (not txn_date/date/dt)"
        """
        if abs(td_error) < 0.1:
            return None  # Not significant enough
        
        # Extract key insights from state and action
        insights = []
        
        # Parse state for domain/query type
        if 'DOMAIN:' in state_desc:
            domain = state_desc.split('DOMAIN:')[1].split('|')[0].strip()
            insights.append(f"Domain={domain}")
        
        if 'QUERY:' in state_desc:
            query = state_desc.split('QUERY:')[1].split('|')[0].strip()[:50]
            insights.append(f"Query='{query}'")
        
        # Parse state for error patterns (MOST VALUABLE!)
        if 'COLS_TRIED:' in state_desc:
            cols_tried = state_desc.split('COLS_TRIED:')[1].split('|')[0].strip()
            insights.append(f"tried [{cols_tried}]")
        
        if 'WORKING_COL:' in state_desc:
            working_col = state_desc.split('WORKING_COL:')[1].split('|')[0].strip()
            insights.append(f"use [{working_col}] instead")
        
        if 'RESOLUTION:' in state_desc:
            resolution = state_desc.split('RESOLUTION:')[1].split('|')[0].strip()
            insights.append(f"Resolution: {resolution}")
        
        # Parse state for tables
        if 'TABLES:' in state_desc:
            tables = state_desc.split('TABLES:')[1].split('|')[0].strip()
            insights.append(f"tables={tables}")
        
        # Parse action for tool/strategy
        if 'TOOL:' in action_desc:
            tool = action_desc.split('TOOL:')[1].split('|')[0].strip()
            insights.append(f"tool={tool}")
        
        if 'PARTITION:' in action_desc:
            partition = action_desc.split('PARTITION:')[1].split('|')[0].strip()
            insights.append(f"partition={partition}")
        
        # Build lesson based on outcome
        insight_str = "; ".join(insights) if insights else f"{state_desc[:60]}... → {action_desc[:40]}..."
        
        if reward > 0.7:
            # Success lesson - capture what worked
            if 'WORKING_COL:' in state_desc or 'RESOLUTION:' in state_desc:
                return f" LEARNED: {insight_str} → SUCCESS (reward={reward:.2f})"
            return f" SUCCESS: {insight_str} (reward={reward:.2f})"
        
        elif reward < 0.3:
            # Failure lesson - capture what to avoid
            if 'COLS_TRIED:' in state_desc:
                return f" AVOID: {insight_str} → FAILED (reward={reward:.2f})"
            return f" FAILED: {insight_str} (reward={reward:.2f})"
        
        elif td_error > 0.2:
            # Better than expected - good strategy found
            return f" DISCOVERY: {insight_str} performed better than expected (actual={reward:.2f}, expected={reward-td_error:.2f})"
        
        elif td_error < -0.2:
            # Worse than expected - strategy didn't generalize
            return f" CAUTION: {insight_str} performed worse than expected (actual={reward:.2f}, expected={reward-td_error:.2f})"
        
        return None
    
    def _get_similar_experiences(self, state: Dict, action: Dict) -> List[Dict]:
        """Get similar past experiences for few-shot learning."""
        state_desc = self._state_to_natural_language(state)
        action_desc = self._action_to_natural_language(action)
        
        similar = []
        for exp in self.experience_buffer:
            exp_state_desc = exp.get('state_desc', '')
            exp_action_desc = exp.get('action_desc', '')
            
            # Check semantic similarity
            if self._are_similar(state_desc, exp_state_desc) or \
               self._are_similar(action_desc, exp_action_desc):
                similar.append(exp)
        
        return sorted(similar, key=lambda x: x.get('timestamp', 0), reverse=True)[:10]  # Top 10
    
    def predict_q_value(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        goal: str = ""
    ) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        Predict Q-value for a state-action pair.

        Modes:
        - "simple": Average reward per actor (fast, reliable, for natural dependencies)
        - "llm": LLM-based prediction (USP - semantic generalization)

        Returns:
            (q_value, confidence, alternative_suggestion)
        """
        try:
            # Check mode from config
            q_mode = getattr(self.config, 'q_value_mode', 'simple')

            if q_mode == 'simple':
                # SIMPLE MODE: Average reward per actor
                # Fast, reliable, perfect for natural dependencies
                actor = action.get('actor', '')
                if actor:
                    # Calculate average reward for this actor from experience buffer
                    actor_experiences = [exp for exp in self.experience_buffer
                                       if exp.get('action', {}).get('actor') == actor]

                    if actor_experiences:
                        rewards = [exp.get('reward', 0.0) for exp in actor_experiences]
                        avg_reward = sum(rewards) / len(rewards)
                        return avg_reward, 0.9, None

            # LLM MODE: Semantic Q-value prediction (fallback for simple mode too)
            # Get similar experiences for few-shot learning
            similar_exps = self._get_similar_experiences(state, action)
            
            # Format historical outcomes
            historical = []
            for exp in similar_exps:  # Top 5
                historical.append({
                    'state': str(exp.get('state', {})),
                    'action': str(exp.get('action', {})),
                    'reward': exp.get('reward', 0.0)
                })
            
            # Prepare inputs
            state_desc = str(state)
            action_desc = str(action)
            historical_json = json.dumps(historical, indent=2)
            
            # Predict using LLM
            result = self.predictor(
                state_description=state_desc,
                proposed_action=action_desc,
                historical_outcomes=historical_json,
                goal_context=goal
            )
            
            # Parse outputs
            try:
                q_val = float(result.q_value)
                q_val = max(0.0, min(1.0, q_val))  # Clamp to [0,1]
            except (ValueError, TypeError, AttributeError) as e:
                logger.debug(f"Q-value parsing failed: {e}")
                q_val = 0.5  # Neutral default

            try:
                conf = float(result.confidence)
                conf = max(0.0, min(1.0, conf))
            except (ValueError, TypeError, AttributeError) as e:
                logger.debug(f"Confidence parsing failed: {e}")
                conf = 0.5
            
            alt = result.alternative_suggestion if hasattr(result, 'alternative_suggestion') else None
            
            return q_val, conf, alt
            
        except Exception as e:
            # Fallback: use Q-table if available
            state_desc = self._state_to_natural_language(state)
            action_desc = self._action_to_natural_language(action)
            q_from_table = self._get_q_value_from_table(state_desc, action_desc)
            
            if q_from_table != 0.5:  # Found in table
                return q_from_table, 0.5, None
            
            # Last resort: historical average
            if similar_exps:
                avg_reward = sum(exp.get('reward', 0.0) for exp in similar_exps) / len(similar_exps)
                return avg_reward, 0.3, None
            return 0.5, 0.1, None
    
    def get_learned_context(self, state: Dict, action: Dict = None) -> str:
        """
        Get learned context to inject into prompts.
        
        THIS IS HOW LEARNING MANIFESTS IN LLM AGENTS!
        
        Returns natural language lessons learned from Q-table.
        Filters by task_type when available to avoid irrelevant lessons
        (e.g., stock analysis lessons polluting code generation tasks).
        """
        state_desc = self._state_to_natural_language(state)
        # Extract task_type for filtering (if provided in state)
        current_task_type = state.get('task_type', '').lower().strip() if isinstance(state, dict) else ''
        
        # Collect relevant lessons
        lessons = []
        
        if action:
            # Get lessons for specific (state, action) pair
            action_desc = self._action_to_natural_language(action)
            key = (state_desc, action_desc)
            
            if key in self.Q:
                q_data = self.Q[key]
                lessons.extend(q_data.get('learned_lessons', []))
        else:
            # Get lessons for similar states, filtered by task_type
            for (s, a), q_data in self.Q.items():
                # If we have a task_type, only match entries of the same type
                if current_task_type:
                    # Parse task_type from stored state description
                    stored_fields = self._parse_structured_fields(s)
                    stored_type = stored_fields.get('TASK_TYPE', '').lower().strip()
                    # Skip if stored entry has a different task_type
                    if stored_type and stored_type != current_task_type:
                        continue
                
                if self._are_similar(state_desc, s):
                    lessons.extend(q_data.get('learned_lessons', []))
        
        if not lessons:
            return ""
        
        # Deduplicate and sort by recency
        unique_lessons = list(dict.fromkeys(lessons))  # Preserve order, remove dupes
        
        context = "# Q-Learning Lessons (Learned from Experience):\n"
        for i, lesson in enumerate(unique_lessons[:5], 1):  # Top 5 (was 10 — too noisy)
            context += f"{i}. {lesson}\n"
        
        # Add Q-value statistics
        if action:
            action_desc = self._action_to_natural_language(action)
            q_val = self._get_q_value_from_table(state_desc, action_desc)
            context += f"\n# Expected Value: Q(state, action) = {q_val:.3f}\n"
        
        return context
    
    def get_best_action(self, state: Dict, available_actions: List[Dict]) -> Tuple[Dict, float, str]:
        """
        Choose best action for given state using learned Q-values.
        
        Returns: (best_action, q_value, reasoning)
        """
        if not available_actions:
            return None, 0.5, "No actions available"
        
        state_desc = self._state_to_natural_language(state)
        
        # Get Q-values for all actions
        action_values = []
        for action in available_actions:
            action_desc = self._action_to_natural_language(action)
            q_val = self._get_q_value_from_table(state_desc, action_desc)
            action_values.append((action, q_val, action_desc))
        
        # Epsilon-greedy
        import random
        if random.random() < self.epsilon:
            # Explore
            chosen = random.choice(action_values)
            reasoning = f"Exploring (ε={self.epsilon}): Trying {chosen[2][:100]}..."
        else:
            # Exploit: choose best
            chosen = max(action_values, key=lambda x: x[1])
            reasoning = f"Exploiting: Best Q-value = {chosen[1]:.3f} for {chosen[2][:100]}..."
        
        return chosen[0], chosen[1], reasoning
    
    def experience_replay(self, batch_size: int = 32) -> int:
        """
        Experience replay: re-learn from past experiences.
        
        This improves sample efficiency (key RL technique).
        
        Returns: number of updates performed
        """
        if len(self.experience_buffer) < batch_size:
            return 0
        
        # A-TEAM: Prioritized sampling (per GRF MARL paper)
        # Sample proportional to priority
        import random
        priorities = [e.get('priority', 1.0) for e in self.experience_buffer]
        total_priority = sum(priorities)
        
        if total_priority > 0:
            # Weighted sampling
            probs = [p / total_priority for p in priorities]
            indices = random.choices(
                range(len(self.experience_buffer)),
                weights=probs,
                k=min(batch_size, len(self.experience_buffer))
            )
            batch = [self.experience_buffer[i] for i in indices]
        else:
            batch = random.sample(self.experience_buffer, min(batch_size, len(self.experience_buffer)))
        
        updates = 0
        for exp in batch:
            self._update_q_value(
                exp['state_desc'],
                exp['action_desc'],
                exp['reward'],
                exp.get('next_state_desc'),
                exp.get('done', False)
            )
            updates += 1
        
        return updates
    
    def get_q_table_summary(self) -> Dict[str, Any]:
        """Get summary statistics of Q-table for debugging/monitoring."""
        if not self.Q:
            return {
                'size': 0,
                'avg_value': 0.0,
                'max_value': 0.0,
                'min_value': 0.0,
                'total_visits': 0
            }
        
        values = [q_data['value'] for q_data in self.Q.values()]
        visits = [q_data['visit_count'] for q_data in self.Q.values()]
        
        return {
            'size': len(self.Q),
            'avg_value': sum(values) / len(values),
            'max_value': max(values),
            'min_value': min(values),
            'total_visits': sum(visits),
            'avg_visits_per_entry': sum(visits) / len(visits),
            'total_lessons': sum(len(q_data.get('learned_lessons', [])) for q_data in self.Q.values()),
            # NeuroChunk stats
            'tier1_size': len(self.tier1_working),
            'tier2_clusters': len(self.tier2_clusters),
            'tier3_size': len(self.tier3_archive),
            'tier1_threshold': self.tier1_threshold
        }
    
    # ===== NEUROCHUNK TIERED MEMORY MANAGEMENT =====
    
    @property
    def chunker(self) -> Any:
        """Lazy-init chunker to avoid circular imports."""
        if self._chunker is None:
            try:
                from Jotty.core.context.chunker import ContextChunker
                lm = getattr(self.config, 'lm', None) or dspy.settings.lm
                self._chunker = ContextChunker(lm=lm)
            except Exception as e:
                # Fallback: no chunking
                self._chunker = None
        return self._chunker
    
    def _compute_retention_score(self, key: Tuple[str, str]) -> float:
        """
        Compute multi-criteria retention score for a Q-table entry.
        
        Score = α·Q-value + β·novelty + γ·causal_impact - δ·staleness
        
        Where:
        - α, β, γ, δ are learnable weights (currently fixed, but can be meta-learned)
        - Q-value: Expected reward
        - Novelty: Inverse visit count (rare states = high novelty)
        - Causal impact: Average absolute TD error (high = impactful)
        - Staleness: Time since last access
        
        Returns: float in [0, 1]
        """
        if key not in self.Q:
            return 0.0
        
        q_data = self.Q[key]
        
        # Q-value component (0-1, normalized)
        q_value = q_data['value']
        
        # Novelty component (inverse visit count, normalized)
        visit_count = q_data['visit_count']
        novelty = 1.0 / (1.0 + visit_count)  # High novelty = low visits
        
        # Causal impact component (average absolute TD error)
        td_errors = q_data.get('td_errors', [])
        causal_impact = sum(abs(e) for e in td_errors) / len(td_errors) if td_errors else 0.0
        causal_impact = min(1.0, causal_impact)  # Clamp to [0, 1]
        
        # Staleness component (time since last update)
        last_updated = q_data.get('last_updated', time.time())
        staleness = min(1.0, (time.time() - last_updated) / 3600.0)  # 1 hour = max staleness

        # A-Team v8.0: Use adaptive learned weights instead of hardcoded values
        alpha = self.retention_weights.get('q_value')       # Q-value weight (reward salience)
        beta = self.retention_weights.get('novelty')        # Novelty weight (rare = important)
        gamma = self.retention_weights.get('causal_impact') # Causal impact weight
        delta = self.retention_weights.get('staleness')     # Staleness penalty

        score = alpha * q_value + beta * novelty + gamma * causal_impact - delta * staleness

        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))
    
    def _promote_demote_memories(self, episode_reward: float = None) -> None:
        """
        Move memories between tiers based on retention scores.
        
        Tier 1: High-value, recently accessed (always in context)
        Tier 2: Medium-value, semantically clustered (retrieval-based)
        Tier 3: Low-value, archived (pruned periodically)
        
        This is called after each episode to reorganize memory.
        """
        if not self.Q:
            return
        
        # Adaptive threshold adjustment
        if episode_reward is not None:
            if episode_reward > self.last_episode_reward:
                # Policy improved - current threshold is good
                self.episodes_without_improvement = 0
            else:
                # Policy plateaued - might need to explore more
                self.episodes_without_improvement += 1
                
                # Promote archived memories if stuck
                if self.episodes_without_improvement >= 10:
                    self._promote_from_archive(top_k=5)
                    self.episodes_without_improvement = 0
            
            self.last_episode_reward = episode_reward
        
        # Compute retention scores for all memories
        scored_memories = []
        for key in self.Q.keys():
            score = self._compute_retention_score(key)
            scored_memories.append((key, score))
        
        # Sort by score (descending)
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        # Tier 1: Top N (working memory)
        self.tier1_working = [key for key, score in scored_memories[:self.tier1_max_size]]
        
        # Tier 2: Next M (semantic clusters)
        tier2_candidates = [key for key, score in scored_memories[self.tier1_max_size:self.tier1_max_size + 100]]
        if tier2_candidates and self.chunker:
            self._cluster_tier2(tier2_candidates)
        
        # Tier 3: Rest (archive)
        self.tier3_archive = [key for key, score in scored_memories[self.tier1_max_size + 100:]]
        
        # Adaptive threshold: If Tier 1 overflows, tighten
        if len(self.tier1_working) >= self.tier1_max_size:
            self.tier1_threshold = min(0.95, self.tier1_threshold * 1.05)
    
    def _cluster_tier2(self, keys: List[Tuple[str, str]]) -> None:
        """
        Cluster Tier 2 memories by semantic similarity using ContextChunker.
        
        This enables efficient retrieval: when state matches cluster centroid,
        retrieve all members of that cluster.
        """
        if not self.chunker or not keys:
            return
        
        # Format memories as text chunks for clustering
        memory_texts = []
        for key in keys:
            state_desc, action_desc = key
            memory_texts.append(f"State: {state_desc}\nAction: {action_desc}")
        
        try:
            # Use ContextChunker to find semantic clusters
            # (Simplified: In reality, we'd use chunker's semantic grouping)
            # For now, simple keyword-based clustering
            
            clusters = {}
            for i, key in enumerate(keys):
                state_desc, action_desc = key
                
                # Extract key concepts for clustering (simple heuristic)
                concepts = self._extract_concepts(state_desc + " " + action_desc)
                cluster_id = "_".join(concepts[:2])  # Use first 2 concepts as cluster ID
                
                if cluster_id not in clusters:
                    clusters[cluster_id] = {
                        'centroid': " ".join(concepts),
                        'members': []
                    }
                
                clusters[cluster_id]['members'].append(key)
            
            # Keep only top N clusters
            self.tier2_clusters = dict(list(clusters.items())[:self.tier2_max_clusters])
            
        except Exception as e:
            # Fallback: no clustering
            pass
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text (simple keyword extraction)."""
        # Remove common words
        stopwords = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'but'}
        words = text.lower().split()
        concepts = [w for w in words if len(w) > 3 and w not in stopwords]
        return concepts[:5]  # Top 5 concepts
    
    def _retrieve_relevant_cluster(self, state_desc: str) -> List[Tuple[str, str]]:
        """
        Retrieve relevant cluster based on state similarity.
        
        Returns: List of (state_desc, action_desc) keys from the matched cluster.
        """
        if not self.tier2_clusters:
            return []
        
        # Find cluster with highest similarity to current state
        best_cluster = None
        best_similarity = 0.0
        
        for cluster_id, cluster_data in self.tier2_clusters.items():
            centroid = cluster_data['centroid']
            similarity = self._compute_similarity(state_desc, centroid)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster_data
        
        # Return members if similarity > threshold
        if best_similarity > 0.7 and best_cluster:
            return best_cluster['members']
        
        return []
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity (simple cosine of word sets)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1 & words2)
        union = len(words1 | words2)
        
        return overlap / union if union > 0 else 0.0
    
    def _promote_from_archive(self, top_k: int = 5) -> None:
        """
        Promote top-K memories from Tier 3 to Tier 2.
        
        Called when policy plateaus (exploration mechanism).
        """
        if not self.tier3_archive:
            return
        
        # Score archived memories by novelty (rare = explore)
        scored = []
        for key in self.tier3_archive:
            if key in self.Q:
                visit_count = self.Q[key]['visit_count']
                novelty = 1.0 / (1.0 + visit_count)
                scored.append((key, novelty))
        
        # Promote top-K novel memories
        scored.sort(key=lambda x: x[1], reverse=True)
        promoted = [key for key, _ in scored[:top_k]]
        
        # Remove from archive, add to Tier 2
        for key in promoted:
            self.tier3_archive.remove(key)
            # They'll be re-clustered in next _cluster_tier2 call
    
    def prune_tier3_by_causal_impact(self, sample_rate: float = 0.1) -> None:
        """
        Prune Tier 3 using causal impact scoring.
        
        Causal Impact = E[Q_with_memory] - E[Q_without_memory]
        
        We sample 10% of Tier 3, compute their causal impact, and extrapolate.
        
        Args:
            sample_rate: Fraction of Tier 3 to sample (default 0.1 = 10%)
        """
        if not self.tier3_archive:
            return
        
        # Sample memories for causal impact assessment
        sample_size = max(1, int(len(self.tier3_archive) * sample_rate))
        sampled = random.sample(self.tier3_archive, min(sample_size, len(self.tier3_archive)))
        
        # Compute causal impact for sampled memories
        low_impact_keys = []
        for key in sampled:
            if key not in self.Q:
                continue
            
            q_data = self.Q[key]
            
            # Proxy for causal impact: average absolute TD error
            # High TD error = high impact on learning
            td_errors = q_data.get('td_errors', [])
            avg_td_error = sum(abs(e) for e in td_errors) / len(td_errors) if td_errors else 0.0
            
            # Also consider visit count (very low = not useful)
            visit_count = q_data['visit_count']
            
            # Low impact = low TD error AND low visits AND low Q-value
            if avg_td_error < 0.05 and visit_count <= 1 and q_data['value'] < 0.4:
                low_impact_keys.append(key)
        
        # Prune low-impact memories from Tier 3 AND Q-table
        for key in low_impact_keys:
            if key in self.tier3_archive:
                self.tier3_archive.remove(key)
            if key in self.Q:
                del self.Q[key]
    
    def get_context_for_state(self, state: Dict, max_tokens: int = 2000) -> str:
        """
        Get tiered context for a given state.
        
        Tier 1: Always included (working memory)
        Tier 2: Retrieve relevant cluster if state matches
        
        This is the KEY method that enables O(1) context size!
        
        Args:
            state: Current state dict
            max_tokens: Maximum context size (soft limit)
        
        Returns: Natural language context string
        """
        state_desc = self._state_to_natural_language(state)
        
        context_parts = []
        
        # === TIER 1: WORKING MEMORY (always include) ===
        if self.tier1_working:
            context_parts.append("# Working Memory (High-Value Recent):")
            for i, key in enumerate(self.tier1_working[:20], 1):  # Top 20
                if key not in self.Q:
                    continue
                state_d, action_d = key
                q_data = self.Q[key]
                q_val = q_data['value']
                visits = q_data['visit_count']
                
                # Truncate for readability
                state_short = state_d[:80] + "..." if len(state_d) > 80 else state_d
                action_short = action_d[:80] + "..." if len(action_d) > 80 else action_d
                
                context_parts.append(f"{i}. [{state_short}] → [{action_short}] | Q={q_val:.2f} (n={visits})")
        
        # === TIER 2: RETRIEVE RELEVANT CLUSTER ===
        relevant_cluster = self._retrieve_relevant_cluster(state_desc)
        if relevant_cluster:
            context_parts.append("\n# Retrieved Relevant Context (Semantic Match):")
            for i, key in enumerate(relevant_cluster[:10], 1):  # Top 10 from cluster
                if key not in self.Q:
                    continue
                state_d, action_d = key
                q_data = self.Q[key]
                q_val = q_data['value']
                
                state_short = state_d[:60] + "..." if len(state_d) > 60 else state_d
                action_short = action_d[:60] + "..." if len(action_d) > 60 else action_d
                
                context_parts.append(f"{i}. [{state_short}] → [{action_short}] | Q={q_val:.2f}")
        
        # === LEARNED LESSONS (from Tier 1 + retrieved) ===
        all_keys = self.tier1_working + relevant_cluster
        all_lessons = []
        for key in all_keys[:30]:  # Limit to prevent explosion
            if key in self.Q:
                all_lessons.extend(self.Q[key].get('learned_lessons', []))
        
        if all_lessons:
            unique_lessons = list(dict.fromkeys(all_lessons))[:10]  # Top 10 unique
            context_parts.append("\n# Learned Lessons:")
            for i, lesson in enumerate(unique_lessons, 1):
                context_parts.append(f"{i}. {lesson}")
        
        # Join and truncate if needed
        full_context = "\n".join(context_parts)
        
        # Simple token estimation (4 chars ≈ 1 token)
        estimated_tokens = len(full_context) // 4
        if estimated_tokens > max_tokens:
            # Truncate to fit
            target_chars = max_tokens * 4
            full_context = full_context[:target_chars] + "\n... (truncated for context window)"
        
        return full_context
    
    # ===== PERSISTENCE METHODS =====
    
    def save_state(self, path: str) -> None:
        """
        Save Q-table and learning state for persistence across runs.
        
         A-TEAM: This is CRITICAL for DQN convergence!
        Without persistence, agent never learns from past runs.
        
        Saves:
        - Q-table (natural language state-action values)
        - Experience buffer (for replay)
        - Tiered memories (NeuroChunk)
        - Learned lessons
        """
        import json
        from pathlib import Path
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert Q-table: tuple keys → string keys
        q_table_serializable = {}
        for key, value in self.Q.items():
            # Key is (state_desc, action_desc) tuple
            key_str = f"{key[0]}|||{key[1]}" if isinstance(key, tuple) else str(key)
            q_table_serializable[key_str] = value
        
        state = {
            'q_table': q_table_serializable,
            'experience_buffer': self.experience_buffer[-self.max_buffer_size:],  # Last N
            'tier1_working': [
                f"{k[0]}|||{k[1]}" if isinstance(k, tuple) else str(k)
                for k in self.tier1_working
            ],
            'tier2_clusters': {
                cid: {
                    'centroid': c['centroid'],
                    'members': [
                        f"{k[0]}|||{k[1]}" if isinstance(k, tuple) else str(k)
                        for k in c['members']
                    ]
                }
                for cid, c in self.tier2_clusters.items()
            },
            'tier3_archive': [
                f"{k[0]}|||{k[1]}" if isinstance(k, tuple) else str(k)
                for k in self.tier3_archive
            ],
            'tier1_threshold': self.tier1_threshold,
            'last_episode_reward': self.last_episode_reward,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            # A-Team v8.0: Persist adaptive retention weights
            'retention_weights': self.retention_weights.to_dict()
        }

        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)

        logger.info(f" Q-learning state saved: {len(self.Q)} Q-entries, {len(self.experience_buffer)} experiences")
    
    def load_state(self, path: str) -> bool:
        """
        Load Q-table and learning state from previous run.
        
         A-TEAM: This enables convergence across runs!
        
        Returns: True if loaded successfully, False otherwise
        """
        import json
        from pathlib import Path
        
        if not Path(path).exists():
            logger.info(f"ℹ No previous Q-learning state at {path}")
            return False
        
        try:
            with open(path, 'r') as f:
                state = json.load(f)
            
            # Restore Q-table: string keys → tuple keys
            self.Q = {}
            for key_str, value in state.get('q_table', {}).items():
                if '|||' in key_str:
                    parts = key_str.split('|||', 1)
                    key = (parts[0], parts[1])
                else:
                    key = (key_str, '')
                self.Q[key] = value
            
            # Restore experience buffer
            self.experience_buffer = state.get('experience_buffer', [])
            
            # Restore tiers
            self.tier1_working = [
                tuple(k.split('|||', 1)) if '|||' in k else (k, '')
                for k in state.get('tier1_working', [])
            ]
            
            # Restore clusters
            self.tier2_clusters = {}
            for cid, c in state.get('tier2_clusters', {}).items():
                self.tier2_clusters[cid] = {
                    'centroid': c['centroid'],
                    'members': [
                        tuple(k.split('|||', 1)) if '|||' in k else (k, '')
                        for k in c['members']
                    ]
                }
            
            self.tier3_archive = [
                tuple(k.split('|||', 1)) if '|||' in k else (k, '')
                for k in state.get('tier3_archive', [])
            ]
            
            # Restore parameters
            self.tier1_threshold = state.get('tier1_threshold', 0.8)
            self.last_episode_reward = state.get('last_episode_reward', 0.0)
            self.alpha = state.get('alpha', self.alpha)
            self.gamma = state.get('gamma', self.gamma)
            self.epsilon = state.get('epsilon', self.epsilon)

            # A-Team v8.0: Restore adaptive retention weights
            if 'retention_weights' in state:
                self.retention_weights = AdaptiveWeightGroup.from_dict(state['retention_weights'])

            logger.info(f" Q-learning state loaded: {len(self.Q)} Q-entries, {len(self.experience_buffer)} experiences")
            return True
            
        except Exception as e:
            logger.warning(f" Failed to load Q-learning state: {e}")
            return False
    
    # ===== END NEUROCHUNK METHODS ====

