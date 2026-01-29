"""
StateActionManager - Manages state representation and action space for RL.

Extracted from conductor.py to improve maintainability.
Handles current state extraction, available actions, and Q-learning state representation.
"""
import logging
import re
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class StateActionManager:
    """
    Centralized state-action management for reinforcement learning.

    Responsibilities:
    - Current state extraction (_get_current_state)
    - Available actions enumeration (_get_available_actions)
    - State representation for Q-learning
    - Action space management
    """

    def __init__(self, config):
        """
        Initialize state-action manager.

        Args:
            config: JottyConfig
        """
        self.config = config

        logger.info("🎯 StateActionManager initialized")

    def get_current_state(
        self,
        todo,
        trajectory,
        shared_context=None,
        context_guard=None,
        io_manager=None
    ) -> Dict[str, Any]:
        """
        Get RICH current state for Q-prediction.

        🔥 A-TEAM CRITICAL: State must capture semantic context!

        Includes:
        1. Query semantics (what user asked)
        2. Metadata context (tables, columns, partitions)
        3. Error patterns (what failed)
        4. Tool usage (what worked)
        5. Actor outputs (what was produced)

        Args:
            todo: TodoManager instance
            trajectory: List of execution steps
            shared_context: Shared context dictionary
            context_guard: SmartContextGuard instance
            io_manager: IOManager instance

        Returns:
            Dict with rich state representation
        """
        from ...foundation.data_structures import TaskStatus

        state = {
            # === 1. TASK PROGRESS ===
            'todo': {
                'completed': len(todo.completed),
                'pending': len([t for t in todo.subtasks.values() if t.status == TaskStatus.PENDING]),
                'failed': len(todo.failed_tasks)
            },
            'trajectory_length': len(trajectory),
            'recent_outcomes': [t.get('passed', False) for t in trajectory[-5:]]
        }

        # === 2. QUERY CONTEXT (CRITICAL!) ===
        # Try multiple sources for query
        query = None

        # Source 1: SharedContext
        if shared_context:
            query = shared_context.get('query') or shared_context.get('goal')

        # Source 2: Context guard buffers (if available)
        if not query and context_guard:
            # SmartContextGuard stores content in buffers
            for priority_buffer in context_guard.buffers.values():
                for key, content, _ in priority_buffer:
                    if key == 'ROOT_GOAL':
                        query = content
                        break
                if query:
                    break

        # Source 3: TODO root task
        if not query and todo:
            query = todo.root_task

        if query:
            state['query'] = str(query)[:200]

        # === 3. METADATA CONTEXT ===
        if shared_context:
            # Get table info
            tables = shared_context.get('table_names') or shared_context.get('relevant_tables')
            if tables:
                state['tables'] = tables if isinstance(tables, list) else [str(tables)]

            # Get filter info
            filters = shared_context.get('filters') or shared_context.get('filter_conditions')
            if filters:
                state['filters'] = filters

            # Get resolved terms
            resolved = shared_context.get('resolved_terms')
            if resolved:
                if isinstance(resolved, dict):
                    state['resolved_terms'] = list(resolved.keys())[:5]

        # === 4. ACTOR OUTPUT CONTEXT ===
        if io_manager:
            all_outputs = io_manager.get_all_outputs()
            output_summary = {}
            for actor_name, output in all_outputs.items():
                if hasattr(output, 'output_fields') and output.output_fields:
                    output_summary[actor_name] = list(output.output_fields.keys())
            if output_summary:
                state['actor_outputs'] = output_summary

        # === 5. ERROR PATTERNS (CRITICAL FOR LEARNING!) ===
        if trajectory:
            errors = []
            columns_tried = []
            working_column = None

            for step in trajectory:
                # Check for errors in trajectory
                if step.get('error'):
                    err = step['error']
                    if 'COLUMN_NOT_FOUND' in str(err):
                        # Extract column name from error
                        match = re.search(r"Column '(\w+)' cannot be resolved", str(err))
                        if match:
                            col = match.group(1)
                            columns_tried.append(col)
                            errors.append({'type': 'COLUMN_NOT_FOUND', 'column': col})

                # Check for success
                if step.get('passed') and step.get('tool_calls'):
                    for tc in step.get('tool_calls', []):
                        if isinstance(tc, dict) and tc.get('success'):
                            # Extract working column if SQL-related
                            if 'query' in str(tc):
                                # Try to find date column that worked
                                query_text = str(tc.get('query', ''))
                                for possible_col in ['dl_last_updated', 'dt', 'date', 'created_at']:
                                    if possible_col in query_text.lower():
                                        working_column = possible_col
                                        break

            if errors:
                state['errors'] = errors[-5:]  # Last 5 errors
            if columns_tried:
                state['columns_tried'] = list(dict.fromkeys(columns_tried))  # Unique
            if working_column:
                state['working_column'] = working_column
                state['error_resolution'] = f"use {working_column} instead of {','.join(columns_tried[:3])}"

        # === 6. TOOL USAGE PATTERNS ===
        successful_tools = []
        failed_tools = []
        tool_calls = []

        for step in trajectory:
            if step.get('tool_calls'):
                for tc in step.get('tool_calls', []):
                    tool_name = tc.get('tool') if isinstance(tc, dict) else str(tc)
                    tool_calls.append(tool_name)

                    if isinstance(tc, dict):
                        if tc.get('success'):
                            successful_tools.append(tool_name)
                        else:
                            failed_tools.append(tool_name)

        if tool_calls:
            state['tool_calls'] = tool_calls[-10:]
        if successful_tools:
            state['successful_tools'] = list(dict.fromkeys(successful_tools))
        if failed_tools:
            state['failed_tools'] = list(dict.fromkeys(failed_tools))

        # === 7. CURRENT ACTOR ===
        if trajectory and trajectory[-1].get('actor'):
            state['current_actor'] = trajectory[-1]['actor']

        # === 8. VALIDATION CONTEXT ===
        for step in trajectory[-3:]:  # Last 3 steps
            if step.get('architect_confidence'):
                state['architect_confidence'] = step['architect_confidence']
            if step.get('auditor_result'):
                state['auditor_result'] = step['auditor_result']
            if step.get('validation_passed') is not None:
                state['validation_passed'] = step['validation_passed']

        # === 9. EXECUTION STATS ===
        state['attempts'] = len(trajectory)
        state['success'] = any(t.get('passed', False) for t in trajectory)

        return state

    def get_available_actions(self, actors: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get available actions for exploration.

        Args:
            actors: Dict of actor configurations

        Returns:
            List of available actions
        """
        actions = []
        for name, config in actors.items():
            actions.append({
                'actor': name,
                'action': 'execute',
                'enabled': config.enabled
            })
        return actions

    def get_stats(self) -> Dict[str, Any]:
        """
        Get state-action statistics.

        Returns:
            Dict with state-action metrics
        """
        return {
            "manager_initialized": True,
        }

    def reset_stats(self):
        """Reset state-action statistics."""
        logger.debug("StateActionManager stats reset")
