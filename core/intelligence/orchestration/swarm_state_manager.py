"""
SwarmStateManager - Comprehensive State Management for V2
=========================================================

Manages state at both swarm-level (shared) and agent-level (per-agent).

Swarm-Level State:
- Task progress (completed, pending, failed)
- Query/Goal context
- Metadata context (tables, columns, filters, etc.)
- Error patterns (what failed, how to fix)
- Tool usage patterns (what worked, what didn't)
- Swarm trajectory (execution history)
- Validation context (architect/auditor results)

Agent-Level State:
- Per-agent outputs
- Per-agent errors
- Per-agent tool usage
- Per-agent trajectory
- Per-agent validation results

Integrates StateManager capabilities into orchestration architecture.
"""

import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from datetime import datetime
from pathlib import Path
import json

if TYPE_CHECKING:
    from .swarm_roadmap import SwarmTaskBoard, SubtaskState, TaskStatus
    from Jotty.core.infrastructure.foundation.agent_config import AgentConfig
    from Jotty.core.infrastructure.data.data_registry import DataRegistry
    from Jotty.core.infrastructure.data.io_manager import IOManager
    from Jotty.core.infrastructure.context.context_guard import LLMContextManager
else:
    SwarmTaskBoard = Any
    SubtaskState = Any
    TaskStatus = Any
    AgentConfig = Any
    DataRegistry = Any
    IOManager = Any
    LLMContextManager = Any

logger = logging.getLogger(__name__)


class AgentStateTracker:
    """
    Tracks state for a single agent.
    
    Maintains per-agent:
    - Outputs and their types
    - Errors and failure patterns
    - Tool usage (successful/failed)
    - Trajectory (execution steps)
    - Validation results (architect/auditor)
    """
    
    def __init__(self, agent_name: str) -> None:
        """Initialize agent state tracker."""
        self.agent_name = agent_name
        self.outputs: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
        self.tool_usage: Dict[str, Dict[str, int]] = {
            'successful': {},
            'failed': {}
        }
        self.trajectory: List[Dict[str, Any]] = []
        self.validation_results: List[Dict[str, Any]] = []
        self.stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_tool_calls': 0,
            'successful_tool_calls': 0,
            'failed_tool_calls': 0
        }
    
    def record_output(self, output: Any, output_type: str = None) -> None:
        """Record agent output."""
        self.outputs.append({
            'output': output,
            'type': output_type or type(output).__name__,
            'timestamp': datetime.now().isoformat()
        })
        self.stats['total_executions'] += 1
        self.stats['successful_executions'] += 1
    
    def record_error(self, error: str, error_type: str = None, context: Dict = None) -> None:
        """Record agent error."""
        self.errors.append({
            'error': error,
            'type': error_type or 'Unknown',
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        })
        self.stats['total_executions'] += 1
        self.stats['failed_executions'] += 1
    
    def record_tool_call(self, tool_name: str, success: bool, metadata: Dict = None) -> None:
        """Record tool usage."""
        if success:
            self.tool_usage['successful'][tool_name] = self.tool_usage['successful'].get(tool_name, 0) + 1
            self.stats['successful_tool_calls'] += 1
        else:
            self.tool_usage['failed'][tool_name] = self.tool_usage['failed'].get(tool_name, 0) + 1
            self.stats['failed_tool_calls'] += 1
        self.stats['total_tool_calls'] += 1
    
    def record_trajectory_step(self, step: Dict[str, Any]) -> None:
        """Record trajectory step."""
        step['timestamp'] = datetime.now().isoformat()
        step['agent'] = self.agent_name
        self.trajectory.append(step)
    
    def record_validation(self, validation_type: str, passed: bool, confidence: float = None, feedback: str = None) -> None:
        """Record validation result."""
        self.validation_results.append({
            'type': validation_type,  # 'architect' or 'auditor'
            'passed': passed,
            'confidence': confidence,
            'feedback': feedback,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state."""
        return {
            'agent_name': self.agent_name,
            'stats': self.stats.copy(),
            'recent_outputs': self.outputs[-5:] if self.outputs else [],
            'recent_errors': self.errors[-5:] if self.errors else [],
            'tool_usage': {
                'successful': dict(self.tool_usage['successful']),
                'failed': dict(self.tool_usage['failed'])
            },
            'recent_trajectory': self.trajectory[-10:] if self.trajectory else [],
            'recent_validation': self.validation_results[-5:] if self.validation_results else [],
            'success_rate': (
                self.stats['successful_executions'] / self.stats['total_executions']
                if self.stats['total_executions'] > 0 else 0.0
            ),
            'tool_success_rate': (
                self.stats['successful_tool_calls'] / self.stats['total_tool_calls']
                if self.stats['total_tool_calls'] > 0 else 0.0
            )
        }
    
    def get_error_patterns(self) -> List[Dict[str, Any]]:
        """Extract error patterns for learning."""
        patterns = []
        for error in self.errors[-10:]:  # Last 10 errors
            error_type = error.get('type', 'Unknown')
            error_msg = str(error.get('error', ''))
            
            # Extract common patterns
            pattern = {
                'type': error_type,
                'message_pattern': error_msg[:100],  # First 100 chars
                'frequency': sum(1 for e in self.errors if e.get('type') == error_type),
                'context': error.get('context', {})
            }
            patterns.append(pattern)
        
        return patterns
    
    def get_successful_patterns(self) -> List[Dict[str, Any]]:
        """Extract successful patterns for learning."""
        patterns = []
        
        # Successful tool usage patterns
        for tool_name, count in self.tool_usage['successful'].items():
            if count >= 2:  # Used successfully at least twice
                patterns.append({
                    'type': 'tool_usage',
                    'tool': tool_name,
                    'success_count': count,
                    'pattern': f"Tool '{tool_name}' works reliably"
                })
        
        # Successful validation patterns
        successful_validations = [v for v in self.validation_results if v.get('passed')]
        if successful_validations:
            patterns.append({
                'type': 'validation',
                'pattern': 'Validation passes consistently',
                'count': len(successful_validations)
            })
        
        return patterns


class SwarmStateManager:
    """
    Comprehensive state management for swarm-level and agent-level state.
    
    Integrates V1 StateManager capabilities:
    - Rich state introspection for Q-prediction
    - Error pattern extraction
    - Tool usage tracking
    - Output type detection and registration
    - Actor signature introspection
    
    Maintains:
    - Swarm-level state (shared across all agents)
    - Agent-level state (per-agent tracking)
    """
    
    def __init__(self, swarm_task_board: SwarmTaskBoard, swarm_memory: Any, io_manager: Optional[IOManager] = None, data_registry: Optional[DataRegistry] = None, shared_context: Optional[Dict[str, Any]] = None, context_guard: Optional[LLMContextManager] = None, config: Optional[Any] = None, agents: Optional[Dict[str, AgentConfig]] = None, agent_signatures: Optional[Dict[str, Dict]] = None) -> None:
        """
        Initialize SwarmStateManager.
        
        Args:
            swarm_task_board: SwarmTaskBoard (SwarmTaskBoard) for task tracking
            swarm_memory: SwarmMemory (SwarmMemory) for memory
            io_manager: IOManager for accessing actor outputs
            data_registry: DataRegistry for output registration
            shared_context: Shared context dictionary
            context_guard: LLMContextManager for context management
            config: SwarmConfig instance
            agents: Dictionary of agent configurations
            agent_signatures: Dictionary of agent signatures
        """
        self.swarm_task_board = swarm_task_board
        self.swarm_memory = swarm_memory
        self.io_manager = io_manager
        self.data_registry = data_registry
        self.shared_context = shared_context or {}
        self.context_guard = context_guard
        self.config = config
        self.agents = agents or {}
        self.agent_signatures = agent_signatures or {}
        
        # Agent-level state trackers
        self.agent_trackers: Dict[str, AgentStateTracker] = {}
        
        # Swarm-level trajectory (execution history)
        self.swarm_trajectory: List[Dict[str, Any]] = []
        
        # Swarm-level error patterns
        self.swarm_error_patterns: List[Dict[str, Any]] = []
        
        # Swarm-level tool usage
        self.swarm_tool_usage: Dict[str, Dict[str, int]] = {
            'successful': {},
            'failed': {}
        }
        
        logger.info(" SwarmStateManager initialized")
    
    def get_agent_tracker(self, agent_name: str) -> AgentStateTracker:
        """Get or create agent state tracker."""
        if agent_name not in self.agent_trackers:
            self.agent_trackers[agent_name] = AgentStateTracker(agent_name)
        return self.agent_trackers[agent_name]
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get RICH current state for Q-prediction (swarm-level).
        
         CRITICAL: State must capture semantic context!
        
        Includes:
        1. Task progress (completed, pending, failed)
        2. Query/Goal context (what user asked)
        3. Metadata context (tables, columns, filters, etc.)
        4. Error patterns (what failed, how to fix)
        5. Tool usage patterns (what worked, what didn't)
        6. Actor outputs (what was produced)
        7. Validation context (architect/auditor results)
        8. Execution stats
        """
        state = {
            # === 1. TASK PROGRESS ===
            'task_progress': {
                'completed': len(self.swarm_task_board.completed_tasks),
                'pending': len([
                    t for t in self.swarm_task_board.subtasks.values()
                    if t.status.name == 'PENDING'
                ]),
                'failed': len(self.swarm_task_board.failed_tasks),
                'total': len(self.swarm_task_board.subtasks)
            },
            'trajectory_length': len(self.swarm_trajectory),
            'recent_outcomes': [
                t.get('success', False) for t in self.swarm_trajectory[-5:]
            ]
        }
        
        # === 2. QUERY/GOAL CONTEXT (CRITICAL!) ===
        query = None
        
        # Source 1: SharedContext
        if self.shared_context:
            # SharedContext has .get() method
            query = self.shared_context.get('query') or self.shared_context.get('goal')
        
        # Source 2: Context guard buffers
        if not query and self.context_guard:
            for priority_buffer in self.context_guard.buffers.values():
                for key, content, _ in priority_buffer:
                    if key == 'ROOT_GOAL':
                        query = content
                        break
                if query:
                    break
        
        # Source 3: Task board root task
        if not query and self.swarm_task_board:
            query = getattr(self.swarm_task_board, 'root_task', None)
        
        if query:
            state['query'] = str(query)[:200]
            state['goal'] = str(query)[:200]
        
        # === 3. METADATA CONTEXT ===
        if self.shared_context:
            # SharedContext has .get() method
            # Tables
            tables = self.shared_context.get('table_names') or self.shared_context.get('relevant_tables')
            if tables:
                state['tables'] = tables if isinstance(tables, list) else [str(tables)]
            
            # Filters
            filters = self.shared_context.get('filters') or self.shared_context.get('filter_conditions')
            if filters:
                state['filters'] = filters
            
            # Resolved terms
            resolved = self.shared_context.get('resolved_terms')
            if resolved and isinstance(resolved, dict):
                state['resolved_terms'] = list(resolved.keys())[:5]
        
        # === 4. ACTOR OUTPUT CONTEXT ===
        if self.io_manager:
            all_outputs = self.io_manager.get_all_outputs()
            output_summary = {}
            for actor_name, output in all_outputs.items():
                if hasattr(output, 'output_fields') and output.output_fields:
                    output_summary[actor_name] = list(output.output_fields.keys())
            if output_summary:
                state['actor_outputs'] = output_summary
        
        # === 5. ERROR PATTERNS (CRITICAL FOR LEARNING!) ===
        if self.swarm_trajectory:
            errors = []
            for step in self.swarm_trajectory:
                if step.get('error'):
                    err = step['error']
                    error_info = {
                        'error': str(err)[:200],
                        'agent': step.get('agent'),
                        'timestamp': step.get('timestamp')
                    }
                    errors.append(error_info)
            
            if errors:
                state['errors'] = errors[-5:]  # Last 5 errors
        
        # Aggregate error patterns from agent trackers
        agent_error_patterns = []
        for tracker in self.agent_trackers.values():
            agent_error_patterns.extend(tracker.get_error_patterns())
        if agent_error_patterns:
            state['agent_error_patterns'] = agent_error_patterns[-10:]
        
        # === 6. TOOL USAGE PATTERNS ===
        if self.swarm_trajectory:
            for step in self.swarm_trajectory:
                if step.get('tool_calls'):
                    for tc in step.get('tool_calls', []):
                        tool_name = tc.get('tool') if isinstance(tc, dict) else str(tc)
                        if isinstance(tc, dict) and tc.get('success'):
                            self.swarm_tool_usage['successful'][tool_name] = (
                                self.swarm_tool_usage['successful'].get(tool_name, 0) + 1
                            )
                        else:
                            self.swarm_tool_usage['failed'][tool_name] = (
                                self.swarm_tool_usage['failed'].get(tool_name, 0) + 1
                            )
        
        if self.swarm_tool_usage['successful']:
            state['successful_tools'] = list(self.swarm_tool_usage['successful'].keys())
        if self.swarm_tool_usage['failed']:
            state['failed_tools'] = list(self.swarm_tool_usage['failed'].keys())
        
        # === 7. CURRENT AGENT ===
        if self.swarm_trajectory:
            last_step = self.swarm_trajectory[-1]
            state['current_agent'] = last_step.get('agent')
        
        # === 8. VALIDATION CONTEXT ===
        for step in self.swarm_trajectory[-3:]:  # Last 3 steps
            if step.get('architect_confidence'):
                state['architect_confidence'] = step['architect_confidence']
            if step.get('auditor_result'):
                state['auditor_result'] = step['auditor_result']
            if step.get('validation_passed') is not None:
                state['validation_passed'] = step['validation_passed']
        
        # === 9. EXECUTION STATS ===
        state['attempts'] = len(self.swarm_trajectory)
        state['success'] = any(t.get('success', False) for t in self.swarm_trajectory)
        
        # === 10. AGENT STATES ===
        agent_states = {}
        for agent_name, tracker in self.agent_trackers.items():
            agent_states[agent_name] = tracker.get_state()
        if agent_states:
            state['agent_states'] = agent_states
        
        return state
    
    def get_agent_state(self, agent_name: str) -> Dict[str, Any]:
        """Get state for a specific agent."""
        tracker = self.get_agent_tracker(agent_name)
        return tracker.get_state()
    
    def get_state_summary(self) -> str:
        """Get human-readable state summary."""
        state = self.get_current_state()
        
        summary = "=== Swarm State Summary ===\n"
        summary += f"Tasks: {state['task_progress']['completed']} completed, "
        summary += f"{state['task_progress']['pending']} pending, "
        summary += f"{state['task_progress']['failed']} failed\n"
        
        if state.get('query'):
            summary += f"Goal: {state['query'][:100]}...\n"
        
        if state.get('errors'):
            summary += f"Recent Errors: {len(state['errors'])}\n"
        
        if state.get('successful_tools'):
            summary += f"Successful Tools: {', '.join(state['successful_tools'][:5])}\n"
        
        if state.get('agent_states'):
            summary += f"\nAgents: {len(state['agent_states'])}\n"
            for agent_name, agent_state in state['agent_states'].items():
                stats = agent_state.get('stats', {})
                summary += f"  {agent_name}: {stats.get('successful_executions', 0)}/{stats.get('total_executions', 0)} successful\n"
        
        return summary
    
    def record_swarm_step(self, step: Dict[str, Any]) -> None:
        """Record swarm-level trajectory step."""
        step['timestamp'] = datetime.now().isoformat()
        self.swarm_trajectory.append(step)
        
        # Also record in agent tracker if agent specified
        if 'agent' in step:
            tracker = self.get_agent_tracker(step['agent'])
            tracker.record_trajectory_step(step)
    
    def get_available_actions(self) -> List[Dict[str, Any]]:
        """Get available actions for exploration."""
        actions = []
        for name, config in self.agents.items():
            actions.append({
                'agent': name,
                'action': 'execute',
                'enabled': getattr(config, 'enabled', True)
            })
        return actions
    
    def save_state(self, file_path: Path) -> None:
        """Save swarm and agent state to file."""
        state_data = {
            'swarm_state': {
                'trajectory': self.swarm_trajectory,
                'tool_usage': self.swarm_tool_usage,
                'error_patterns': self.swarm_error_patterns
            },
            'agent_states': {
                name: tracker.get_state()
                for name, tracker in self.agent_trackers.items()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(state_data, f, indent=2, default=str)
        
        logger.info(f" Saved swarm state to {file_path}")
    
    def load_state(self, file_path: Path) -> None:
        """Load swarm and agent state from file."""
        with open(file_path, 'r') as f:
            state_data = json.load(f)
        
        # Restore swarm state
        if 'swarm_state' in state_data:
            swarm_state = state_data['swarm_state']
            self.swarm_trajectory = swarm_state.get('trajectory', [])
            self.swarm_tool_usage = swarm_state.get('tool_usage', {'successful': {}, 'failed': {}})
            self.swarm_error_patterns = swarm_state.get('error_patterns', [])
        
        # Restore agent states
        if 'agent_states' in state_data:
            for agent_name, agent_state in state_data['agent_states'].items():
                tracker = self.get_agent_tracker(agent_name)
                # Restore agent state (simplified - full restoration would need more logic)
                tracker.stats = agent_state.get('stats', tracker.stats)
        
        logger.info(f" Loaded swarm state from {file_path}")
