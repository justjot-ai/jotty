"""
Policy Explorer for ReVal.

Provides exploration strategies for action selection.
"""
import dspy
from typing import Dict, List, Any
import random
import logging

logger = logging.getLogger(__name__)

from .roadmap import MarkovianTODO, TodoItem

from ..foundation.data_structures import JottyConfig

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False


class PolicyExplorerSignature(dspy.Signature):
    """LLM-based policy exploration for finding better action sequences."""
    
    current_state = dspy.InputField(desc="Current TODO state and failures")
    failed_actions = dspy.InputField(desc="Actions that have failed and why")
    available_actions = dspy.InputField(desc="Available alternative actions")
    goal = dspy.InputField(desc="Root goal to achieve")
    
    analysis = dspy.OutputField(desc="Analysis of why current approach is failing")
    recommended_exploration = dspy.OutputField(desc="Specific alternative approach to try")
    new_todo_items = dspy.OutputField(desc="JSON list of new TODO items to try")
    confidence = dspy.OutputField(desc="Confidence this will work 0.0-1.0")


class PolicyExplorer:
    """
    Explores alternative policies when stuck.
    
    Instead of giving up when tasks fail:
    1. Analyzes failure patterns
    2. Generates alternative approaches
    3. Updates TODO with new exploration paths
    4. Tracks exploration history to avoid loops
    """
    
    def __init__(self, config: JottyConfig):
        self.config = config
        self.explorer = dspy.ChainOfThought(PolicyExplorerSignature) if DSPY_AVAILABLE else None
        
        # Exploration state
        self.explored_paths: List[List[str]] = []
        self.exploration_count = 0
        self.max_explorations = self.config.max_exploration_iterations  # 🔧 STANFORD FIX
    
    def should_explore(self, todo: MarkovianTODO) -> bool:
        """Check if we should try a new exploration."""
        # Explore if: failed tasks exist, haven't explored too much
        has_failures = len(todo.failed) > 0
        can_explore = self.exploration_count < self.max_explorations
        not_repeated = todo.current_path not in self.explored_paths
        
        return has_failures and can_explore and not_repeated
    
    def explore(
        self, 
        todo: MarkovianTODO, 
        available_actions: List[Dict[str, Any]],
        goal: str
    ) -> List[TodoItem]:
        """
        Generate new TODO items through exploration.
        
        Returns:
            List of new TodoItems to try
        """
        if not self.explorer:
            return []
        
        self.exploration_count += 1
        self.explored_paths.append(todo.current_path.copy())
        
        # Gather failure info
        failed_info = []
        for task_id in todo.failed:
            task = todo.items.get(task_id)
            if task:
                failed_info.append({
                    'task': task.description,
                    'actor': task.actor,
                    'reasons': task.failure_reasons
                })
        
        try:
            result = self.explorer(
                current_state=todo.get_state_summary(),
                failed_actions=json.dumps(failed_info, default=str),
                available_actions=json.dumps(available_actions, default=str),
                goal=goal
            )
            
            # Parse new TODO items
            new_items = []
            try:
                items_json = json.loads(result.new_todo_items or "[]")
                for item in items_json:
                    new_items.append(TodoItem(
                        id=f"explore_{self.exploration_count}_{len(new_items)}",
                        description=item.get('description', 'Exploration task'),
                        actor=item.get('actor', 'unknown'),
                        status="pending",
                        priority=0.8,  # High priority for explorations
                        estimated_reward=self.config.default_estimated_reward  # ✅ FROM CONFIG! (was 0.6)
                    ))
            except json.JSONDecodeError:
                pass
            
            logger.info(f"🔍 Policy exploration generated {len(new_items)} new tasks")
            return new_items
            
        except Exception as e:
            logger.warning(f"Exploration failed: {e}")
            return []


# =============================================================================
# SWARM LEARNER - Prompt Updates as Weight Updates
# =============================================================================

class SwarmLearnerSignature(dspy.Signature):
    """Update system prompts based on episode outcomes (online learning)."""
    
    current_prompt = dspy.InputField(desc="Current PreVal/PostVal prompt")
    episode_trajectory = dspy.InputField(desc="What happened this episode")
    outcome = dspy.InputField(desc="Success/failure and why")
    patterns_observed = dspy.InputField(desc="Patterns that led to success/failure")
    
    updated_prompt = dspy.OutputField(desc="Updated prompt incorporating learnings")
    changes_made = dspy.OutputField(desc="List of specific changes made")
    learning_summary = dspy.OutputField(desc="What the system learned")


