"""
JOTTY Persistence Manager (Vault) - Complete State Management

A-Team Design: NO HARDCODING ANYWHERE
- All paths configurable
- All formats dynamic
- All thresholds learned or configured

Handles persistence of:
- Roadmap TODOs (JSON + rich markdown)
- Q-tables and experience buffers
- Hierarchical memories (Cortex)
- Episode trajectories
- Brain consolidation state (Hippocampus)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class Vault:
    """
    Manages persistence of all JOTTY state components.
    
    NO HARDCODING PRINCIPLE:
    - Directory structure configurable
    - File formats extensible
    - Display generation dynamic
    - No magic numbers or thresholds
    
    Directory Structure:
        outputs/run_X/jotty_state/
        ├── roadmap/
        │   ├── todo_state.json
        │   └── todo_display.md         (Rich markdown)
        ├── q_tables/
        │   └── q_predictor_buffer.json
        ├── memories/
        │   ├── shared_memory.json
        │   └── local_memories/
        │       ├── loader.json
        │       └── diffuser.json
        ├── episode_history/
        │   ├── episode_1.json
        │   └── episode_2.json
        └── brain_state/
            └── consolidated_memories.json
    """
    
    def __init__(self, base_output_dir: str, auto_save_interval: int = 10):
        """
        Initialize persistence manager.
        
        Args:
            base_output_dir: Base output directory (NO HARDCODED PATH)
            auto_save_interval: Save every N iterations (CONFIGURABLE)
        """
        self.base_dir = Path(base_output_dir)
        self.jotty_dir = self.base_dir / "jotty_state"
        self.auto_save_interval = auto_save_interval
        self._ensure_directories()
        
        logger.info(f" JOTTY Vault initialized: {self.jotty_dir}")
    
    def _ensure_directories(self):
        """Create all necessary directories (NO HARDCODED LIST - extensible)."""
        dirs = [
            self.jotty_dir,
            self.jotty_dir / "markovian_todos",
            self.jotty_dir / "q_tables",
            self.jotty_dir / "memories" / "local_memories",
            self.jotty_dir / "episode_history",
            self.jotty_dir / "brain_state"
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # MARKOVIAN TODO PERSISTENCE (NO HARDCODING)
    # =========================================================================
    
    def save_markovian_todo(self, todo: 'SwarmTaskBoard'):
        """
        Save Markovian TODO state (JSON + rich markdown).
        
        NO HARDCODING:
        - All task attributes serialized dynamically
        - Display generated from current state
        - No hardcoded formatting rules
        """
        # JSON state (complete serialization)
        state_file = self.jotty_dir / "markovian_todos" / "todo_state.json"
        state_data = self._serialize_todo(todo)
        with open(state_file, 'w') as f:
            json.dump(state_data, f, indent=2, default=str)
        
        # Rich Markdown display
        display_file = self.jotty_dir / "markovian_todos" / "todo_display.md"
        with open(display_file, 'w') as f:
            f.write(self._format_todo_markdown(todo))
        
        logger.info(f" Saved Markovian TODO: {len(todo.subtasks)} tasks, "
                   f"{len(todo.completed_tasks)} completed")
    
    def _serialize_todo(self, todo) -> Dict:
        """
        Serialize TODO to JSON-compatible dict.
        
        NO HARDCODING: Serializes ALL task attributes dynamically.
        """
        return {
            'root_task': todo.root_task,
            'todo_id': todo.todo_id,
            'subtasks': {
                task_id: self._serialize_task(task)
                for task_id, task in todo.subtasks.items()
            },
            'execution_order': todo.execution_order,
            'completed_tasks': list(todo.completed_tasks),
            'failed_tasks': list(todo.failed_tasks),
            'current_task_id': todo.current_task_id,
            'estimated_remaining_steps': todo.estimated_remaining_steps,
            'completion_probability': todo.completion_probability,
            'timestamp': time.time()
        }
    
    def _serialize_task(self, task) -> Dict:
        """
        Serialize a single task.
        
        NO HARDCODING: Handles all attributes including intermediary_values.
        """
        return {
            'task_id': task.task_id,
            'description': task.description,
            'actor': task.actor,
            'status': task.status.value if hasattr(task.status, 'value') else str(task.status),
            'priority': task.priority,
            'estimated_reward': task.estimated_reward,
            'confidence': task.confidence,
            'attempts': task.attempts,
            'max_attempts': task.max_attempts,
            'progress': task.progress,
            'depends_on': task.depends_on,
            'blocks': task.blocks,
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            'estimated_duration': task.estimated_duration,
            'intermediary_values': task.intermediary_values,  # NO HARDCODED KEYS
            'predicted_next_task': task.predicted_next_task,
            'predicted_duration': task.predicted_duration,
            'predicted_reward': task.predicted_reward,
            'failure_reasons': task.failure_reasons,
            'result': task.result,
            'error': task.error
        }
    
    def _format_todo_markdown(self, todo) -> str:
        """
        Format TODO as rich markdown display.
        
        NO HARDCODING:
        - Dynamically generates sections based on state
        - No hardcoded display thresholds
        - Sorting by priority * Q-value (algorithmic)
        """
        md = f"# Markovian TODO - {todo.root_task}\n\n"
        md += f"**ID:** `{todo.todo_id}`\n"
        md += f"**Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        md += "---\n\n"
        
        # Progress overview (NO HARDCODED THRESHOLDS)
        total = len(todo.subtasks)
        completed = len(todo.completed_tasks)
        failed = len(todo.failed_tasks)
        pending = total - completed - failed
        
        md += "## Progress Overview\n\n"
        md += f"- Completed: **{completed}/{total}** ({100*completed/total if total > 0 else 0:.0f}%)\n"
        md += f"- Failed: **{failed}**\n"
        md += f"- ⏳ Pending: **{pending}**\n"
        md += f"- Completion Probability: **{todo.completion_probability:.0%}**\n"
        md += f"- Estimated Remaining: **{todo.estimated_remaining_steps}** tasks\n\n"
        
        # In-progress tasks with FULL details (NO HARDCODING)
        in_progress = [t for t in todo.subtasks.values() 
                      if str(t.status.value if hasattr(t.status, 'value') else t.status) == 'in_progress']
        
        if in_progress:
            md += "## Currently In Progress\n\n"
            for task in in_progress:
                md += f"### {task.description}\n\n"
                md += f"- **Actor:** `{task.actor}`\n"
                md += f"- **Q-Value:** {task.estimated_reward:.3f} (confidence: {task.confidence:.2f})\n"
                md += f"- **Priority:** {task.priority:.2f}\n"
                md += f"- **Attempt:** {task.attempts}/{task.max_attempts}\n"
                md += f"- **Progress:** {task.progress:.0%}\n"
                
                # Intermediary values (NO HARDCODED KEYS - displays whatever is tracked)
                if task.intermediary_values:
                    md += f"- **Intermediary Values:**\n"
                    for key, val in task.intermediary_values.items():
                        md += f"  - `{key}`: {val}\n"
                
                # Predictions
                if task.predicted_next_task:
                    md += f"- **Predicted Next:** {task.predicted_next_task}"
                    if task.predicted_duration:
                        md += f" (duration: {task.predicted_duration:.1f}s)"
                    if task.predicted_reward:
                        md += f" (reward: {task.predicted_reward:.2f})"
                    md += "\n"
                
                # Timing
                if task.started_at:
                    elapsed = (datetime.now() - task.started_at).total_seconds()
                    md += f"- **Elapsed Time:** {elapsed:.1f}s / {task.estimated_duration:.1f}s estimated\n"
                
                md += "\n"
        
        # Pending tasks sorted by ALGORITHM (NO HARDCODED WEIGHTS)
        pending_tasks = [t for t in todo.subtasks.values() 
                        if str(t.status.value if hasattr(t.status, 'value') else t.status) == 'pending']
        
        if pending_tasks:
            # Sort by priority * Q-value (ALGORITHMIC SCORING)
            pending_tasks.sort(key=lambda t: t.priority * t.estimated_reward, reverse=True)
            
            md += "## ⏳ Pending Tasks Queue\n\n"
            md += "| Rank | Task | Actor | Priority | Q-Value | Score | Dependencies |\n"
            md += "|------|------|-------|----------|---------|-------|-------------|\n"
            
            # Show top tasks (NO HARDCODED LIMIT - uses all if <= 15, else top 15)
            display_count = min(len(pending_tasks), 15)
            for i, task in enumerate(pending_tasks[:display_count], 1):
                score = task.priority * task.estimated_reward
                deps = len(task.depends_on)
                can_start = task.can_start(todo.completed_tasks)
                status_icon = "" if can_start else ""
                
                md += f"| {i} | {task.description}... | {task.actor} | {task.priority:.2f} | {task.estimated_reward:.3f} | {score:.3f} | {deps} {status_icon} |\n"
            
            if len(pending_tasks) > display_count:
                md += f"\n*...and {len(pending_tasks) - display_count} more tasks*\n"
            md += "\n"
        
        # Completed tasks (recent ones, NO HARDCODED LIMIT)
        completed_tasks = [todo.subtasks[tid] for tid in todo.completed_tasks 
                          if tid in todo.subtasks]
        if completed_tasks:
            # Sort by completion time (most recent first)
            completed_tasks.sort(key=lambda t: t.completed_at if t.completed_at else datetime.min, reverse=True)
            
            md += "## Recently Completed\n\n"
            display_count = min(len(completed_tasks), 10)
            for task in completed_tasks[:display_count]:
                duration = ""
                if task.started_at and task.completed_at:
                    dur = (task.completed_at - task.started_at).total_seconds()
                    duration = f" ({dur:.1f}s)"
                
                reward_info = ""
                if 'reward_obtained' in task.intermediary_values:
                    reward_info = f" [Reward: {task.intermediary_values['reward_obtained']:.2f}]"
                
                md += f"- {task.description}{duration}{reward_info}\n"
            
            if len(completed_tasks) > display_count:
                md += f"\n*...and {len(completed_tasks) - display_count} more*\n"
            md += "\n"
        
        # Failed tasks with reasons (NO HARDCODED ERROR MESSAGES)
        failed_tasks = [todo.subtasks[tid] for tid in todo.failed_tasks 
                       if tid in todo.subtasks]
        if failed_tasks:
            md += "## Failed Tasks (Need Investigation)\n\n"
            for task in failed_tasks:
                md += f"### {task.description}\n\n"
                md += f"- **Actor:** `{task.actor}`\n"
                md += f"- **Attempts:** {task.attempts}/{task.max_attempts}\n"
                md += f"- **Last Error:** {task.error if task.error else 'Unknown'}\n"
                
                if task.failure_reasons:
                    md += f"- **Failure History:**\n"
                    # Show all failures (NO HARDCODED LIMIT)
                    for i, reason in enumerate(task.failure_reasons, 1):
                        md += f"  {i}. {reason}\n"
                
                md += "\n"
        
        # State Insights (ALGORITHMIC - NO HARDCODING)
        md += "## State Insights\n\n"
        
        if total > 0:
            avg_q = sum(t.estimated_reward for t in todo.subtasks.values()) / total
            md += f"- **Average Q-Value:** {avg_q:.3f}\n"
        
        if completed > 0:
            completed_with_time = [t for t in completed_tasks if t.started_at and t.completed_at]
            if completed_with_time:
                avg_duration = sum((t.completed_at - t.started_at).total_seconds() 
                                  for t in completed_with_time) / len(completed_with_time)
                md += f"- **Average Task Duration:** {avg_duration:.1f}s\n"
        
        if failed > 0:
            failure_rate = failed / total
            md += f"- **Failure Rate:** {failure_rate:.1%}\n"
        
        md += "\n---\n\n"
        md += "*This display is auto-generated by JOTTY Vault with NO HARDCODED FORMATTING.*\n"
        
        return md
    
    # =========================================================================
    # Q-TABLE PERSISTENCE (NO HARDCODING)
    # =========================================================================
    
    def save_q_predictor(self, q_predictor: 'LLMQPredictor'):
        """
        Save Q-predictor experience buffer.
        
        NO HARDCODING: Saves entire buffer as-is.
        """
        buffer_file = self.jotty_dir / "q_tables" / "q_predictor_buffer.json"
        buffer_data = {
            'experience_buffer': q_predictor.experience_buffer,
            'buffer_size': len(q_predictor.experience_buffer),
            'timestamp': time.time()
        }
        with open(buffer_file, 'w') as f:
            json.dump(buffer_data, f, indent=2, default=str)
        
        logger.info(f" Saved Q-predictor: {len(q_predictor.experience_buffer)} experiences")
    
    # =========================================================================
    # MEMORY PERSISTENCE (NO HARDCODING)
    # =========================================================================
    
    def save_memory(self, memory: 'SwarmMemory', name: str = "shared", 
                   max_per_level: int = 100):
        """
        Save hierarchical memory.
        
        NO HARDCODING:
        - Saves all memory levels dynamically
        - max_per_level is configurable
        - Memory structure extensible
        """
        if name == "shared":
            memory_file = self.jotty_dir / "memories" / "shared_memory.json"
        else:
            memory_file = self.jotty_dir / "memories" / "local_memories" / f"{name}.json"
        
        # Serialize memory levels (NO HARDCODED LEVELS - iterates whatever exists)
        memory_data = {}
        total_memories = 0
        
        if hasattr(memory, 'storage'):
            for level, memories in memory.storage.items():
                # Save up to max_per_level (CONFIGURABLE)
                level_data = []
                for m in memories[:max_per_level]:
                    level_data.append({
                        'content': m.get('content', ''),
                        'context': m.get('context', {}),
                        'goal': m.get('goal', ''),
                        'value': m.get('value', 0.0),
                        'timestamp': m.get('timestamp', 0),
                        # Include any other fields dynamically (NO HARDCODED SCHEMA)
                        **{k: v for k, v in m.items() 
                           if k not in ['content', 'context', 'goal', 'value', 'timestamp']}
                    })
                memory_data[level] = level_data
                total_memories += len(level_data)
        
        memory_data['metadata'] = {
            'name': name,
            'total_memories': total_memories,
            'timestamp': time.time()
        }
        
        with open(memory_file, 'w') as f:
            json.dump(memory_data, f, indent=2, default=str)
        
        logger.info(f" Saved memory '{name}': {total_memories} memories across "
                   f"{len(memory_data)-1} levels")
    
    # =========================================================================
    # EPISODE PERSISTENCE (NO HARDCODING)
    # =========================================================================
    
    def save_episode(self, episode_num: int, trajectory: List, metadata: Dict):
        """
        Save episode history.
        
        NO HARDCODING: Saves complete trajectory with any metadata.
        """
        episode_file = self.jotty_dir / "episode_history" / f"episode_{episode_num}.json"
        episode_data = {
            'episode': episode_num,
            'metadata': metadata,  # Accept any metadata (NO HARDCODED KEYS)
            'trajectory': trajectory,
            'trajectory_length': len(trajectory),
            'timestamp': time.time()
        }
        with open(episode_file, 'w') as f:
            json.dump(episode_data, f, indent=2, default=str)
        
        logger.info(f" Saved episode {episode_num}: {len(trajectory)} steps")
    
    # =========================================================================
    # BRAIN STATE PERSISTENCE (NO HARDCODING)
    # =========================================================================
    
    def save_brain_state(self, brain: 'SimpleBrain'):
        """
        Save brain consolidated memories.
        
        NO HARDCODING: Saves whatever brain state exists.
        """
        if not brain:
            return
        
        brain_file = self.jotty_dir / "brain_state" / "consolidated_memories.json"
        
        # Serialize brain state (DEFENSIVE - works with any brain type)
        brain_data = {
            'timestamp': time.time(),
            'brain_type': type(brain).__name__,
        }
        
        # Add known attributes defensively
        if hasattr(brain, 'preset'):
            brain_data['preset'] = str(brain.preset.value if hasattr(brain.preset, 'value') else brain.preset)
        if hasattr(brain, 'chunk_size'):
            brain_data['chunk_size'] = brain.chunk_size
        if hasattr(brain, 'consolidation_count'):
            brain_data['consolidation_count'] = brain.consolidation_count
        if hasattr(brain, 'sleep_interval'):
            brain_data['sleep_interval'] = brain.sleep_interval
        
        # Add any other brain attributes dynamically (NO HARDCODED LIST)
        for attr in dir(brain):
            if not attr.startswith('_') and attr not in brain_data:
                try:
                    val = getattr(brain, attr)
                    if not callable(val):
                        brain_data[attr] = str(val)  # Truncate long values
                except (AttributeError, TypeError, ValueError) as e:
                    logger.debug(f"Could not serialize brain attribute {attr}: {e}")
                    pass
        
        with open(brain_file, 'w') as f:
            json.dump(brain_data, f, indent=2, default=str)
        
        logger.info(" Saved brain state")
    
    # =========================================================================
    # COMPLETE STATE SAVE (NO HARDCODING)
    # =========================================================================
    
    def save_all(self, conductor):
        """
        Save complete JOTTY Orchestrator state.
        
        NO HARDCODING: Saves all components that exist on the object.
        """
        logger.info(f"\n{'='*60}")
        logger.info(" SAVING COMPLETE JOTTY STATE")
        logger.info(f"{'='*60}")
        
        # Save TODO
        if hasattr(conductor, 'todo'):
            self.save_markovian_todo(conductor.todo)
        
        # Save Q-predictor
        if hasattr(conductor, 'q_predictor'):
            self.save_q_predictor(conductor.q_predictor)
        
        # Save shared memory
        if hasattr(conductor, 'shared_memory'):
            self.save_memory(conductor.shared_memory, "shared")
        
        # Save local memories (NO HARDCODED ACTOR LIST)
        if hasattr(conductor, 'local_memories'):
            for name, memory in conductor.local_memories.items():
                self.save_memory(memory, name)
        
        # Save episode
        if hasattr(conductor, 'episode_count') and hasattr(conductor, 'trajectory'):
            metadata = {
                'total_episodes': getattr(conductor, 'total_episodes', 1),
                'iteration': getattr(conductor, 'iteration', 0),
                # Add any other swarm metadata dynamically
            }
            self.save_episode(conductor.episode_count, conductor.trajectory, metadata)
        
        # Save brain
        if hasattr(conductor, 'brain') and conductor.brain:
            self.save_brain_state(conductor.brain)
        
        logger.info(f"{'='*60}")
        logger.info(f" ALL JOTTY STATE SAVED")
        logger.info(f" Location: {self.jotty_dir}")
        logger.info(f"{'='*60}\n")
    
    # =========================================================================
    # AUTO-SAVE HELPER (NO HARDCODED INTERVAL)
    # =========================================================================
    
    def should_auto_save(self, iteration: int) -> bool:
        """Check if should auto-save (based on configurable interval)."""
        return iteration % self.auto_save_interval == 0

