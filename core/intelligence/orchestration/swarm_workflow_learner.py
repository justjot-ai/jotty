"""
SwarmWorkflowLearner - Workflow Pattern Learning and Reuse

Learns from successful workflows and reuses patterns.
Follows DRY: Reuses existing memory system.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WorkflowPattern:
    """Learned workflow pattern."""

    pattern_id: str
    task_type: str
    operations: List[str]
    tools_used: List[str]
    success_rate: float
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class SwarmWorkflowLearner:
    """
    Learns workflow patterns and enables reuse.

    DRY Principle: Reuses existing SwarmMemory for storage.
    """

    def __init__(self, swarm_memory: Any = None) -> None:
        """
        Initialize SwarmWorkflowLearner.

        Args:
            swarm_memory: Optional SwarmMemory instance (reuses existing)
        """
        self.swarm_memory = swarm_memory
        self._patterns: Dict[str, WorkflowPattern] = {}
        self._pattern_index: Dict[str, List[str]] = {}  # task_type -> pattern_ids

    def learn_from_execution(
        self,
        task_type: str,
        operations: List[str],
        tools_used: List[str],
        success: bool,
        execution_time: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Learn from a workflow execution.

        Args:
            task_type: Type of task (e.g., "research", "automation")
            operations: List of operations performed
            tools_used: List of tools/skills used
            success: Whether execution was successful
            execution_time: Time taken
            metadata: Optional additional metadata
        """
        # Create pattern signature
        pattern_signature = self._create_pattern_signature(task_type, operations, tools_used)

        # Update or create pattern
        if pattern_signature in self._patterns:
            pattern = self._patterns[pattern_signature]
            # Update success rate
            total_executions = pattern.metadata.get("total_executions", 0) + 1
            successful_executions = pattern.metadata.get("successful_executions", 0)
            if success:
                successful_executions += 1

            pattern.success_rate = successful_executions / total_executions
            pattern.execution_time = (pattern.execution_time + execution_time) / 2
            pattern.metadata["total_executions"] = total_executions
            pattern.metadata["successful_executions"] = successful_executions
            pattern.metadata["last_used"] = datetime.now().isoformat()
        else:
            # Create new pattern
            pattern = WorkflowPattern(
                pattern_id=pattern_signature,
                task_type=task_type,
                operations=operations,
                tools_used=tools_used,
                success_rate=1.0 if success else 0.0,
                execution_time=execution_time,
                metadata={
                    "total_executions": 1,
                    "successful_executions": 1 if success else 0,
                    "created": datetime.now().isoformat(),
                    "last_used": datetime.now().isoformat(),
                    **(metadata or {}),
                },
            )
            self._patterns[pattern_signature] = pattern

            # Index by task type
            if task_type not in self._pattern_index:
                self._pattern_index[task_type] = []
            self._pattern_index[task_type].append(pattern_signature)

        # Store in memory if available (DRY: reuse existing memory)
        if self.swarm_memory:
            try:
                self.swarm_memory.store(
                    content=f"Workflow pattern: {pattern_signature}",
                    level="procedural",  # Procedural memory for workflows
                    context={
                        "pattern_id": pattern_signature,
                        "task_type": task_type,
                        "operations": operations,
                        "tools_used": tools_used,
                    },
                    goal="workflow_learning",
                )
            except Exception as e:
                logger.debug(f"Failed to store pattern in memory: {e}")

        logger.info(
            f" Learned pattern: {pattern_signature} (success_rate: {pattern.success_rate:.2f})"
        )

    def find_similar_pattern(
        self, task_type: str, operations: List[str], tools_available: List[str]
    ) -> Optional[WorkflowPattern]:
        """
        Find similar workflow pattern.

        Args:
            task_type: Type of task
            operations: Desired operations
            tools_available: Available tools

        Returns:
            Best matching WorkflowPattern or None
        """
        # Look for patterns of same task type
        candidate_ids = self._pattern_index.get(task_type, [])

        if not candidate_ids:
            return None

        best_match = None
        best_score = 0.0

        for pattern_id in candidate_ids:
            pattern = self._patterns[pattern_id]

            # Calculate similarity score
            score = self._calculate_similarity(pattern, operations, tools_available)

            # Weight by success rate
            score *= pattern.success_rate

            if score > best_score:
                best_score = score
                best_match = pattern

        if best_match and best_score > 0.5:  # Threshold
            logger.info(
                f" Found similar pattern: {best_match.pattern_id} (score: {best_score:.2f})"
            )
            return best_match

        return None

    def _create_pattern_signature(
        self, task_type: str, operations: List[str], tools_used: List[str]
    ) -> str:
        """Create unique signature for pattern."""
        ops_str = "_".join(sorted(operations))
        tools_str = "_".join(sorted(tools_used))
        return f"{task_type}:{ops_str}:{tools_str}"

    def _calculate_similarity(
        self, pattern: WorkflowPattern, operations: List[str], tools_available: List[str]
    ) -> float:
        """Calculate similarity score between pattern and requirements."""
        # Operation overlap
        ops_set = set(pattern.operations)
        req_ops_set = set(operations)
        ops_overlap = len(ops_set & req_ops_set) / max(len(ops_set | req_ops_set), 1)

        # Tool availability (can we use the same tools?)
        tools_set = set(pattern.tools_used)
        avail_tools_set = set(tools_available)
        tools_overlap = len(tools_set & avail_tools_set) / max(len(tools_set), 1)

        # Combined score
        score = ops_overlap * 0.6 + tools_overlap * 0.4
        return score

    def get_patterns_by_task_type(self, task_type: str) -> List[WorkflowPattern]:
        """Get all patterns for a task type."""
        pattern_ids = self._pattern_index.get(task_type, [])
        return [self._patterns[pid] for pid in pattern_ids if pid in self._patterns]

    def get_best_patterns(self, limit: int = 5) -> List[WorkflowPattern]:
        """Get best patterns by success rate."""
        patterns = list(self._patterns.values())
        patterns.sort(key=lambda p: (p.success_rate, -p.execution_time), reverse=True)
        return patterns[:limit]
