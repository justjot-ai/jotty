"""
Workflow Orchestrator

Handles agent selection and routing for workflow tasks.
"""

import logging
from typing import Any, Dict, List, Optional

from .workflow_context import WorkflowContext, WorkflowTask

logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """
    Orchestrates workflow execution by selecting and routing to appropriate agents.
    """

    def __init__(
        self, conductor: Any, mode: str = "dynamic", agent_order: Optional[List[str]] = None
    ) -> None:
        """
        Initialize workflow orchestrator.

        Args:
            conductor: Jotty Conductor instance
            mode: Orchestration mode ("static" or "dynamic")
            agent_order: Required for static mode - list of agent IDs in execution order
        """
        self.conductor = conductor
        self.mode = mode
        self.agent_order = agent_order

        # Validate mode
        if mode not in ["static", "dynamic"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'static' or 'dynamic'")

        if mode == "static" and not agent_order:
            raise ValueError("agent_order is required for static mode")

    def select_agent(self, task: WorkflowTask, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Select agent for handling the task.

        Args:
            task: Workflow task
            context: Additional context

        Returns:
            Agent ID or name
        """
        # Static mode: use predefined order
        if self.mode == "static":
            # Find next agent in order that hasn't been used
            if self.agent_order:
                # Simple round-robin for now
                task_index = hash(task.id) % len(self.agent_order)
                agent_name = self.agent_order[task_index]
                logger.debug(f"Static mode: Selected agent: {agent_name}")
                return agent_name

        # Dynamic routing (delegate to conductor)
        if hasattr(self.conductor, "select_agent"):
            agent = self.conductor.select_agent(task.goal, None, context)
            logger.debug(f"Dynamic mode: Selected agent: {agent}")
            return agent

        # Fallback: use conductor's agent selection logic
        if hasattr(self.conductor, "actors") and self.conductor.actors:
            # Use Q-predictor if available for intelligent routing
            if hasattr(self.conductor, "q_predictor") and self.conductor.q_predictor:
                agent = self._select_with_q_predictor(task, context)
                if agent:
                    return agent

            # Default: first agent
            agent_name = (
                self.conductor.actors[0].name
                if hasattr(self.conductor.actors[0], "name")
                else str(self.conductor.actors[0])
            )
            logger.debug(f"Fallback: Using first agent: {agent_name}")
            return agent_name

        raise RuntimeError("No agents available for workflow")

    def _select_with_q_predictor(
        self, task: WorkflowTask, context: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Select agent using Q-predictor."""
        if not hasattr(self.conductor, "q_predictor") or not self.conductor.q_predictor:
            return None

        # Build state description
        state = {
            "goal": task.goal,
            "task_id": task.id,
            "dependencies": task.dependencies,
            "status": task.status.value,
            "context_keys": list(context.keys()) if context else [],
        }

        # Evaluate each agent
        best_agent = None
        best_q_value = -1.0

        for actor in self.conductor.actors:
            agent_name = actor.name if hasattr(actor, "name") else str(actor)
            action = {"agent": agent_name, "type": "workflow", "task_id": task.id}

            try:
                q_value, confidence, _ = self.conductor.q_predictor.predict_q_value(
                    state=state, action=action, goal=task.goal
                )

                if q_value and q_value > best_q_value:
                    best_q_value = q_value
                    best_agent = agent_name
            except Exception as e:
                logger.warning(f"Q-prediction failed for {agent_name}: {e}")

        return best_agent

    def prepare_agent_context(
        self,
        task: WorkflowTask,
        workflow_context: WorkflowContext,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Prepare context for agent execution.

        Args:
            task: Workflow task
            workflow_context: Workflow context manager
            context: Additional context

        Returns:
            Context dictionary for agent
        """
        agent_context = {
            "goal": task.goal,
            "query": task.goal,  # For compatibility
            "task_id": task.id,
            **(context or {}),
        }

        # Add dependency results
        if task.dependencies:
            dep_results = {}
            for dep_id in task.dependencies:
                dep_task = workflow_context.get_task(dep_id)
                if dep_task and dep_task.result is not None:
                    dep_results[dep_id] = dep_task.result

            if dep_results:
                agent_context["dependency_results"] = dep_results

        # Add workflow metadata
        agent_context["workflow_id"] = workflow_context.workflow_id
        agent_context["workflow_summary"] = workflow_context.get_summary()

        return agent_context
