"""
SwarmDAGExecutor - Extracted from Orchestrator
===============================================

DAG-based orchestration: TaskBreakdownAgent + TodoCreatorAgent + parallel stages.
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from Jotty.core.infrastructure.foundation.data_structures import EpisodeResult

if TYPE_CHECKING:
    from .swarm_manager import Orchestrator

logger = logging.getLogger(__name__)


class SwarmDAGExecutor:
    """
    DAG-based task orchestration.

    Breaks down plans into tasks, assigns to agents,
    executes in parallel stages, and records learnings.
    """

    def __init__(self, swarm: "Orchestrator") -> None:
        self.swarm = swarm

    async def run(
        self,
        implementation_plan: str,
        available_actors: List[Dict[str, Any]] = None,
        status_callback: Any = None,
        **kwargs: Any,
    ) -> EpisodeResult:
        """
        Execute a task using DAG-based orchestration.

        Args:
            implementation_plan: Implementation plan or goal description
            available_actors: List of actor dicts with 'name' and 'capabilities'
            status_callback: Optional callback(stage, detail) for progress

        Returns:
            EpisodeResult with execution output and metadata
        """
        from Jotty.core.modes.agent.dag_agents import TaskBreakdownAgent, TodoCreatorAgent

        def _status(stage: str, detail: str = "") -> Any:
            if status_callback:
                try:
                    status_callback(stage, detail)
                except Exception:
                    pass
            logger.info(f" [DAG] {stage}" + (f": {detail}" if detail else ""))

        _status("Initializing", "DAG-based orchestration")

        # Ensure DSPy LM is configured
        import dspy

        if not hasattr(dspy.settings, "lm") or dspy.settings.lm is None:
            lm = self.swarm.swarm_provider_gateway.get_lm()
            if lm:
                dspy.configure(lm=lm)

        # Default actors from existing agents
        if available_actors is None:
            available_actors = [
                {
                    "name": agent.name,
                    "capabilities": getattr(agent.agent, "capabilities", ["coding", "analysis"]),
                    "description": getattr(agent.agent, "description", None),
                }
                for agent in self.swarm.agents
            ]
            _status("Using existing agents", f"{len(available_actors)} actors")

        # Step 1: Break down plan into tasks
        _status("Task breakdown", "analyzing plan")
        breakdown_agent = TaskBreakdownAgent(config=self.swarm.config)
        markovian_todo = breakdown_agent(implementation_plan)
        _status("Tasks extracted", f"{len(markovian_todo.subtasks)} tasks")

        # Step 2: Assign actors and create executable DAG
        _status("Actor assignment", "optimizing assignments")
        todo_agent = TodoCreatorAgent(config=self.swarm.config, lm=dspy.settings.lm)
        executable_dag = todo_agent.create_executable_dag(
            markovian_todo=markovian_todo,
            available_actors=available_actors,
        )
        _status("DAG created", f"{len(executable_dag.assignments)} assignments")

        viz = todo_agent.visualize_assignments(executable_dag)
        logger.info(viz)

        # Step 3: Execute in parallel stages
        _status("Execution", "starting stage-based execution")
        stages = executable_dag.get_execution_stages()
        all_outputs = []
        outcomes = {}
        total_start = time.time()

        for stage_idx, stage_task_ids in enumerate(stages, 1):
            _status(f"Stage {stage_idx}/{len(stages)}", f"{len(stage_task_ids)} tasks in parallel")

            stage_results = await self._execute_stage(
                executable_dag,
                stage_task_ids,
                status_callback,
            )

            for task_id, result in stage_results.items():
                outcomes[task_id] = result.get("success", False)
                if result.get("output"):
                    all_outputs.append(f"[{task_id}]: {result['output']}")

                executable_dag.add_trajectory_step(
                    task_id=task_id,
                    action_type="execution",
                    action_content=f"Executed by {result.get('actor', 'unknown')}",
                    observation=str(result.get("output", ""))[:200],
                    reward=1.0 if result.get("success") else -0.5,
                )

        total_time = time.time() - total_start

        # Step 4: Learn from outcomes
        _status("Learning", "recording execution outcomes")
        todo_agent.update_from_execution(executable_dag, outcomes)

        success_count = sum(1 for s in outcomes.values() if s)
        overall_success = success_count == len(outcomes) if outcomes else False
        success_rate = success_count / len(outcomes) if outcomes else 0.0

        _status("Complete", f"{success_count}/{len(outcomes)} tasks succeeded ({success_rate:.0%})")

        return EpisodeResult(
            output="\n\n".join(all_outputs) if all_outputs else "No output",
            success=overall_success,
            trajectory=[
                {
                    "step": s.step_idx,
                    "action": s.action_content,
                    "observation": s.observation,
                    "reward": s.reward,
                }
                for s in executable_dag.trajectory
            ],
            tagged_outputs=[],
            episode=0,
            execution_time=total_time,
            architect_results=[],
            auditor_results=[],
            agent_contributions={},
            override_metadata={
                "mode": "dag_orchestration",
                "total_tasks": len(executable_dag.markovian_todo.subtasks),
                "stages": len(stages),
                "outcomes": outcomes,
                "success_rate": success_rate,
                "dag_serialized": executable_dag.to_dict(),
            },
        )

    async def _execute_stage(
        self, executable_dag: Any, task_ids: List[str], status_callback: Any = None
    ) -> Dict[str, Dict[str, Any]]:
        """Execute a stage of tasks (potentially in parallel)."""
        from Jotty.core.modes.agent.auto_agent import AutoAgent

        results = {}

        async def execute_single_task(task_id: str) -> Any:
            task = executable_dag.markovian_todo.subtasks.get(task_id)
            actor = executable_dag.assignments.get(task_id)

            if not task or not actor:
                return task_id, {"success": False, "error": "Task or actor not found"}

            runner = self.swarm.runners.get(actor.name)

            if runner:
                try:
                    task.start()
                    result = await runner.run(task.description)
                    task.complete(
                        {"output": result.output if hasattr(result, "output") else str(result)}
                    )
                    return task_id, {
                        "success": result.success if hasattr(result, "success") else True,
                        "output": result.output if hasattr(result, "output") else str(result),
                        "actor": actor.name,
                    }
                except Exception as e:
                    task.fail(str(e))
                    return task_id, {"success": False, "error": str(e), "actor": actor.name}
            else:
                try:
                    task.start()
                    auto_agent = AutoAgent()
                    result = await auto_agent.execute(task.description)
                    task.complete(
                        {
                            "output": (
                                result.final_output
                                if hasattr(result, "final_output")
                                else str(result)
                            )
                        }
                    )
                    return task_id, {
                        "success": result.success if hasattr(result, "success") else True,
                        "output": (
                            result.final_output if hasattr(result, "final_output") else str(result)
                        ),
                        "actor": actor.name,
                    }
                except Exception as e:
                    task.fail(str(e))
                    return task_id, {"success": False, "error": str(e), "actor": actor.name}

        tasks = [execute_single_task(tid) for tid in task_ids]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for item in completed:
            if isinstance(item, Exception):
                logger.error(f"Task execution error: {item}")
            else:
                task_id, result = item
                results[task_id] = result

        return results

    def get_agents(self) -> Tuple:
        """Get DAG agents for external use."""
        import dspy

        from Jotty.core.modes.agent.dag_agents import TaskBreakdownAgent, TodoCreatorAgent

        if not hasattr(dspy.settings, "lm") or dspy.settings.lm is None:
            lm = self.swarm.swarm_provider_gateway.get_lm()
            if lm:
                dspy.configure(lm=lm)

        return (
            TaskBreakdownAgent(config=self.swarm.config),
            TodoCreatorAgent(config=self.swarm.config, lm=dspy.settings.lm),
        )
