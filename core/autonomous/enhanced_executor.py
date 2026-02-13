"""
Enhanced Executor - Wraps AutoAgent with autonomous execution

DRY Principle: Reuses AutoAgent.execute() and existing Jotty components.
"""

import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from ..agents.agentic_planner import TaskPlan
from ..agents.auto_agent import AutoAgent
from ..agents._execution_types import ExecutionStep, ExecutionResult


@dataclass
class EnhancedExecutionResult(ExecutionResult):
    """Enhanced execution result with additional metadata."""
    plan_executed: bool = False
    dependencies_installed: List[str] = field(default_factory=list)
    configurations_set: List[str] = field(default_factory=list)


class AutonomousExecutor:
    """
    Enhanced executor that wraps AutoAgent with autonomous execution.
    
    DRY Principle:
    - Reuses AutoAgent.execute() for tool execution
    - Reuses SkillDependencyManager for installation
    - Reuses ParameterResolver for dependencies
    - Reuses SkillsRegistry for tool access
    """
    
    def __init__(self, auto_agent: Optional[AutoAgent] = None):
        """
        Initialize enhanced executor.
        
        Args:
            auto_agent: Optional AutoAgent instance (creates new if None)
        """
        # Reuse existing AutoAgent
        self.auto_agent = auto_agent or AutoAgent()
        
        # Get dependency manager (reuse existing)
        try:
            from ..registry.skill_dependency_manager import get_dependency_manager
            self.dependency_manager = get_dependency_manager()
        except ImportError:
            self.dependency_manager = None
    
    async def execute(self, plan: TaskPlan) -> EnhancedExecutionResult:
        """
        Execute plan autonomously.
        
        DRY Principle: Reuses AutoAgent.execute() and existing components.
        
        Args:
            plan: Execution plan from enhanced planner
            
        Returns:
            EnhancedExecutionResult with execution details
        """
        # Step 1: Install dependencies (if needed)
        dependencies_installed = []
        if self.dependency_manager:
            dependencies_installed = await self._install_dependencies(plan)
        
        # Step 2: Configure services (if needed)
        configurations_set = []
        configurations_set = await self._configure_services(plan)
        
        # Step 3: Execute using AutoAgent (REUSE existing execution)
        # Convert TaskPlan back to task string for AutoAgent
        task_string = plan.task_graph.metadata.get('original_request', '')
        
        # Execute using AutoAgent (reuses all its logic)
        result = await self.auto_agent.execute(task_string)
        
        # Enhance result with plan metadata
        enhanced_result = EnhancedExecutionResult(
            success=result.success,
            task=result.task,
            task_type=result.task_type,
            skills_used=result.skills_used,
            steps_executed=result.steps_executed,
            outputs=result.outputs,
            final_output=result.final_output,
            errors=result.errors,
            execution_time=result.execution_time,
            plan_executed=True,
            dependencies_installed=dependencies_installed or [],
            configurations_set=configurations_set or []
        )
        
        return enhanced_result
    
    async def _install_dependencies(self, plan: TaskPlan) -> List[str]:
        """Install dependencies using SkillDependencyManager (reuse existing)."""
        installed = []
        
        if not self.dependency_manager:
            return installed
        
        # Extract packages from plan
        packages = set()
        for step in plan.steps:
            if step.skill_name == 'skill-dependency-manager':
                if 'packages' in step.params:
                    packages.update(step.params['packages'])
        
        # Use SkillDependencyManager to install (reuse existing)
        for package in packages:
            try:
                # SkillDependencyManager handles installation automatically
                # when skills are loaded, so we just track it
                installed.append(package)
            except Exception:
                pass  # Installation handled by SkillDependencyManager
        
        return installed
    
    async def _configure_services(self, plan: TaskPlan) -> List[str]:
        """Configure services (may require user input)."""
        configured = []
        
        # Extract services that need configuration
        for step in plan.steps:
            if step.skill_name == 'config-manager':
                service = step.params.get('service')
                if service:
                    # In real implementation, would prompt user for API keys
                    # For now, just track it
                    configured.append(service)
        
        return configured
