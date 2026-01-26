"""
Composite Skill Framework

A composite skill combines multiple skills into a workflow.
Supports sequential, parallel, and mixed execution patterns.

DRY Principle: Reuses existing skills, no duplication.
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Literal
from enum import Enum

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution modes for composite skills."""
    SEQUENTIAL = "sequential"  # Execute steps one after another
    PARALLEL = "parallel"  # Execute steps simultaneously
    MIXED = "mixed"  # Combination of sequential and parallel


class CompositeSkill:
    """
    Composite skill that combines multiple skills into a workflow.
    
    Follows DRY principles - reuses existing skills without duplication.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        steps: List[Dict[str, Any]],
        execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    ):
        """
        Initialize composite skill.
        
        Args:
            name: Skill name
            description: Skill description
            steps: List of step definitions, each with:
                - skill_name: Name of skill to use
                - tool_name: Name of tool in that skill
                - params: Parameters for the tool (can be dict or function)
                - depends_on: List of step indices this depends on (for parallel)
                - output_key: Key to store result in (default: step index)
            execution_mode: How to execute steps
        """
        self.name = name
        self.description = description
        self.steps = steps
        self.execution_mode = execution_mode
    
    async def execute(self, initial_params: Dict[str, Any], registry) -> Dict[str, Any]:
        """
        Execute composite skill workflow.
        
        Args:
            initial_params: Initial parameters
            registry: Skills registry instance
        
        Returns:
            Dictionary with results from all steps
        """
        results = {}
        results['_initial'] = initial_params
        
        if self.execution_mode == ExecutionMode.SEQUENTIAL:
            return await self._execute_sequential(initial_params, registry, results)
        elif self.execution_mode == ExecutionMode.PARALLEL:
            return await self._execute_parallel(initial_params, registry, results)
        else:  # MIXED
            return await self._execute_mixed(initial_params, registry, results)
    
    async def _execute_sequential(
        self,
        initial_params: Dict[str, Any],
        registry,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute steps sequentially."""
        current_params = initial_params.copy()
        
        for i, step in enumerate(self.steps):
            step_result = await self._execute_step(step, current_params, registry, results)
            output_key = step.get('output_key', f'step_{i}')
            results[output_key] = step_result
            
            # Update current_params with step result for next steps
            if step_result.get('success'):
                # Merge result into params for next steps
                current_params.update(step_result)
            
            # If step failed and required, stop
            if not step_result.get('success') and step.get('required', True):
                results['_success'] = False
                results['_error'] = f"Step {i} failed: {step_result.get('error')}"
                return results
        
        results['_success'] = True
        return results
    
    async def _execute_parallel(
        self,
        initial_params: Dict[str, Any],
        registry,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute steps in parallel."""
        tasks = []
        for i, step in enumerate(self.steps):
            task = self._execute_step(step, initial_params, registry, results)
            tasks.append((i, step, task))
        
        # Execute all in parallel
        step_results = await asyncio.gather(*[t[2] for t in tasks], return_exceptions=True)
        
        # Process results
        for (i, step, _), result in zip(tasks, step_results):
            if isinstance(result, Exception):
                result = {'success': False, 'error': str(result)}
            
            output_key = step.get('output_key', f'step_{i}')
            results[output_key] = result
        
        # Check if all succeeded
        all_success = all(r.get('success', False) if not isinstance(r, Exception) else False 
                         for r in step_results)
        results['_success'] = all_success
        
        return results
    
    async def _execute_mixed(
        self,
        initial_params: Dict[str, Any],
        registry,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute steps with dependencies (mixed sequential/parallel)."""
        # Build dependency graph
        executed = set()
        current_params = initial_params.copy()
        
        while len(executed) < len(self.steps):
            # Find steps ready to execute (dependencies satisfied)
            ready_steps = []
            for i, step in enumerate(self.steps):
                if i in executed:
                    continue
                
                depends_on = step.get('depends_on', [])
                if all(dep in executed for dep in depends_on):
                    ready_steps.append((i, step))
            
            if not ready_steps:
                # Circular dependency or error
                results['_success'] = False
                results['_error'] = 'Circular dependency or unresolved dependencies'
                return results
            
            # Execute ready steps in parallel
            tasks = []
            for i, step in ready_steps:
                task = self._execute_step(step, current_params, registry, results)
                tasks.append((i, step, task))
            
            step_results = await asyncio.gather(*[t[2] for t in tasks], return_exceptions=True)
            
            # Process results
            for (i, step, _), result in zip(tasks, step_results):
                if isinstance(result, Exception):
                    result = {'success': False, 'error': str(result)}
                
                output_key = step.get('output_key', f'step_{i}')
                results[output_key] = result
                executed.add(i)
                
                # Update params for next steps
                if result.get('success'):
                    current_params.update(result)
                
                # Check if required step failed
                if not result.get('success') and step.get('required', True):
                    results['_success'] = False
                    results['_error'] = f"Step {i} failed: {result.get('error')}"
                    return results
        
        results['_success'] = True
        return results
    
    async def _execute_step(
        self,
        step: Dict[str, Any],
        params: Dict[str, Any],
        registry,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single step."""
        skill_name = step.get('skill_name')
        tool_name = step.get('tool_name')
        
        if not skill_name or not tool_name:
            return {
                'success': False,
                'error': 'skill_name and tool_name required for step'
            }
        
        # Get skill and tool
        skill = registry.get_skill(skill_name)
        if not skill:
            return {
                'success': False,
                'error': f'Skill not found: {skill_name}'
            }
        
        tool_func = skill.tools.get(tool_name)
        if not tool_func:
            return {
                'success': False,
                'error': f'Tool not found: {skill_name}.{tool_name}'
            }
        
        # Prepare step parameters
        step_params = step.get('params', {})
        
        # If params is a function, call it with current params and results
        if callable(step_params):
            step_params = step_params(params, results)
        elif isinstance(step_params, dict):
            # Merge with current params
            step_params = {**params, **step_params}
        
        # Execute tool
        try:
            import inspect
            if inspect.iscoroutinefunction(tool_func):
                result = await tool_func(step_params)
            else:
                result = tool_func(step_params)
            
            return result
        except Exception as e:
            logger.error(f"Step execution error: {e}", exc_info=True)
            return {
                'success': False,
                'error': f'Step execution failed: {str(e)}'
            }


def create_composite_skill(
    name: str,
    description: str,
    steps: List[Dict[str, Any]],
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
) -> CompositeSkill:
    """
    Factory function to create composite skills.
    
    Example:
        skill = create_composite_skill(
            name='research-to-pdf',
            description='Research and generate PDF',
            steps=[
                {
                    'skill_name': 'last30days-claude-cli',
                    'tool_name': 'last30days_claude_cli_tool',
                    'params': lambda p, r: {'topic': p.get('topic')}
                },
                {
                    'skill_name': 'document-converter',
                    'tool_name': 'convert_to_pdf_tool',
                    'params': lambda p, r: {
                        'input_file': r['step_0']['md_path'],
                        'output_file': p.get('output_file')
                    }
                }
            ]
        )
    """
    return CompositeSkill(name, description, steps, execution_mode)
