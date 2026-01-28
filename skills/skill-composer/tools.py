"""
Skill Composer - Dynamically compose skills into workflows

Supports:
- Sequential execution (A → B → C)
- Parallel execution (A || B || C)
- Conditional execution (if X then A else B)
- Loops (repeat A N times)
- Error handling (try A, catch error, do B)
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


async def compose_skills_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compose multiple skills into a workflow.
    
    Args:
        params: Dictionary containing:
            - workflow (list, required): List of workflow steps
            - Each step can be:
                - Sequential: {"type": "sequential", "skills": ["skill1", "skill2"]}
                - Parallel: {"type": "parallel", "skills": ["skill1", "skill2"]}
                - Conditional: {"type": "conditional", "condition": {...}, "then": [...], "else": [...]}
                - Loop: {"type": "loop", "count": 3, "skill": "skill1"}
                - Single: {"type": "single", "skill": "skill1", "params": {...}}
    
    Returns:
        Dictionary with execution results
    """
    try:
        from core.registry.skills_registry import get_skills_registry
        
        workflow = params.get('workflow', [])
        if not workflow:
            return {
                'success': False,
                'error': 'workflow parameter is required'
            }
        
        registry = get_skills_registry()
        registry.init()
        
        results = {}
        outputs = {}
        
        logger.info(f"🎼 Composing {len(workflow)} workflow steps...")
        
        for i, step in enumerate(workflow):
            step_type = step.get('type', 'single')
            step_name = step.get('name', f'step_{i+1}')
            
            logger.info(f"  Step {i+1}/{len(workflow)}: {step_type} - {step_name}")
            
            try:
                if step_type == 'sequential':
                    result = await _execute_sequential(step, registry, outputs)
                elif step_type == 'parallel':
                    result = await _execute_parallel(step, registry, outputs)
                elif step_type == 'conditional':
                    result = await _execute_conditional(step, registry, outputs)
                elif step_type == 'loop':
                    result = await _execute_loop(step, registry, outputs)
                elif step_type == 'single':
                    result = await _execute_single(step, registry, outputs)
                else:
                    result = {
                        'success': False,
                        'error': f'Unknown step type: {step_type}'
                    }
                
                results[step_name] = result
                
                # Store outputs for next steps
                if result.get('success') and result.get('output'):
                    outputs[step_name] = result['output']
                    # Also store by output_key if specified
                    if 'output_key' in step:
                        outputs[step[step['output_key']]] = result['output']
                
            except Exception as e:
                logger.error(f"  ❌ Step {step_name} failed: {e}")
                results[step_name] = {
                    'success': False,
                    'error': str(e)
                }
                
                # Check if step has error handling
                if 'on_error' in step:
                    logger.info(f"  🔄 Executing error handler for {step_name}")
                    error_result = await _execute_single(
                        {'skill': step['on_error'], 'params': step.get('error_params', {})},
                        registry,
                        outputs
                    )
                    results[f"{step_name}_error_handler"] = error_result
        
        # Check overall success
        all_success = all(r.get('success', False) for r in results.values())
        
        return {
            'success': all_success,
            'results': results,
            'outputs': outputs,
            'steps_executed': len(workflow)
        }
        
    except Exception as e:
        logger.error(f"Skill composition failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


async def _execute_single(step: Dict[str, Any], registry, outputs: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single skill."""
    skill_name = step.get('skill')
    if not skill_name:
        return {'success': False, 'error': 'skill name required'}
    
    skill = registry.get_skill(skill_name)
    if not skill:
        return {'success': False, 'error': f'skill {skill_name} not found'}
    
    # Get tool name (default to first tool or specified)
    tool_name = step.get('tool') or list(skill.tools.keys())[0]
    tool = skill.tools.get(tool_name)
    if not tool:
        return {'success': False, 'error': f'tool {tool_name} not found in {skill_name}'}
    
    # Prepare params (can reference previous outputs)
    params = step.get('params', {})
    params = _resolve_params(params, outputs)
    
    # Execute tool
    if asyncio.iscoroutinefunction(tool):
        result = await tool(params)
    else:
        result = tool(params)
    
    return {
        'success': result.get('success', True),
        'output': result,
        'skill': skill_name,
        'tool': tool_name
    }


async def _execute_sequential(step: Dict[str, Any], registry, outputs: Dict[str, Any]) -> Dict[str, Any]:
    """Execute skills sequentially (one after another)."""
    skills = step.get('skills', [])
    results = []
    
    for skill_config in skills:
        if isinstance(skill_config, str):
            skill_config = {'skill': skill_config}
        
        result = await _execute_single(skill_config, registry, outputs)
        results.append(result)
        
        # Stop on first failure if stop_on_error is True
        if not result.get('success') and step.get('stop_on_error', True):
            break
        
        # Update outputs for next step
        if result.get('success') and result.get('output'):
            outputs[skill_config.get('name', skill_config.get('skill'))] = result['output']
    
    return {
        'success': all(r.get('success', False) for r in results),
        'results': results
    }


async def _execute_parallel(step: Dict[str, Any], registry, outputs: Dict[str, Any]) -> Dict[str, Any]:
    """Execute skills in parallel (all at once)."""
    skills = step.get('skills', [])
    
    # Create tasks for parallel execution
    tasks = []
    for skill_config in skills:
        if isinstance(skill_config, str):
            skill_config = {'skill': skill_config}
        tasks.append(_execute_single(skill_config, registry, outputs.copy()))
    
    # Execute all in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                'success': False,
                'error': str(result)
            })
        else:
            processed_results.append(result)
            # Update outputs
            if result.get('success') and result.get('output'):
                skill_config = skills[i] if isinstance(skills[i], dict) else {'skill': skills[i]}
                outputs[skill_config.get('name', skill_config.get('skill'))] = result['output']
    
    return {
        'success': all(r.get('success', False) for r in processed_results),
        'results': processed_results
    }


async def _execute_conditional(step: Dict[str, Any], registry, outputs: Dict[str, Any]) -> Dict[str, Any]:
    """Execute conditional logic (if-then-else)."""
    condition = step.get('condition', {})
    then_steps = step.get('then', [])
    else_steps = step.get('else', [])
    
    # Evaluate condition
    condition_met = _evaluate_condition(condition, outputs)
    
    steps_to_execute = then_steps if condition_met else else_steps
    
    if not steps_to_execute:
        return {'success': True, 'skipped': True}
    
    # Execute selected steps sequentially
    results = []
    for step_config in steps_to_execute:
        if isinstance(step_config, str):
            step_config = {'skill': step_config}
        result = await _execute_single(step_config, registry, outputs)
        results.append(result)
    
    return {
        'success': all(r.get('success', False) for r in results),
        'results': results,
        'branch': 'then' if condition_met else 'else'
    }


async def _execute_loop(step: Dict[str, Any], registry, outputs: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a skill multiple times (loop)."""
    count = step.get('count', 1)
    skill_config = step.get('skill', {})
    
    if isinstance(skill_config, str):
        skill_config = {'skill': skill_config}
    
    results = []
    for i in range(count):
        logger.debug(f"    Loop iteration {i+1}/{count}")
        result = await _execute_single(skill_config, registry, outputs)
        results.append(result)
        
        # Stop on failure if stop_on_error is True
        if not result.get('success') and step.get('stop_on_error', True):
            break
    
    return {
        'success': all(r.get('success', False) for r in results),
        'results': results,
        'iterations': len(results)
    }


def _resolve_params(params: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve parameter references to previous outputs."""
    resolved = {}
    for key, value in params.items():
        if isinstance(value, str) and value.startswith('${'):
            # Reference to previous output: ${step_name} or ${step_name.field} or ${step_name.output.field}
            ref = value[2:-1]  # Remove ${}
            
            # Handle nested paths: step_name.output.field or step_name.field
            parts = ref.split('.')
            if len(parts) == 1:
                # ${step_name} - return entire output
                resolved[key] = outputs.get(ref, value)
            elif len(parts) == 2:
                # ${step_name.field} - get field from output
                step_name, field = parts
                if step_name in outputs:
                    output = outputs[step_name]
                    if isinstance(output, dict):
                        resolved[key] = output.get(field, value)
                    else:
                        resolved[key] = value
                else:
                    resolved[key] = value
            elif len(parts) == 3 and parts[1] == 'output':
                # ${step_name.output.field} - get field from output dict
                step_name, _, field = parts
                if step_name in outputs:
                    output = outputs[step_name]
                    if isinstance(output, dict) and field in output:
                        resolved[key] = output[field]
                    else:
                        resolved[key] = value
                else:
                    resolved[key] = value
            else:
                resolved[key] = value
        elif isinstance(value, dict):
            resolved[key] = _resolve_params(value, outputs)
        elif isinstance(value, list):
            resolved[key] = [
                _resolve_params(item, outputs) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            resolved[key] = value
    return resolved


def _evaluate_condition(condition: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
    """Evaluate a condition based on outputs."""
    condition_type = condition.get('type', 'exists')
    
    if condition_type == 'exists':
        key = condition.get('key')
        return key in outputs and outputs[key] is not None
    
    elif condition_type == 'equals':
        key = condition.get('key')
        value = condition.get('value')
        return outputs.get(key) == value
    
    elif condition_type == 'contains':
        key = condition.get('key')
        value = condition.get('value')
        output_value = outputs.get(key, '')
        return value in str(output_value)
    
    elif condition_type == 'greater_than':
        key = condition.get('key')
        value = condition.get('value')
        return outputs.get(key, 0) > value
    
    else:
        return False
