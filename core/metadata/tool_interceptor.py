"""
Generic Tool Interceptor for DSPy ReAct Agents

This module provides infrastructure to intercept tool calls made by DSPy ReAct agents,
allowing ReVal to track attempts, log executions, and extract structured outputs.

Design Philosophy:
- Generic: Works with ANY DSPy ReAct agent, not just SQLGenerator
- Transparent: Actors don't need to know they're being intercepted
- Flexible: Supports custom tagging logic per tool type
"""

import logging
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Record of a single tool invocation."""
    tool_name: str
    args: Dict[str, Any]
    result: Any
    success: bool
    error: Optional[str] = None
    attempt_number: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolInterceptor:
    """
    Wraps DSPy tools to intercept and log all calls.
    
    Usage in ReVal:
        # When setting up actor with tools
        interceptor = ToolInterceptor(actor_name="SQLGenerator")
        wrapped_tools = interceptor.wrap_tools(original_tools)
        
        # Pass wrapped_tools to actor
        actor.module.tools = wrapped_tools
        
        # After actor execution, retrieve calls
        all_calls = interceptor.get_all_calls()
    """
    
    def __init__(self, actor_name: str) -> None:
        """
        Initialize tool interceptor for an actor.
        
        Args:
            actor_name: Name of the actor (for logging/tracking)
        """
        self.actor_name = actor_name
        self._calls: List[ToolCall] = []
        self._attempt_counters: Dict[str, int] = {}  # Track attempts per tool
        self._lock = Lock()  # Thread-safe
        
        logger.info(f" [INTERCEPTOR] Initialized for actor '{actor_name}'")
    
    def wrap_tools(self, tools: Dict[str, Callable]) -> Dict[str, Callable]:
        """
        Wrap all tools in a dict with interception logic.
        
        Args:
            tools: Dict of {tool_name: tool_function}
            
        Returns:
            Dict of {tool_name: wrapped_function}
        """
        wrapped = {}
        
        for tool_name, tool_func in tools.items():
            wrapped[tool_name] = self._create_wrapper(tool_name, tool_func)
            logger.debug(f" [INTERCEPTOR] Wrapped tool '{tool_name}' for {self.actor_name}")
        
        logger.info(f" [INTERCEPTOR] Wrapped {len(wrapped)} tools for {self.actor_name}")
        return wrapped
    
    def _create_wrapper(self, tool_name: str, tool_func: Callable) -> Callable:
        """
        Create a wrapper function that intercepts calls to a tool.
        
        Args:
            tool_name: Name of the tool
            tool_func: Original tool function
            
        Returns:
            Wrapped function with interception
        """
        def wrapped_tool(**kwargs: Any) -> Any:
            """Wrapper that logs call and delegates to original."""
            with self._lock:
                # Increment attempt counter
                self._attempt_counters[tool_name] = self._attempt_counters.get(tool_name, 0) + 1
                attempt_num = self._attempt_counters[tool_name]
            
            logger.info(f" [INTERCEPTOR] Tool call #{attempt_num}: {tool_name}({list(kwargs.keys())})")
            
            # Call original tool
            success = False
            result = None
            error = None
            
            try:
                result = tool_func(**kwargs)
                success = True
                logger.info(f" [INTERCEPTOR] Tool '{tool_name}' succeeded (attempt #{attempt_num})")
            except Exception as e:
                success = False
                error = str(e)
                result = f"Error: {error}"
                logger.error(f" [INTERCEPTOR] Tool '{tool_name}' failed: {error}")
            
            # Record the call
            call_record = ToolCall(
                tool_name=tool_name,
                args=kwargs,
                result=result,
                success=success,
                error=error,
                attempt_number=attempt_num,
                metadata={
                    'actor': self.actor_name
                }
            )
            
            with self._lock:
                self._calls.append(call_record)
            
            # Return result (or raise exception to match original behavior)
            if not success and error:
                # Re-raise for DSPy to handle retry logic
                raise Exception(error)
            
            return result
        
        # Preserve function name and docstring
        wrapped_tool.__name__ = tool_func.__name__
        wrapped_tool.__doc__ = tool_func.__doc__
        
        return wrapped_tool
    
    def get_all_calls(self) -> List[ToolCall]:
        """
        Get all tool calls recorded by this interceptor.
        
        Returns:
            List of ToolCall objects in chronological order
        """
        with self._lock:
            return list(self._calls)
    
    def get_calls_for_tool(self, tool_name: str) -> List[ToolCall]:
        """
        Get all calls for a specific tool.
        
        Args:
            tool_name: Name of the tool to filter by
            
        Returns:
            List of ToolCall objects for that tool
        """
        with self._lock:
            return [call for call in self._calls if call.tool_name == tool_name]
    
    def get_successful_calls(self) -> List[ToolCall]:
        """Get only successful tool calls."""
        with self._lock:
            return [call for call in self._calls if call.success]
    
    def get_failed_calls(self) -> List[ToolCall]:
        """Get only failed tool calls."""
        with self._lock:
            return [call for call in self._calls if not call.success]
    
    def clear(self) -> None:
        """Clear all recorded calls (useful for multi-episode scenarios)."""
        with self._lock:
            self._calls.clear()
            self._attempt_counters.clear()
        logger.debug(f" [INTERCEPTOR] Cleared all calls for {self.actor_name}")
    
    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of tool usage.
        
        Returns:
            Dict with statistics about tool calls
        """
        with self._lock:
            total = len(self._calls)
            successful = sum(1 for call in self._calls if call.success)
            failed = total - successful
            
            by_tool = {}
            for call in self._calls:
                if call.tool_name not in by_tool:
                    by_tool[call.tool_name] = {'total': 0, 'successful': 0, 'failed': 0}
                by_tool[call.tool_name]['total'] += 1
                if call.success:
                    by_tool[call.tool_name]['successful'] += 1
                else:
                    by_tool[call.tool_name]['failed'] += 1
            
            return {
                'actor': self.actor_name,
                'total_calls': total,
                'successful': successful,
                'failed': failed,
                'by_tool': by_tool
            }
    
    def to_tagged_attempts(self, tool_name: str = 'execute_query') -> List[Any]:
        """
        Convert tool calls to TaggedAttempt objects (for SQL use case).
        
        This is a convenience method for the common pattern of tagging SQL attempts.
        
        Args:
            tool_name: Name of tool to extract (default: 'execute_query')
            
        Returns:
            List of TaggedAttempt objects
        """
        from jotty.data_structures import TaggedAttempt
        
        attempts = []
        tool_calls = self.get_calls_for_tool(tool_name)
        
        for call in tool_calls:
            # Extract SQL query from args
            if 'query' in call.args:
                sql_query = call.args['query']
            else:
                # Fallback: stringify all args
                sql_query = str(call.args)
            
            # Determine tag based on success
            if call.success:
                # Check if result indicates actual success or just no error
                result_str = str(call.result).lower()
                if 'result' in result_str or '=' in result_str:
                    tag = 'correct'
                else:
                    tag = 'exploratory'
            else:
                tag = 'wrong'
            
            # Create TaggedAttempt
            attempts.append(TaggedAttempt(
                output=sql_query,
                tag=tag,
                execution_status='success' if call.success else 'failed',
                execution_result=str(call.result),
                reasoning=f"Intercepted tool call #{call.attempt_number}",
                attempt_number=call.attempt_number
            ))
        
        logger.info(f" [INTERCEPTOR] Converted {len(attempts)} tool calls to TaggedAttempts")
        return attempts


class ToolCallRegistry:
    """
    Global registry for all tool interceptors in a ReVal swarm.
    
    This allows us to track tool calls across all actors and aggregate results.
    """
    
    def __init__(self) -> None:
        self._interceptors: Dict[str, ToolInterceptor] = {}
        self._lock = Lock()
        logger.info(" [REGISTRY] ToolCallRegistry initialized")
    
    def get_or_create_interceptor(self, actor_name: str) -> ToolInterceptor:
        """
        Get existing interceptor for actor or create new one.
        
        Args:
            actor_name: Name of the actor
            
        Returns:
            ToolInterceptor instance
        """
        with self._lock:
            if actor_name not in self._interceptors:
                self._interceptors[actor_name] = ToolInterceptor(actor_name)
                logger.info(f" [REGISTRY] Created interceptor for '{actor_name}'")
            return self._interceptors[actor_name]
    
    def get_all_calls(self) -> List[ToolCall]:
        """Get all tool calls from all actors."""
        with self._lock:
            all_calls = []
            for interceptor in self._interceptors.values():
                all_calls.extend(interceptor.get_all_calls())
            return all_calls
    
    def summary(self) -> Dict[str, Any]:
        """Get aggregate summary across all actors."""
        with self._lock:
            actor_summaries = {
                name: interceptor.summary()
                for name, interceptor in self._interceptors.items()
            }
            
            total_calls = sum(s['total_calls'] for s in actor_summaries.values())
            total_successful = sum(s['successful'] for s in actor_summaries.values())
            total_failed = sum(s['failed'] for s in actor_summaries.values())
            
            return {
                'total_calls': total_calls,
                'successful': total_successful,
                'failed': total_failed,
                'by_actor': actor_summaries
            }
    
    def clear_all(self) -> None:
        """Clear all interceptors."""
        with self._lock:
            for interceptor in self._interceptors.values():
                interceptor.clear()
        logger.info(" [REGISTRY] Cleared all interceptors")

