"""
ToolShed - Agentic Tool Discovery and Management
=================================================

 NO HARDCODING - All tool mappings are discovered agentically.
 SWARM INTELLIGENCE - Internal micro-agents handle tool selection.

Key Features:
- Automatic tool discovery from metadata providers
- LLM-based tool selection (no regex/keyword matching)
- Capability index (type/schema → producer)
- Tool I/O schema tracking
- Caching to prevent redundant tool calls

Research Foundations:
- GRF MARL: Role assignment and credit (Song et al., 2023)
- No hardcoded role → tool mappings
"""

import logging
import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Type, Tuple
import time

logger = logging.getLogger(__name__)

# Try DSPy for agentic selection
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False


# =============================================================================
# TOOL SCHEMA
# =============================================================================

@dataclass
class ToolShedSchema:
    """
    Schema describing a tool's inputs and outputs.
    
     A-TEAM: Enables intelligent tool selection and chaining.
    """
    name: str
    description: str
    
    # Input parameters
    input_params: Dict[str, Type] = field(default_factory=dict)
    required_params: List[str] = field(default_factory=list)
    
    # Output schema
    output_type: Optional[Type] = None
    output_fields: Dict[str, Type] = field(default_factory=dict)
    
    # Discovery metadata
    producer_of: List[str] = field(default_factory=list)  # What this tool produces
    consumer_of: List[str] = field(default_factory=list)  # What this tool needs
    
    # Usage statistics
    call_count: int = 0
    success_rate: float = 1.0
    avg_latency: float = 0.0
    
    def to_prompt_string(self) -> str:
        """Convert to LLM-readable description."""
        params_str = ", ".join([
            f"{k}: {v.__name__ if hasattr(v, '__name__') else str(v)}"
            for k, v in self.input_params.items()
        ])
        output_str = str(self.output_type.__name__) if self.output_type and hasattr(self.output_type, '__name__') else "Any"
        
        return (
            f"TOOL: {self.name}\n"
            f"  Description: {self.description}\n"
            f"  Parameters: ({params_str})\n"
            f"  Returns: {output_str}\n"
            f"  Produces: {', '.join(self.producer_of) or 'N/A'}\n"
            f"  Needs: {', '.join(self.consumer_of) or 'nothing'}"
        )


@dataclass
class ToolResult:
    """Result from a tool call with metadata."""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    latency: float = 0.0
    cached: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tool': self.tool_name,
            'success': self.success,
            'result': str(self.result)[:200],
            'error': self.error,
            'latency': self.latency,
            'cached': self.cached,
        }


# =============================================================================
# AGENTIC TOOL SELECTOR (LLM-based, no regex/keywords)
# =============================================================================

if DSPY_AVAILABLE:
    class ToolSelectionSignature(dspy.Signature):
        """
        Given a task and available tools, select the best tool(s).
        
        NO HARDCODED MAPPINGS - Pure LLM reasoning.
        """
        task_description = dspy.InputField(desc="What needs to be done")
        required_output = dspy.InputField(desc="What type of output is needed (e.g., 'list of tables', 'SQL query', 'DataFrame')")
        available_tools = dspy.InputField(desc="List of available tools with their descriptions")
        current_context = dspy.InputField(desc="What data/outputs are already available")
        
        selected_tools = dspy.OutputField(desc="Comma-separated list of tool names to use, in order")
        reasoning = dspy.OutputField(desc="Why these tools were selected")


class AgenticToolSelector:
    """
     NO HARDCODING - LLM selects tools based on task.
    
    This replaces regex/keyword matching with pure reasoning.
    """
    
    def __init__(self) -> None:
        if DSPY_AVAILABLE:
            self.selector = dspy.ChainOfThought(ToolSelectionSignature)
        else:
            self.selector = None
        logger.info(" AgenticToolSelector initialized (pure LLM, no regex)")
    
    def select_tools(
        self,
        task: str,
        required_output: str,
        available_tools: List[ToolShedSchema],
        current_context: Dict[str, Any]
    ) -> List[str]:
        """
        Select tools agentically (no hardcoding).
        
        Returns list of tool names in execution order.
        """
        if not self.selector:
            # Fallback: return all tools
            return [t.name for t in available_tools]
        
        try:
            # Format tools for LLM
            tools_str = "\n\n".join([t.to_prompt_string() for t in available_tools])
            
            # Format context
            context_str = ", ".join([
                f"{k}: {type(v).__name__}" for k, v in current_context.items()
            ])
            
            # LLM selection
            result = self.selector(
                task_description=task,
                required_output=required_output,
                available_tools=tools_str,
                current_context=context_str or "empty"
            )
            
            # Parse selected tools
            selected = [t.strip() for t in result.selected_tools.split(",")]
            
            # Validate against available
            valid_names = {t.name for t in available_tools}
            selected = [t for t in selected if t in valid_names]
            
            logger.debug(f" Selected tools: {selected}")
            logger.debug(f"   Reasoning: {result.reasoning}")
            
            return selected
            
        except Exception as e:
            logger.warning(f" Agentic selection failed: {e}, returning all tools")
            return [t.name for t in available_tools]


# =============================================================================
# CAPABILITY INDEX
# =============================================================================

class CapabilityIndex:
    """
     A-TEAM: Maps (output_type, schema) → producers.
    
    Enables:
    - "I need a DataFrame with columns [a,b,c]" → "Use SQLGenerator"
    - Automatic tool chaining based on I/O compatibility
    """
    
    def __init__(self) -> None:
        # type/field → list of producers
        self.producers: Dict[str, List[str]] = {}
        # producer → list of what it produces
        self.produces: Dict[str, List[str]] = {}
        # producer → list of what it consumes
        self.consumes: Dict[str, List[str]] = {}
        
        logger.info(" CapabilityIndex initialized")
    
    def register_tool(self, schema: ToolShedSchema) -> None:
        """Register a tool's capabilities."""
        # What it produces
        for output in schema.producer_of:
            if output not in self.producers:
                self.producers[output] = []
            if schema.name not in self.producers[output]:
                self.producers[output].append(schema.name)
        
        self.produces[schema.name] = schema.producer_of.copy()
        self.consumes[schema.name] = schema.consumer_of.copy()
        
        logger.debug(f" Registered {schema.name}: produces={schema.producer_of}, consumes={schema.consumer_of}")
    
    def find_producers(self, output_type: str) -> List[str]:
        """Find tools that can produce a given output type."""
        return self.producers.get(output_type, [])
    
    def can_chain(self, producer: str, consumer: str) -> bool:
        """Check if producer's output can feed consumer's input."""
        produced = set(self.produces.get(producer, []))
        needed = set(self.consumes.get(consumer, []))
        return bool(produced & needed)
    
    def find_chain(self, start: str, end: str, max_depth: int = 5) -> List[str]:
        """
        Find a chain of tools from start capability to end capability.
        
        Returns list of tool names, or empty if no chain found.
        """
        # BFS to find shortest path
        from collections import deque
        
        queue = deque([(self.find_producers(start), [start])])
        visited = set()
        
        while queue and len(visited) < max_depth * 10:
            producers, path = queue.popleft()
            
            for producer in producers:
                if producer in visited:
                    continue
                visited.add(producer)
                
                new_path = path + [producer]
                
                # Check if this producer can produce the end
                if end in self.produces.get(producer, []):
                    return new_path[1:]  # Skip start capability
                
                # Add downstream capabilities
                for produced in self.produces.get(producer, []):
                    if produced not in visited:
                        downstream = self.find_producers(produced)
                        if downstream:
                            queue.append((downstream, new_path + [produced]))
        
        return []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'producers': self.producers,
            'produces': self.produces,
            'consumes': self.consumes,
        }


# =============================================================================
# TOOL SHED (Main Class)
# =============================================================================

class ToolShed:
    """
    Central repository for tools with agentic discovery.
    
     NO HARDCODING - All mappings are discovered.
     SWARM INTELLIGENCE - LLM selects appropriate tools.
    
    Features:
    - Automatic schema extraction from callables
    - Capability-based tool discovery
    - LLM-based tool selection
    - Call caching to prevent redundant calls
    - Usage statistics for learning
    """
    
    def __init__(self) -> None:
        self.tools: Dict[str, Callable] = {}
        self.schemas: Dict[str, ToolShedSchema] = {}
        self.capability_index = CapabilityIndex()
        self.selector = AgenticToolSelector()
        
        # Call cache (prevents redundant tool calls)
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.cache_ttl: float = 300.0  # 5 minutes
        
        # Usage statistics
        self.call_stats: Dict[str, Dict[str, Any]] = {}
        
        logger.info(" ToolShed initialized (agentic discovery, no hardcoding)")
    
    def register(self, tool: Callable, name: Optional[str] = None, description: Optional[str] = None, produces: Optional[List[str]] = None, consumes: Optional[List[str]] = None) -> Any:
        """
        Register a tool with automatic schema extraction.
        
        Parameters:
        -----------
        tool : Callable
            The function/method to register.
            
        name : str, optional
            Override the tool's name (default: function name).
            
        description : str, optional
            Override description (default: docstring).
            
        produces : List[str], optional
            What this tool produces (auto-inferred if not provided).
            
        consumes : List[str], optional
            What this tool needs (auto-inferred from parameters).
        """
        tool_name = name or tool.__name__
        
        # Extract schema automatically
        schema = self._extract_schema(
            tool=tool,
            name=tool_name,
            description=description,
            produces=produces,
            consumes=consumes,
        )
        
        self.tools[tool_name] = tool
        self.schemas[tool_name] = schema
        self.capability_index.register_tool(schema)
        
        logger.debug(f" Registered tool: {tool_name}")
    
    def _extract_schema(
        self,
        tool: Callable,
        name: str,
        description: Optional[str] = None,
        produces: Optional[List[str]] = None,
        consumes: Optional[List[str]] = None,
    ) -> ToolShedSchema:
        """Extract schema from callable."""
        # Get signature
        sig = inspect.signature(tool)
        
        # Extract input params
        input_params = {}
        required_params = []
        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'cls', 'kwargs', 'args'):
                continue
            
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any
            input_params[param_name] = param_type
            
            if param.default == inspect.Parameter.empty:
                required_params.append(param_name)
        
        # Get return type
        return_type = sig.return_annotation if sig.return_annotation != inspect.Signature.empty else None
        
        # Get description from docstring
        desc = description or (tool.__doc__ or "").split("\n")[0].strip() or f"Tool: {name}"
        
        # Infer produces from return type
        if produces is None:
            produces = []
            if return_type:
                produces.append(str(return_type.__name__) if hasattr(return_type, '__name__') else str(return_type))
        
        # Infer consumes from required params
        if consumes is None:
            consumes = required_params.copy()
        
        return ToolShedSchema(
            name=name,
            description=desc,
            input_params=input_params,
            required_params=required_params,
            output_type=return_type,
            producer_of=produces,
            consumer_of=consumes,
        )
    
    def select_tools(
        self,
        task: str,
        required_output: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Select tools for a task (agentically, no hardcoding).
        
        Uses LLM to reason about which tools to use.
        """
        schemas = list(self.schemas.values())
        return self.selector.select_tools(
            task=task,
            required_output=required_output,
            available_tools=schemas,
            current_context=context or {},
        )
    
    def call(self, tool_name: str, use_cache: bool = True, **kwargs: Any) -> ToolResult:
        """
        Call a tool with caching and statistics.
        
        Parameters:
        -----------
        tool_name : str
            Name of tool to call.
            
        use_cache : bool, default=True
            Use cached result if available.
            
        **kwargs : Any
            Parameters to pass to tool.
        
        Returns:
        --------
        ToolResult with success, result, error, timing.
        """
        if tool_name not in self.tools:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Tool '{tool_name}' not found",
            )
        
        # Check cache
        cache_key = f"{tool_name}:{hash(frozenset(kwargs.items()))}"
        if use_cache and cache_key in self.cache:
            cached_result, cached_time = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                logger.debug(f" Cache hit: {tool_name}")
                return ToolResult(
                    tool_name=tool_name,
                    success=True,
                    result=cached_result,
                    cached=True,
                )
        
        # Call tool
        start_time = time.time()
        try:
            result = self.tools[tool_name](**kwargs)
            latency = time.time() - start_time
            
            # Update cache
            self.cache[cache_key] = (result, time.time())
            
            # Update stats
            self._update_stats(tool_name, success=True, latency=latency)
            
            return ToolResult(
                tool_name=tool_name,
                success=True,
                result=result,
                latency=latency,
            )
            
        except Exception as e:
            latency = time.time() - start_time
            self._update_stats(tool_name, success=False, latency=latency)
            
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
                latency=latency,
            )
    
    def _update_stats(self, tool_name: str, success: bool, latency: float) -> Any:
        """Update usage statistics."""
        if tool_name not in self.call_stats:
            self.call_stats[tool_name] = {
                'calls': 0,
                'successes': 0,
                'total_latency': 0.0,
            }
        
        stats = self.call_stats[tool_name]
        stats['calls'] += 1
        stats['successes'] += 1 if success else 0
        stats['total_latency'] += latency
        
        # Update schema
        if tool_name in self.schemas:
            schema = self.schemas[tool_name]
            schema.call_count = stats['calls']
            schema.success_rate = stats['successes'] / stats['calls']
            schema.avg_latency = stats['total_latency'] / stats['calls']
    
    def get_tools_for_agent(self, agent_name: str, signature: Any = None) -> List[Callable]:
        """
        Get tools for an agent based on its signature.
        
         NO HARDCODING - Uses capability matching.
        """
        if signature is None:
            return list(self.tools.values())
        
        # Extract required inputs from signature
        required = []
        if hasattr(signature, 'input_fields'):
            required = list(signature.input_fields.keys())
        elif hasattr(signature, '__annotations__'):
            required = list(signature.__annotations__.keys())
        
        # Find tools that produce what agent needs
        matching_tools = []
        for req in required:
            producers = self.capability_index.find_producers(req)
            for producer in producers:
                if producer in self.tools:
                    matching_tools.append(self.tools[producer])
        
        return list(set(matching_tools)) or list(self.tools.values())
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        self.cache.clear()
        logger.info(" ToolShed cache cleared")
    
    def get_all_schemas(self) -> List[ToolShedSchema]:
        """Get all tool schemas."""
        return list(self.schemas.values())
    
    def to_prompt_string(self) -> str:
        """Format all tools for LLM prompt."""
        return "\n\n".join([
            schema.to_prompt_string() 
            for schema in self.schemas.values()
        ])


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ToolShed',
    'ToolShedSchema',
    'ToolResult',
    'CapabilityIndex',
    'AgenticToolSelector',
]

