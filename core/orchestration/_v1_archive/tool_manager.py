"""
Tool Manager - Extracted from Conductor
========================================

Handles tool creation, caching, and injection for actors:
- Auto-discovery of DSPy tools
- Tool filtering for Architect/Auditor
- Tool caching
- Error handling and descriptions

JOTTY Framework Enhancement - Fix #1 (Part 2/3)
Extracted from 4,708-line Conductor to improve maintainability.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

# Conditional imports for type hints
if TYPE_CHECKING:
    from ..foundation.agent_config import AgentConfig
    from .conductor import TodoItem
    ActorConfig = AgentConfig  # Alias for backward compatibility
else:
    ActorConfig = Any  # Runtime fallback
    TodoItem = Any  # Runtime fallback

logger = logging.getLogger(__name__)


class ToolManager:
    """
    Manages tools for actors (Architect/Auditor/Main).

    Extracted from Conductor to follow Single Responsibility Principle.

    Responsibilities:
    - Auto-discover tools from metadata provider
    - Filter tools for Architect vs Auditor
    - Cache tool calls
    - Build tool descriptions and error messages
    """

    def __init__(
        self,
        metadata_tool_registry,
        data_registry_tool,
        metadata_fetcher,
        config
    ):
        """
        Initialize ToolManager with dependencies.

        Args:
            metadata_tool_registry: MetadataToolRegistry for tool discovery
            data_registry_tool: DataRegistryTool for data discovery
            metadata_fetcher: MetaDataFetcher for metadata access
            config: JottyConfig/SwarmConfig instance
        """
        self.metadata_tool_registry = metadata_tool_registry
        self.data_registry_tool = data_registry_tool
        self.metadata_fetcher = metadata_fetcher
        self.config = config
        self._tool_cache = {}  # Cache for tool call results

        logger.info("✅ ToolManager initialized")

    # =========================================================================
    # EXTRACTED METHODS FROM CONDUCTOR
    # =========================================================================

    def _get_auto_discovered_dspy_tools(self) -> List[Any]:
        """
        Get ALL auto-discovered metadata tools as dspy.Tool objects.
        
        🔥 A-TEAM CONSENSUS SOLUTION (Post-Debate):
        - Individual DSPy tools (one per metadata method) ✅
        - Smart parameter resolution (4-level fallback) ✅
        - Caching via SharedScratchpad ✅
        - Enhanced descriptions for LLM reasoning ✅
        - FULLY GENERIC - works for ANY tool! ✅
        
        Returns:
            List of dspy.Tool objects for all @jotty_method decorated methods
        """
        import dspy
        import json
        
        if not hasattr(self, 'metadata_tool_registry'):
            logger.warning("⚠️  No metadata_tool_registry found, returning empty tool list")
            return []
        
        tools = []
        tool_names = self.metadata_tool_registry.list_tools()
        
        logger.info(f"🔧 Creating {len(tool_names)} individual DSPy tools with smart param resolution...")
        
        for tool_name in tool_names:
            tool_info = self.metadata_tool_registry.get_tool_info(tool_name)
            
            # Create closure to capture tool_name and tool_info
            def make_smart_tool_func(tname, tinfo):
                def smart_tool_func(*args, **kwargs):
                    """
                    🔥 A-TEAM SMART WRAPPER with 4-level parameter resolution:
                    1. Explicit override (user provides param)
                    2. IOManager auto-resolution (exact name match from actor outputs)
                    3. SharedContext auto-resolution (exact name match from global data)
                    4. Type-based resolution (generic! works for ANY type)
                    """
                    try:
                        # Get parameter specs
                        signature = tinfo.get('signature', {})
                        params_spec = signature.get('parameters', {})
                        
                        # Build final kwargs with smart resolution
                        final_kwargs = dict(kwargs)
                        
                        for param_name, param_info in params_spec.items():
                            if param_name in final_kwargs:
                                # Level 1: Explicit override (highest priority)
                                logger.debug(f"✅ {tname}({param_name}): Using explicit value")
                                continue
                            
                            # Level 2: Exact match in IOManager (actor outputs)
                            resolved_val = self._resolve_param_from_iomanager(param_name)
                            if resolved_val is not None:
                                final_kwargs[param_name] = resolved_val
                                logger.info(f"✅ {tname}({param_name}): Auto-resolved from IOManager")
                                continue
                            
                            # Level 3: Exact match in SharedContext (global data)
                            if hasattr(self, 'shared_context') and self.shared_context.has(param_name):
                                resolved_val = self.shared_context.get(param_name)
                                final_kwargs[param_name] = resolved_val
                                logger.info(f"✅ {tname}({param_name}): Auto-resolved from SharedContext")
                                continue
                            
                            # Level 4: Type-based resolution (GENERIC!)
                            param_type = param_info.get('annotation', '')
                            if param_type:
                                resolved_val = self._resolve_param_by_type(param_name, param_type)
                                if resolved_val is not None:
                                    final_kwargs[param_name] = resolved_val
                                    logger.info(f"✅ {tname}({param_name}): Auto-resolved by type ({param_type})")
                                    continue
                            
                            # Not resolved - let tool fail with clear error
                            if param_info.get('required', True):
                                logger.warning(f"⚠️  {tname}({param_name}): Required parameter not resolved")
                        
                        # Check cache first (via SharedScratchpad)
                        result = self._call_tool_with_cache(tname, **final_kwargs)
                        
                        # Ensure result is string (DSPy tools return strings)
                        if not isinstance(result, str):
                            return json.dumps(result, indent=2)
                        return result
                        
                    except Exception as e:
                        error_msg = self._build_helpful_error_message(tname, tinfo, e)
                        logger.error(f"❌ Tool {tname} error: {e}")
                        return json.dumps({"error": error_msg})
                
                return smart_tool_func
            
            # Build enhanced tool description for LLM reasoning
            tool_desc = self._build_enhanced_tool_description(tool_name, tool_info)
            
            # Create dspy.Tool with smart wrapper
            tool = dspy.Tool(
                func=make_smart_tool_func(tool_name, tool_info),
                name=tool_name,
                desc=tool_desc
            )
            
            # 🔥 A-TEAM CRITICAL FIX: Attach Val agent flags to dspy.Tool!
            tool._jotty_for_architect = tool_info.get('for_architect', False)
            tool._jotty_for_auditor = tool_info.get('for_auditor', False)
            
            tools.append(tool)
            logger.debug(f"  ✅ {tool_name} (architect={tool._jotty_for_architect}, auditor={tool._jotty_for_auditor})")
        
        logger.info(f"✅ Auto-discovered {len(tools)} tools for Val agents")
        for tool in tools:
            logger.info(f"   - {tool.name}: {tool.desc[:80]}...")
        
        return tools
    
    def _resolve_param_from_iomanager(self, param_name: str) -> Any:
        """
        Resolve parameter from IOManager (previous actor outputs).
        
        🔥 A-TEAM: Level 2 resolution - searches actor outputs for param
        
        Args:
            param_name: Name of parameter to resolve (e.g., 'tables')
        
        Returns:
            Resolved value or None if not found
        """
        if not hasattr(self, 'io_manager') or not self.io_manager:
            return None
        
        # Try exact name match in all actor outputs
        all_outputs = self.io_manager.get_all_outputs()
        for actor_name, output in all_outputs.items():
            if hasattr(output, 'output_fields') and isinstance(output.output_fields, dict):
                if param_name in output.output_fields:
                    value = output.output_fields[param_name]
                    logger.debug(f"   📦 Found '{param_name}' in {actor_name} output")
                    return value
        
        return None
    
    def _resolve_param_by_type(self, param_name: str, param_type: Any) -> Any:
        """
        Resolve parameter by type matching in IOManager.
        
        🔥 A-TEAM: Level 4 resolution - GENERIC type-based matching!
        Works for ANY type, not just hardcoded names.
        
        Args:
            param_name: Name of parameter (for logging)
            param_type: Type annotation (can be type object or string)
        
        Returns:
            Resolved value or None if not found
        """
        if not hasattr(self, 'io_manager') or not self.io_manager:
            return None
        
        # 🔥 A-TEAM FIX: Convert type annotation to string for comparison
        # Handle both type objects and string annotations
        type_str = str(param_type) if not isinstance(param_type, str) else param_type
        
        # Parse type (simplified - handles common cases)
        target_type = None
        if 'List' in type_str or 'list' in type_str:
            target_type = list
        elif 'Dict' in type_str or 'dict' in type_str:
            target_type = dict
        elif type_str in ['str', 'string'] or 'str' in type_str:
            target_type = str
        elif type_str in ['int', 'integer'] or 'int' in type_str:
            target_type = int
        elif type_str in ['float'] or 'float' in type_str:
            target_type = float
        elif type_str in ['bool', 'boolean'] or 'bool' in type_str:
            target_type = bool
        
        if not target_type:
            return None
        
        # Search all actor outputs for matching type
        all_outputs = self.io_manager.get_all_outputs()
        for actor_name, output in all_outputs.items():
            if hasattr(output, 'output_fields') and isinstance(output.output_fields, dict):
                for field_name, field_value in output.output_fields.items():
                    if isinstance(field_value, target_type):
                        logger.debug(f"   🎯 Type match for '{param_name}': found {field_name} ({type(field_value).__name__}) in {actor_name}")
                        return field_value
        
        return None
    
    def _call_tool_with_cache(self, tool_name: str, **kwargs) -> Any:
        """
        Call tool with caching via SharedScratchpad.
        
        🔥 A-TEAM: Prevents duplicate tool calls across Val agents!
        
        Args:
            tool_name: Name of tool to call
            **kwargs: Parameters for tool
        
        Returns:
            Tool result (cached if available)
        """
        import json
        
        # Create cache key
        cache_key = f"tool_call:{tool_name}:{json.dumps(kwargs, sort_keys=True)}"
        
        # Check cache
        if hasattr(self, 'shared_scratchpad') and self.shared_scratchpad:
            if cache_key in self.shared_scratchpad:
                logger.debug(f"💾 Cache HIT: {tool_name}({list(kwargs.keys())})")
                return self.shared_scratchpad[cache_key]
        
        # Call actual tool
        logger.debug(f"📞 Calling {tool_name}({list(kwargs.keys())})")
        result = self.metadata_tool_registry.call_tool(tool_name, **kwargs)
        
        # Store in cache
        if hasattr(self, 'shared_scratchpad') and self.shared_scratchpad:
            self.shared_scratchpad[cache_key] = result
        
        return result
    
    def _build_helpful_error_message(self, tool_name: str, tool_info: Dict, error: Exception) -> str:
        """
        Build helpful error message for Val agents when tool call fails.
        
        🔥 A-TEAM: Shows available data and how to fix the issue!
        
        Args:
            tool_name: Name of tool that failed
            tool_info: Tool metadata
            error: Exception that occurred
        
        Returns:
            Helpful error message string
        """
        import json
        
        # Extract missing parameters if it's a TypeError
        error_str = str(error)
        
        # Build helpful message
        msg_parts = [f"❌ {tool_name}() failed: {error_str}"]
        
        # If missing parameters, show available data
        if 'missing' in error_str.lower() or 'required' in error_str.lower():
            msg_parts.append("\n📦 AVAILABLE DATA (IOManager):")
            
            if hasattr(self, 'io_manager') and self.io_manager:
                all_outputs = self.io_manager.get_all_outputs()
                for actor_name, output in all_outputs.items():
                    if hasattr(output, 'output_fields'):
                        fields = list(output.output_fields.keys()) if isinstance(output.output_fields, dict) else []
                        msg_parts.append(f"  • {actor_name}: {fields}")
            else:
                msg_parts.append("  (No IOManager available)")
            
            msg_parts.append("\n💡 TIP: You can provide parameters explicitly in your tool call.")
            msg_parts.append(f"   Example: {tool_name}(param_name=value)")
        
        return "\n".join(msg_parts)
    
    def _build_enhanced_tool_description(self, tool_name: str, tool_info: Dict) -> str:
        """
        Build enhanced tool description for LLM reasoning.
        
        🔥 A-TEAM: Includes parameters, when to use, auto-resolution hints!
        
        Args:
            tool_name: Name of tool
            tool_info: Tool metadata
        
        Returns:
            Enhanced description string
        """
        desc_parts = [tool_info['desc']]
        
        # Add "when to use"
        desc_parts.append(f"\n🎯 WHEN TO USE:\n{tool_info['when']}")
        
        # Add parameters
        signature = tool_info.get('signature', {})
        params = signature.get('parameters', {})
        
        if params:
            desc_parts.append("\n📋 PARAMETERS:")
            for param_name, param_info in params.items():
                ptype = param_info.get('annotation', 'Any')
                required = 'REQUIRED' if param_info.get('required', True) else 'OPTIONAL'
                pdesc = param_info.get('desc', '')
                default = param_info.get('default')
                
                if default is not None:
                    desc_parts.append(f"  • {param_name} ({ptype}) [{required}]: {pdesc} (default: {default})")
                else:
                    desc_parts.append(f"  • {param_name} ({ptype}) [{required}]: {pdesc}")
        else:
            desc_parts.append("\n📋 PARAMETERS: None (simple getter)")
        
        # Add returns
        returns = tool_info.get('returns', 'str')
        desc_parts.append(f"\n↩️  RETURNS: {returns}")
        
        # Add auto-resolution hint
        if params:
            desc_parts.append("""
🤖 AUTO-RESOLUTION:
Parameters you don't provide will be auto-resolved from:
1. Previous actor outputs (IOManager) - exact name match
2. Shared global context - exact name match
3. Type-based matching - finds any matching type

You can override any parameter by providing it explicitly.""")
        
        return "\n".join(desc_parts)
    
    def _get_architect_tools(self, all_tools: List[Any]) -> List[Any]:
        """
        Filter tools marked for Architect by metadata manager.
        
        🔥 A-TEAM CRITICAL FIX: NO HARDCODING!
        
        This method checks the `for_architect` flag set by @jotty_method decorator.
        The METADATA MANAGER (user-provided, domain-specific) decides which tools
        are for Architect, NOT JOTTY core (which is generic).
        
        This keeps JOTTY domain-agnostic and reusable for ANY use case!
        """
        filtered = []
        for tool in all_tools:
            # 🔥 A-TEAM FIX: Tools are dictionaries from MetadataToolRegistry
            if isinstance(tool, dict) and tool.get('for_architect', False):
                filtered.append(tool)
            # Legacy support for function objects
            elif hasattr(tool, 'func') and hasattr(tool.func, '_jotty_for_architect'):
                if tool.func._jotty_for_architect:
                    filtered.append(tool)
            elif hasattr(tool, '_jotty_for_architect'):
                if tool._jotty_for_architect:
                    filtered.append(tool)
        
        logger.info(f"🔍 Architect tools: {len(filtered)}/{len(all_tools)} tools (marked by metadata manager)")
        for tool in filtered:
            tool_name = tool.get('name') if isinstance(tool, dict) else (tool.name if hasattr(tool, 'name') else tool.__name__)
            logger.debug(f"   ✅ {tool_name}")
        
        return filtered
    
    def _get_auditor_tools(self, all_tools: List[Any]) -> List[Any]:
        """
        Filter tools marked for Auditor by metadata manager.
        
        🔥 A-TEAM CRITICAL FIX: NO HARDCODING!
        
        This method checks the `for_auditor` flag set by @jotty_method decorator.
        The METADATA MANAGER (user-provided, domain-specific) decides which tools
        are for Auditor, NOT JOTTY core (which is generic).
        
        This keeps JOTTY domain-agnostic and reusable for ANY use case!
        """
        filtered = []
        for tool in all_tools:
            # 🔥 A-TEAM FIX: Tools are dictionaries from MetadataToolRegistry
            if isinstance(tool, dict) and tool.get('for_auditor', False):
                filtered.append(tool)
            # Legacy support for function objects
            elif hasattr(tool, 'func') and hasattr(tool.func, '_jotty_for_auditor'):
                if tool.func._jotty_for_auditor:
                    filtered.append(tool)
            elif hasattr(tool, '_jotty_for_auditor'):
                if tool._jotty_for_auditor:
                    filtered.append(tool)
        
        logger.info(f"✅ Auditor tools: {len(filtered)}/{len(all_tools)} tools (marked by metadata manager)")
        for tool in filtered:
            tool_name = tool.get('name') if isinstance(tool, dict) else (tool.name if hasattr(tool, 'name') else tool.__name__)
            logger.debug(f"   ✅ {tool_name}")
        
        return filtered
    
    def _should_inject_registry_tool(self, actor_name: str) -> bool:
        """Check if actor signature requests data_registry."""
        signature = self.actor_signatures.get(actor_name, {})
        return 'data_registry' in signature
    
    async def _run_auditor(
        self,
        actor_config: ActorConfig,
        result: Any,
        task: TodoItem
    ) -> Tuple[bool, float, str]:
        """
        Run Auditor for actor result.
        
        Incorporates:
        - Auditor prompts
        - Annotations
        - Learned patterns
        
        Returns:
            (passed, reward, feedback)
        """
        # For now, simple check - in full impl would use Auditor agents
        # TODO: Integrate full Auditor with prompts and annotations
        
        # Check if result indicates success
