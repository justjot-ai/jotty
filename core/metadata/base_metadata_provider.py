"""
Base Metadata Provider for ReVal
================================

Provides common functionality for MetadataProvider implementations:
- Auto-registration of methods as callable tools
- Tool exposure for agents to discover and use
- Signature introspection and documentation
- NO hardcoding - works for ANY domain

A-Team Approved: Makes agents SMART - they can discover what metadata methods exist!

Usage:
    class MyMetadataProvider(BaseMetadataProvider):
        def get_schema(self, table_name: str) -> Dict:
            '''Get schema for a table.'''  # ← Docstring becomes tool description
            return self.schemas[table_name]
        
        def get_columns(self, table_name: str) -> List[str]:
            '''Get columns for a table.'''
            return self.columns[table_name]
    
    provider = MyMetadataProvider(...)
    tools = provider.get_tools()  # Agents can now discover and call these methods!
"""

import inspect
import functools
import logging
from typing import Dict, Any, List, Callable, Optional

logger = logging.getLogger(__name__)


class BaseMetadataProvider:
    """
    Base class for MetadataProvider implementations.
    
    Provides automatic tool exposure for agents to discover and use metadata methods.
    
    Key Features:
    1. **Auto-registration**: All public methods automatically become callable
    2. **Tool exposure**: `get_tools()` returns list of tools for agents
    3. **Signature preservation**: Tools maintain proper type hints and docstrings
    4. **Discovery**: Agents can list available methods
    5. **NO hardcoding**: Works for SQL, code-gen, marketing, ANY domain
    
    Subclassing:
        1. Inherit from BaseMetadataProvider
        2. Define public methods (not starting with _)
        3. Add docstrings (they become tool descriptions)
        4. Call super().__init__() to auto-register
        5. Use get_tools() to expose to agents
    
    Example:
        ```python
        class SQLMetadataProvider(BaseMetadataProvider):
            def __init__(self, metadata_manager):
                self.metadata_manager = metadata_manager
                super().__init__()  # ← Auto-registers methods!
            
            def get_schema(self, table_name: str) -> Dict:
                '''Get complete schema for a table including columns and types.'''
                return self.metadata_manager.get_schema(table_name)
            
            def get_columns_for_table(self, table_name: str, filter_type: str = None) -> List[str]:
                '''
                Get column names for a specific table.
                
                Args:
                    table_name: Name of the table
                    filter_type: Optional type filter (e.g., "string", "numeric")
                
                Returns:
                    List of column names
                '''
                cols = self.metadata_manager.get_columns(table_name)
                if filter_type:
                    cols = [c for c in cols if c['type'] == filter_type]
                return [c['name'] for c in cols]
            
            def get_partition_info(self, table_names: List[str]) -> Dict[str, Dict]:
                '''Get partition information for multiple tables.'''
                return {t: self.metadata_manager.get_partitions(t) for t in table_names}
        
        # Usage:
        provider = SQLMetadataProvider(metadata_manager)
        metadata_tools = provider.get_tools() # All methods exposed as tools!
        
        ActorConfig(
            name="ColumnSelector",
            actor=cs_agent,
            architect_tools=metadata_tools, # Agent can discover and call!
            ...
        )
        ```
    
    A-Team Consensus:
        This makes agents SMART - they can discover what metadata exists and use it intelligently.
        NO manual wrapping, NO hardcoding, COMPLETE genericity.
    """
    
    def __init__(self, token_budget: int = 50000, enable_caching: bool = True, **kwargs: Any) -> None:
        """
        Initialize base provider with auto-registration.
        
        Args:
            token_budget: Maximum tokens that can be fetched via tools (default: 50k)
            enable_caching: Whether to cache tool results (default: True)
            **kwargs: Additional provider-specific arguments
        
        Call this from subclass __init__() AFTER setting instance attributes.
        """
        self._registered_methods: Dict[str, Callable] = {}
        self._method_descriptions: Dict[str, str] = {}
        self._method_signatures: Dict[str, inspect.Signature] = {}
        
        # A-Team Safeguards: Token budget and caching
        self._token_budget = token_budget
        self._tokens_used = 0
        self._enable_caching = enable_caching
        self._tool_cache: Dict[str, Any] = {}
        self._tool_call_count = 0
        
        # Auto-register all public methods
        self._auto_register_methods()
        
        logger.info(f" {self.__class__.__name__} initialized")
        logger.info(f"   Callable methods: {len(self._registered_methods)}")
        logger.info(f"   Token budget: {self._token_budget:,}")
        logger.info(f"   Caching: {'enabled' if self._enable_caching else 'disabled'}")
    
    def _auto_register_methods(self) -> Any:
        """
        Auto-register all public methods as callable tools.
        
        Registers methods that:
        - Don't start with underscore (_)
        - Are not inherited base methods
        - Are not reserved protocol methods
        """
        reserved_methods = {
            'get_context_for_actor',  # Protocol method
            'get_swarm_context',      # Protocol method
            'get_tools',              # Base class method
            'list_available_methods', # Base class method
            'get_method_info'         # Base class method
        }
        
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            # Skip private, reserved, and base class methods
            if name.startswith('_') or name in reserved_methods:
                continue
            
            # Register method
            self._registered_methods[name] = method
            self._method_descriptions[name] = inspect.getdoc(method) or f"Call {name} on metadata provider"
            self._method_signatures[name] = inspect.signature(method)
            
            logger.debug(f" Registered metadata method: {name}{self._method_signatures[name]}")
    
    def get_tools(self) -> List[Callable]:
        """
        Expose all registered methods as SAFEGUARDED tools for agents.
        
        Returns:
            List of callable tools with:
            - Token budget tracking (prevents context explosion)
            - Automatic caching (prevents redundant fetches)
            - Extreme logging (for debugging)
            - DSPy-compatible attributes (name, description, signature)
        
        A-Team Safeguards:
            1. **Budget**: Tools can't exceed token_budget (default: 50k)
            2. **Cache**: Identical calls return cached result (free!)
            3. **Logging**: Every call logged with tokens used
        
        Example:
            ```python
            provider = SQLMetadataProvider(token_budget=50000)
            tools = provider.get_tools()
            
            ActorConfig(
                name="ColumnSelector",
                architect_tools=tools, # All metadata methods available!
                ...
            )
            
            # Agent calls tool:
            schema = metadata_get_schema("data_source_1")
            # Log: metadata_get_schema("data_source_1") [call #1]
            # Result: Dict (2,500 tokens)
            # Budget: 2,500 / 50,000 (5.0% used)
            
            # Agent calls same tool again:
            schema = metadata_get_schema("data_source_1")
            # Log: metadata_get_schema("data_source_1") → CACHE HIT
            # (No tokens used, instant return!)
            ```
        """
        tools = []
        for name, method in self._registered_methods.items():
            tool = self._wrap_as_tool(name, method)
            tools.append(tool)
        
        logger.info(f" Exposed {len(tools)} metadata methods as safeguarded tools")
        logger.info(f"   Token budget: {self._token_budget:,}")
        logger.info(f"   Caching: {'enabled' if self._enable_caching else 'disabled'}")
        return tools
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """
        Get statistics about tool usage.
        
        Returns:
            Dictionary with:
            - calls: Total tool calls
            - tokens_used: Total tokens consumed
            - tokens_budget: Total budget
            - cache_size: Number of cached results
            - cache_hit_rate: Percentage of cache hits
        
        Example:
            ```python
            stats = provider.get_tool_stats()
            print(f"Tool calls: {stats['calls']}")
            print(f"Tokens used: {stats['tokens_used']:,} / {stats['tokens_budget']:,}")
            print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
            ```
        """
        return {
            'calls': self._tool_call_count,
            'tokens_used': self._tokens_used,
            'tokens_budget': self._token_budget,
            'budget_remaining': self._token_budget - self._tokens_used,
            'usage_pct': (self._tokens_used / self._token_budget) * 100 if self._token_budget > 0 else 0,
            'cache_size': len(self._tool_cache),
            'caching_enabled': self._enable_caching
        }
    
    def reset_tool_stats(self) -> None:
        """Reset tool usage statistics (useful for testing)."""
        self._tokens_used = 0
        self._tool_call_count = 0
        self._tool_cache.clear()
        logger.info(" Tool statistics reset")
    
    def _wrap_as_tool(self, name: str, method: Callable) -> Callable:
        """
        Wrap method as a tool with safeguards (budget, caching, logging).
        
        Args:
            name: Method name
            method: Method to wrap
        
        Returns:
            Wrapped callable with:
            - Token budget tracking (A-Team: prevent explosion)
            - Caching (A-Team: prevent redundant calls)
            - Extreme logging (A-Team: for debugging)
            - DSPy-compatible attributes
        """
        description = self._method_descriptions.get(name, f"Call {name}")
        signature = self._method_signatures.get(name)
        
        # Create wrapper with safeguards
        @functools.wraps(method)
        def safeguarded_tool(*args: Any, **kwargs: Any) -> Any:
            """Tool wrapper with budget, caching, and logging."""
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # SAFEGUARD 1: Check cache (A-Team: Dr. Chen's cooperation)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if self._enable_caching:
                cache_key = f"{name}:{args}:{tuple(sorted(kwargs.items()))}"
                if cache_key in self._tool_cache:
                    cached_result = self._tool_cache[cache_key]
                    logger.info(f" metadata_{name}(*{args[:2] if args else []}, **{list(kwargs.keys())[:2]}) → CACHE HIT")
                    return cached_result
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # SAFEGUARD 2: Check token budget (A-Team: Alex's concern)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if self._tokens_used >= self._token_budget:
                error_msg = (
                    f"Token budget exceeded for metadata tools: "
                    f"{self._tokens_used:,} / {self._token_budget:,} tokens used. "
                    f"Attempted to call: metadata_{name}"
                )
                logger.error(f" {error_msg}")
                raise ValueError(error_msg)
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Execute tool
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            self._tool_call_count += 1
            logger.info(f" metadata_{name}(*{args[:2] if args else []}, **{list(kwargs.keys())[:2]}) [call #{self._tool_call_count}]")
            
            try:
                result = method(*args, **kwargs)
                
                # Count tokens (approximate)
                result_str = str(result)
                tokens = len(result_str) // 4
                self._tokens_used += tokens
                
                usage_pct = (self._tokens_used / self._token_budget) * 100
                logger.info(f" Result: {type(result).__name__} ({tokens:,} tokens)")
                logger.info(f" Budget: {self._tokens_used:,} / {self._token_budget:,} ({usage_pct:.1f}% used)")
                
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # SAFEGUARD 3: Cache result (A-Team: Dr. Chen's cooperation)
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                if self._enable_caching:
                    cache_key = f"{name}:{args}:{tuple(sorted(kwargs.items()))}"
                    self._tool_cache[cache_key] = result
                    logger.debug(f" Cached result for future calls")
                
                return result
                
            except Exception as e:
                logger.error(f" Error calling metadata_{name}: {e}")
                logger.error(f" Args: {args}")
                logger.error(f" Kwargs: {kwargs}")
                raise
        
        # Set tool attributes for DSPy
        safeguarded_tool.name = f"metadata_{name}"
        safeguarded_tool.description = description
        safeguarded_tool.__signature__ = signature  # Preserve signature for introspection
        
        return safeguarded_tool
    
    def list_available_methods(self) -> Dict[str, str]:
        """
        List all callable methods with their descriptions.
        
        Useful for agents to discover what metadata methods exist.
        
        Returns:
            Dictionary mapping method_name → description
        
        Example:
            ```python
            methods = provider.list_available_methods()
            # {
            #     'get_schema': 'Get complete schema for a table including...',
            #     'get_columns_for_table': 'Get column names for a specific table...',
            #     'get_partition_info': 'Get partition information for multiple tables.'
            # }
            ```
        """
        return self._method_descriptions.copy()
    
    def get_method_info(self, method_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific method.
        
        Args:
            method_name: Name of the method
        
        Returns:
            Dictionary with method details or None if not found:
            - name: Method name
            - description: Docstring
            - signature: Method signature
            - parameters: Parameter names and types
        
        Example:
            ```python
            info = provider.get_method_info('get_schema')
            # {
            #     'name': 'get_schema',
            #     'description': 'Get complete schema for a table...',
            #     'signature': '(table_name: str) -> Dict',
            #     'parameters': {
            #         'table_name': {'type': 'str', 'required': True}
            #     }
            # }
            ```
        """
        if method_name not in self._registered_methods:
            return None
        
        signature = self._method_signatures[method_name]
        params = {}
        for param_name, param in signature.parameters.items():
            if param_name == 'self':
                continue
            params[param_name] = {
                'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'Any',
                'required': param.default == inspect.Parameter.empty,
                'default': param.default if param.default != inspect.Parameter.empty else None
            }
        
        return {
            'name': method_name,
            'description': self._method_descriptions[method_name],
            'signature': str(signature),
            'parameters': params
        }
    
    # =========================================================================
    # PROTOCOL METHODS (Must be implemented by subclass)
    # =========================================================================
    
    def get_context_for_actor(self, actor_name: str, query: str, previous_outputs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
        """
        Return context/metadata for a specific actor.
        
        MUST be implemented by subclass.
        
        This is called by ReVal when an actor is about to execute.
        Return dictionary with keys matching actor's signature parameters.
        
        See MetadataProvider protocol for full documentation.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_context_for_actor()"
        )
    
    def get_swarm_context(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Return global context for the entire swarm (optional).
        
        This is called ONCE when swarm initializes.
        
        Default implementation returns empty dict (no swarm context).
        Override if you need swarm-level metadata.
        
        See MetadataProvider protocol for full documentation.
        """
        return {}

    # =========================================================================
    # HELPER: Register fields for simple providers
    # =========================================================================
    
    def register_field(self, field_name: str, field_value: Any) -> None:
        """
        Register a metadata field that can be accessed by actors.
        
        This is a helper for simple providers where you just want to
        expose data fields without writing methods.
        
        Args:
            field_name: Name of the field (e.g., 'business_context')
            field_value: Value of the field (string, dict, list, etc.)
        
        Example:
            ```python
            class SimpleProvider(BaseMetadataProvider):
                def __init__(self, data):
                    self.register_field('business_context', data.business_context)
                    self.register_field('term_glossary', data.glossary)
                    super().__init__()
            ```
        """
        setattr(self, f"{field_name}_str", field_value)
        logger.debug(f" Registered field: {field_name}_str")


# =============================================================================
# SIMPLE INTERFACE: Factory Function (90% of users)
# =============================================================================

def create_metadata_provider(
    data_directory: str,
    actor_mappings: Optional[Dict[str, List[str]]] = None,
    token_budget: int = 50000,
    enable_caching: bool = True,
    file_extension: str = ".md"
) -> BaseMetadataProvider:
    """
    Create a metadata provider from a directory of files (SIMPLE INTERFACE).
    
    This is the RECOMMENDED way for most users. No subclassing needed!
    
# A-Team Consensus: 90% of users should use this factory function.
    
    What it does:
    1. Scans directory for metadata files
    2. Loads each file as a field (business_context.md → business_context_str)
    3. Auto-exposes methods as tools
    4. Sets up actor mappings
    5. Enables safeguards (budget, caching, logging)
    
    Args:
        data_directory: Path to directory containing metadata files
        actor_mappings: Dictionary mapping actor names to list of field names
        token_budget: Maximum tokens for tool calls (default: 50k)
        enable_caching: Whether to cache tool results (default: True)
        file_extension: File extension to scan for (default: .md)
    
    Returns:
        Configured BaseMetadataProvider ready to use
    
    Example (Simple):
        ```python
        # Directory structure:
        # /metadata/
        #   business_context.md
        #   term_glossary.md
        #   table_metadata.md
        
        provider = create_metadata_provider(
            data_directory="/metadata",
            actor_mappings={
                'BusinessTermResolver': ['business_context', 'term_glossary'],
                'ColumnSelector': ['table_metadata']
            }
        )
        
        # Done! Use with ReVal:
        swarm = SwarmReVal(actors, provider, config)
        ```
    
    Example (Minimal):
        ```python
        # Even simpler - just provide directory!
        provider = create_metadata_provider("/metadata")
        # All actors get all fields by default
        ```
    """
    import os
    import glob
    
    # Validate directory
    if not os.path.isdir(data_directory):
        raise ValueError(f"Directory not found: {data_directory}")
    
    # Scan for files
    pattern = os.path.join(data_directory, f"*{file_extension}")
    files = glob.glob(pattern)
    
    if not files:
        logger.warning(f" No {file_extension} files found in {data_directory}")
    
    # Create dynamic class
    class AutoMetadataProvider(BaseMetadataProvider):
        """Auto-generated metadata provider from directory."""
        
        def __init__(self) -> None:
            # Load all files
            self._fields = {}
            for filepath in files:
                # Extract field name from filename
                filename = os.path.basename(filepath)
                field_name = filename.replace(file_extension, "")
                
                # Load file content
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Register field
                self._fields[field_name] = content
                setattr(self, f"{field_name}_str", content)
            
            # Store actor mappings
            self._actor_mappings = actor_mappings or {}
            self._default_fields = list(self._fields.keys())
            
            # Initialize base class
            super().__init__(token_budget=token_budget, enable_caching=enable_caching)
            
            logger.info(f" Auto-created metadata provider from: {data_directory}")
            logger.info(f"   Fields: {list(self._fields.keys())}")
            logger.info(f"   Actor mappings: {len(self._actor_mappings)}")
        
        def get_context_for_actor(self, actor_name: str, query: str, previous_outputs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
            """Return fields for this actor based on mappings."""
            # Get fields for this actor
            fields = self._actor_mappings.get(actor_name, self._default_fields)
            
            # Build context
            context = {'current_query': query}
            for field in fields:
                field_attr = f"{field}_str"
                if hasattr(self, field_attr):
                    context[field_attr] = getattr(self, field_attr)
            
            return context
    
    # Create and return instance
    provider = AutoMetadataProvider()
    logger.info(f" Created metadata provider with {len(provider._fields)} fields")
    
    return provider

