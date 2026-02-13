"""
Jotty Metadata Protocol - User's metadata class interface.

This module defines how users provide metadata to Jotty.
NO hardcoded requirements - Jotty discovers methods via introspection!

Design Philosophy:
- Like DSPy agents: user defines structure, JOTTY discovers it
- Protocol over inheritance: user can write ANY class
- Introspection-based: JOTTY finds methods automatically
- LLM-driven: agents decide when to call methods

Author: A-Team
Date: Dec 26, 2025
"""

import inspect
import asyncio
import logging
from typing import Protocol, Any, Dict, List, Optional, Callable, runtime_checkable
from dataclasses import dataclass, field
from functools import wraps
import time

logger = logging.getLogger(__name__)


# =============================================================================
# METADATA METHOD DESCRIPTOR
# =============================================================================

@dataclass
class MethodMetadata:
    """
    Metadata about a metadata method (yes, meta-meta!).
    
    This describes what a method does, when to use it, and how to call it.
    LLM uses this to decide when agents should call the method.
    """
    name: str
    description: str
    when_to_use: str
    signature: str
    parameters: List[str]
    parameter_types: Dict[str, Any]
    return_type: Any
    docstring: Optional[str] = None
    cache: bool = True
    timeout: float = 30.0
    is_async: bool = False
    
    def to_tool_description(self) -> str:
        """
        Convert to LLM-readable tool description.
        
        This is what agents see when deciding whether to call this method.
        """
        desc = f"""
Tool: {self.name}
Description: {self.description}
When to use: {self.when_to_use}
Signature: {self.signature}
Parameters: {', '.join(f'{p}: {self.parameter_types.get(p, "Any")}' for p in self.parameters)}
Returns: {self.return_type}
"""
        if self.docstring:
            desc += f"\nDetails: {self.docstring}\n"
        return desc.strip()


# =============================================================================
# DECORATOR FOR METADATA METHODS
# =============================================================================

def jotty_method(
    desc: str,
    when: str = "Use when relevant to agent's task",
    cache: bool = True,
    timeout: float = 30.0,
    for_architect: bool = False,
    for_auditor: bool = False
):
    """
    Decorator to add metadata to user's methods.
    
     A-TEAM PHASE 2 FIX: Added for_architect and for_auditor flags!
    
    Usage:
        # Tool for Architect only (exploration)
        @jotty_method(
            desc="Get all available tables",
            when="Agent needs to discover what tables exist",
            for_architect=True,
            for_auditor=False
        )
        def get_all_tables(self):
            return list(self.tables.keys())
        
        # Tool for Auditor only (verification)
        @jotty_method(
            desc="Get table schema",
            when="Agent needs to verify table structure",
            for_architect=False,
            for_auditor=True
        )
        def get_table_schema(self, table_name: str):
            return self.schemas[table_name]
        
        # Tool for both (if needed)
        @jotty_method(
            desc="Get current datetime",
            when="Agent needs current date for context",
            for_architect=True,
            for_auditor=True
        )
        def get_current_datetime(self):
            return datetime.now().isoformat()
    
    Args:
        desc: Human-readable description of what method does
        when: When should agents use this method
        cache: Whether to cache results
        timeout: Timeout in seconds
        for_architect: If True, this tool is available to Architect agents (DEFAULT: False)
        for_auditor: If True, this tool is available to Auditor agents (DEFAULT: False)
        
    Note:
        - If both for_architect=False and for_auditor=False, tool is available to actors only
        - Metadata manager decides tool routing, NOT Jotty (keeps Jotty generic!)
    """
    def decorator(func):
        # Store metadata on function
        func._jotty_meta = {
            'desc': desc,
            'when': when,
            'cache': cache,
            'timeout': timeout,
            'for_architect': for_architect, # A-TEAM PHASE 2
            'for_auditor': for_auditor # A-TEAM PHASE 2
        }
        # A-TEAM PHASE 2: Also store as top-level attributes for easy access
        func._jotty_for_architect = for_architect
        func._jotty_for_auditor = for_auditor
        return func
    return decorator


# =============================================================================
# METADATA INTROSPECTOR
# =============================================================================

class MetadataIntrospector:
    """
    Discovers methods in user's metadata class.
    
    Uses Python introspection to find all public methods,
    extract their signatures, and create tool descriptions.
    
    NO hardcoded method names! Discovers whatever user provides!
    """
    
    def __init__(self):
        self.cache = {}
    
    def discover(self, metadata_obj: Any) -> List[MethodMetadata]:
        """
        Discover all public methods in metadata object.
        
        Args:
            metadata_obj: User's metadata instance
            
        Returns:
            List of MethodMetadata describing each method
        """
        # Check cache
        obj_id = id(metadata_obj)
        if obj_id in self.cache:
            return self.cache[obj_id]
        
        methods = []
        for name in dir(metadata_obj):
            # Skip private/magic methods
            if name.startswith('_'):
                continue
            
            attr = getattr(metadata_obj, name)
            
            # Only process callables
            if not callable(attr):
                continue
            
            # Analyze method
            try:
                method_meta = self._analyze_method(name, attr)
                methods.append(method_meta)
                logger.debug(f" Discovered metadata method: {name}")
            except Exception as e:
                logger.warning(f" Could not analyze method {name}: {e}")
        
        logger.info(f" Discovered {len(methods)} metadata methods")
        
        # Cache results
        self.cache[obj_id] = methods
        return methods
    
    def _analyze_method(self, name: str, method: Callable) -> MethodMetadata:
        """
        Extract metadata from a method.
        
        Looks at:
        - Function signature (parameters, types, return type)
        - Docstring
        - @jotty_method decorator metadata (if present)
        """
        # Get signature
        sig = inspect.signature(method)
        
        # Get docstring
        doc = inspect.getdoc(method) or f"Method: {name}"
        
        # Get decorator metadata (if present)
        decorator_meta = getattr(method, '_jotty_meta', {})
        
        # Extract parameters (skip 'self')
        params = []
        param_types = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            params.append(param_name)
            param_types[param_name] = param.annotation if param.annotation != inspect.Parameter.empty else Any
        
        # Get return type
        return_type = sig.return_annotation if sig.return_annotation != inspect.Signature.empty else Any
        
        # Check if async
        is_async = inspect.iscoroutinefunction(method)
        
        # Build description
        description = decorator_meta.get('desc', doc.split('\n')[0] if doc else f"Method: {name}")
        when_to_use = decorator_meta.get('when', f"Use when agent needs: {name.replace('_', ' ')}")
        
        return MethodMetadata(
            name=name,
            description=description,
            when_to_use=when_to_use,
            signature=str(sig),
            parameters=params,
            parameter_types=param_types,
            return_type=return_type,
            docstring=doc,
            cache=decorator_meta.get('cache', True),
            timeout=decorator_meta.get('timeout', 30.0),
            is_async=is_async
        )


# =============================================================================
# METADATA TOOL WRAPPER
# =============================================================================

class MetadataToolWrapper:
    """
    Wraps a metadata method as a callable tool for agents.
    
    Handles:
    - Caching (if enabled)
    - Timeouts
    - Error handling
    - Async/sync methods
    """
    
    def __init__(self, metadata_obj: Any, method_meta: MethodMetadata):
        self.metadata = metadata_obj
        self.meta = method_meta
        self.cache = {} if method_meta.cache else None
        self.call_count = 0
        self.total_time = 0.0
    
    async def __call__(self, **kwargs) -> Any:
        """
        Call the wrapped method with caching and error handling.
        
        Args:
            **kwargs: Parameters for the method
            
        Returns:
            Method result or None if error
        """
        self.call_count += 1
        start_time = time.time()
        
        # Check cache
        if self.cache is not None:
            cache_key = (self.meta.name, frozenset(kwargs.items()))
            if cache_key in self.cache:
                logger.debug(f" Cache hit for {self.meta.name}({kwargs})")
                return self.cache[cache_key]
        
        # Get method
        method = getattr(self.metadata, self.meta.name)
        
        try:
            # Call method (handle async/sync)
            if self.meta.is_async:
                result = await asyncio.wait_for(
                    method(**kwargs),
                    timeout=self.meta.timeout
                )
            else:
                # Run sync method in executor to avoid blocking
                loop = asyncio.get_running_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: method(**kwargs)),
                    timeout=self.meta.timeout
                )
            
            # Cache result
            if self.cache is not None:
                self.cache[cache_key] = result
            
            elapsed = time.time() - start_time
            self.total_time += elapsed
            logger.debug(f" {self.meta.name}({kwargs}) -> {type(result).__name__} ({elapsed:.3f}s)")
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"⏱ Timeout calling {self.meta.name}({kwargs}) after {self.meta.timeout}s")
            return None
        except Exception as e:
            logger.error(f" Error calling {self.meta.name}({kwargs}): {e}")
            return None
    
    def get_description(self) -> str:
        """Get LLM-readable description of this tool."""
        return self.meta.to_tool_description()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            'name': self.meta.name,
            'call_count': self.call_count,
            'total_time': self.total_time,
            'avg_time': self.total_time / self.call_count if self.call_count > 0 else 0,
            'cache_size': len(self.cache) if self.cache is not None else 0
        }


# =============================================================================
# METADATA PROTOCOL
# =============================================================================

@runtime_checkable
class MetadataProtocol(Protocol):
    """
    Protocol for user's metadata class.
    
    User can implement ANY class - NO inheritance required!
    
    Requirements:
    1. Must have at least one public method (not starting with _)
    2. Methods should have docstrings (for LLM understanding)
    3. Methods should have type hints (for validation)
    4. Methods can be sync or async
    
    JOTTY will:
    - Discover all public methods via introspection
    - Expose them as tools to agents
    - Cache results (if method decorated with cache=True)
    - Handle errors gracefully
    
    Optional Methods:
    - __jotty_validate__(): Validate metadata on load
    - __jotty_context__(actor_name, task): Provide context-specific metadata
    """
    
    def __jotty_validate__(self) -> bool:
        """
        Optional: User can implement validation logic.
        Called by JOTTY before using metadata.
        
        Returns:
            True if metadata is valid, False otherwise
        """
        ...
    
    def __jotty_context__(self, actor_name: str, task: str) -> Dict[str, Any]:
        """
        Optional: User can provide context-specific metadata.
        If implemented, JOTTY calls this to get relevant subset.
        
        Args:
            actor_name: Name of the actor requesting context
            task: Description of the task
            
        Returns:
            Dictionary of context-specific metadata
        """
        ...


# =============================================================================
# OPTIONAL BASE CLASS (Convenience)
# =============================================================================

class JottyMetadataBase:
    """
    Optional base class for user's metadata.
    
    Provides convenience methods:
    - load_from_directory(): Load metadata files
    - get_tools(): Get all methods as tools
    - validate(): Validate metadata structure
    
    User can inherit from this for convenience, but it's NOT required!
    
    JOTTY v1.0 - Metadata Base Class
    """
    
    def __init__(self):
        self._introspector = MetadataIntrospector()
        self._tools = None
        self._validated = False
    
    def load_from_directory(self, data_dir: str):
        """
        Helper: Load metadata from directory.
        
        Looks for .md, .txt, .json files and loads them as attributes.
        """
        from pathlib import Path
        
        data_path = Path(data_dir)
        if not data_path.is_dir():
            raise ValueError(f"Data directory not found: {data_dir}")
        
        for file_path in data_path.iterdir():
            if file_path.suffix in ['.md', '.txt', '.json']:
                field_name = file_path.stem
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    setattr(self, field_name, content)
                    logger.info(f" Loaded metadata file: {file_path.name}")
                except Exception as e:
                    logger.error(f" Error loading {file_path.name}: {e}")
    
    def get_tools(self) -> List[MetadataToolWrapper]:
        """
        Get all methods as callable tools.
        
        Returns:
            List of MetadataToolWrapper instances
        """
        if self._tools is None:
            methods = self._introspector.discover(self)
            self._tools = [MetadataToolWrapper(self, m) for m in methods]
        return self._tools
    
    def validate(self) -> bool:
        """
        Validate metadata structure.
        
        Checks:
        - At least one public method exists
        - Methods have type hints
        - Methods have docstrings
        
        Returns:
            True if valid, False otherwise
        """
        if self._validated:
            return True
        
        methods = self._introspector.discover(self)
        
        if not methods:
            logger.error(" No public methods found in metadata class")
            return False
        
        for method in methods:
            if not method.docstring:
                logger.warning(f" Method {method.name} has no docstring")
            if not method.parameters:
                logger.warning(f" Method {method.name} has no parameters")
        
        self._validated = True
        logger.info(f" Metadata validation passed: {len(methods)} methods")
        return True
    
    def __jotty_validate__(self) -> bool:
        """Implement protocol method."""
        return self.validate()
    
    # Backward compatibility alias
    # Backward compatibility alias
    __reval_validate__ = __jotty_validate__


# =============================================================================
# METADATA VALIDATOR (Testing Tool)
# =============================================================================

class MetadataValidator:
    """
    Tool for users to test their metadata class.
    
    Usage:
        validator = MetadataValidator()
        validator.check(MyMetadata())
        validator.test_discovery(MyMetadata())
        validator.test_call(MyMetadata(), 'get_schema', table='users')
    """
    
    def __init__(self):
        self.introspector = MetadataIntrospector()
    
    def check(self, metadata_obj: Any) -> bool:
        """
        Validate metadata object structure.
        
        Args:
            metadata_obj: User's metadata instance
            
        Returns:
            True if valid, False otherwise
        """
        logger.info(" Validating metadata class...")
        
        # Check if has __jotty_validate__ (or legacy __reval_validate__)
        validate_method = getattr(metadata_obj, '__jotty_validate__', None) or getattr(metadata_obj, '__reval_validate__', None)
        if validate_method:
            if not validate_method():
                logger.error(" Metadata validation failed (__jotty_validate__ returned False)")
                return False
        
        # Discover methods
        methods = self.introspector.discover(metadata_obj)
        
        if not methods:
            logger.error(" No public methods found")
            return False
        
        logger.info(f" Found {len(methods)} methods")
        
        # Check each method
        for method in methods:
            logger.info(f" {method.name}: {method.description}")
            if not method.docstring:
                logger.warning(f" No docstring")
            if not method.parameters:
                logger.warning(f" No parameters")
        
        return True
    
    def discover_tools(self, metadata_obj: Any) -> List[MetadataToolWrapper]:
        """
        Discover and wrap all methods as tools.
        
        Args:
            metadata_obj: User's metadata instance
            
        Returns:
            List of MetadataToolWrapper instances
        """
        methods = self.introspector.discover(metadata_obj)
        tools = [MetadataToolWrapper(metadata_obj, m) for m in methods]
        
        logger.info(f" Created {len(tools)} tools:")
        for tool in tools:
            logger.info(f"  {tool.get_description()}")
        
        return tools
    
    async def test_call(self, metadata_obj: Any, method_name: str, **kwargs) -> Any:
        """
        Test calling a specific method.
        
        Args:
            metadata_obj: User's metadata instance
            method_name: Name of method to call
            **kwargs: Parameters for method
            
        Returns:
            Method result
        """
        logger.info(f" Testing {method_name}({kwargs})...")
        
        # Find method
        methods = self.introspector.discover(metadata_obj)
        method_meta = next((m for m in methods if m.name == method_name), None)
        
        if not method_meta:
            logger.error(f" Method {method_name} not found")
            return None
        
        # Wrap and call
        tool = MetadataToolWrapper(metadata_obj, method_meta)
        result = await tool(**kwargs)
        
        logger.info(f" Result: {result}")
        logger.info(f" Stats: {tool.get_stats()}")
        
        return result


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'MethodMetadata',
    'jotty_method',
    'MetadataIntrospector',
    'MetadataToolWrapper',
    'MetadataProtocol',
    'JottyMetadataBase',
    'MetadataValidator'
]

