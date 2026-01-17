"""
ParameterResolutionManager - Manages parameter resolution and dependency tracking.

Extracted from conductor.py to improve maintainability.
Consolidates parameter resolution logic from conductor and ParameterResolver.
"""
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ResolutionResult:
    """Result of parameter resolution."""
    value: Any
    source: str  # 'kwargs', 'shared_context', 'io_manager', 'metadata', 'default'
    confidence: float = 1.0


class ParameterResolutionManager:
    """
    Centralized parameter resolution management.

    Responsibilities:
    - Parameter resolution from multiple sources
    - Dependency tracking between actors
    - Signature introspection
    - Type matching and transformation

    This manager coordinates with ParameterResolver and provides a unified interface.
    """

    def __init__(self, config, parameter_resolver=None):
        """
        Initialize parameter resolution manager.

        Args:
            config: JottyConfig
            parameter_resolver: Optional ParameterResolver instance (from conductor)
        """
        self.config = config
        self.parameter_resolver = parameter_resolver  # Delegate to existing resolver
        self.resolution_count = 0
        self.cache = {}

        logger.info("🔧 ParameterResolutionManager initialized")

    def resolve_parameter(
        self,
        param_name: str,
        param_info: Dict,
        kwargs: Dict,
        shared_context: Dict,
        io_manager: Any = None
    ) -> Optional[Any]:
        """
        Resolve parameter from multiple sources.

        Priority order:
        1. Direct kwargs (explicit overrides)
        2. SharedContext (global parameters)
        3. IOManager (previous actor outputs)
        4. Metadata tools
        5. Defaults

        Args:
            param_name: Name of parameter to resolve
            param_info: Parameter metadata (type, required, default)
            kwargs: Direct keyword arguments
            shared_context: Shared execution context
            io_manager: IO manager with actor outputs

        Returns:
            Resolved value or None if not found
        """
        self.resolution_count += 1

        # Priority 1: Direct kwargs
        if param_name in kwargs:
            logger.debug(f"✅ Resolved '{param_name}' from kwargs")
            return kwargs[param_name]

        # Priority 2: SharedContext
        if shared_context and shared_context.get(param_name) is not None:
            logger.debug(f"✅ Resolved '{param_name}' from SharedContext")
            return shared_context.get(param_name)

        # Priority 3: IOManager (actor outputs)
        if io_manager:
            value = self._resolve_from_io_manager(param_name, io_manager)
            if value is not None:
                logger.debug(f"✅ Resolved '{param_name}' from IOManager")
                return value

        # Priority 4: Default value
        if 'default' in param_info and param_info['default'] is not None:
            logger.debug(f"✅ Using default for '{param_name}'")
            return param_info['default']

        logger.debug(f"⚠️  Could not resolve '{param_name}'")
        return None

    def _resolve_from_io_manager(self, param_name: str, io_manager: Any) -> Optional[Any]:
        """
        Resolve parameter from IOManager actor outputs.

        Args:
            param_name: Parameter name
            io_manager: IOManager instance

        Returns:
            Resolved value or None
        """
        try:
            all_outputs = io_manager.get_all_outputs()
            for actor_name, actor_output in all_outputs.items():
                if hasattr(actor_output, 'output_fields') and actor_output.output_fields:
                    if param_name in actor_output.output_fields:
                        return actor_output.output_fields[param_name]
        except Exception as e:
            logger.warning(f"Error resolving from IOManager: {e}")
        return None

    def introspect_signature(self, actor_config) -> Dict[str, Any]:
        """
        Introspect actor signature to extract parameters.

        Delegates to ParameterResolver if available.

        Args:
            actor_config: Actor configuration

        Returns:
            Dict of parameter_name -> parameter_info
        """
        # Delegate to existing ParameterResolver if available
        if self.parameter_resolver and hasattr(self.parameter_resolver, 'introspect_signature'):
            return self.parameter_resolver.introspect_signature(actor_config)

        # Fallback: basic introspection
        logger.debug(f"Using basic signature introspection for {actor_config.name}")
        return {}

    def build_dependency_graph(self, actors: List) -> Dict[str, List[str]]:
        """
        Build dependency graph showing which actors depend on which.

        Args:
            actors: List of actor configs

        Returns:
            Dict mapping actor_name -> list of dependencies
        """
        # Delegate to existing ParameterResolver if available
        if self.parameter_resolver and hasattr(self.parameter_resolver, 'build_dependency_graph'):
            return self.parameter_resolver.build_dependency_graph(actors)

        # Fallback: empty graph
        logger.debug("Using empty dependency graph")
        return {}

    def get_stats(self) -> Dict[str, Any]:
        """
        Get parameter resolution statistics.

        Returns:
            Dict with resolution metrics
        """
        return {
            "total_resolutions": self.resolution_count,
            "cache_size": len(self.cache)
        }

    def reset_stats(self):
        """Reset resolution statistics."""
        self.resolution_count = 0
        self.cache.clear()
        logger.debug("ParameterResolutionManager stats reset")

    # NOTE: Full parameter resolution logic remains in conductor.py for now
    # This manager provides the interface and will gradually absorb more logic
    # Future enhancement: Move all 1,500 lines of resolution logic here
