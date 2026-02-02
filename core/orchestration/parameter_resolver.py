"""
Parameter Resolver - Extracted from Conductor
==============================================

Handles parameter resolution from multiple sources:
- Actor outputs (IOManager)
- Metadata provider
- Semantic extraction
- Type-based matching
- LLM-based field matching

JOTTY Framework Enhancement - Fix #1 (Part 1/3)
Extracted from 4,708-line Conductor to improve maintainability.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import json
from pathlib import Path

# Import required types from core
try:
    from ..foundation.agent_config import AgentConfig, AgentSpec
    # Use AgentSpec as the primary type (AgentConfig is backward compat alias)
    ActorConfig = AgentSpec
except ImportError:
    # Fallback for testing - define minimal type
    class ActorConfig:
        pass

# Import centralized parameter aliases
from ..foundation.data_structures import DEFAULT_PARAM_ALIASES

# Type hints only - avoid circular imports
if TYPE_CHECKING:
    from .conductor import TodoItem
else:
    # Runtime fallback
    TodoItem = Any

logger = logging.getLogger(__name__)


class ParameterResolver:
    """
    Resolves actor input parameters from multiple sources.

    Extracted from Conductor to follow Single Responsibility Principle.

    Resolution hierarchy:
    1. Direct from kwargs
    2. From IOManager (previous actor outputs)
    3. From metadata provider
    4. Type-based matching
    5. Semantic extraction (LLM)
    6. Field name matching (LLM)
    """

    def __init__(
        self,
        io_manager,
        param_resolver,  # AgenticParameterResolver
        metadata_fetcher,
        actors: Dict,
        actor_signatures: Dict,
        param_mappings: Dict,
        data_registry,
        registration_orchestrator,
        data_transformer,
        shared_context,
        config
    ):
        """
        Initialize ParameterResolver with dependencies.

        Args:
            io_manager: IOManager instance for actor outputs
            param_resolver: AgenticParameterResolver for LLM-based resolution
            metadata_fetcher: MetaDataFetcher for metadata access
            actors: Dict of actor configurations
            actor_signatures: Dict of introspected actor signatures
            param_mappings: Dict of parameter name mappings
            data_registry: DataRegistry for data discovery
            registration_orchestrator: RegistrationOrchestrator for data registration
            data_transformer: SmartDataTransformer for format conversion
            shared_context: SharedContext for shared data
            config: JottyConfig/SwarmConfig instance
        """
        self.io_manager = io_manager
        self.param_resolver = param_resolver
        self.metadata_fetcher = metadata_fetcher
        self.actors = actors
        self.actor_signatures = actor_signatures
        self.param_mappings = param_mappings
        self.data_registry = data_registry
        self.registration_orchestrator = registration_orchestrator
        self.data_transformer = data_transformer
        self.shared_context = shared_context
        self.config = config

        logger.info("✅ ParameterResolver initialized")

    # =========================================================================
    # EXTRACTED METHODS FROM CONDUCTOR
    # =========================================================================

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

    def _build_param_mappings(self, custom_mappings: Optional[Dict[str, List[str]]]) -> Dict[str, List[str]]:
        """
        Build parameter mappings: defaults + config + user-provided.
        
        🆕 A-TEAM: Makes JOTTY truly generic by allowing users to define
        domain-specific parameter name mappings!
        
        Priority: user_mappings > config_mappings > defaults
        """
        # Minimal generic defaults (domain-agnostic)
        default_mappings = {
            # Core generic patterns
            'content': ['content', 'text', 'body'],
            'data': ['data', 'output_data', 'results'],
            'file': ['file', 'filepath', 'path', 'file_path'],
            'url': ['url', 'uri', 'link', 'href'],
        }
        
        # Start with defaults
        mappings = default_mappings.copy()
        
        # Merge config mappings (if provided)
        if hasattr(self.config, 'custom_param_mappings') and self.config.custom_param_mappings:
            for key, aliases in self.config.custom_param_mappings.items():
                if key in mappings:
                    # Extend existing (remove duplicates)
                    mappings[key] = list(set(mappings[key] + aliases))
                else:
                    # Add new
                    mappings[key] = aliases
        
        # Merge user mappings (highest priority)
        if custom_mappings:
            for key, aliases in custom_mappings.items():
                if key in mappings:
                    # Extend existing (remove duplicates)
                    mappings[key] = list(set(mappings[key] + aliases))
                else:
                    # Add new
                    mappings[key] = aliases
        
        logger.info(f"📋 Parameter mappings: {len(mappings)} keys, {sum(len(v) for v in mappings.values())} total aliases")
        if mappings:
            logger.debug(f"   Mappings: {list(mappings.keys())}")
        
        return mappings
    
    
    async def _attempt_parameter_recovery(
        self,
        actor_config: ActorConfig,
        missing_params: List[str],
        context: Dict[str, Any],
        shared_context: Dict[str, Any],
        max_attempts: int = 2
    ) -> Dict[str, Any]:
        """
        🛠️ INTELLIGENT PARAMETER RECOVERY
        
        PRIORITY ORDER (A-TEAM FINAL):
        1. Check IOManager (actor outputs) ← WHERE AGENT DATA LIVES!
        2. Check SharedContext['metadata'] (only if not found in IOManager)
        3. Invoke MetaDataFetcher (on-demand fetch)
        4. (Future) Re-invoke failed dependencies
        """
        recovered = {}
        logger.info(f"🔍 Recovery Strategy for {actor_config.name}: analyzing {len(missing_params)} missing parameters")
        
        for param in missing_params:
            logger.info(f"   🔍 Attempting recovery for '{param}'...")
            
            # 🔥 STRATEGY 1: Check IOManager (ACTOR OUTPUTS) - THIS IS THE PRIMARY DATA FLOW!
            if hasattr(self, 'io_manager') and self.io_manager:
                all_outputs = self.io_manager.get_all_outputs()
                for actor_name, actor_output in all_outputs.items():
                    if hasattr(actor_output, 'output_fields') and actor_output.output_fields:
                        # Exact match
                        if param in actor_output.output_fields:
                            recovered[param] = actor_output.output_fields[param]
                            logger.info(f"   ✅ Found '{param}' in IOManager['{actor_name}']")
                            break
                        
                        # Check aliases using centralized DEFAULT_PARAM_ALIASES
                        # Merge with config custom mappings if available
                        aliases = DEFAULT_PARAM_ALIASES.copy()
                        if hasattr(self.config, 'custom_param_mappings') and self.config.custom_param_mappings:
                            for key, vals in self.config.custom_param_mappings.items():
                                if key in aliases:
                                    aliases[key] = list(set(aliases[key] + vals))
                                else:
                                    aliases[key] = vals

                        if param in aliases:
                            for alias in aliases[param]:
                                if alias in actor_output.output_fields:
                                    recovered[param] = actor_output.output_fields[alias]
                                    logger.info(f"   ✅ Found '{param}' as '{alias}' in IOManager['{actor_name}']")
                                    break
                        
                        if param in recovered:
                            break
            
            if param in recovered:
                continue
            
            # Strategy 2: Check SharedContext['metadata'] (ONLY if not in IOManager!)
            if self.shared_context and 'metadata' in self.shared_context.data:
                metadata = self.shared_context.get('metadata')
                if metadata and isinstance(metadata, dict):
                    # Direct match
                    if param in metadata:
                        recovered[param] = metadata[param]
                        logger.info(f"   ✅ Found '{param}' in SharedContext['metadata']")
                        continue
                    
                    # 🔥 A-TEAM: ALIAS MATCHING using centralized DEFAULT_PARAM_ALIASES
                    aliases = DEFAULT_PARAM_ALIASES.copy()
                    if hasattr(self.config, 'custom_param_mappings') and self.config.custom_param_mappings:
                        for key, vals in self.config.custom_param_mappings.items():
                            if key in aliases:
                                aliases[key] = list(set(aliases[key] + vals))
                            else:
                                aliases[key] = vals

                    if param in aliases:
                        for alias in aliases[param]:
                            if alias in metadata:
                                recovered[param] = metadata[alias]
                                logger.info(f"   ✅ Found '{param}' via alias '{alias}' in SharedContext['metadata']")
                                break
                    
                    if param in recovered:
                        continue
                    
                    # 🔥 A-TEAM: AGENTIC SEMANTIC SEARCH (NO FUZZY MATCHING!)
                    # Use AgenticParameterResolver for intelligent, LLM-based matching
                    logger.info(f"   🔍 Using agentic search for '{param}' in metadata...")
                    
                    if hasattr(self, 'param_resolver') and self.param_resolver:
                        # Get parameter info from signature
                        param_info = {}
                        param_type_str = 'str'
                        if actor_config.name in self.actor_signatures:
                            sig = self.actor_signatures[actor_config.name]
                            if isinstance(sig, dict) and param in sig:
                                param_info = sig[param]
                                param_type = param_info.get('annotation')
                                if param_type and hasattr(param_type, '__name__'):
                                    param_type_str = param_type.__name__
                        
                        # Prepare available data with rich descriptions
                        available_data = {}
                        for key, value in metadata.items():
                            available_data[key] = {
                                'value': value,
                                'type': type(value).__name__,
                                'description': f"Metadata key '{key}' containing {type(value).__name__}",
                                'tags': ['metadata', 'available'],
                                'source': 'SharedContext[metadata]'
                            }
                        
                        try:
                            # Use agentic resolver with LLM intelligence  
                            matched_key, confidence, reasoning = self.param_resolver.resolve_parameter(
                                actor_name=actor_config.name,
                                parameter_name=param,
                                parameter_type=param_type_str,
                                parameter_purpose=f"Required parameter for {actor_config.name}",
                                available_data=available_data
                            )
                            
                            if matched_key and matched_key in metadata:
                                if confidence > 0.7:  # High confidence threshold
                                    recovered[param] = metadata[matched_key]
                                    logger.info(f"   ✅ Agentic match: '{param}' → '{matched_key}' (confidence: {confidence:.2f})")
                                    logger.info(f"   📝 Reasoning: {reasoning[:150]}...")
                                else:
                                    logger.warning(f"   ⚠️ Low confidence match: '{param}' → '{matched_key}' (conf: {confidence:.2f})")
                                    logger.info(f"   📝 Reasoning: {reasoning[:100]}...")
                            else:
                                logger.warning(f"   ❌ No agentic match found for '{param}'")
                        
                        except Exception as e:
                            logger.error(f"   ❌ Agentic resolver failed: {e}")
                            import traceback
                            logger.error(f"Traceback: {traceback.format_exc()}")
                    else:
                        logger.warning(f"   ⚠️ AgenticParameterResolver not available - parameter '{param}' cannot be resolved from metadata")
            
            if param in recovered:
                continue
            
            # Strategy 3: Invoke MetaDataFetcher for on-demand fetch
            if self.metadata_fetcher and hasattr(self.metadata_fetcher, 'tools'):
                logger.info(f"   🔍 Asking MetaDataFetcher to fetch '{param}'...")
                try:
                    # Check if any tool matches this parameter
                    for tool in self.metadata_fetcher.tools:
                        tool_name = tool.name.lower() if hasattr(tool, 'name') else str(tool)
                        if param.lower() in tool_name or tool_name in param.lower():
                            logger.info(f"   📞 Calling tool '{tool.name}' for '{param}'")
                            try:
                                result = tool.func() if hasattr(tool, 'func') else None
                                if result:
                                    recovered[param] = result
                                    logger.info(f"   ✅ Fetcher retrieved '{param}'")
                                    break
                            except Exception as e:
                                logger.warning(f"   ⚠️  Tool call failed: {e}")
                except Exception as e:
                    logger.warning(f"   ⚠️  MetaDataFetcher recovery failed: {e}")
            
            if param not in recovered:
                logger.warning(f"   ❌ Could not recover '{param}'")
        
        still_missing = [p for p in missing_params if p not in recovered]
        
        logger.info(f"🏁 Recovery complete: {len(recovered)}/{len(missing_params)} recovered")
        if still_missing:
            logger.warning(f"   Still missing: {still_missing}")
        
        return {
            'recovered': recovered,
            'still_missing': still_missing
        }
    
    def _find_parameter_producer(self, parameter_name: str, requesting_actor: str) -> Optional[str]:
        """
        Find which actor produces a given parameter using dependency graph.
        
        🔥 A-TEAM: Uses signature introspection + IOManager to route requests intelligently.
        """
        logger.info(f"🔍 [DEP GRAPH] Finding producer for '{parameter_name}' (requested by {requesting_actor})")
        
        # Check actor signatures for outputs
        for actor_name, sig in self.actor_signatures.items():
            if actor_name == requesting_actor:
                continue  # Don't ask yourself
            
            # Check if signature has this in outputs
            if isinstance(sig, dict) and 'outputs' in sig:
                if parameter_name in sig['outputs']:
                    logger.info(f"✅ [DEP GRAPH] {actor_name} produces '{parameter_name}' (from signature)")
                    return actor_name
        
        # Check IOManager for actual outputs
        if hasattr(self, 'io_manager') and self.io_manager:
            all_outputs = self.io_manager.get_all_outputs()
            for actor_name in all_outputs:
                if actor_name == requesting_actor:
                    continue
                output_fields = self.io_manager.get_output_fields(actor_name)
                if parameter_name in output_fields:
                    logger.info(f"✅ [DEP GRAPH] {actor_name} produces '{parameter_name}' (from IOManager)")
                    return actor_name
        
        logger.warning(f"⚠️ [DEP GRAPH] No producer found for '{parameter_name}'")
        return None
    
    async def _route_to_producer(
        self,
        producer_actor: str,
        parameters_needed: List[str],
        requesting_actor: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Route request to producer actor and retrieve data.
        
        🔥 A-TEAM: Intelligent routing using dependency graph.
        If producer already executed, retrieve from IOManager.
        If not, execute producer first.
        """
        logger.info(f"📢 [ROUTING] {requesting_actor} → {producer_actor}")
        logger.info(f"   Requesting: {parameters_needed}")
        
        # Check if producer already executed
        if hasattr(self, 'io_manager') and self.io_manager:
            all_outputs = self.io_manager.get_all_outputs()
            if producer_actor in all_outputs:
                logger.info(f"✅ [ROUTING] {producer_actor} already executed")
                output_fields = self.io_manager.get_output_fields(producer_actor)
                
                recovered = {}
                for param in parameters_needed:
                    if param in output_fields:
                        recovered[param] = output_fields[param]
                        logger.info(f"   ✅ Retrieved '{param}': {type(output_fields[param]).__name__}")
                
                if recovered:
                    return recovered
        
        # Producer hasn't executed - need to execute it first
        logger.info(f"⚠️ [ROUTING] {producer_actor} not executed yet, executing now...")
        
        if producer_actor not in self.actors:
            logger.error(f"❌ [ROUTING] Producer actor '{producer_actor}' not found in swarm")
            return {}
        
        producer_config = self.actors[producer_actor]
        
        # Build kwargs for producer
        try:
            producer_kwargs = {}
            
            # Get producer's signature
            if producer_actor in self.actor_signatures:
                sig = self.actor_signatures[producer_actor]
                if isinstance(sig, dict):
                    for param_name, param_info in sig.items():
                        # Resolve producer's parameters
                        value = self._resolve_parameter(
                            param_name,
                            param_info,
                            context.get('kwargs', {}),
                            self.shared_context.data if hasattr(self, 'shared_context') else {}
                        )
                        if value is not None:
                            producer_kwargs[param_name] = value
            
            # Execute producer
            logger.info(f"🔄 [ROUTING] Executing {producer_actor} to generate data...")
            
            # Get or create JOTTY-wrapped actor
            actor = producer_config.actor
            if hasattr(actor, 'arun'):
                result = await actor.arun(context.get('goal', ''), **producer_kwargs)
            else:
                # Not wrapped, execute directly
                result = actor(**producer_kwargs)
                # Wrap in EpisodeResult
                from ..foundation.data_structures import EpisodeResult
                result = EpisodeResult(
                    output=result,
                    success=True,
                    trajectory=[],
                    tagged_outputs=[],
                    episode=0
                )
            
            if result.success and result.output:
                # Register output in IOManager
                if hasattr(self, 'io_manager'):
                    self.io_manager.register_output(
                        actor_name=producer_actor,
                        output=result.output,
                        actor=actor,
                        success=True
                    )
                
                # Extract requested parameters
                recovered = {}
                for param in parameters_needed:
                    if hasattr(result.output, param):
                        recovered[param] = getattr(result.output, param)
                    elif hasattr(result.output, '_store') and param in result.output._store:
                        recovered[param] = result.output._store[param]
                
                logger.info(f"✅ [ROUTING] Producer executed, recovered {len(recovered)} params")
                return recovered
            else:
                logger.error(f"❌ [ROUTING] Producer {producer_actor} failed")
                return {}
        
        except Exception as e:
            logger.error(f"❌ [ROUTING] Failed to execute producer {producer_actor}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {}
    
    async def _intelligent_recovery_with_routing(
        self,
        actor_config: ActorConfig,
        missing_params: List[str],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Intelligent parameter recovery using dependency graph routing.
        
        🔥 A-TEAM: Full intelligence stack:
        1. Check IOManager (already executed actors)
        2. Check SharedContext metadata
        3. Use dependency graph to find producers
        4. Route requests to producer actors
        5. Re-execute producers if needed
        6. Fallback to MetaDataFetcher
        7. Fallback to smart defaults
        """
        recovered = {}
        logger.info(f"🛠️  [INTELLIGENT RECOVERY] for {actor_config.name}")
        logger.info(f"   Missing: {missing_params}")
        
        for param in missing_params:
            # Strategy 1: IOManager (already executed)
            if hasattr(self, 'io_manager') and self.io_manager:
                all_outputs = self.io_manager.get_all_outputs()
                for actor_name in all_outputs:
                    output_fields = self.io_manager.get_output_fields(actor_name)
                    if param in output_fields:
                        recovered[param] = output_fields[param]
                        logger.info(f"   ✅ Found '{param}' in IOManager[{actor_name}]")
                        break
            
            if param in recovered:
                continue
            
            # Strategy 2: SharedContext metadata
            if hasattr(self, 'shared_context') and self.shared_context:
                if self.shared_context.has('metadata'):
                    metadata = self.shared_context.get('metadata')
                    if isinstance(metadata, dict) and param in metadata:
                        recovered[param] = metadata[param]
                        logger.info(f"   ✅ Found '{param}' in SharedContext[metadata]")
                        continue
            
            # Strategy 3: 🔥 DEPENDENCY GRAPH ROUTING
            producer = self._find_parameter_producer(param, actor_config.name)
            if producer:
                logger.info(f"   🎯 Dependency graph: '{param}' from {producer}")
                
                # Route request to producer
                producer_data = await self._route_to_producer(
                    producer_actor=producer,
                    parameters_needed=[param],
                    requesting_actor=actor_config.name,
                    context=context
                )
                
                if param in producer_data:
                    recovered[param] = producer_data[param]
                    logger.info(f"   ✅ Routed to {producer}, retrieved '{param}'")
                    continue
            
            # Strategy 4: MetaDataFetcher (fallback)
            if hasattr(self, 'metadata_fetcher') and self.metadata_fetcher:
                logger.info(f"   🔍 Asking MetaDataFetcher for '{param}'...")
                # ... (existing MetaDataFetcher logic if needed)
        
        logger.info(f"🏁 [INTELLIGENT RECOVERY] Complete: {len(recovered)}/{len(missing_params)} recovered")
        if len(recovered) < len(missing_params):
            still_missing = [p for p in missing_params if p not in recovered]
            logger.warning(f"   Still missing: {still_missing}")
        
        return recovered
    
    def resolve_input(self, input_spec: str, resolution_context: Dict) -> Any:
        """
        🆕 DECLARATIVE INPUT RESOLUTION (A-Team FINAL Solution)
        
        Resolve input from specification string with natural syntax:
        
        Examples:
            "input.query" → From swarm.run() kwargs
            "BusinessTermResolver.required_tables" → From previous actor output
            "context.current_date" → From context_providers
            "metadata.get_all_validations()" → Call metadata_manager method
            "metadata.get_partition_info_for_tables(BusinessTermResolver.required_tables)" → Method with arg
        
        Args:
            input_spec: Specification string describing where to get the value
            resolution_context: Dict containing:
                - 'inputs': kwargs from swarm.run()
                - 'actor_outputs': Dict of previous actor outputs
                - 'context_providers': Dict of context providers (metadata_manager, etc.)
        
        Returns:
            Resolved value or None
        """
        input_spec = input_spec.strip()
        
        # Pattern 1: input.{field} - From swarm.run() kwargs
        if input_spec.startswith("input."):
            field = input_spec.split(".", 1)[1]
            value = resolution_context.get('inputs', {}).get(field)
            if value is not None:
                logger.debug(f"   ✅ Resolved from input.{field}")
            return value
        
        # Pattern 2: context.{field} - From context_providers
        elif input_spec.startswith("context."):
            field = input_spec.split(".", 1)[1]
            value = resolution_context.get('context_providers', {}).get(field)
            if value is not None:
                logger.debug(f"   ✅ Resolved from context.{field}")
            return value
        
        # Pattern 3: metadata.{method}(...) or metadata.{attr} - From metadata_manager
        elif input_spec.startswith("metadata."):
            method_or_attr = input_spec.split(".", 1)[1]
            context_providers = resolution_context.get('context_providers', {})
            # ✅ A-TEAM FIX: Look for 'metadata' (new) or 'metadata_manager' (legacy)
            metadata_manager = context_providers.get('metadata') or context_providers.get('metadata_manager')
            
            logger.info(f"   🔍 Trying to resolve metadata.{method_or_attr}")
            logger.info(f"   🔍 context_providers keys: {list(context_providers.keys())}")
            logger.info(f"   🔍 metadata_manager found: {metadata_manager is not None}")
            
            if not metadata_manager:
                logger.warning(f"⚠️  metadata_manager not found in context_providers for '{input_spec}'")
                logger.warning(f"⚠️  Available providers: {list(context_providers.keys())}")
                return None
            
            # Check if it's a method call (has parentheses)
            if "(" in method_or_attr:
                method_name = method_or_attr.split("(")[0]
                args_str = method_or_attr.split("(")[1].rstrip(")")
                
                if not hasattr(metadata_manager, method_name):
                    logger.warning(f"⚠️  Method '{method_name}' not found on metadata_manager")
                    return None
                
                method = getattr(metadata_manager, method_name)
                
                # Parse and resolve arguments
                try:
                    if args_str.strip():
                        args = []
                        for arg_spec in args_str.split(","):
                            arg_spec = arg_spec.strip()
                            # Recursively resolve the argument
                            arg_value = self.resolve_input(arg_spec, resolution_context)
                            args.append(arg_value)
                        
                        logger.info(f"   🔧 Calling {method_name}({args})")
                        result = method(*args)
                        logger.info(f"   ✅ Result: {str(result)[:200]}")
                        return result
                    else:
                        logger.info(f"   🔧 Calling {method_name}()")
                        result = method()
                        logger.info(f"   ✅ Result: {str(result)[:200]}")
                        return result
                except Exception as e:
                    logger.error(f"   ❌ Error calling metadata.{method_name}: {e}")
                    import traceback
                    logger.error(f"   ❌ Traceback: {traceback.format_exc()}")
                    return None
            else:
                # Attribute access
                if hasattr(metadata_manager, method_or_attr):
                    value = getattr(metadata_manager, method_or_attr)
                    logger.debug(f"   ✅ Resolved from metadata.{method_or_attr}")
                    return value
                else:
                    logger.warning(f"⚠️  Attribute '{method_or_attr}' not found on metadata_manager")
                    return None
        
        # Pattern 4: {Actor}.{field} - From previous actor output
        elif "." in input_spec:
            actor_name, field = input_spec.split(".", 1)
            actor_outputs = resolution_context.get('actor_outputs', {})
            actor_output = actor_outputs.get(actor_name)
            
            if not actor_output:
                logger.debug(f"   ⚠️  Actor '{actor_name}' output not found")
                return None
            
            # Extract field from output using existing extraction logic
            value = self._extract_from_output(actor_output, field)
            if value is not None:
                logger.debug(f"   ✅ Resolved from {actor_name}.{field}")
            return value
        
        # Pattern 5: Direct value (fallback)
        logger.warning(f"⚠️  Unknown input specification format: '{input_spec}'")
        return None
    
    def _resolve_parameter(self, param_name: str, param_info: Dict, kwargs: Dict, shared_context: Dict) -> Any:
        """
        Resolve parameter from multiple sources.
        
#         🔥 A-TEAM FIX: CORRECT priority order!
        1. Direct kwargs (explicit overrides)
        2. SharedContext (for GLOBAL parameters like current_date)
        3. IOManager (previous actor outputs) ← THE MAIN DATA FLOW
        4. Metadata tools
        5. Defaults
        
#         ❌ NEVER resolve domain data from SharedContext['metadata'] - agents should call tools!
        """
        
        # Priority 1: Direct kwargs (explicit overrides)
        if param_name in kwargs:
            logger.debug(f"✅ Resolved '{param_name}' from kwargs")
            return kwargs[param_name]
        
        # Priority 2: SharedContext for GLOBAL parameters (current_date, current_datetime, etc.)
        # These are system-wide constants that should be shared across all actors
        if param_name in ['current_date', 'current_datetime', 'timezone']:
            if hasattr(self, 'shared_context') and self.shared_context:
                if self.shared_context.has(param_name):
                    value = self.shared_context.get(param_name)
                    logger.info(f"✅ Resolved '{param_name}' from SharedContext (global)")
                    return value
        
        # Priority 3: IOManager - previous actor outputs (MAIN DATA FLOW!)
        if hasattr(self, 'io_manager') and self.io_manager:
            all_outputs = self.io_manager.get_all_outputs()
            for actor_name, actor_output in all_outputs.items():
                if hasattr(actor_output, 'output_fields') and actor_output.output_fields:
                    # Exact match
                    if param_name in actor_output.output_fields:
                        value = actor_output.output_fields[param_name]
                        logger.info(f"✅ Resolved '{param_name}' from IOManager['{actor_name}']")
                        
                        # 🔥 A-TEAM FIX: AGENTIC TYPE TRANSFORMATION (ReAct agent with sandbox!)
                        expected_type = param_info.get('annotation', Any)
                        if expected_type != Any and value is not None:
                            actual_type = type(value)
                            # Get origin for typing generics (List[str] → list, Dict[str, Any] → dict)
                            import typing
                            expected_origin = typing.get_origin(expected_type) or expected_type
                            
                            # 🔥 A-TEAM FIX: Skip transformation for Union types IMMEDIATELY!
                            # Union types ALWAYS fail isinstance() checks and waste tokens
                            expected_str = str(expected_type)
                            if 'Union' in expected_str or expected_origin is typing.Union:
                                logger.debug(f"⏩ Skipping transformation for Union type: {expected_str}")
                                logger.debug(f"   Using raw value: {actual_type.__name__} (DSPy will coerce)")
                                continue  # Skip to next parameter
                            
                            if actual_type != expected_origin:
                                actual_name = actual_type.__name__
                                expected_name = getattr(expected_origin, '__name__', str(expected_type))
                                logger.info(f"🔄 Type mismatch detected: {actual_name} → {expected_name}")
                                
                                # 🔥 A-TEAM LEARNING FIX: Check memory for past transformation failures!
                                should_skip_transformation = False
                                if hasattr(self, 'shared_memory') and self.shared_memory:
                                    try:
                                        past_failures = self.shared_memory.retrieve(
                                            query=f"SmartDataTransformer {param_name} {actor_name} transformation failure",
                                            top_k=3
                                        )
                                        
                                        if past_failures:
                                            logger.info(f"🧠 LEARNING: Found {len(past_failures)} past transformation failures for {param_name}")
                                            
                                            # Check if Union isinstance error happened before
                                            for failure in past_failures:
                                                failure_content = str(failure.content) if hasattr(failure, 'content') else str(failure)
                                                if "Union cannot be used with isinstance" in failure_content or "Union" in failure_content:
                                                    logger.info(f"🧠 LEARNING: Skipping SmartDataTransformer for Union type (learned from past failure)")
                                                    logger.info(f"   📦 Using raw value: {type(value).__name__}")
                                                    should_skip_transformation = True
                                                    break
                                    except Exception as mem_e:
                                        logger.debug(f"Memory query failed: {mem_e}")
                                
                                if should_skip_transformation:
                                    # Skip transformation, use raw value
                                    logger.info(f"   📦 DSPy ReAct will handle type conversion")
                                else:
                                    logger.info(f"   🤖 Invoking SmartDataTransformer (ReAct agent with sandbox)")
                                    try:
                                        # Call transformer synchronously (it handles async internally)
                                        transformed = self.data_transformer.transform(
                                            source=value,
                                            target_type=expected_origin,  # Use origin type (list, dict, not List[str])
                                            context=f"Parameter '{param_name}' from {actor_name}. Agent needs {expected_name} to proceed.",
                                            param_name=param_name
                                        )
                                        if transformed is not None:
                                            value = transformed
                                            logger.info(f"✅ Agentic transformation successful: {type(value).__name__}")
                                        else:
                                            logger.warning(f"⚠️  Transformer returned None - using original value")
                                    except Exception as e:
                                        # 🔥 A-TEAM FIX: Don't fail the entire actor execution on transformation failure!
                                        # Just log the error and continue with the original value
                                        # DSPy ReAct signature system will handle type coercion
                                        logger.warning(f"⚠️  Agentic transformation failed: {e}")
                                        logger.info(f"   📦 Continuing with original value, DSPy will handle type conversion")
                        
                        return value
        
                    # 🔥 A-TEAM FIX: GENERIC PARAMETER ALIASING using centralized DEFAULT_PARAM_ALIASES
                    aliases = DEFAULT_PARAM_ALIASES.copy()
                    if hasattr(self.config, 'custom_param_mappings') and self.config.custom_param_mappings:
                        for key, vals in self.config.custom_param_mappings.items():
                            if key in aliases:
                                aliases[key] = list(set(aliases[key] + vals))
                            else:
                                aliases[key] = vals

                    # Check if param_name has known aliases
                    if param_name in aliases:
                        for alias in aliases[param_name]:
                            if alias in actor_output.output_fields:
                                value = actor_output.output_fields[alias]
                                logger.info(f"✅ Resolved '{param_name}' from IOManager['{actor_name}']['{alias}'] (alias)")
                                
                                # 🔥 CRITICAL FIX: Return immediately! Don't wait for type transformation!
                                # Type transformation is optional - if types match, we still need to return!
                                return value
                                
                                # NOTE: Code below (type transformation) is now unreachable but kept for reference
                                # TODO: Remove this dead code in future cleanup
                                # 🔥 A-TEAM FIX: AGENTIC TYPE TRANSFORMATION (ReAct agent with sandbox!)
                                expected_type = param_info.get('annotation', Any)
                                if expected_type != Any and value is not None:
                                    actual_type = type(value)
                                    # Get origin for typing generics
                                    import typing
                                    expected_origin = typing.get_origin(expected_type) or expected_type
                                    
                                    # 🔥 A-TEAM FIX: Skip transformation for Union types IMMEDIATELY!
                                    expected_str = str(expected_type)
                                    if 'Union' in expected_str or expected_origin is typing.Union:
                                        logger.debug(f"⏩ Skipping transformation for Union type: {expected_str}")
                                        logger.debug(f"   Using raw value: {actual_type.__name__} (DSPy will coerce)")
                                    elif actual_type != expected_origin:
                                        actual_name = actual_type.__name__
                                        expected_name = getattr(expected_origin, '__name__', str(expected_type))
                                        logger.info(f"🔄 Type mismatch detected: {actual_name} → {expected_name}")
                                        logger.info(f"   🤖 Invoking SmartDataTransformer (ReAct agent)")
                                        try:
                                            # Call transformer synchronously
                                            transformed = self.data_transformer.transform(
                                                source=value,
                                                target_type=expected_origin,
                                                context=f"Parameter '{param_name}' from {actor_name}. Agent needs {expected_name}.",
                                                param_name=param_name
                                            )
                                            if transformed is not None:
                                                value = transformed
                                                logger.info(f"✅ Agentic transformation successful: {type(value).__name__}")
                                            else:
                                                logger.error(f"❌ Transformer returned None!")
                                        except Exception as e:
                                            logger.error(f"❌ Agentic transformation failed: {e}")
                                            raise RuntimeError(f"SmartDataTransformer failed: {e}") from e
                                
                                return value
        
        # Priority 3: Previous actor outputs from shared_context['actor_outputs'] (legacy path)
        actor_outputs = shared_context.get('actor_outputs', {})
        for actor_name, output in actor_outputs.items():
            value = self._extract_from_output(output, param_name)
            if value is not None:
                logger.info(f"✅ Resolved '{param_name}' from actor '{actor_name}' output")
                return value
        
        # Priority 4: SharedContext - ONLY for 'goal', 'query', 'conversation_history' (NOT metadata!)
        if hasattr(self, 'shared_context') and self.shared_context:
            # Whitelist of allowed SharedContext keys for parameters
            allowed_keys = {'goal', 'query', 'conversation_history', 'session_id'}
            
            if param_name in allowed_keys:
                # Try exact match
                if self.shared_context.has(param_name):
                    value = self.shared_context.get(param_name)
                    logger.info(f"✅ Resolved '{param_name}' from SharedContext")
                    return value
                
                # Try semantic match ONLY within allowed keys
                for key in allowed_keys:
                    if self.shared_context.has(key) and (param_name.lower() in key.lower() or key.lower() in param_name.lower()):
                        value = self.shared_context.get(key)
                        logger.info(f"✅ Resolved '{param_name}' from SharedContext['{key}'] (semantic match)")
                        return value  # 🔧 A-TEAM FIX: Return immediately when found!
        
        # Priority 5: Context providers (direct match)
        if param_name in self.context_providers:
            logger.info(f"✅ Resolved '{param_name}' from context_providers")
            return self.context_providers[param_name]
        
        # Priority 6: Shared context dict (non-metadata keys only)
        if param_name in shared_context and param_name != 'metadata':
            logger.info(f"✅ Resolved '{param_name}' from shared_context dict")
            return shared_context[param_name]
        
        # ✅ Priority 6.5: SharedContext['metadata'] - USER FEEDBACK: This was missing!
        # Check metadata directly instead of waiting for recovery
        metadata = shared_context.get('metadata', {})
        if param_name in metadata:
            logger.info(f"✅ Resolved '{param_name}' from SharedContext['metadata']")
            return metadata[param_name]
        
        # Check metadata with aliases using centralized DEFAULT_PARAM_ALIASES
        aliases = DEFAULT_PARAM_ALIASES.copy()
        if hasattr(self.config, 'custom_param_mappings') and self.config.custom_param_mappings:
            for key, vals in self.config.custom_param_mappings.items():
                if key in aliases:
                    aliases[key] = list(set(aliases[key] + vals))
                else:
                    aliases[key] = vals
        param_aliases = aliases.get(param_name, [])
        for alias in param_aliases:
            if alias in metadata:
                logger.info(f"✅ Resolved '{param_name}' via alias '{alias}' from SharedContext['metadata']")
                return metadata[alias]
        
        # Priority 7: Default value from signature
        if param_info['default'] != inspect.Parameter.empty:
            logger.debug(f"✅ Using default value for '{param_name}'")
            return param_info['default']
        
        # No resolution found
        logger.debug(f"❌ Cannot resolve parameter '{param_name}'")
        return None
    
    def _extract_from_metadata_manager(self, metadata_manager, param_name: str) -> Any:
        """
        Extract parameter from metadata_manager (user-provided metadata).
        
        NO HARDCODING - just simple attribute mapping!
        """
        # Generic attribute mapping (NO domain-specific logic!)
        metadata_mappings = {
            'validation_criterias': 'validations',
            'business_context': 'business_context',
            'term_glossary': 'term_glossary',
            'filter_conditions': 'filter_conditions',
            'joining_conditions': 'joining_conditions',
            'table_metadata': 'table_metadata',
            'column_metadata': 'column_metadata',
            'widgets_context': 'widgets_context',
        }
        
        # Try direct mapping first
        attr_name = metadata_mappings.get(param_name, param_name)
        
        if hasattr(metadata_manager, attr_name):
            value = getattr(metadata_manager, attr_name)
            if value:
                logger.debug(f"   Extracted from metadata_manager.{attr_name}")
                return value
        
        return None
    
    def _semantic_extract(self, output: Any, param_name: str) -> Any:
        """
        Semantic extraction using LLM understanding.
        
        Replaces fuzzy matching with LLM-based semantic understanding.
        Handles synonyms, typos, variations.
        
        A-Team Decision: This is the NEW way to resolve parameters.
        """
        try:
            # Build available fields
            available = {}
            if hasattr(output, '__dict__'):
                available = {k: v for k, v in vars(output).items() if not k.startswith('_')}
            elif isinstance(output, dict):
                available = output
            else:
                return None  # Can't extract from non-object/non-dict
            
            if not available:
                return None
            
            # Quick check: exact match (skip LLM)
            if param_name in available:
                return available[param_name]
            
            # Ask LLM which field matches param_name
            match_result = self._llm_match_field(param_name, available)
            
            if match_result and match_result.confidence > 0.5:
                matched_field = match_result.field_name
                if matched_field in available:
                    logger.info(f"✅ Semantic match: '{param_name}' → '{matched_field}' (confidence={match_result.confidence:.2f})")
                    logger.info(f"   Reasoning: {match_result.reasoning}")
                    return available[matched_field]
            
            return None
        
        except Exception as e:
            logger.error(f"❌ Semantic extraction error: {e}")
            return None
    
    def _llm_match_field(self, param_name: str, available_fields: Dict) -> Optional[Any]:
        """Use LLM to match parameter name to available fields."""
        try:
            import dspy
            from dataclasses import dataclass
            
            class FieldMatchSignature(dspy.Signature):
                """Match parameter name to available field."""
                
                parameter_needed = dspy.InputField(desc="Parameter name being requested")
                available_fields = dspy.InputField(desc="Available fields with previews")
                
                best_match = dspy.OutputField(desc="""
                    Best matching field name from available fields.
                    
                    Consider:
                    - Semantic similarity (e.g., 'schema' matches 'structure')
                    - Typos (e.g., 'meta' matches 'metadata')
                    - Variations (e.g., 'user_data' matches 'userData')
                    
                    Return the exact field name or 'none' if no good match.
                """)
                
                confidence = dspy.OutputField(desc="Confidence in match (0.0-1.0)")
                reasoning = dspy.OutputField(desc="Brief explanation")
            
            # Format available fields for LLM
            fields_str = "\n".join([
                f"  - {name}: {type(value).__name__} = {str(value)[:100]}..."
                for name, value in available_fields.items()
            ])
            
            matcher = dspy.ChainOfThought(FieldMatchSignature)
            result = matcher(
                parameter_needed=param_name,
                available_fields=fields_str
            )
            
            @dataclass
            class MatchResult:
                field_name: str
                confidence: float
                reasoning: str
            
            # Parse confidence
            try:
                confidence = float(result.confidence)
                confidence = max(0.0, min(1.0, confidence))  # Clamp
            except (ValueError, TypeError, AttributeError):
                # Default to 0.5 if confidence parsing fails
                confidence = 0.5
            
            return MatchResult(
                field_name=result.best_match,
                confidence=confidence,
                reasoning=result.reasoning
            )
        
        except Exception as e:
            logger.warning(f"⚠️ LLM field matching unavailable: {e}")
            return None
    
    def _extract_from_output(self, output, param_name: str) -> Any:
        """Extract parameter from previous actor's output.
        
        Handles:
        1. EpisodeResult (from wrapped JOTTY actors) → extract from .output or tagged_outputs
        2. DSPy Prediction objects
        3. Dict-like outputs
        4. Parameter name variations (generic patterns)
        5. DataFrames, files, web content (extensible)
        
        🔑 A-TEAM FIX: Unwrap nested EpisodeResults WITHOUT recursion to prevent stack overflow!
        🆕 A-TEAM: Now uses SEMANTIC extraction instead of fuzzy matching!
        """
        # 🔍 A-TEAM DEBUG: Log extraction attempt (DEBUG level to reduce noise)
        logger.debug(f"🔍 [EXTRACT] Attempting to extract '{param_name}' from output type: {type(output)}")
        
        # 🔑 CRITICAL FIX: Unwrap nested EpisodeResults iteratively (NO RECURSION!)
        unwrapped = output
        max_unwrap = 10  # Safety limit
        unwrap_count = 0
        
        # ✅ A-TEAM FIX: Check if it's ACTUALLY an EpisodeResult, not just has 'output' attribute
        # DSPy Predictions may have 'output' field in signature, but they're not EpisodeResults!
        from ..foundation.data_structures import EpisodeResult
        
        logger.debug(f"🔍 [EXTRACT] Is EpisodeResult: {isinstance(output, EpisodeResult)}")
        
        while isinstance(unwrapped, EpisodeResult) and unwrap_count < max_unwrap:
            logger.debug(f"🔍 [EXTRACT] Unwrapping EpisodeResult #{unwrap_count+1}")
            # This is an EpisodeResult
            if unwrapped.output is not None:
                logger.debug(f"🔍 [EXTRACT] EpisodeResult.output is NOT None, unwrapping...")
                unwrapped = unwrapped.output
                unwrap_count += 1
            else:
                logger.warning(f"⚠️  EpisodeResult.output is None, no trajectory output for {param_name}")
                # EpisodeResult.output is None, try tagged_outputs
                if hasattr(unwrapped, 'tagged_outputs') and unwrapped.tagged_outputs:
                    logger.info(f"🔍 Checking {len(unwrapped.tagged_outputs)} tagged outputs for {param_name}")
                    for tagged in unwrapped.tagged_outputs:
                        if hasattr(tagged, 'content'):
                            # Recursively unwrap tagged content (but with depth limit now)
                            value = self._extract_from_output(tagged.content, param_name)
                            if value is not None:
                                logger.info(f"✅ Found {param_name} in tagged output")
                                return value
                
                # Try trajectory if available
                if hasattr(unwrapped, 'trajectory'):
                    for step in unwrapped.trajectory:
                        if step.get('step') == 'actor' and 'output' in step:
                            step_output = step['output']
                            if step_output is not None:
                                logger.info(f"🔄 Found output in trajectory, extracting '{param_name}'")
                                # Don't recurse, just set unwrapped and let the loop continue
                                unwrapped = step_output
                                unwrap_count += 1
                                break
                    else:
                        # No output in trajectory either
                        logger.warning(f"⚠️  EpisodeResult.output is None, no trajectory output for {param_name}")
                        return None
                else:
                    logger.warning(f"⚠️  EpisodeResult.output is None, cannot extract {param_name}")
                    return None
        
        if unwrap_count >= max_unwrap:
            logger.error(f"❌ Max unwrap depth ({max_unwrap}) reached for '{param_name}' - possible circular reference!")
            return None
        
        # Now unwrapped is the actual data (Prediction, dict, etc.)
        output = unwrapped
        
        # Direct attribute access
        if hasattr(output, param_name):
            return getattr(output, param_name)
        
        # Dict access
        if isinstance(output, dict) and param_name in output:
            return output[param_name]
        
        # 🆕 A-TEAM: Use user-configurable mappings (NOT hardcoded!)
        # Try pattern matching with user-defined mappings
        if param_name in self.param_mappings:
            for attr in self.param_mappings[param_name]:
                if hasattr(output, attr):
                    value = getattr(output, attr)
                    if value is not None:  # Changed: accept 0, False, empty string
                        return value
                if isinstance(output, dict) and attr in output:
                    value = output[attr]
                    if value is not None:
                        return value
        
        # 🆕 A-TEAM: Semantic extraction (LLM-based, pure agentic)
        if self.registration_orchestrator:
            try:
                value = self._semantic_extract(output, param_name)
                if value is not None:
                    return value
                else:
                    # A-TEAM FIX: This is normal - trying to extract input params from output
                    # Should be DEBUG, not ERROR
                    logger.debug(f"🔍 Semantic extraction: '{param_name}' not found in {type(output).__name__}")
                    logger.debug(f"   (This is normal - tried to extract from previous actor output)")
                    return None
            except Exception as e:
                logger.error(f"❌ Semantic extraction failed for '{param_name}': {e}")
                logger.error(f"   Fix the semantic extraction logic or actor output structure!")
                raise RuntimeError(f"Parameter extraction failed for '{param_name}' - no fallbacks allowed!") from e
        
        # No registration orchestrator - this is a configuration error
        logger.error(f"❌ RegistrationOrchestrator not available - cannot extract '{param_name}'")
        raise RuntimeError(f"RegistrationOrchestrator must be enabled for semantic extraction!")
        
        return None
    
    async def _execute_actor(
        self,
        actor_config: ActorConfig,
        task: TodoItem,
        context: str,
        kwargs: Dict,
        actor_context_dict: Optional[Dict] = None  # 🔥 A-TEAM: Actor-specific context from metadata_provider
    ) -> Any:
        """
        Execute actor with parameter resolution.
        
        🆕 DECLARATIVE MODE (A-Team FINAL): If actor_config.inputs is provided,
        use declarative resolution. Otherwise, fall back to signature introspection.
        
        🤝 COORDINATION: Check for pending feedback from other agents before execution.
        """
        actor = actor_config.agent
        
        # 🤝 NEW: Check for feedback from other agents
        if self.feedback_channel and self.feedback_channel.has_feedback(actor_config.name):
            messages = self.feedback_channel.get_for_actor(actor_config.name, clear=True)
            logger.info(f"📧 Actor '{actor_config.name}' received {len(messages)} feedback message(s)")
            
            # Format messages for injection
            feedback_context = self.feedback_channel.format_messages_for_agent(actor_config.name, messages)
            
            # Inject feedback into context for actor to see
            if 'feedback' not in kwargs:
                kwargs['feedback'] = feedback_context
            else:
                kwargs['feedback'] += "\n\n" + feedback_context
            
            logger.debug(f"📧 Feedback injected for '{actor_config.name}':\n{feedback_context[:200]}...")
        
        # Get shared context
        shared_context = kwargs.get('_shared_context', {'actor_outputs': {}})
        
        # 🔍 DEBUG: Log current state
        logger.info(f"🔍 DEBUG: Executing actor '{actor_config.name}'")
        logger.info(f"🔍 DEBUG: Available actor_outputs: {list(shared_context.get('actor_outputs', {}).keys())}")
        logger.info(f"🔍 DEBUG: _shared_context in kwargs: {'_shared_context' in kwargs}")
        
        # 🔄 AUTO-RESOLVE parameters with optional mappings override (A-Team FINAL - User Corrected)
        logger.info(f"✅ Resolving parameters for '{actor_config.name}'")
        signature = self.actor_signatures.get(actor_config.name, {})
        
        if not signature:
            # Fallback: basic kwargs
            logger.warning(f"⚠️  No signature for {actor_config.name}, using basic kwargs")
            # 🔥 A-TEAM CRITICAL FIX: Map 'goal' to 'query' for first actor!
            resolved_kwargs = {}
            for k, v in kwargs.items():
                if k in ['query', 'conversation_history', 'session_id', 'goal']:
                    resolved_kwargs[k] = v
            # Auto-map goal → query if query is missing
            if 'query' not in resolved_kwargs and 'goal' in kwargs:
                resolved_kwargs['query'] = kwargs['goal']
                logger.info(f"🔄 Auto-mapped 'goal' → 'query' for {actor_config.name}")
        else:
            # Build resolution context for mappings
            resolution_context = {
                'inputs': kwargs,  # swarm.run() kwargs
                'actor_outputs': shared_context.get('actor_outputs', {}),
                'context_providers': self.context_providers,
            }
            
            # Build resolved kwargs
            resolved_kwargs = {}
            missing_required = []
            
            for param_name, param_info in signature.items():
                # 🆕 Check if there's an explicit mapping for this parameter
                if actor_config.parameter_mappings and param_name in actor_config.parameter_mappings:
                    # Use explicit mapping (for special cases)
                    mapping_spec = actor_config.parameter_mappings[param_name]
                    logger.info(f"   🔑 Using EXPLICIT mapping: {param_name} = {mapping_spec}")
                    try:
                        value = self.resolve_input(mapping_spec, resolution_context)
                        if value is not None:
                            logger.info(f"   ✅ Resolved '{param_name}' = {str(value)[:100]}")
                        else:
                            logger.warning(f"   ⚠️  Mapping for '{param_name}' returned None!")
                    except Exception as e:
                        logger.error(f"   ❌ Error resolving '{param_name}': {e}")
                        value = None
                else:
                    # Use auto-resolution (default)
                    value = self._resolve_parameter(param_name, param_info, kwargs, shared_context)
                
                if value is not None:
                    # 🔄 A-TEAM: SMART TYPE TRANSFORMATION
                    # Check if value type matches expected type from signature
                    expected_type = param_info.get('annotation', type(None))
                    if expected_type != inspect.Parameter.empty and expected_type != type(None):
                        actual_type = type(value)
                        if actual_type != expected_type and expected_type in (dict, list, str, int, float, bool):
                            logger.debug(f"🔄 Type mismatch for '{param_name}': {actual_type.__name__} → {expected_type.__name__}")
                            try:
                                value = self.data_transformer.transform(
                                    source=value,
                                    target_type=expected_type,
                                    context=f"Parameter for {actor_config.name}",
                                    param_name=param_name
                                )
                                logger.info(f"✅ Transformed '{param_name}' to {expected_type.__name__}")
                            except Exception as e:
                                logger.warning(f"⚠️  Transformation failed for '{param_name}': {e}")
                    
                    resolved_kwargs[param_name] = value
                elif param_info['required']:
                    missing_required.append(param_name)
            
            if missing_required:
                logger.warning(f"⚠️  {actor_config.name} missing: {missing_required}")
                
                # 🔥 A-TEAM INTELLIGENT RECOVERY: Attempt to recover missing parameters
                enable_recovery = getattr(self.config, 'enable_auto_recovery', True)  # Default True
                if enable_recovery:
                    logger.info(f"🛠️  Attempting intelligent recovery for {len(missing_required)} missing parameters...")
                    recovery_result = await self._attempt_parameter_recovery(
                        actor_config=actor_config,
                        missing_params=missing_required,
                        context=kwargs,
                        shared_context=shared_context
                    )
                    
                    if recovery_result['recovered']:
                        logger.info(f"✅ Recovered {len(recovery_result['recovered'])} parameters!")
                        resolved_kwargs.update(recovery_result['recovered'])
                        # Update missing list
                        missing_required = [p for p in missing_required if p not in recovery_result['recovered']]
                    
                    if missing_required:
                        logger.error(f"❌ Still missing after recovery: {missing_required}")
                        # Return early with failure if critical parameters still missing
                        allow_partial = getattr(self.config, 'allow_partial_execution', False)
                        if not allow_partial:
                            logger.error(f"❌ Cannot execute {actor_config.name} - missing required parameters: {missing_required}")
                            # Return a minimal EpisodeResult indicating failure
                            return EpisodeResult(
                                output=None,
                                success=False,
                                trajectory=[],
                                tagged_outputs=[],
                                episode=0,
                                execution_time=0.0,
                                architect_results=[],
                                auditor_results=[],
                                agent_contributions={},
                                alerts=[f"Missing required parameters: {missing_required}"]
                            )
        
        # 🔥 A-TEAM CRITICAL FIX: Merge actor_context_dict into resolved_kwargs!
        # This is the context from metadata_provider.get_context_for_actor()
        if actor_context_dict and isinstance(actor_context_dict, dict):
            logger.info(f"🔧 Merging {len(actor_context_dict)} context items from metadata_provider into resolved_kwargs")
            for key, value in actor_context_dict.items():
                if key not in resolved_kwargs:  # Don't override existing
                    resolved_kwargs[key] = value
                    logger.debug(f"   ✅ Added '{key}' from actor_context")
                else:
                    logger.debug(f"   ⏭️  Skipped '{key}' (already in resolved_kwargs)")
        
        # 🔧 A-TEAM FIX: DON'T inject _metadata_tools as kwarg
        # The metadata tools are already available via metadata_manager
        # Injecting them as kwarg causes TypeError if forward() doesn't accept **kwargs
        if hasattr(self, 'metadata_tool_registry') and self.metadata_tool_registry:
            logger.debug(f"🔧 Metadata tools available ({len(self.metadata_tool_registry.tools)} tools) for '{actor_config.name}'")
        
        # Execute
        # 🔥 A-TEAM CRITICAL FIX: Call DSPy modules correctly!
        # DSPy modules should be called via __call__, NOT .forward()
        # The __call__ method sets up tracking and calls forward() internally
        if asyncio.iscoroutinefunction(getattr(actor, 'run', None)):
            result = await actor.run(**resolved_kwargs)
        elif asyncio.iscoroutinefunction(getattr(actor, 'arun', None)):
            result = await actor.arun(**resolved_kwargs)
        elif hasattr(actor, '__call__'):
            # ✅ CORRECT: Call actor directly (works for DSPy modules and regular callables)
            if asyncio.iscoroutinefunction(actor):
                result = await actor(**resolved_kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: actor(**resolved_kwargs))
        else:
            raise ValueError(f"Actor {actor_config.name} has no callable method")
        
        # 🔑 Store output in shared context
        shared_context['actor_outputs'][actor_config.name] = result
        kwargs['_shared_context'] = shared_context
        
        # 🔥 A-TEAM: RETRY LOGIC FOR EMPTY EPISODES
        # If actor failed or returned None, attempt intelligent recovery and retry
        if (hasattr(result, 'success') and not result.success) or \
           (hasattr(result, 'output') and result.output is None):
            logger.warning(f"⚠️ [{actor_config.name}] Failed or returned None - attempting intelligent recovery")
            
            # Extract missing parameters from error or signature
            missing_params = []
            
            # Check if result has error message with missing params
            if hasattr(result, 'error') and result.error:
                import re
                matches = re.findall(r"missing.*?'(\w+)'|(\w+).*required", str(result.error), re.IGNORECASE)
                for match in matches:
                    param = match[0] or match[1]
                    if param and param not in missing_params:
                        missing_params.append(param)
            
            # Check signature for required params not in resolved_kwargs
            if actor_config.name in self.actor_signatures:
                sig = self.actor_signatures[actor_config.name]
                if isinstance(sig, dict):
                    for param_name, param_info in sig.items():
                        if param_info.get('required') and param_name not in resolved_kwargs:
                            if param_name not in missing_params:
                                missing_params.append(param_name)
            
            if missing_params:
                logger.info(f"🔄 [RETRY] Attempting recovery for {len(missing_params)} missing params: {missing_params}")
                
                # Use intelligent recovery with routing
                recovered = await self._intelligent_recovery_with_routing(
                    actor_config,
                    missing_params,
                    {'goal': kwargs.get('goal', kwargs.get('query', '')), 'kwargs': kwargs}
                )
                
                if recovered:
                    logger.info(f"✅ [RETRY] Recovered {len(recovered)} parameters")
                    
                    # 🔥 A-TEAM: BUILD RETRY CONTEXT + REASON for agent agency!
                    retry_context = f"""
🔄 RETRY ATTEMPT - You are being re-executed with additional context and data.

📋 REASON FOR RETRY:
- Previous execution failed or returned None
- Missing parameters were identified: {', '.join(missing_params)}

✅ WHAT WE DID TO FIX IT:
- Recovered {len(recovered)} parameter(s) from dependency graph and data sources
- Parameters now available: {', '.join(recovered.keys())}

📊 RECOVERED DATA:
"""
                    for param, value in recovered.items():
                        value_preview = str(value)[:100] if value else "None"
                        retry_context += f"- {param}: {type(value).__name__} = {value_preview}\n"
                    
                    retry_context += f"""
🎯 WHAT YOU SHOULD DO NOW:
- Use the newly provided parameters: {', '.join(recovered.keys())}
- Re-analyze the query with complete context
- Generate output based on ALL available data
- Previous attempt lacked: {', '.join(missing_params)}

💡 ADDITIONAL GUIDANCE:
- All required data is now available
- Focus on producing valid, complete output
- If you still encounter issues, clearly state what's missing
"""
                    
                    # Inject retry context into actor's context
                    if 'retry_context' not in resolved_kwargs:
                        resolved_kwargs['retry_context'] = retry_context
                    
                    # Update resolved_kwargs with recovered data
                    resolved_kwargs.update(recovered)
                    
                    # Log the retry context for debugging
                    logger.info(f"📝 [RETRY CONTEXT]:\n{retry_context}")
                    
                    # RETRY actor execution WITH CONTEXT
                    logger.info(f"🔄 [RETRY] Re-executing {actor_config.name} with context + recovered data...")
                    
                    try:
                        if asyncio.iscoroutinefunction(getattr(actor, 'run', None)):
                            result = await actor.run(**resolved_kwargs)
                        elif asyncio.iscoroutinefunction(getattr(actor, 'arun', None)):
                            result = await actor.arun(**resolved_kwargs)
                        elif hasattr(actor, '__call__'):
                            if asyncio.iscoroutinefunction(actor):
                                result = await actor(**resolved_kwargs)
                            else:
                                loop = asyncio.get_event_loop()
                                result = await loop.run_in_executor(None, lambda: actor(**resolved_kwargs))
                        
                        # Update shared context with retry result
                        shared_context['actor_outputs'][actor_config.name] = result
                        kwargs['_shared_context'] = shared_context
                        
                        if hasattr(result, 'success') and result.success:
                            logger.info(f"✅ [RETRY] Retry successful for {actor_config.name}")
                        else:
                            logger.warning(f"⚠️ [RETRY] Retry failed for {actor_config.name}")
                    
                    except Exception as e:
                        logger.error(f"❌ [RETRY] Retry execution failed: {e}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                else:
                    logger.warning(f"⚠️ [RETRY] Could not recover any parameters")
            else:
                logger.warning(f"⚠️ [RETRY] No missing parameters identified for recovery")
        
        # ✅ A-TEAM: Register output in IOManager (typed outputs!)
        # 🔥 CRITICAL FIX: Extract the ACTUAL output (DSPy Prediction) from EpisodeResult wrapper!
        try:
            # 🔍 A-TEAM DEBUG: Log EVERYTHING about the result
            logger.info(f"🔍 [IOManager PREP] result type: {type(result)}")
            logger.info(f"🔍 [IOManager PREP] result has 'output': {hasattr(result, 'output')}")
            if hasattr(result, 'output'):
                logger.info(f"🔍 [IOManager PREP] result.output type: {type(result.output)}")
                logger.info(f"🔍 [IOManager PREP] result.output is None: {result.output is None}")
                if result.output and hasattr(result.output, '_store'):
                    logger.info(f"🔍 [IOManager PREP] result.output has _store with keys: {list(result.output._store.keys())}")
            if hasattr(result, 'success'):
                logger.info(f"🔍 [IOManager PREP] result.success: {result.success}")
            
            actual_output = result.output if hasattr(result, 'output') else result
            logger.info(f"🔍 [IOManager PREP] actual_output type: {type(actual_output)}")
            logger.info(f"🔍 [IOManager PREP] actual_output is None: {actual_output is None}")
            
            # 🆕 A-TEAM: Extract tagged attempts from trajectory (if available)
            tagged_attempts = []
            if hasattr(result, 'trajectory') and isinstance(result.trajectory, list):
                # Get the last trajectory entry (actor execution)
                for traj_entry in reversed(result.trajectory):
                    if isinstance(traj_entry, dict) and traj_entry.get('step') == 'actor':
                        tagged_attempts = traj_entry.get('tagged_attempts', [])
                        if tagged_attempts:
                            logger.info(f"🏷️  Retrieved {len(tagged_attempts)} tagged attempts from trajectory")
                            break
            
            self.io_manager.register_output(
                actor_name=actor_config.name,
                output=actual_output,  # ← FIXED! Pass the actual DSPy Prediction, not EpisodeResult!
                actor=actor,  # ✅ Pass actor for signature extraction!
                success=result.success if hasattr(result, 'success') else True,
                tagged_attempts=tagged_attempts  # 🆕 Pass tagged attempts!
            )
            logger.info(f"📦 Registered '{actor_config.name}' output in IOManager with {len(tagged_attempts)} tagged attempts")
        except Exception as e:
            logger.warning(f"⚠️  IOManager registration failed for '{actor_config.name}': {e}")
            import traceback as tb
            logger.warning(f"⚠️  Full traceback: {tb.format_exc()}")
        
        # =================================================================
        # 🎯 Q-LEARNING UPDATE: Natural Language Q-Table
        # =================================================================
        if hasattr(self, 'q_learner') and self.q_learner:
            try:
                # Generate natural language state description
                completed_actors = list(self.io_manager.outputs.keys()) if hasattr(self, 'io_manager') and self.io_manager else []
                state = {
                    "goal": kwargs.get('goal', '')[:100],
                    "actor": actor_config.name,
                    "completed": completed_actors,
                    "attempts": len(tagged_attempts)
                }
                
                action = {
                    "actor": actor_config.name,
                    "task": f"Execute {actor_config.name}"
                }
                
                next_state = state.copy()
                next_state["completed"] = completed_actors + [actor_config.name]
                
                # Compute reward (1.0 if Auditor passed, 0.0 otherwise)
                auditor_success = all(r.is_valid for r in result.auditor_results) if hasattr(result, 'auditor_results') and result.auditor_results else True
                reward = 1.0 if auditor_success else 0.0
                
                # Check if terminal (all actors done)
                is_terminal = len(next_state["completed"]) == len(self.actors)
                
                # Add experience (this updates Q-table AND stores in buffer)
                self.q_learner.add_experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=is_terminal
                )
                
                logger.debug(f"🎯 Q-Learning updated: {actor_config.name} reward={reward:.2f}, terminal={is_terminal}")
                
                # Get learned context for injection into actor prompts
                learned_context = self.q_learner.get_learned_context(state, action)
                if learned_context:
                    logger.debug(f"📚 Learned context ({len(learned_context)} chars):")
                    logger.debug(learned_context[:200] + "..." if len(learned_context) > 200 else learned_context)
                
            except Exception as e:
                logger.warning(f"⚠️  Q-Learning update failed: {e}")
        
        # =================================================================
        # 📊 TD(λ) LEARNING UPDATE: Temporal Difference with Eligibility Traces
        # =================================================================
        # 🔥 FIX: TD(λ) updates happen at episode end via end_episode(), not per-actor.
        # The TDLambdaLearner doesn't have an update() method - it uses:
        # - record_access() during execution  
        # - end_episode() at terminal state
        # This per-actor update was incorrect - TD(λ) is called in run() at episode end.
        
        # 🔍 DEBUG: Log after storage
        logger.info(f"🔍 DEBUG: Stored output for '{actor_config.name}'")
        logger.info(f"🔍 DEBUG: Updated actor_outputs: {list(shared_context['actor_outputs'].keys())}")
        logger.info(f"🔍 DEBUG: Output type: {type(result)}")
        if hasattr(result, '__dict__'):
            logger.info(f"🔍 DEBUG: Output fields: {list(vars(result).keys())[:10]}")
        
        # 🔑 NEW: Register output in Data Registry (AGENTIC DISCOVERY)
        if self.data_registry and self.registration_orchestrator:
            # 🎯 A-TEAM FIX: Use await (not asyncio.run) since we're in async function
            logger.info("🎯 REGISTERING OUTPUT - START")
            logger.info(f"  Actor: {actor_config.name}")
            logger.info(f"  Output type: {type(result).__name__}")
            logger.info(f"  Registry: {self.data_registry is not None}")
            logger.info(f"  Orchestrator: {self.registration_orchestrator is not None}")
            
            registration_context = {
                'task': self.todo.root_task if hasattr(self.todo, 'root_task') else None,
                'goal': kwargs.get('goal', ''),
                'iteration': len(self.trajectory),
                'actor_config': actor_config
            }
            
            try:
                # ✅ A-TEAM: Use await (we're already in async function!)
                artifact_ids = await self.registration_orchestrator.register_output(
                    actor_name=actor_config.name,
                    output=result,
                    context=registration_context
                )
                logger.info("🎯 REGISTERING OUTPUT - COMPLETE")
                logger.info(f"  Artifacts registered: {len(artifact_ids)}")
                logger.info(f"  Artifact IDs: {artifact_ids}")
            except Exception as e:
                logger.error(f"❌ Agentic registration failed: {e}")
                logger.error(f"   Actor: {actor_config.name}")
                logger.error(f"   Output type: {type(result)}")
                raise RuntimeError(
                    f"Agentic registration failed for {actor_config.name}. "
                    f"This is a critical error - fix the registration logic or actor output!"
                ) from e
        
        return result
    
