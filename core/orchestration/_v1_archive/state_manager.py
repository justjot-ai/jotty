"""
StateManager Component - Extracted from Conductor
=================================================

Handles all state management, introspection, and output registration.

Responsibilities:
1. State introspection (_get_current_state, _get_available_actions)
2. Actor signature analysis (_introspect_actor_signature)
3. Output type detection (_detect_output_type, _extract_schema)
4. Output preview/tagging (_generate_preview, _generate_tags)
5. Registry management (_register_output_in_registry, get_actor_outputs)

Part of Conductor refactoring to separate concerns.
"""

import inspect
import json
import re
import traceback
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, TYPE_CHECKING
from datetime import datetime
from pathlib import Path

# Import required types
from ..foundation.data_structures import OutputTag
from ..data.io_manager import ActorOutput
from ..data.data_registry import DataArtifact
from .roadmap import TaskStatus  # For _get_current_state

# Conditional imports for type hints
if TYPE_CHECKING:
    from .conductor import TodoItem
    from ..foundation.agent_config import AgentConfig
    from ..data.data_registry import DataRegistry
    from ..data.io_manager import IOManager
    from .roadmap import MarkovianTODO
    ActorConfig = AgentConfig  # Alias for backward compatibility
else:
    TodoItem = Any
    AgentConfig = Any
    DataRegistry = Any
    IOManager = Any
    MarkovianTODO = Any
    ActorConfig = Any  # Runtime fallback

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class StateManager:
    """
    Manages state introspection, actor signatures, and output registration.

    Extracted from Conductor to separate state management concerns.
    """

    def __init__(
        self,
        io_manager,
        data_registry,
        metadata_provider,
        context_guard,
        shared_context: Dict[str, Any],
        todo,
        trajectory: List[Dict[str, Any]],
        config,
        # Additional dependencies
        actors: Dict = None,
        actor_signatures: Dict = None,
        metadata_fetcher = None,
        param_resolver = None
    ):
        """
        Initialize StateManager with required dependencies.

        Args:
            io_manager: IOManager instance for accessing actor outputs
            data_registry: DataRegistry for output registration
            metadata_provider: Provider for metadata context
            context_guard: SmartContextGuard for context management
            shared_context: Shared context dictionary
            todo: MarkovianTODO instance
            trajectory: Execution trajectory list
            config: JottyConfig instance
            actors: Dictionary of actor instances
            actor_signatures: Dictionary of actor signatures
            metadata_fetcher: Metadata fetcher instance
            param_resolver: Parameter resolver instance
        """
        self.io_manager = io_manager
        self.data_registry = data_registry
        self.metadata_provider = metadata_provider
        self.context_guard = context_guard
        self.shared_context = shared_context
        self.todo = todo
        self.trajectory = trajectory
        self.config = config
        self.actors = actors or {}
        self.actor_signatures = actor_signatures or {}
        self.metadata_fetcher = metadata_fetcher
        self.param_resolver = param_resolver

        logger.info("✅ StateManager initialized")

    def _get_current_state(self) -> Dict[str, Any]:
        """
        Get RICH current state for Q-prediction.
        
        🔥 A-TEAM CRITICAL: State must capture semantic context!
        
        Includes:
        1. Query semantics (what user asked)
        2. Metadata context (tables, columns, partitions)
        3. Error patterns (what failed)
        4. Tool usage (what worked)
        5. Actor outputs (what was produced)
        """
        state = {
            # === 1. TASK PROGRESS ===
            'todo': {
                'completed': len(self.todo.completed),
                'pending': len([t for t in self.todo.subtasks.values() if t.status == TaskStatus.PENDING]),
                'failed': len(self.todo.failed_tasks)
            },
            'trajectory_length': len(self.trajectory),
            'recent_outcomes': [t.get('passed', False) for t in self.trajectory[-5:]]
        }
        
        # === 2. QUERY CONTEXT (CRITICAL!) ===
        # Try multiple sources for query
        query = None
        
        # Source 1: SharedContext
        if hasattr(self, 'shared_context') and self.shared_context:
            query = self.shared_context.get('query') or self.shared_context.get('goal')
        
        # Source 2: Context guard buffers (if available)
        if not query and hasattr(self, 'context_guard') and self.context_guard:
            # SmartContextGuard stores content in buffers
            for priority_buffer in self.context_guard.buffers.values():
                for key, content, _ in priority_buffer:
                    if key == 'ROOT_GOAL':
                        query = content
                        break
                if query:
                    break
        
        # Source 3: TODO root task
        if not query and hasattr(self, 'todo') and self.todo:
            query = self.todo.root_task
        
        if query:
            state['query'] = str(query)[:200]
        
        # === 3. METADATA CONTEXT ===
        if hasattr(self, 'shared_context') and self.shared_context:
            # Get table info
            tables = self.shared_context.get('table_names') or self.shared_context.get('relevant_tables')
            if tables:
                state['tables'] = tables if isinstance(tables, list) else [str(tables)]
            
            # Get filter info
            filters = self.shared_context.get('filters') or self.shared_context.get('filter_conditions')
            if filters:
                state['filters'] = filters
            
            # Get resolved terms
            resolved = self.shared_context.get('resolved_terms')
            if resolved:
                if isinstance(resolved, dict):
                    state['resolved_terms'] = list(resolved.keys())[:5]
        
        # === 4. ACTOR OUTPUT CONTEXT ===
        if hasattr(self, 'io_manager') and self.io_manager:
            all_outputs = self.io_manager.get_all_outputs()
            output_summary = {}
            for actor_name, output in all_outputs.items():
                if hasattr(output, 'output_fields') and output.output_fields:
                    output_summary[actor_name] = list(output.output_fields.keys())
            if output_summary:
                state['actor_outputs'] = output_summary
        
        # === 5. ERROR PATTERNS (CRITICAL FOR LEARNING!) ===
        if self.trajectory:
            errors = []
            columns_tried = []
            working_column = None
            
            for step in self.trajectory:
                # Check for errors in trajectory
                if step.get('error'):
                    err = step['error']
                    if 'COLUMN_NOT_FOUND' in str(err):
                        # Extract column name from error
                        import re
                        match = re.search(r"Column '(\w+)' cannot be resolved", str(err))
                        if match:
                            col = match.group(1)
                            columns_tried.append(col)
                            errors.append({'type': 'COLUMN_NOT_FOUND', 'column': col})
                
                # Check for success
                if step.get('passed') and step.get('tool_calls'):
                    for tc in step.get('tool_calls', []):
                        if isinstance(tc, dict) and tc.get('success'):
                            # Extract working column if SQL-related
                            if 'query' in str(tc):
                                # Try to find date column that worked
                                query = str(tc.get('query', ''))
                                for possible_col in ['dl_last_updated', 'dt', 'date', 'created_at']:
                                    if possible_col in query.lower():
                                        working_column = possible_col
                                        break
            
            if errors:
                state['errors'] = errors[-5:]  # Last 5 errors
            if columns_tried:
                state['columns_tried'] = list(dict.fromkeys(columns_tried))  # Unique
            if working_column:
                state['working_column'] = working_column
                state['error_resolution'] = f"use {working_column} instead of {','.join(columns_tried[:3])}"
        
        # === 6. TOOL USAGE PATTERNS ===
        successful_tools = []
        failed_tools = []
        tool_calls = []
        
        for step in self.trajectory:
            if step.get('tool_calls'):
                for tc in step.get('tool_calls', []):
                    tool_name = tc.get('tool') if isinstance(tc, dict) else str(tc)
                    tool_calls.append(tool_name)
                    
                    if isinstance(tc, dict):
                        if tc.get('success'):
                            successful_tools.append(tool_name)
                        else:
                            failed_tools.append(tool_name)
        
        if tool_calls:
            state['tool_calls'] = tool_calls[-10:]
        if successful_tools:
            state['successful_tools'] = list(dict.fromkeys(successful_tools))
        if failed_tools:
            state['failed_tools'] = list(dict.fromkeys(failed_tools))
        
        # === 7. CURRENT ACTOR ===
        if self.trajectory and self.trajectory[-1].get('actor'):
            state['current_actor'] = self.trajectory[-1]['actor']
        
        # === 8. VALIDATION CONTEXT ===
        for step in self.trajectory[-3:]:  # Last 3 steps
            if step.get('architect_confidence'):
                state['architect_confidence'] = step['architect_confidence']
            if step.get('auditor_result'):
                state['auditor_result'] = step['auditor_result']
            if step.get('validation_passed') is not None:
                state['validation_passed'] = step['validation_passed']
        
        # === 9. EXECUTION STATS ===
        state['attempts'] = len(self.trajectory)
        state['success'] = any(t.get('passed', False) for t in self.trajectory)
        
        return state
    

    def _get_available_actions(self) -> List[Dict[str, Any]]:
        """Get available actions for exploration."""
        actions = []
        for name, config in self.actors.items():
            actions.append({
                'actor': name,
                'action': 'execute',
                'enabled': config.enabled
            })
        return actions
    
    def _introspect_actor_signature(self, actor_config: ActorConfig):
        """
        Introspect actor's signature for auto-resolution.
        🔥 A-TEAM FIX: Use DSPy Signature object, NOT forward() method directly!
        """
        actor = actor_config.agent
        
        # Strategy 1: DSPy module with signature attribute (BEST)
        if hasattr(actor, 'signature') and hasattr(actor.signature, 'input_fields'):
            try:
                params = {}
                for field_name in actor.signature.input_fields:
                    # 🔥 A-TEAM CRITICAL FIX: Extract REAL type from DSPy field!
                    field = actor.signature.input_fields[field_name]
                    field_type = Any  # default
                    
                    # DSPy fields have a 'annotation' or '_type' attribute
                    if hasattr(field, 'annotation'):
                        field_type = field.annotation
                    elif hasattr(field, '_type'):
                        field_type = field._type
                    elif hasattr(field, '__annotations__'):
                        # Check class annotations
                        for cls in type(field).__mro__:
                            if hasattr(cls, '__annotations__') and field_name in cls.__annotations__:
                                field_type = cls.__annotations__[field_name]
                                break
                    
                    # If still Any, try to infer from field's json_schema_extra or desc
                    if field_type is Any:
                        logger.debug(f"   ⚠️  Could not extract type for '{field_name}', defaulting to Any")
                    
                    params[field_name] = {
                        'annotation': field_type,
                        'default': inspect.Parameter.empty,
                        'required': True
                    }
                self.actor_signatures[actor_config.name] = params
                self.dependency_graph[actor_config.name] = []
                logger.info(f"  📋 {actor_config.name}: {len(params)} params (DSPy signature), deps=[]")
                return
            except Exception as e:
                logger.warning(f"⚠️  Failed to extract DSPy signature for {actor_config.name}: {e}")
        
        # Strategy 2: Inspect forward method (WITHOUT calling it)
        forward_method = getattr(actor, 'forward', None)
        if forward_method:
            try:
                sig = inspect.signature(forward_method)
                params = {}
                dependencies = []
                
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue
                    
                    params[param_name] = {
                        'annotation': param.annotation,
                        'default': param.default,
                        'required': param.default == inspect.Parameter.empty
                    }
                
                self.actor_signatures[actor_config.name] = params
                self.dependency_graph[actor_config.name] = dependencies
                
                logger.info(f"  📋 {actor_config.name}: {len(params)} params, deps={dependencies}")
                return
            except Exception as e:
                logger.warning(f"⚠️  Failed to introspect forward method for {actor_config.name}: {e}")
        
        # Strategy 3: Fallback - inspect __call__
        if hasattr(actor, '__call__'):
            try:
                sig = inspect.signature(actor.__call__)
                params = {}
                for param_name, param in sig.parameters.items():
                    if param_name in ('self', 'args', 'kwargs'):
                        continue
                    params[param_name] = {
                        'annotation': param.annotation,
                        'default': param.default,
                        'required': param.default == inspect.Parameter.empty
                    }
                self.actor_signatures[actor_config.name] = params
                self.dependency_graph[actor_config.name] = []
                logger.info(f"  📋 {actor_config.name}: {len(params)} params (__call__), deps=[]")
                return
            except Exception as e:
                logger.warning(f"⚠️  Failed to introspect __call__ for {actor_config.name}: {e}")
        
        # Fallback: No signature
        logger.warning(f"⚠️  Actor {actor_config.name} has no inspectable signature")
        self.actor_signatures[actor_config.name] = {}
        self.dependency_graph[actor_config.name] = []
    
    def _detect_output_type(self, output: Any) -> str:
        """Auto-detect output type."""
        if hasattr(output, 'to_dict'):  # DataFrame
            return 'dataframe'
        elif isinstance(output, str):
            if len(output) > 100:
                if '<html' in output[:100].lower():
                    return 'html'
                elif '#' in output[:100]:
                    return 'markdown'
            return 'text'
        elif isinstance(output, bytes):
            return 'binary'
        elif isinstance(output, dict):
            return 'json'
        elif hasattr(output, 'output'):  # EpisodeResult
            return 'episode_result'
        elif hasattr(output, '__dict__'):
            return 'prediction'
        return 'unknown'
    

    def _extract_schema(self, output: Any) -> Dict[str, str]:
        """Extract schema from output."""
        schema = {}
        
        # Handle EpisodeResult
        if hasattr(output, 'output') and hasattr(output, 'success'):
            if output.output is not None:
                return self._extract_schema(output.output)
            return {}
        
        if hasattr(output, '__dict__'):
            for field_name, field_value in vars(output).items():
                if not field_name.startswith('_'):
                    schema[field_name] = type(field_value).__name__
        
        elif isinstance(output, dict):
            for key, value in output.items():
                schema[key] = type(value).__name__
        
        elif hasattr(output, 'columns'):  # DataFrame
            schema = {col: 'column' for col in output.columns}
        
        return schema
    

    def _generate_preview(self, output: Any) -> str:
        """Generate preview of output."""
        try:
            if isinstance(output, str):
                return output[:200]
            elif hasattr(output, '__str__'):
                return str(output)[:200]
            elif hasattr(output, 'head'):  # DataFrame
                return str(output.head(3))[:200]
            return f"<{type(output).__name__}>"
        except (AttributeError, TypeError, ValueError, Exception) as e:
            # Preview generation failed, return safe fallback
            logger.debug(f"Preview generation failed: {e}")
            return "<preview unavailable>"
    

    def _generate_tags(self, actor_name: str, output: Any, output_type: str) -> List[str]:
        """Generate semantic tags for output."""
        tags = [output_type, actor_name.lower()]
        
        # Handle EpisodeResult
        if hasattr(output, 'output') and hasattr(output, 'success'):
            if output.output is not None:
                return self._generate_tags(actor_name, output.output, output_type)
            return tags
        
        # Add field names as tags
        if hasattr(output, '__dict__'):
            field_names = [f for f in vars(output).keys() if not f.startswith('_')]
            tags.extend(field_names[:5])  # Top 5 fields
        
        elif isinstance(output, dict):
            tags.extend(list(output.keys())[:5])
        
        return tags
    

    def _register_output_in_registry(self, actor_name: str, output: Any):
        """Register output in Data Registry."""
        try:
            # Detect type
            output_type = self._detect_output_type(output)
            
            # Extract schema
            schema = self._extract_schema(output)
            
            # Generate tags
            tags = self._generate_tags(actor_name, output, output_type)
            
            # Generate preview
            preview = self._generate_preview(output)
            
            # Calculate size
            try:
                size = len(str(output))
            except (TypeError, AttributeError):
                # Size calculation failed, use 0
                size = 0
            
            # Create artifact
            artifact = DataArtifact(
                id=f"{actor_name}_{int(time.time() * 1000)}",
                name=actor_name,
                source_actor=actor_name,
                data=output,
                data_type=output_type,
                schema=schema,
                tags=tags,
                description=f"Output from {actor_name}",
                timestamp=time.time(),
                depends_on=[],
                size=size,
                preview=preview
            )
            
            # Register
            self.data_registry.register(artifact)
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to register output in registry: {e}")
    

    def _register_output_in_registry_fallback(self, actor_name: str, output: Any):
        """
        DEPRECATED: Fallback registration removed - use semantic registration only!
        
        This method is kept for backwards compatibility but will raise an error.
        All registration MUST go through the agentic RegistrationOrchestrator.
        """
        raise RuntimeError(
            f"❌ Fallback registration called for {actor_name}! "
            f"This is not allowed - use RegistrationOrchestrator for semantic registration."
        )
    
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
        if isinstance(result, dict):
            if result.get('success', True):
                return True, 1.0, "Auditor passed"
            else:
                return False, 0.0, result.get('error', 'Auditor failed')
        
        # Default: assume success
        return True, 0.8, "Result received"
    

    def get_actor_outputs(self) -> Dict[str, Any]:
        """
        Extract all actor outputs from trajectory.
        
        Returns:
            Dict mapping actor_name -> latest output
        """
        outputs = {}
        for step in self.trajectory:
            actor = step.get('actor')
            if actor and 'actor_output' in step:
                outputs[actor] = step['actor_output']
        return outputs
    

    def get_output_from_actor(self, actor_name: str, field: Optional[str] = None) -> Any:
        """
        Get specific output from an actor.
        
        Args:
            actor_name: Name of the actor
            field: Optional field to extract from output dict
        
        Returns:
            Actor output or specific field value
        """
        # Search from most recent to oldest
        for step in reversed(self.trajectory):
            if step.get('actor') == actor_name and 'actor_output' in step:
                output = step['actor_output']
                if field and isinstance(output, dict):
                    return output.get(field)
                return output
        return None
    
