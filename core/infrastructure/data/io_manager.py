"""
IOManager - Typed Output Management for ReVal

Manages actor outputs with:
- Type safety from DSPy signatures
- Automatic tagging and discovery
- Tool exposure for agents
- No manual extraction needed
- Tagged predictions from ReAct exploration

# GENERIC: No domain-specific logic
# NO HARDCODING: Works with any actors/signatures
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Type, Callable
from datetime import datetime
import logging
import dspy

# Import TaggedOutput for special handling (GENERIC for any actor)
try:
    from jotty.data_structures import TaggedOutput
    TAGGED_OUTPUT_AVAILABLE = True
except ImportError:
    TAGGED_OUTPUT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeProvenance:
    """
     A-TEAM: Track who knew what when (per GRF MARL paper).
    
    Enables:
    - Asymmetric learning (agents learn what others don't know)
    - Information value estimation (for Nash comms)
    - Credit assignment (who contributed what knowledge)
    """
    source_agent: str
    knowledge_type: str  # 'metadata', 'tool_result', 'actor_output', 'shared_context'
    timestamp: float
    keys_accessed: List[str] = field(default_factory=list)
    data_summary: str = ""
    was_useful: Optional[bool] = None  # Set post-hoc based on outcome
    
    def to_dict(self) -> Dict:
        return {
            'source': self.source_agent,
            'type': self.knowledge_type,
            'time': self.timestamp,
            'keys': self.keys_accessed,
            'summary': self.data_summary[:100],
            'useful': self.was_useful
        }


@dataclass
class ActorOutput:
    """
    Typed output from a single actor WITH tagged attempts.
    
    Extracted automatically from DSPy signature output fields.
    Tagged attempts allow filtering 'answer' vs 'error' vs 'exploratory'.
    
     A-TEAM: Includes knowledge provenance tracking.
    """
    actor_name: str
    output_fields: Dict[str, Any]  # field_name -> value
    tagged_attempts: List[Any] = field(default_factory=list)  # TaggedAttempt objects
    signature: Optional[Type[dspy.Signature]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # A-TEAM: Knowledge provenance
    provenance: List[KnowledgeProvenance] = field(default_factory=list)
    
    def get(self, field_name: str, default: Any = None) -> Any:
        """Get output field value."""
        return self.output_fields.get(field_name, default)
    
    def __getitem__(self, field_name: str) -> Any:
        """Allow dict-like access: output['field_name']"""
        return self.output_fields[field_name]
    
    def __contains__(self, field_name: str) -> bool:
        """Check if field exists: 'field_name' in output"""
        return field_name in self.output_fields
    
    def get_answers(self) -> List[Any]:
        """Get only successful attempts (tag='answer')."""
        return [a for a in self.tagged_attempts if hasattr(a, 'is_answer') and a.is_answer()]
    
    def get_errors(self) -> List[Any]:
        """Get only failed attempts (tag='error')."""
        return [a for a in self.tagged_attempts if hasattr(a, 'is_error') and a.is_error()]
    
    def get_exploratory(self) -> List[Any]:
        """Get only exploratory attempts."""
        return [a for a in self.tagged_attempts if hasattr(a, 'is_exploratory') and a.is_exploratory()]
    
    def hide_non_answers(self) -> 'ActorOutput':
        """
        Return copy with only 'answer' attempts visible.
        
        This is critical for swarm coordination:
        - Downstream actors should only see successful attempts
        - Failed/exploratory attempts shouldn't pollute their context
        """
        return ActorOutput(
            actor_name=self.actor_name,
            output_fields=self.output_fields,
            tagged_attempts=self.get_answers(),  # Only answers
            signature=self.signature,
            timestamp=self.timestamp,
            success=self.success,
            error=self.error,
            metadata=self.metadata
        )


@dataclass
class ReValSwarmResult:
    """
    Complete result from SwarmReVal execution.
    
    Provides typed access to all outputs and metadata.
    """
    success: bool
    final_output: Any
    actor_outputs: Dict[str, ActorOutput]  # actor_name -> ActorOutput
    trajectory: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def get_actor_output(self, actor_name: str) -> Optional[ActorOutput]:
        """Get output from specific actor."""
        return self.actor_outputs.get(actor_name)
    
    def get_field(self, actor_name: str, field_name: str, default: Any = None) -> Any:
        """Get specific field from specific actor."""
        actor_output = self.actor_outputs.get(actor_name)
        if actor_output:
            return actor_output.get(field_name, default)
        return default
    
    def list_actors(self) -> List[str]:
        """List all actors that produced outputs."""
        return list(self.actor_outputs.keys())
    
    def list_fields(self, actor_name: str) -> List[str]:
        """List all output fields from specific actor."""
        actor_output = self.actor_outputs.get(actor_name)
        if actor_output:
            return list(actor_output.output_fields.keys())
        return []


class IOManager:
    """
    Manages actor outputs with type safety and discovery.
    
    Features:
    - Automatic output registration from DSPy signatures
    - Typed access to outputs
    - Discovery tools for agents
    - Tool exposure for DSPy actors
    - No manual extraction needed
    
# GENERIC: Works with any actors/signatures
# NO HARDCODING: Discovers structure automatically
    """
    
    def __init__(self) -> None:
        self.outputs: Dict[str, ActorOutput] = {}  # actor_name -> ActorOutput
        self.execution_order: List[str] = []  # Track execution order
        logger.info(" IOManager initialized - typed output management enabled")
    
    def register_output(self, actor_name: str, output: Any, actor: Optional[Any] = None, signature: Optional[Type[dspy.Signature]] = None, success: bool = True, error: Optional[str] = None, tagged_attempts: Optional[List[Any]] = None) -> Any:
        """
        Register actor output automatically WITH tagged attempts.
        
        Extracts output fields from DSPy output based on signature.
        Stores tagged attempts for filtering (answer/error/exploratory).
        
        Args:
            actor_name: Name of the actor
            output: Raw output from actor.forward()
            actor: Actor instance (for automatic signature extraction)
            signature: DSPy signature (optional if actor provided)
            success: Whether execution succeeded
            error: Error message if failed
            tagged_attempts: List of TaggedAttempt objects from trajectory parser
        """
        # A-TEAM DEBUG: Log EVERYTHING received
        logger.info(f" [IOManager RECV] actor_name: {actor_name}")
        logger.info(f" [IOManager RECV] output type: {type(output)}")
        logger.info(f" [IOManager RECV] output is None: {output is None}")
        logger.info(f" [IOManager RECV] success: {success}")
        logger.info(f" [IOManager RECV] tagged_attempts: {len(tagged_attempts) if tagged_attempts else 0}")
        if output and hasattr(output, '_store'):
            logger.info(f" [IOManager RECV] output has _store with keys: {list(output._store.keys())}")
        if output and hasattr(output, '__dict__'):
            logger.info(f" [IOManager RECV] output.__dict__ keys: {list(output.__dict__.keys())[:10]}")
        
        # Extract signature from actor if not provided
        if signature is None and actor is not None:
            signature = self._extract_signature_from_actor(actor)
        
        # Extract output fields
        output_fields = self._extract_output_fields(output, signature)
        
        # A-TEAM DEBUG: Log extraction results
        logger.info(f" [IOManager EXTRACT] Extracted {len(output_fields)} fields: {list(output_fields.keys())}")
        if output_fields:
            for key, value in list(output_fields.items())[:5]:  # Show first 5
                value_preview = str(value)[:100] if value is not None else "None"
                logger.info(f" [IOManager EXTRACT] - {key}: {type(value).__name__} = {value_preview}")
        
        # Create ActorOutput WITH tagged attempts
        actor_output = ActorOutput(
            actor_name=actor_name,
            output_fields=output_fields,
            tagged_attempts=tagged_attempts or [],  # NEW: Store tagged attempts
            signature=signature,
            timestamp=datetime.now(),
            success=success,
            error=error
        )
        
        # Store
        self.outputs[actor_name] = actor_output
        if actor_name not in self.execution_order:
            self.execution_order.append(actor_name)
        
        logger.info(
            f" Registered output from '{actor_name}': "
            f"{len(output_fields)} fields, {len(tagged_attempts or [])} tagged attempts, success={success}"
        )
        logger.debug(f"   Fields: {list(output_fields.keys())}")
    
    def _extract_signature_from_actor(self, actor: Any) -> Optional[Type[dspy.Signature]]:
        """
        Extract DSPy signature from actor automatically.
        
# GENERIC: Works with ANY DSPy module structure
        
        Strategies:
        1. actor.resolver.signature (ChainOfThought)
        2. actor.predictor.signature (Predict)
        3. actor.generator.signature (ReAct)
        4. actor.signature (direct)
        5. Introspect forward() return annotation
        """
        # Strategy 1: actor.resolver.signature (ChainOfThought)
        if hasattr(actor, 'resolver') and hasattr(actor.resolver, 'signature'):
            logger.debug(f"   Extracted signature from actor.resolver")
            return actor.resolver.signature
        
        # Strategy 2: actor.predictor.signature (Predict)
        if hasattr(actor, 'predictor') and hasattr(actor.predictor, 'signature'):
            logger.debug(f"   Extracted signature from actor.predictor")
            return actor.predictor.signature
        
        # Strategy 3: actor.generator.signature (ReAct)
        if hasattr(actor, 'generator') and hasattr(actor.generator, 'signature'):
            logger.debug(f"   Extracted signature from actor.generator")
            return actor.generator.signature
        
        # Strategy 4: actor.generate.signature (SQLGenerator pattern)
        if hasattr(actor, 'generate') and hasattr(actor.generate, 'signature'):
            logger.debug(f"   Extracted signature from actor.generate")
            return actor.generate.signature
        
        # Strategy 5: actor.signature (direct)
        if hasattr(actor, 'signature'):
            logger.debug(f"   Extracted signature from actor.signature")
            return actor.signature
        
        # Strategy 6: Introspect forward() return type annotation
        if hasattr(actor, 'forward'):
            import inspect
            sig = inspect.signature(actor.forward)
            if sig.return_annotation != inspect.Parameter.empty:
                logger.debug(f"   Extracted signature from forward() annotation")
                return sig.return_annotation
        
        logger.debug(f" Could not extract signature from {type(actor).__name__}")
        return None
    
    def _extract_output_fields(
        self,
        output: Any,
        signature: Optional[Type[dspy.Signature]] = None
    ) -> Dict[str, Any]:
        """
        Extract output fields from DSPy output.
        
        Handles:
        - TaggedSQLOutput (special handling for ReAct exploration)
        - DSPy Prediction objects
        - Dict outputs
        - Dataclass outputs
        - Plain objects with attributes
        """
        output_fields = {}
        
        # Strategy 0: TaggedOutput (GENERIC handling for ReAct exploration)
        if TAGGED_OUTPUT_AVAILABLE and isinstance(output, TaggedOutput):
            best = output.get_best_attempt()
            
            if best and best.tag == 'correct':
                output_fields = {
                    # GENERIC fields (works for ANY actor)
                    'final_output': best.output,  # Could be query, code, config, etc.
                    'execution_result': best.execution_result,
                    'execution_status': best.execution_status,
                    'all_attempts': output.all_attempts,  # Keep for learning
                    'reasoning': output.reasoning,
                    'explanation': output.explanation,
                    'validation_notes': output.validation_notes,
                    
                    # BACKWARD COMPATIBILITY: For SQL actors
                    'sql_query': best.output if isinstance(best.output, str) else str(best.output),
                    'query_explanation': output.explanation
                }
                
                logger.info(f" [TAGGED EXTRACTION] Extracted output with tag='{best.tag}'")
                logger.info(f"   Total attempts: {len(output.all_attempts)}")
                logger.info(f"   Successful attempts: {len(output.get_correct_attempts())}")
                logger.info(f"   Failed attempts: {len(output.get_wrong_attempts())}")
                
                return output_fields
            else:
                # A-TEAM CRITICAL FIX: When no attempts or all failed, use TaggedOutput's direct fields!
                # SQLGenerator populates final_output even when attempts=[] by extracting from _store
                logger.warning(f" [TAGGED EXTRACTION] No correct attempts found (total: {len(output.all_attempts)})")
                logger.info(f" [FALLBACK] Checking TaggedOutput direct fields: final_output={output.final_output is not None}")
                
                # Use TaggedOutput's direct fields (populated from _store)
                output_fields = {
                    'final_output': output.final_output,  # ← This is populated from _store in SQLGenerator!
                    'execution_result': output.final_result if hasattr(output, 'final_result') else (best.error_message if best else "No attempts"),
                    'execution_status': 'success' if output.final_output else 'failed',
                    'all_attempts': output.all_attempts,
                    'reasoning': output.reasoning,
                    'explanation': output.explanation,
                    'validation_notes': output.validation_notes,
                    
                    # Backward compatibility
                    'sql_query': output.final_output if isinstance(output.final_output, str) else "",
                    'query_explanation': output.explanation
                }
                
                if output.final_output:
                    logger.info(f" [FALLBACK SUCCESS] Extracted from TaggedOutput.final_output: {len(str(output.final_output))} chars")
                else:
                    logger.warning(f" [FALLBACK FAILED] TaggedOutput.final_output is None/empty!")
                
                return output_fields
        
        # Strategy 1: DSPy Prediction (has _store)
        # A-TEAM CRITICAL FIX: DSPy Prediction stores output in _store dict!
        if hasattr(output, '_store') and isinstance(output._store, dict):
            output_fields = output._store.copy()
            logger.debug(f" Extracted {len(output_fields)} fields from DSPy Prediction._store")
            return output_fields
        
        # Strategy 2: Dict
        if isinstance(output, dict):
            output_fields = output.copy()
            logger.debug(f" Extracted {len(output_fields)} fields from dict")
            return output_fields
        
        # Strategy 3: Dataclass
        if hasattr(output, '__dataclass_fields__'):
            output_fields = {k: getattr(output, k) for k in output.__dataclass_fields__}
            logger.debug(f" Extracted {len(output_fields)} fields from dataclass")
            return output_fields
        
        # Strategy 4: Object attributes
        if hasattr(output, '__dict__'):
            for key, value in output.__dict__.items():
                if not key.startswith('_'):  # Skip private attributes (except _store which we handled)
                    output_fields[key] = value
            logger.debug(f" Extracted {len(output_fields)} fields from object attributes")
            return output_fields
        
        logger.warning(f" Could not extract fields from {type(output).__name__}")
        return output_fields

    
    def list_available(self) -> Dict[str, List[str]]:
        """
        List all available outputs.
        
        Returns:
            Dict mapping actor_name -> list of output field names
        """
        return {
            actor_name: list(output.output_fields.keys())
            for actor_name, output in self.outputs.items()
        }
    
    def as_tools(self) -> Dict[str, Callable]:
        """
        Expose outputs as DSPy tools.
        
        Agents can call:
        - get_output(actor_name, field_name) -> value
        - list_outputs() -> Dict[actor, fields]
        
        Returns:
            Dict of tool_name -> callable
        """
        def get_output(actor_name: str, field_name: str) -> Any:
            """Get output field from specific actor."""
            return self.get(actor_name, field_name)
        
        def list_outputs() -> Dict[str, List[str]]:
            """List all available outputs."""
            return self.list_available()
        
        return {
            'get_output': get_output,
            'list_outputs': list_outputs,
        }
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Convert to dict for serialization.
        
        Returns:
            Dict mapping actor_name -> output_fields
        """
        return {
            actor_name: output.output_fields
            for actor_name, output in self.outputs.items()
        }
    
    def get_all_outputs(self) -> Dict[str, ActorOutput]:
        """Get all actor outputs."""
        return self.outputs.copy()

    def get_output_fields(self, actor_name: str) -> Dict[str, Any]:
        """
        Get output fields from a specific actor.

        Returns:
            Dict mapping field names to their values.
            Empty dict if actor not found.
        """
        actor_output = self.outputs.get(actor_name)
        if actor_output:
            return actor_output.output_fields.copy()
        return {}

    def clear(self) -> None:
        """Clear all outputs."""
        self.outputs.clear()
        self.execution_order.clear()
        logger.info(" IOManager cleared")
    
    def __repr__(self) -> str:
        return (
            f"IOManager("
            f"actors={len(self.outputs)}, "
            f"execution_order={self.execution_order})"
        )

