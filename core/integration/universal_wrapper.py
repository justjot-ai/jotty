"""
Jotty v6.3 - Universal Wrapper
==============================

A-Team Critical Fix: Make JOTTY generic enough for ANY agent.

Supports:
- DSPy modules
- Plain Python functions
- Class methods  
- Async functions
- LangChain/AutoGen agents
- ANY callable

No required prompts, no required tools - sensible defaults provided.
"""

import asyncio
import warnings
from typing import Union, Callable, Dict, Any, List
from pathlib import Path
import yaml
import dspy

from ..foundation.data_structures import SwarmLearningConfig
from .framework_decorators import ContextGuard


# =============================================================================
# TYPE ALIASES
# =============================================================================

Actor = Union[Callable, dspy.Module, Any]
ValidatorSpec = Union[str, Callable, List[str], List[Callable], None]
ConfigSpec = Union[SwarmConfig, Dict, str, None]


# =============================================================================
# SMART CONFIG - TIERED CONFIGURATION
# =============================================================================

class SmartConfig:
    """
    Auto-configures JOTTY based on simple settings.
    
    Three levels:
    1. Simple: Just quality + model (3 settings)
    2. Intermediate: 10-15 key settings
    3. Advanced: Full 90+ parameters
    """
    
    # Quality presets (only parameters that exist in SwarmConfig)
    PRESETS = {
        "fast": {
            "max_actor_iters": 30,
            "max_eval_iters": 10,
            "episodic_capacity": 500,
            "semantic_capacity": 200,
            "procedural_capacity": 100,
            "meta_capacity": 50,
            "causal_capacity": 100,
            "enable_multi_round": False,
            "max_validation_rounds": 1,
            "enable_causal_learning": False,
            "offline_update_interval": 100,
            "consolidation_threshold": 100,
            "verbose": 0
        },
        "balanced": {
            "max_actor_iters": 100,
            "max_eval_iters": 50,
            "episodic_capacity": 2000,
            "semantic_capacity": 1000,
            "procedural_capacity": 400,
            "meta_capacity": 200,
            "causal_capacity": 300,
            "enable_multi_round": True,
            "max_validation_rounds": 3,
            "enable_causal_learning": True,
            "offline_update_interval": 25,
            "consolidation_threshold": 50,
            "verbose": 1
        },
        "thorough": {
            "max_actor_iters": 200,
            "max_eval_iters": 100,
            "episodic_capacity": 10000,
            "semantic_capacity": 5000,
            "procedural_capacity": 2000,
            "meta_capacity": 1000,
            "causal_capacity": 1500,
            "enable_multi_round": True,
            "max_validation_rounds": 5,
            "enable_causal_learning": True,
            "offline_update_interval": 10,
            "consolidation_threshold": 25,
            "verbose": 2
        }
    }
    
    # Model-specific configurations
    MODEL_CONFIGS = {
        "gpt-4.1": {
            "max_context_tokens": 28000,
            "chunk_size": 6000,
            "chunk_overlap": 400,
            "system_prompt_budget": 4000,
            "current_input_budget": 8000,
            "trajectory_budget": 8000,
            "tool_output_budget": 4000,
            "min_memory_budget": 2000,
            "max_memory_budget": 8000
        },
        "gpt-4o": {
            "max_context_tokens": 120000,
            "chunk_size": 20000,
            "chunk_overlap": 2000,
            "system_prompt_budget": 10000,
            "current_input_budget": 40000,
            "trajectory_budget": 40000,
            "tool_output_budget": 20000,
            "min_memory_budget": 10000,
            "max_memory_budget": 40000
        },
        "gpt-4o-mini": {
            "max_context_tokens": 120000,
            "chunk_size": 20000,
            "chunk_overlap": 2000,
            "system_prompt_budget": 10000,
            "current_input_budget": 40000,
            "trajectory_budget": 40000,
            "tool_output_budget": 20000,
            "min_memory_budget": 10000,
            "max_memory_budget": 40000
        },
        "claude-3.5-sonnet": {
            "max_context_tokens": 180000,
            "chunk_size": 30000,
            "chunk_overlap": 3000,
            "system_prompt_budget": 15000,
            "current_input_budget": 60000,
            "trajectory_budget": 60000,
            "tool_output_budget": 30000,
            "min_memory_budget": 15000,
            "max_memory_budget": 60000
        },
        "default": {
            "max_context_tokens": 28000,
            "chunk_size": 6000,
            "chunk_overlap": 400,
            "system_prompt_budget": 4000,
            "current_input_budget": 8000,
            "trajectory_budget": 8000,
            "tool_output_budget": 4000,
            "min_memory_budget": 2000,
            "max_memory_budget": 8000
        }
    }
    
    @classmethod
    def from_simple(cls, 
                    quality: str = "balanced",
                    model: str = "gpt-4.1",
                    base_path: str = "agent_generated/memory/jotty") -> SwarmConfig:
        """
        Create full config from simple settings.
        
        Args:
            quality: "fast", "balanced", or "thorough"
            model: Model name for auto-configuring context limits
            base_path: Where to store JOTTY state
            
        Returns:
            Fully configured SwarmConfig
        """
        # Get preset
        preset = cls.PRESETS.get(quality, cls.PRESETS["balanced"])
        
        # Get model config
        model_key = model.lower().replace("openai/azure/", "").replace("openai/", "")
        model_cfg = cls.MODEL_CONFIGS.get(model_key, cls.MODEL_CONFIGS["default"])
        
        # Merge - only include valid SwarmConfig parameters
        full_config = {
            "base_path": base_path,
            **preset,
            **model_cfg,
            # Standard RL parameters
            "gamma": 0.95,
            "lambda_trace": 0.9,
            "alpha": 0.03,
            "enable_adaptive_alpha": True,
            "alpha_min": 0.005,
            "alpha_max": 0.15,
            "enable_intermediate_rewards": True,
            "epsilon_start": 0.3,
            "epsilon_end": 0.08,
            "epsilon_decay_episodes": 200,
            "enable_adaptive_exploration": True,
            # Protection
            "protected_memory_threshold": 0.75,
            "min_confidence": 0.5,
            # Persistence
            "auto_load": True,
            "auto_save": True,
            "save_interval": 5,
            "enable_backups": True,
            "backup_interval": 50,
            "max_backups": 5,
            # Execution
            "async_timeout": 300.0,
            "max_concurrent_agents": 5,
            # Budget
            "enable_dynamic_budget": True,
            # Deduplication
            "enable_deduplication": True,
            "similarity_threshold": 0.85,
            # LLM RAG
            "enable_llm_rag": True,
            "rag_use_cot": True,
            # Goal hierarchy
            "enable_goal_hierarchy": True,
            # Inter-agent
            "enable_agent_communication": True
        }
        
        return SwarmConfig(**full_config)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> SwarmConfig:
        """Load config from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Check for simple_mode
        if data.get('simple_mode', {}).get('enabled', False):
            simple = data['simple_mode']
            return cls.from_simple(
                quality=simple.get('quality', 'balanced'),
                model=simple.get('context_model', 'gpt-4.1')
            )
        
        # Check for intermediate mode
        if 'intermediate' in data and not data.get('advanced_mode', False):
            return cls._from_intermediate(data['intermediate'])
        
        # Full advanced mode
        if data.get('advanced_mode', False) and 'advanced' in data:
            return SwarmConfig(**data['advanced'])
        
        # Direct config
        if 'jotty_config' in data:
            return SwarmConfig(**data['jotty_config'])
        
        return SwarmConfig(**data)
    
    @classmethod
    def _from_intermediate(cls, intermediate: Dict) -> SwarmConfig:
        """Build config from intermediate settings."""
        # Start with balanced preset
        base = cls.from_simple("balanced")
        
        # Apply intermediate overrides
        memory_sizes = {
            "small": 500,
            "medium": 2000,
            "large": 10000
        }
        
        if "memory_size" in intermediate:
            size = memory_sizes.get(intermediate["memory_size"], 2000)
            base.episodic_capacity = size
            base.semantic_capacity = size // 2
            base.procedural_capacity = size // 5
            base.meta_capacity = size // 10
            base.causal_capacity = size // 6
        
        if "validation_strictness" in intermediate:
            strictness = intermediate["validation_strictness"]
            if strictness == "lenient":
                base.min_confidence = 0.3
                base.require_all_architect = False
                base.require_all_auditor = False
            elif strictness == "strict":
                base.min_confidence = 0.7
                base.require_all_architect = True
                base.require_all_auditor = True
        
        if "multi_round_enabled" in intermediate:
            base.enable_multi_round = intermediate["multi_round_enabled"]
        
        if "max_context_percentage" in intermediate:
            pct = intermediate["max_context_percentage"] / 100.0
            base.max_context_tokens = int(base.max_context_tokens * pct)
        
        return base


# =============================================================================
# ACTOR TYPE DETECTION
# =============================================================================

class ActorType:
    """Enum-like for actor types."""
    DSPY_MODULE = "dspy_module"
    ASYNC_FUNCTION = "async_function"
    SYNC_FUNCTION = "sync_function"
    LANGCHAIN_CHAIN = "langchain_chain"
    CLASS_INSTANCE = "class_instance"
    UNKNOWN = "unknown"


def detect_actor_type(actor: Actor) -> str:
    """Detect what kind of actor we're dealing with."""
    # DSPy Module
    if isinstance(actor, dspy.Module):
        return ActorType.DSPY_MODULE
    
    # Async function
    if asyncio.iscoroutinefunction(actor):
        return ActorType.ASYNC_FUNCTION
    
    # Sync function
    if callable(actor) and not isinstance(actor, type):
        return ActorType.SYNC_FUNCTION
    
    # LangChain (check by attribute)
    if hasattr(actor, 'invoke') or hasattr(actor, '_call'):
        return ActorType.LANGCHAIN_CHAIN
    
    # Class instance with __call__
    if hasattr(actor, '__call__'):
        return ActorType.CLASS_INSTANCE
    
    return ActorType.UNKNOWN


# =============================================================================
# UNIVERSAL JOTTY WRAPPER
# =============================================================================

class JottyUniversal:
    """
    Wrap ANY callable with JOTTY validation.
    
    This is the recommended entry point for JOTTY.
    
    Usage:
        # Minimal - just wrap your function
        jotty = JottyUniversal(my_function)
        result = await jotty(query="...")
        
        # With custom validation
        jotty = JottyUniversal(
            my_function,
            architect="Check input is valid",   # Inline prompt
            auditor="Verify result is correct"  # Inline prompt
        )
        
        # With config
        jotty = JottyUniversal(
            my_function,
            config={"quality": "thorough"}  # Simple config
        )
    
    JOTTY v1.0 - Universal Wrapper
    """
    
    def __init__(self, actor: Actor, architect: ValidatorSpec = None, auditor: ValidatorSpec = None, tools: List[Callable] = None, config: ConfigSpec = None, name: str = None) -> None:
        """
        Initialize universal wrapper.
        
        Args:
            actor: The function/agent to wrap (required)
            architect: Architect specification - can be:
                    - None: Use default pass-through
                    - str: Inline prompt text OR path to file
                    - Callable: Custom validation function
                    - List: Multiple validators
            auditor: Auditor specification (same formats as architect)
            tools: Optional tools for validation agents
            config: Configuration - can be:
                    - None: Use balanced defaults
                    - SwarmConfig: Full config object
                    - Dict: {"quality": "...", "model": "..."}
                    - str: Path to YAML file
            name: Optional name for this wrapped actor
        """
        self.actor = actor
        self.actor_type = detect_actor_type(actor)
        self.name = name or self._infer_name()
        
        # Load config
        self.config = self._load_config(config)
        
        # Context guard
        self.context_guard = ContextGuard(
            max_context_tokens=self.config.max_context_tokens,
            safety_margin=0.9
        )
        
        # Normalize validators
        self.architect_fn = self._normalize_validator(architect, is_architect=True)
        self.auditor_fn = self._normalize_validator(auditor, is_architect=False)
        
        # Tools
        self.tools = tools or []
        
        # State tracking
        self.call_count = 0
        self.success_count = 0
        
        # Memory (lazy init)
        self._memory = None
        self._q_table = {}
    
    def _infer_name(self) -> str:
        """Infer a name for the actor."""
        if hasattr(self.actor, '__name__'):
            return self.actor.__name__
        if hasattr(self.actor, '__class__'):
            return self.actor.__class__.__name__
        return "UnknownActor"
    
    def _load_config(self, config: ConfigSpec) -> SwarmConfig:
        """Load config from various formats."""
        if config is None:
            return SmartConfig.from_simple("balanced")
        
        if isinstance(config, SwarmConfig):
            return config
        
        if isinstance(config, str):
            # Path to YAML
            return SmartConfig.from_yaml(config)
        
        if isinstance(config, dict):
            # Simple config dict
            if "quality" in config:
                return SmartConfig.from_simple(
                    quality=config.get("quality", "balanced"),
                    model=config.get("model", "gpt-4.1"),
                    base_path=config.get("base_path", "agent_generated/memory/jotty")
                )
            # Full config dict
            return SwarmConfig(**config)
        
        return SmartConfig.from_simple("balanced")
    
    def _normalize_validator(self, 
                             spec: ValidatorSpec, 
                             is_architect: bool) -> Callable:
        """
        Normalize validator spec into a callable.
        
        Supports:
        - None: Default pass-through
        - str: Inline prompt or file path
        - Callable: Use directly
        - List: Chain multiple validators
        """
        if spec is None:
            # Default pass-through with warning
            warnings.warn(
                f"JOTTY: No {'architect' if is_architect else 'auditor'} specified. "
                "Using default pass-through validation.",
                UserWarning
            )
            return self._default_validator(is_architect)
        
        if callable(spec):
            return spec
        
        if isinstance(spec, str):
            # Check if file path
            if Path(spec).exists():
                return self._create_file_validator(spec, is_architect)
            # Inline prompt
            return self._create_inline_validator(spec, is_architect)
        
        if isinstance(spec, list):
            # Chain multiple validators
            validators = [self._normalize_validator(s, is_architect) for s in spec]
            return self._chain_validators(validators, is_architect)
        
        return self._default_validator(is_architect)
    
    def _default_validator(self, is_architect: bool) -> Callable:
        """Create default pass-through validator."""
        async def default_architect(**kwargs: Any) -> Dict:
            return {
                "should_proceed": True,
                "confidence": 0.5,
                "reasoning": "Default pass-through (no custom validation)",
                "injected_instructions": ""
            }
        
        async def default_auditor(**kwargs: Any) -> Dict:
            return {
                "is_valid": True,
                "confidence": 0.5,
                "reasoning": "Default pass-through (no custom validation)",
                "output_tag": "useful"
            }
        
        return default_architect if is_architect else default_auditor
    
    def _create_inline_validator(self, prompt: str, is_architect: bool) -> Callable:
        """Create validator from inline prompt."""
        # Create DSPy signature dynamically
        if is_architect:
            class InlineArchitect(dspy.Signature):
                """Custom Architect."""
                context: str = dspy.InputField()
                custom_instructions: str = dspy.InputField(default=prompt)
                should_proceed: bool = dspy.OutputField()
                confidence: float = dspy.OutputField()
                reasoning: str = dspy.OutputField()
            
            validator = dspy.ChainOfThought(InlineArchitect)
        else:
            class InlineAuditor(dspy.Signature):
                """Custom Auditor."""
                output: str = dspy.InputField()
                custom_instructions: str = dspy.InputField(default=prompt)
                is_valid: bool = dspy.OutputField()
                confidence: float = dspy.OutputField()
                reasoning: str = dspy.OutputField()
            
            validator = dspy.ChainOfThought(InlineAuditor)
        
        async def run_validator(**kwargs: Any) -> Dict:
            try:
                result = validator(**kwargs)
                return vars(result)
            except Exception as e:
                return {
                    "should_proceed" if is_architect else "is_valid": True,
                    "confidence": 0.3,
                    "reasoning": f"Validation failed: {e}"
                }
        
        return run_validator
    
    def _create_file_validator(self, path: str, is_architect: bool) -> Callable:
        """Create validator from file prompt."""
        # Read prompt from file
        with open(path, 'r') as f:
            prompt = f.read()
        
        return self._create_inline_validator(prompt, is_architect)
    
    def _chain_validators(self, validators: List[Callable], is_architect: bool) -> Callable:
        """Chain multiple validators."""
        async def chained(**kwargs: Any) -> Dict:
            results = []
            for v in validators:
                result = await v(**kwargs) if asyncio.iscoroutinefunction(v) else v(**kwargs)
                results.append(result)
            
            # Aggregate results
            if is_architect:
                # All must proceed
                should_proceed = all(r.get("should_proceed", True) for r in results)
                confidence = sum(r.get("confidence", 0.5) for r in results) / len(results)
                reasoning = " | ".join(r.get("reasoning", "") for r in results)
                return {
                    "should_proceed": should_proceed,
                    "confidence": confidence,
                    "reasoning": reasoning
                }
            else:
                # Majority vote
                valid_count = sum(1 for r in results if r.get("is_valid", True))
                is_valid = valid_count > len(results) / 2
                confidence = sum(r.get("confidence", 0.5) for r in results) / len(results)
                reasoning = " | ".join(r.get("reasoning", "") for r in results)
                return {
                    "is_valid": is_valid,
                    "confidence": confidence,
                    "reasoning": reasoning
                }
        
        return chained
    
    # =========================================================================
    # EXECUTION
    # =========================================================================
    
    async def __call__(self, **kwargs: Any) -> Any:
        """Execute with Jotty validation."""
        return await self.arun(**kwargs)
    
    async def arun(self, **kwargs: Any) -> Dict:
        """
        Run actor with full Jotty validation.
        
        Returns dict with:
        - output: Actor result
        - success: Whether validation passed
        - architect_result: Architect details
        - auditor_result: Auditor details
        - metrics: Performance metrics
        """
        self.call_count += 1
        
        # Context guard check
        context_str = str(kwargs)
        if self.context_guard.estimate_tokens(context_str) > self.config.max_context_tokens * 0.9:
            # Auto-truncate
            truncated = self.context_guard.check_and_truncate(
                system_prompt="",
                user_input=context_str,
                context="",
                memories=[],
                trajectory=""
            )
            kwargs = {"_truncated_input": truncated["user_input"]}
        
        # === ARCHITECT ===
        architect_result = await self._run_architect(kwargs)
        
        if not architect_result.get("should_proceed", True):
            return {
                "output": None,
                "success": False,
                "blocked": True,
                "architect_result": architect_result,
                "auditor_result": None,
                "reasoning": architect_result.get("reasoning", "Blocked by Architect")
            }
        
        # === ACTOR EXECUTION ===
        try:
            output = await self._run_actor(kwargs, architect_result)
        except Exception as e:
            return {
                "output": None,
                "success": False,
                "error": str(e),
                "architect_result": architect_result,
                "auditor_result": None
            }
        
        # === AUDITOR ===
        auditor_result = await self._run_auditor(output, kwargs)
        
        success = auditor_result.get("is_valid", True)
        if success:
            self.success_count += 1
        
        # === LEARNING UPDATE ===
        self._update_learning(kwargs, output, architect_result, auditor_result, success)
        
        return {
            "output": output,
            "success": success,
            "architect_result": architect_result,
            "auditor_result": auditor_result,
            "metrics": {
                "call_count": self.call_count,
                "success_rate": self.success_count / self.call_count
            }
        }
    
    def run(self, **kwargs: Any) -> Dict:
        """Synchronous wrapper."""
        return asyncio.run(self.arun(**kwargs))
    
    async def _run_architect(self, kwargs: Dict) -> Dict:
        """Run Architect."""
        context = str(kwargs)  # Truncate for safety
        
        if asyncio.iscoroutinefunction(self.architect_fn):
            return await self.architect_fn(context=context, **kwargs)
        return self.architect_fn(context=context, **kwargs)
    
    async def _run_actor(self, kwargs: Dict, architect_result: Dict) -> Any:
        """Run the actor."""
        # Inject instructions from architect
        if architect_result.get("injected_instructions"):
            kwargs["_instructions"] = architect_result["injected_instructions"]
        
        # Execute based on actor type
        if self.actor_type == ActorType.DSPY_MODULE:
            return self.actor(**kwargs)
        
        if self.actor_type == ActorType.ASYNC_FUNCTION:
            return await self.actor(**kwargs)
        
        if self.actor_type == ActorType.LANGCHAIN_CHAIN:
            if hasattr(self.actor, 'ainvoke'):
                return await self.actor.ainvoke(kwargs)
            return self.actor.invoke(kwargs)
        
        # Default: call directly
        if asyncio.iscoroutinefunction(self.actor):
            return await self.actor(**kwargs)
        return self.actor(**kwargs)
    
    async def _run_auditor(self, output: Any, kwargs: Dict) -> Dict:
        """Run Auditor."""
        output_str = str(output)  # Truncate for safety
        
        if asyncio.iscoroutinefunction(self.auditor_fn):
            return await self.auditor_fn(output=output_str, **kwargs)
        return self.auditor_fn(output=output_str, **kwargs)
    
    def _update_learning(self, kwargs: Dict, output: Any, architect: Dict, auditor: Dict, success: bool) -> Any:
        """Update Q-table from experience."""
        # Simple state key
        state_key = f"{self.name}:{hash(str(kwargs)) % 10000}"
        
        # Reward
        reward = 1.0 if success else -0.5
        if architect.get("confidence", 0) > 0.8:
            reward += 0.1
        if auditor.get("confidence", 0) > 0.8:
            reward += 0.1
        
        # TD update
        old_value = self._q_table.get(state_key, 0.5)
        new_value = old_value + 0.1 * (reward - old_value)
        self._q_table[state_key] = new_value


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def jotty_universal(actor: Actor = None, architect: ValidatorSpec = None, auditor: ValidatorSpec = None, config: ConfigSpec = None, **kwargs: Any) -> Union['JottyUniversal', Callable]:
    """
    Decorator/function to wrap anything with JOTTY.
    
    Usage as decorator:
        @jotty_universal(architect="Check input", config={"quality": "fast"})
        def my_function(query):
            return query.upper()
    
    Usage as function:
        wrapped = jotty_universal(my_function)
    """
    if actor is None:
        # Being used as decorator with args
        def decorator(fn: Any) -> Any:
            return JottyUniversal(fn, architect=architect, auditor=auditor, config=config, **kwargs)
        return decorator
    
    # Being used as function
    return JottyUniversal(actor, architect=architect, auditor=auditor, config=config, **kwargs)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'JottyUniversal',
    'SmartConfig',
    'jotty_universal',
    'detect_actor_type',
    'ActorType'
]

