"""
LOTUS Integration - Connect Optimizer to Jotty v2 Infrastructure

DRY Principle: Integrate without duplicating existing logic.
Uses composition and delegation to enhance existing components.

Integration Points:
1. Orchestrator - Add LOTUS optimizer as optional enhancement
2. AgentRunner - Wire adaptive validation
3. DSPy LM - Wrap with cascade and caching

Usage:
    from Jotty.core.lotus.integration import enhance_swarm_manager

    # Enhance existing Orchestrator
    swarm = Orchestrator(agents=[...])
    enhanced_swarm = enhance_swarm_manager(swarm)

    # Or use LotusSwarmManager directly
    swarm = LotusSwarmManager(agents=[...], enable_lotus=True)
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from functools import wraps

from .config import LotusConfig, ModelTier
from .model_cascade import ModelCascade
from .semantic_cache import SemanticCache
from .batch_executor import BatchExecutor
from .adaptive_validator import AdaptiveValidator
from .optimizer import LotusOptimizer

logger = logging.getLogger(__name__)


class LotusEnhancement:
    """
    Enhancement mixin that adds LOTUS optimization to any swarm.

    DRY: Composes with existing swarm without modifying its code.
    """

    def __init__(
        self,
        config: Optional[LotusConfig] = None,
        enable_cascade: bool = True,
        enable_cache: bool = True,
        enable_adaptive_validation: bool = True,
    ):
        """
        Initialize LOTUS enhancement.

        Args:
            config: LOTUS configuration
            enable_cascade: Enable model cascading
            enable_cache: Enable semantic caching
            enable_adaptive_validation: Enable adaptive validation
        """
        self.lotus_config = config or LotusConfig()
        self.enable_cascade = enable_cascade
        self.enable_cache = enable_cache
        self.enable_adaptive_validation = enable_adaptive_validation

        # Core optimizer
        self.lotus_optimizer = LotusOptimizer(self.lotus_config)

        # Individual components for fine-grained control
        self.cascade = self.lotus_optimizer.cascade
        self.cache = self.lotus_optimizer.cache
        self.batch_executor = self.lotus_optimizer.batch_executor
        self.adaptive_validator = self.lotus_optimizer.validator

        logger.info(
            f"LotusEnhancement initialized: "
            f"cascade={enable_cascade}, cache={enable_cache}, "
            f"adaptive_validation={enable_adaptive_validation}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get LOTUS optimization statistics."""
        return self.lotus_optimizer.get_stats()

    def get_savings(self) -> Dict[str, float]:
        """Get estimated cost savings."""
        return self.lotus_optimizer.get_savings_estimate()


def enhance_swarm_manager(swarm_manager: Any, config: Optional[LotusConfig] = None) -> Any:
    """
    Enhance an existing Orchestrator with LOTUS optimization.

    DRY: Patches existing instance without modifying class.

    Args:
        swarm_manager: Existing Orchestrator instance
        config: LOTUS configuration

    Returns:
        Enhanced Orchestrator with lotus_optimizer attached
    """
    enhancement = LotusEnhancement(config)

    # Attach enhancement to swarm
    swarm_manager.lotus = enhancement
    swarm_manager.lotus_optimizer = enhancement.lotus_optimizer

    # Enhance agent runners with adaptive validation
    for name, runner in getattr(swarm_manager, 'runners', {}).items():
        _enhance_agent_runner(runner, enhancement)

    logger.info(f"Orchestrator enhanced with LOTUS optimization")

    return swarm_manager


def _enhance_agent_runner(runner: Any, enhancement: LotusEnhancement) -> Tuple:
    """
    Enhance an AgentRunner with adaptive validation.

    DRY: Wraps validation calls without modifying runner code.
    """
    # Store original validators
    original_architect = getattr(runner, 'architect_validator', None)
    original_auditor = getattr(runner, 'auditor_validator', None)

    if not enhancement.enable_adaptive_validation:
        return

    # Import ValidationResult for creating skip results
    try:
        from ..foundation.data_structures import ValidationResult, OutputTag, ValidationRound
    except ImportError:
        logger.warning("Could not import ValidationResult, skipping adaptive validation enhancement")
        return

    # Wrap validation calls with adaptive skipping
    if original_architect:
        runner._original_architect_validator = original_architect
        original_validate = original_architect.validate

        async def adaptive_architect_validate(
            goal: str,
            inputs: dict,
            trajectory: list,
            is_architect: bool = True,
        ):
            """Wrapped architect validation with adaptive skipping."""
            agent_name = runner.agent_name
            decision = enhancement.adaptive_validator.should_validate(agent_name, "architect")

            if not decision.should_validate:
                logger.debug(f"Skipping architect validation for {agent_name}: {decision.reason}")
                # Return ValidationResult with proceed=True to skip validation
                skip_result = ValidationResult(
                    agent_name=f"{agent_name}_lotus_skip",
                    is_valid=True,
                    confidence=decision.confidence,
                    reasoning=f"LOTUS: Skipped validation ({decision.reason})",
                    should_proceed=True,
                    validation_round=ValidationRound.INITIAL,
                )
                return [skip_result], True

            # Call original validate
            result = await original_validate(
                goal=goal,
                inputs=inputs,
                trajectory=trajectory,
                is_architect=is_architect,
            )

            # Record outcome (result is tuple of (results_list, proceed_bool))
            is_success = result[1] if isinstance(result, tuple) else True
            enhancement.adaptive_validator.record_result(agent_name, "architect", is_success)

            return result

        runner.architect_validator.validate = adaptive_architect_validate

    if original_auditor:
        runner._original_auditor_validator = original_auditor
        original_validate = original_auditor.validate

        async def adaptive_auditor_validate(
            goal: str,
            inputs: dict,
            trajectory: list,
            is_architect: bool = False,
        ):
            """Wrapped auditor validation with adaptive skipping."""
            agent_name = runner.agent_name
            decision = enhancement.adaptive_validator.should_validate(agent_name, "auditor")

            if not decision.should_validate:
                logger.debug(f"Skipping auditor validation for {agent_name}: {decision.reason}")
                skip_result = ValidationResult(
                    agent_name=f"{agent_name}_lotus_skip",
                    is_valid=True,
                    confidence=decision.confidence,
                    reasoning=f"LOTUS: Skipped validation ({decision.reason})",
                    should_proceed=True,
                    output_tag=OutputTag.USEFUL,
                    validation_round=ValidationRound.INITIAL,
                )
                return [skip_result], True

            # Call original validate
            result = await original_validate(
                goal=goal,
                inputs=inputs,
                trajectory=trajectory,
                is_architect=is_architect,
            )

            # Record outcome
            is_success = result[1] if isinstance(result, tuple) else True
            enhancement.adaptive_validator.record_result(agent_name, "auditor", is_success)

            return result

        runner.auditor_validator.validate = adaptive_auditor_validate


class CascadedLM:
    """
    DSPy LM wrapper with model cascade.

    DRY: Wraps any DSPy LM with cascade logic.
    """

    def __init__(
        self,
        base_lm: Any,
        cascade: ModelCascade,
        cache: Optional[SemanticCache] = None,
        default_operation: str = "default",
    ):
        """
        Initialize cascaded LM.

        Args:
            base_lm: Base DSPy LM
            cascade: ModelCascade instance
            cache: Optional SemanticCache
            default_operation: Default operation type for cascade thresholds
        """
        self.base_lm = base_lm
        self.cascade = cascade
        self.cache = cache
        self.default_operation = default_operation

        # Copy attributes from base LM
        for attr in ['model_name', 'api_key', 'provider']:
            if hasattr(base_lm, attr):
                setattr(self, attr, getattr(base_lm, attr))

    def __call__(self, prompt: Any, **kwargs) -> Any:
        """
        Call LM with cascade optimization.

        For single prompts, uses cache + base LM.
        For batch prompts, uses full cascade.
        """
        # Handle both string and list prompts
        is_batch = isinstance(prompt, list)
        prompts = prompt if is_batch else [prompt]

        results = []
        for p in prompts:
            # Check cache first
            if self.cache:
                hit, cached = self.cache.get("lm_call", p)
                if hit:
                    results.append(cached)
                    continue

            # Call base LM
            result = self.base_lm(p, **kwargs)

            # Cache result
            if self.cache:
                self.cache.put("lm_call", p, result)

            results.append(result)

        return results if is_batch else results[0]


def create_cascaded_lm(
    config: Optional[LotusConfig] = None,
    enable_cache: bool = True,
) -> CascadedLM:
    """
    Create a cascaded LM from current DSPy settings.

    Usage:
        import dspy
        cascaded_lm = create_cascaded_lm()
        dspy.configure(lm=cascaded_lm)
    """
    try:
        import dspy

        config = config or LotusConfig()
        base_lm = dspy.settings.lm

        if base_lm is None:
            raise ValueError("No DSPy LM configured. Call dspy.configure(lm=...) first.")

        cascade = ModelCascade(config)
        cache = SemanticCache(config) if enable_cache else None

        return CascadedLM(base_lm, cascade, cache)

    except ImportError:
        raise ImportError("DSPy not available. Install with: pip install dspy-ai")


class LotusSwarmMixin:
    """
    Mixin class to add LOTUS optimization to Orchestrator.

    Usage:
        class MySwarm(LotusSwarmMixin, Orchestrator):
            pass

        swarm = MySwarm(agents=[...], enable_lotus=True)
    """

    def _init_lotus(
        self,
        config: Optional[LotusConfig] = None,
        enable_cascade: bool = True,
        enable_cache: bool = True,
        enable_adaptive_validation: bool = True,
    ):
        """Initialize LOTUS optimization components."""
        self.lotus = LotusEnhancement(
            config=config,
            enable_cascade=enable_cascade,
            enable_cache=enable_cache,
            enable_adaptive_validation=enable_adaptive_validation,
        )

        # Enhance runners
        for name, runner in getattr(self, 'runners', {}).items():
            _enhance_agent_runner(runner, self.lotus)

    def lotus_stats(self) -> Dict[str, Any]:
        """Get LOTUS optimization statistics."""
        if hasattr(self, 'lotus'):
            return self.lotus.get_stats()
        return {}

    def lotus_savings(self) -> Dict[str, float]:
        """Get estimated cost savings from LOTUS optimization."""
        if hasattr(self, 'lotus'):
            return self.lotus.get_savings()
        return {}


# Convenience function to wrap DSPy signatures with caching
def cached_signature(cache: SemanticCache):
    """
    Decorator to cache DSPy signature calls.

    Usage:
        cache = SemanticCache()

        @cached_signature(cache)
        class MySignature(dspy.Signature):
            query = dspy.InputField()
            answer = dspy.OutputField()
    """
    def decorator(sig_class):
        original_init = sig_class.__init__

        def new_init(self, *args, **kwargs) -> None:
            original_init(self, *args, **kwargs)
            self._cache = cache

        sig_class.__init__ = new_init
        return sig_class

    return decorator


# Export helper for easy integration
def setup_lotus_optimization(
    swarm_manager: Any = None,
    config: Optional[Dict[str, Any]] = None,
) -> LotusOptimizer:
    """
    One-liner setup for LOTUS optimization.

    Usage:
        from Jotty.core.lotus import setup_lotus_optimization

        # Standalone optimizer
        optimizer = setup_lotus_optimization()

        # With existing swarm
        optimizer = setup_lotus_optimization(swarm_manager=my_swarm)
    """
    lotus_config = LotusConfig()

    if config:
        # Apply custom config
        if 'skip_threshold' in config:
            lotus_config.validation_skip_threshold = config['skip_threshold']
        if 'cache_enabled' in config:
            lotus_config.cache.enabled = config['cache_enabled']
        if 'batch_size' in config:
            lotus_config.batch.max_batch_size = config['batch_size']

    optimizer = LotusOptimizer(lotus_config)

    if swarm_manager:
        enhance_swarm_manager(swarm_manager, lotus_config)

    logger.info("LOTUS optimization setup complete")

    return optimizer
