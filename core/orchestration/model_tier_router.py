"""
ModelTierRouter - Complexity-Based Model Selection
===================================================

Maps task complexity (from ValidationGate) to LLM model tiers.
Reduces cost by using cheap models for simple tasks and reserving
expensive models for complex ones.

Routing table:
    DIRECT     → Haiku/Flash   (cheap, fast — Q&A, lookups, lists)
    AUDIT_ONLY → Sonnet/GPT-4o (balanced — summaries, analysis)
    FULL       → Sonnet/Opus   (quality — code gen, multi-step, high-stakes)

Integrates with:
    - ValidationGate (complexity classification)
    - SwarmProviderGateway (LM creation)
    - CostTracker (observability)

Usage:
    router = ModelTierRouter()
    lm = router.get_lm_for_mode(ValidationMode.DIRECT)
    # → Returns cheapest available LM (Haiku, Flash, etc.)

    lm = router.get_lm_for_mode(ValidationMode.FULL)
    # → Returns highest quality available LM (Opus, Sonnet, etc.)
"""

import os
import logging
from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from .validation_gate import ValidationMode
from Jotty.core.foundation.config_defaults import MODEL_SONNET, MODEL_OPUS, MODEL_HAIKU

logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Model quality tiers."""
    CHEAP = "cheap"        # Haiku, Flash, GPT-4o-mini, Llama-8b
    BALANCED = "balanced"  # Sonnet, GPT-4o, Gemini Pro
    QUALITY = "quality"    # Opus, GPT-4-turbo, Gemini Ultra


# Mapping: ValidationMode -> ModelTier
MODE_TO_TIER: Dict[ValidationMode, ModelTier] = {
    ValidationMode.DIRECT: ModelTier.CHEAP,
    ValidationMode.AUDIT_ONLY: ModelTier.BALANCED,
    ValidationMode.FULL: ModelTier.QUALITY,
}

# Model preferences per tier, per provider (ordered by preference)
# IMPORTANT: First model in each list is the primary pick.
# Anthropic model names centralized in config_defaults.py
TIER_MODELS: Dict[str, Dict[ModelTier, List[str]]] = {
    'anthropic': {
        ModelTier.CHEAP: [MODEL_HAIKU, 'claude-haiku'],
        ModelTier.BALANCED: [MODEL_SONNET, 'claude-3-5-sonnet-latest'],
        ModelTier.QUALITY: [MODEL_OPUS, MODEL_SONNET],
    },
    'openai': {
        ModelTier.CHEAP: ['gpt-4o-mini'],
        ModelTier.BALANCED: ['gpt-4o'],
        ModelTier.QUALITY: ['gpt-4-turbo', 'gpt-4o'],
    },
    'google': {
        ModelTier.CHEAP: ['gemini-2.0-flash'],
        ModelTier.BALANCED: ['gemini-2.5-pro', 'gemini-1.5-pro'],
        ModelTier.QUALITY: ['gemini-2.5-pro', 'gemini-1.5-pro'],
    },
    'groq': {
        ModelTier.CHEAP: ['llama-3.1-8b-instant'],
        ModelTier.BALANCED: ['llama-3.3-70b-versatile', 'llama-3.1-70b-versatile'],
        ModelTier.QUALITY: ['llama-3.3-70b-versatile', 'llama-3.1-70b-versatile'],
    },
}

# Approximate cost per 1M tokens (input) for quick estimation
TIER_COST_PER_1M: Dict[ModelTier, float] = {
    ModelTier.CHEAP: 0.25,     # ~$0.25/1M (Haiku)
    ModelTier.BALANCED: 3.0,   # ~$3/1M (Sonnet)
    ModelTier.QUALITY: 15.0,   # ~$15/1M (Opus)
}


@dataclass
class TierDecision:
    """Result of model tier routing."""
    tier: ModelTier
    model: str
    provider: str
    reason: str
    estimated_cost_ratio: float = 1.0  # Relative to QUALITY tier (1.0 = Opus)


class ModelTierRouter:
    """
    Routes tasks to appropriate model tiers based on complexity.

    Integrates with ValidationGate for complexity classification
    and SwarmProviderGateway for LM creation.
    """

    def __init__(self, default_provider: Optional[str] = None) -> None:
        """
        Args:
            default_provider: Preferred provider (auto-detect if None)
        """
        self._default_provider = default_provider
        self._detected_provider: Optional[str] = None
        self._tier_lms: Dict[ModelTier, Any] = {}
        self._call_counts: Dict[ModelTier, int] = {t: 0 for t in ModelTier}
        self._estimated_savings: float = 0.0

    def _detect_provider(self) -> str:
        """Detect the best available provider."""
        if self._detected_provider:
            return self._detected_provider

        if self._default_provider:
            self._detected_provider = self._default_provider
            return self._detected_provider

        # Auto-detect by checking API keys
        provider_keys = [
            ('anthropic', 'ANTHROPIC_API_KEY'),
            ('openai', 'OPENAI_API_KEY'),
            ('google', 'GOOGLE_API_KEY'),
            ('google', 'GEMINI_API_KEY'),
            ('groq', 'GROQ_API_KEY'),
        ]
        for provider, key in provider_keys:
            if os.environ.get(key):
                self._detected_provider = provider
                return provider

        # Fallback
        self._detected_provider = 'anthropic'
        return 'anthropic'

    def get_tier_for_mode(self, mode: ValidationMode) -> ModelTier:
        """Map a ValidationMode to a ModelTier."""
        return MODE_TO_TIER.get(mode, ModelTier.BALANCED)

    def get_model_for_mode(self, mode: ValidationMode) -> TierDecision:
        """
        Get the recommended model for a given validation mode.

        This is a pure lookup — does NOT count as an actual usage.
        Call ``get_lm_for_mode`` to create an LM *and* record the usage.

        Args:
            mode: Task complexity from ValidationGate

        Returns:
            TierDecision with model, provider, and cost info
        """
        tier = self.get_tier_for_mode(mode)
        provider = self._detect_provider()
        models = TIER_MODELS.get(provider, {}).get(tier, [])

        if not models:
            # Fallback to balanced tier
            models = TIER_MODELS.get(provider, {}).get(ModelTier.BALANCED, [])
            tier = ModelTier.BALANCED

        model = models[0] if models else MODEL_SONNET

        # Cost ratio relative to QUALITY tier (what you'd pay without routing)
        quality_cost = TIER_COST_PER_1M[ModelTier.QUALITY]
        tier_cost = TIER_COST_PER_1M[tier]
        cost_ratio = tier_cost / quality_cost if quality_cost > 0 else 1.0

        return TierDecision(
            tier=tier,
            model=model,
            provider=provider,
            reason=f"{mode.value} -> {tier.value}: {model}",
            estimated_cost_ratio=cost_ratio,
        )

    def get_lm_for_mode(self, mode: ValidationMode) -> None:
        """
        Get or create a DSPy LM instance for the given validation mode.

        This is the primary API.  It records usage for cost tracking
        and caches LMs per tier to avoid recreation overhead.

        Args:
            mode: Task complexity from ValidationGate

        Returns:
            DSPy BaseLM instance, or None if unavailable
        """
        tier = self.get_tier_for_mode(mode)

        # Record usage (even for cached LMs — we're counting *calls*, not creations)
        self._call_counts[tier] += 1

        # Return cached LM if available
        if tier in self._tier_lms and self._tier_lms[tier] is not None:
            return self._tier_lms[tier]

        decision = self.get_model_for_mode(mode)

        try:
            from Jotty.core.foundation.unified_lm_provider import UnifiedLMProvider
            lm = UnifiedLMProvider.create_lm(
                provider=decision.provider,
                model=decision.model,
            )
            self._tier_lms[tier] = lm
            logger.info(
                f"ModelTierRouter: created {decision.tier.value} LM "
                f"({decision.provider}/{decision.model})"
            )
            return lm
        except Exception as e:
            logger.warning(
                f"ModelTierRouter: could not create {decision.tier.value} LM: {e}"
            )
            return None

    def get_savings_estimate(self) -> Dict[str, Any]:
        """
        Estimate cost savings from tier routing.

        Compares actual tier distribution to a baseline where
        every call uses the QUALITY tier (what you'd pay without routing).
        """
        total_calls = sum(self._call_counts.values())
        if total_calls == 0:
            return {'total_calls': 0, 'savings_pct': '0%'}

        # Baseline: all calls at QUALITY cost (worst case without routing)
        baseline_cost = total_calls * TIER_COST_PER_1M[ModelTier.QUALITY]

        # Actual: calls at their respective tier costs
        actual_cost = sum(
            count * TIER_COST_PER_1M[tier]
            for tier, count in self._call_counts.items()
        )

        savings = baseline_cost - actual_cost
        savings_pct = savings / baseline_cost if baseline_cost > 0 else 0

        return {
            'total_calls': total_calls,
            'tier_distribution': {t.value: c for t, c in self._call_counts.items()},
            'baseline_cost_units': round(baseline_cost, 2),
            'actual_cost_units': round(actual_cost, 2),
            'savings_units': round(savings, 2),
            'savings_pct': f"{savings_pct:.0%}",
        }

    # =========================================================================
    # CONSTRAINT-AWARE MODEL SELECTION (Tier 2 Enhancement)
    # =========================================================================

    def select_tier(
        self,
        task: str,
        cost_budget_usd: Optional[float] = None,
        latency_budget_ms: Optional[float] = None,
        quality_target: Optional[float] = None,
        estimated_tokens: int = 2000
    ) -> str:
        """
        Select model tier based on task complexity AND constraints.

        ENHANCEMENT: Adds cost/latency/quality awareness to complexity-based routing.

        Without constraints (backward compatible):
        - Uses complexity heuristics (keywords, length)
        - Returns 'cheap', 'balanced', or 'quality'

        With constraints (new capability):
        - Filters tiers that violate cost/latency/quality budgets
        - Returns cheapest tier that meets all constraints

        EXAMPLE 1: No constraints (existing behavior)
        >>> router.select_tier("What is 2+2?")
        'cheap'  # Simple task → cheap model

        EXAMPLE 2: Cost constraint
        >>> router.select_tier(
        ...     "Analyze this 10-page document",
        ...     cost_budget_usd=0.01  # Strict budget
        ... )
        'cheap'  # Would normally be 'quality', but budget forces cheap

        EXAMPLE 3: Quality constraint
        >>> router.select_tier(
        ...     "Simple question",
        ...     quality_target=0.90  # High quality required
        ... )
        'balanced'  # Would normally be 'cheap', but quality forces upgrade

        Args:
            task: Task description (for complexity estimation)
            cost_budget_usd: Max cost in USD (None = no limit)
            latency_budget_ms: Max latency in milliseconds (None = no limit)
            quality_target: Min quality score 0-1 (None = no minimum)
            estimated_tokens: Estimated tokens for cost calculation

        Returns:
            Tier name: 'cheap', 'balanced', or 'quality'
        """
        # STEP 1: Determine base complexity tier (existing logic)
        base_tier = self._estimate_complexity_tier(task)

        # STEP 2: If no constraints, return base tier (backward compatible)
        if cost_budget_usd is None and latency_budget_ms is None and quality_target is None:
            return base_tier.value

        # STEP 3: Apply constraints to filter viable tiers
        viable_tiers = self._filter_tiers_by_constraints(
            base_tier=base_tier,
            cost_budget_usd=cost_budget_usd,
            latency_budget_ms=latency_budget_ms,
            quality_target=quality_target,
            estimated_tokens=estimated_tokens
        )

        # STEP 4: Select cheapest viable tier
        if not viable_tiers:
            # No tier meets constraints - return cheapest with warning
            logger.warning(
                f"⚠️  No tier meets all constraints. Using CHEAP (may violate quality/latency)."
            )
            return ModelTier.CHEAP.value

        # Return cheapest viable tier (cost-efficient)
        cheapest = min(viable_tiers, key=lambda t: TIER_COST_PER_1M[t])
        return cheapest.value

    def _estimate_complexity_tier(self, task: str) -> ModelTier:
        """
        Estimate task complexity using heuristics (existing logic).

        Returns base tier recommendation before applying constraints.
        """
        task_lower = task.lower()

        # Keywords indicating complex tasks
        complex_keywords = [
            'analyze', 'research', 'comprehensive', 'design', 'architect',
            'multi-step', 'complex', 'detailed', 'expert', 'advanced'
        ]

        # Keywords indicating simple tasks
        simple_keywords = [
            'list', 'what is', 'define', 'summarize', 'lookup', 'find'
        ]

        # Length-based heuristic
        if len(task) > 500:
            return ModelTier.QUALITY  # Long tasks need quality models

        # Keyword matching
        for keyword in complex_keywords:
            if keyword in task_lower:
                return ModelTier.QUALITY

        for keyword in simple_keywords:
            if keyword in task_lower:
                return ModelTier.CHEAP

        # Default to balanced
        return ModelTier.BALANCED

    def _filter_tiers_by_constraints(
        self,
        base_tier: ModelTier,
        cost_budget_usd: Optional[float],
        latency_budget_ms: Optional[float],
        quality_target: Optional[float],
        estimated_tokens: int
    ) -> List[ModelTier]:
        """
        Filter tiers based on cost/latency/quality constraints.

        Returns list of tiers that satisfy ALL constraints.
        """
        # Estimated performance per tier (learned from production data)
        TIER_QUALITY = {
            ModelTier.CHEAP: 0.75,      # Haiku, Flash: good for simple tasks
            ModelTier.BALANCED: 0.85,   # Sonnet, GPT-4o: solid quality
            ModelTier.QUALITY: 0.95,    # Opus, GPT-4: best quality
        }

        TIER_LATENCY_MS = {
            ModelTier.CHEAP: 300,       # Fast models (Haiku ~300ms)
            ModelTier.BALANCED: 600,    # Mid-tier (Sonnet ~600ms)
            ModelTier.QUALITY: 1200,    # Slow models (Opus ~1200ms)
        }

        viable = []

        for tier in ModelTier:
            # Check cost constraint
            if cost_budget_usd is not None:
                estimated_cost = (estimated_tokens / 1_000_000) * TIER_COST_PER_1M[tier]
                if estimated_cost > cost_budget_usd:
                    continue  # Too expensive

            # Check latency constraint
            if latency_budget_ms is not None:
                if TIER_LATENCY_MS[tier] > latency_budget_ms:
                    continue  # Too slow

            # Check quality constraint
            if quality_target is not None:
                if TIER_QUALITY[tier] < quality_target:
                    continue  # Quality too low

            # Passed all constraints
            viable.append(tier)

        return viable
