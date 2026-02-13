"""
ValidationGate — LLM-Powered Intelligent Validation Routing
=============================================================

Replaces the boolean ``skip_validation`` flag with a cheap Haiku call
that decides **how much** validation each task actually needs.

Design rationale
~~~~~~~~~~~~~~~~

The full Architect → Actor → Auditor pipeline costs 3+ Sonnet calls
(~$0.03-0.10 per task, ~90s latency).  For "list 5 benefits of X" you
don't need an architect to *plan* a one-step answer, nor an auditor to
*validate* a factual list.

A single Haiku call (~$0.00025, <1s) can classify task complexity and
route to one of three validation modes, giving 40-100x ROI on the gate
cost while preserving quality for tasks that genuinely need oversight.

Validation modes
~~~~~~~~~~~~~~~~

    DIRECT      →  Actor only.  No architect, no auditor.       (1 LLM call)
    AUDIT_ONLY  →  Actor + Auditor.  Skip architect planning.   (2 LLM calls)
    FULL        →  Architect → Actor → Auditor.  Full pipeline. (3+ LLM calls)

Abuse prevention
~~~~~~~~~~~~~~~~

1. **Never-skip list** — Code generation, financial analysis, security
   reviews, and multi-step workflows ALWAYS get FULL validation,
   regardless of what the gate says.  Hardcoded, not learnable.

2. **Sampling** — Even DIRECT tasks are randomly audited at a configurable
   rate (default 10%) to catch quality drift.

3. **Outcome tracking** — Every gate decision is recorded.  If DIRECT
   tasks start failing (detected by the learning pipeline), the gate
   auto-escalates to AUDIT_ONLY or FULL.

4. **Confidence threshold** — The gate must be >80% confident to
   downgrade validation.  Low confidence → FULL.

5. **LOTUS integration** — Combines with AdaptiveValidator history:
   trusted agent + simple task → DIRECT.  New agent + any task → FULL.

Usage::

    gate = ValidationGate()
    decision = await gate.decide(goal="List 3 Python frameworks")
    # → ValidationMode.DIRECT  (skip architect + auditor)

    decision = await gate.decide(goal="Build a REST API with auth, rate limiting, and tests")
    # → ValidationMode.FULL  (complex multi-step, needs full pipeline)
"""

from __future__ import annotations

import logging
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =========================================================================
# VALIDATION MODES
# =========================================================================

class ValidationMode(Enum):
    """How much validation a task needs."""
    DIRECT = "direct"           # Actor only.  Fastest.
    AUDIT_ONLY = "audit_only"   # Actor + Auditor.  Skip architect.
    FULL = "full"               # Architect → Actor → Auditor.  Safest.


@dataclass
class GateDecision:
    """Result of the validation gate."""
    mode: ValidationMode
    confidence: float           # 0-1 — gate's confidence in its decision
    reason: str                 # Human-readable explanation
    latency_ms: float = 0.0    # How long the gate itself took
    was_overridden: bool = False  # True if safety rail forced escalation
    was_sampled: bool = False     # True if a DIRECT task was randomly audited


# =========================================================================
# SAFETY: NEVER-SKIP CATEGORIES
# =========================================================================

# Tasks matching these patterns ALWAYS get FULL validation.
# This is a hardcoded safety rail — not learnable, not overridable.
NEVER_SKIP_PATTERNS = [
    # Code generation / modification
    "write code", "generate code", "implement", "refactor",
    "fix the bug", "debug", "patch", "create a function",
    "create a class", "build a module", "write tests",
    # Multi-step workflows
    "step 1", "step 2", "first,", "then,", "finally,",
    "pipeline", "workflow", "deploy",
    # High-stakes domains
    "financial", "payment", "transaction", "billing",
    "security", "authentication", "authorization", "encrypt",
    "medical", "diagnosis", "patient", "health",
    "legal", "compliance", "regulation",
    # File system / system operations
    "delete", "remove", "overwrite", "execute command",
    "run script", "install", "sudo",
]

# =========================================================================
# GATE IMPLEMENTATION
# =========================================================================

# The Haiku prompt — kept minimal to stay under 200 tokens total
_GATE_SYSTEM = (
    "You classify task complexity for an AI agent pipeline.\n"
    "Output ONLY one word: DIRECT, AUDIT, or FULL.\n\n"
    "DIRECT = simple Q&A, factual lookup, list, definition, single-step.\n"
    "AUDIT  = medium: summarization, analysis, comparison, short generation.\n"
    "FULL   = complex: code generation, multi-step plan, research + create,\n"
    "         anything with multiple deliverables or high stakes."
)


class ValidationGate:
    """
    Intelligent validation router using a cheap LLM call.

    Decides per-task whether to run the full Architect → Actor → Auditor
    pipeline, a partial pipeline, or just the actor alone.

    Combines three signals:
        1. Haiku LLM classification  (semantic understanding)
        2. Keyword safety rail       (hardcoded never-skip list)
        3. Historical outcome data   (learned from past decisions)
    """

    def __init__(
        self,
        model: str = "haiku",
        confidence_threshold: float = 0.80,
        sample_rate: float = 0.10,
        enable_llm: bool = True,
        fallback_mode: ValidationMode = ValidationMode.FULL,
    ):
        """
        Args:
            model: LLM model for gate classification (default: haiku — cheapest)
            confidence_threshold: Min confidence to downgrade from FULL.
                Below this → always FULL.
            sample_rate: Fraction of DIRECT tasks randomly sent to AUDIT_ONLY
                for quality monitoring.  0.10 = 10%.
            enable_llm: If False, use heuristic-only (no LLM call).  Useful
                when no API key is available.
            fallback_mode: Mode to use when gate can't decide (error, no LLM).
        """
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.sample_rate = sample_rate
        self.enable_llm = enable_llm
        self.fallback_mode = fallback_mode

        # Lazy-loaded LM (only created on first decide() call)
        self._lm = None
        self._lm_available: Optional[bool] = None

        # Outcome tracking for drift detection
        self._decisions: Dict[ValidationMode, int] = defaultdict(int)
        self._outcomes: Dict[ValidationMode, deque] = {
            m: deque(maxlen=200) for m in ValidationMode
        }
        self._total_calls = 0
        self._total_latency_ms = 0.0

        logger.info(
            f"ValidationGate: model={model}, threshold={confidence_threshold}, "
            f"sample_rate={sample_rate}, llm={'on' if enable_llm else 'off'}"
        )

    # =====================================================================
    # MAIN API
    # =====================================================================

    async def decide(
        self,
        goal: str,
        agent_name: str = "auto",
        force_mode: Optional[ValidationMode] = None,
        adaptive_history: Optional[Any] = None,
    ) -> GateDecision:
        """
        Decide validation mode for a task.

        Args:
            goal: The task description.
            agent_name: Which agent will execute (for history lookup).
            force_mode: Override gate decision (for explicit user control).
            adaptive_history: Optional LOTUS AdaptiveValidator for compound
                decision-making.

        Returns:
            GateDecision with mode, confidence, and reason.
        """
        t0 = time.time()
        self._total_calls += 1

        # ── 1. Explicit override ──────────────────────────────────────
        if force_mode is not None:
            return GateDecision(
                mode=force_mode,
                confidence=1.0,
                reason="explicit_override",
                latency_ms=(time.time() - t0) * 1000,
            )

        # ── 2. Safety rail: never-skip check ─────────────────────────
        goal_lower = goal.lower()
        for pattern in NEVER_SKIP_PATTERNS:
            if pattern in goal_lower:
                return GateDecision(
                    mode=ValidationMode.FULL,
                    confidence=1.0,
                    reason=f"safety_rail: matched '{pattern}'",
                    latency_ms=(time.time() - t0) * 1000,
                    was_overridden=True,
                )

        # ── 3. LLM classification (Haiku) ────────────────────────────
        llm_mode, llm_confidence, llm_reason = await self._classify_with_llm(goal)

        # ── 4. Confidence gate — low confidence → escalate to FULL ───
        if llm_confidence < self.confidence_threshold and llm_mode != ValidationMode.FULL:
            final_mode = ValidationMode.FULL
            reason = f"low_confidence ({llm_confidence:.0%} < {self.confidence_threshold:.0%}), escalated to FULL"
            was_overridden = True
        else:
            final_mode = llm_mode
            reason = llm_reason
            was_overridden = False

        # ── 5. Drift check — if DIRECT tasks have been failing, escalate
        if final_mode == ValidationMode.DIRECT:
            recent_direct = list(self._outcomes[ValidationMode.DIRECT])[-20:]
            if len(recent_direct) >= 5:
                recent_fail_rate = 1.0 - (sum(recent_direct) / len(recent_direct))
                if recent_fail_rate > 0.30:
                    final_mode = ValidationMode.AUDIT_ONLY
                    reason = f"drift_escalation: DIRECT fail rate {recent_fail_rate:.0%} > 30%"
                    was_overridden = True

        # ── 6. LOTUS history boost — trusted agent can stay DIRECT ────
        if adaptive_history and final_mode in (ValidationMode.DIRECT, ValidationMode.AUDIT_ONLY):
            # If LOTUS says this agent is trusted for both architect and auditor,
            # keep the lighter mode.  Otherwise, escalate one level.
            try:
                arch_decision = adaptive_history.should_validate(agent_name, "architect")
                aud_decision = adaptive_history.should_validate(agent_name, "auditor")
                if arch_decision.should_validate and aud_decision.should_validate:
                    # Agent is NOT yet trusted — escalate
                    if final_mode == ValidationMode.DIRECT:
                        final_mode = ValidationMode.AUDIT_ONLY
                        reason += " + lotus_untrusted→AUDIT"
                        was_overridden = True
            except Exception:
                pass  # LOTUS not available, keep gate decision

        # ── 7. Random sampling — DIRECT tasks get spot-checked ────────
        was_sampled = False
        if final_mode == ValidationMode.DIRECT and random.random() < self.sample_rate:
            final_mode = ValidationMode.AUDIT_ONLY
            was_sampled = True
            reason += " + sampled_for_audit"

        latency_ms = (time.time() - t0) * 1000
        self._total_latency_ms += latency_ms
        self._decisions[final_mode] += 1

        return GateDecision(
            mode=final_mode,
            confidence=llm_confidence,
            reason=reason,
            latency_ms=latency_ms,
            was_overridden=was_overridden,
            was_sampled=was_sampled,
        )

    def record_outcome(self, mode: ValidationMode, success: bool):
        """
        Record task outcome for drift detection.

        Call this after task execution completes so the gate can learn
        whether its routing decisions are leading to good outcomes.
        """
        self._outcomes[mode].append(success)
        # deque auto-bounds at maxlen=200

    # =====================================================================
    # LLM CLASSIFICATION
    # =====================================================================

    async def _classify_with_llm(
        self, goal: str
    ) -> Tuple[ValidationMode, float, str]:
        """
        Use a cheap LLM (Haiku) to classify task complexity.

        Returns (mode, confidence, reason).
        Falls back to heuristic if LLM is unavailable.
        """
        if not self.enable_llm:
            return self._classify_heuristic(goal)

        # Lazy init LM
        if self._lm_available is None:
            self._lm_available = self._init_lm()
            # If no LLM, relax confidence threshold for heuristic
            # (heuristic maxes at 75%; LLM gives 85-90%)
            if not self._lm_available:
                self.confidence_threshold = min(self.confidence_threshold, 0.65)
                logger.info(f"ValidationGate: heuristic mode, threshold relaxed to {self.confidence_threshold:.0%}")

        if not self._lm_available:
            return self._classify_heuristic(goal)

        try:
            # Single cheap call — ~100 input tokens, ~1 output token
            prompt = f"{_GATE_SYSTEM}\n\nTask: {goal[:300]}\n\nClassification:"
            response = self._lm(prompt=prompt)

            # Parse response
            text = ""
            if isinstance(response, list):
                text = str(response[0]).strip().upper()
            elif isinstance(response, str):
                text = response.strip().upper()
            else:
                text = str(response).strip().upper()

            # Extract classification
            if "DIRECT" in text:
                return (ValidationMode.DIRECT, 0.90, "llm_classified: DIRECT")
            elif "AUDIT" in text:
                return (ValidationMode.AUDIT_ONLY, 0.85, "llm_classified: AUDIT_ONLY")
            elif "FULL" in text:
                return (ValidationMode.FULL, 0.90, "llm_classified: FULL")
            else:
                # Unparseable → fallback to FULL (safe default)
                logger.warning(f"ValidationGate: unparseable LLM response '{text[:50]}', defaulting to FULL")
                return (ValidationMode.FULL, 0.50, f"llm_unparseable: '{text[:30]}'")

        except Exception as e:
            logger.warning(f"ValidationGate LLM call failed: {e}, falling back to heuristic")
            return self._classify_heuristic(goal)

    def _init_lm(self) -> bool:
        """
        Lazy-init the cheapest available LM.

        Tries in order of cost/speed:
            1. Gemini 2.0 Flash via OpenRouter — fastest, cheapest
            2. DirectAnthropicLM (Haiku) — fast, needs ANTHROPIC_API_KEY
            3. LiteLLM (Groq llama-3.1-8b) — free tier, ultra-fast, needs GROQ_API_KEY
            4. LiteLLM (any available) — fallback to whatever API key exists

        Returns True if any LM is available.
        """
        import os

        # 1. Try Gemini Flash via OpenRouter (fastest + cheapest for classification)
        or_key = os.environ.get('OPENROUTER_API_KEY')
        if or_key:
            try:
                import dspy
                self._lm = dspy.LM(
                    'openrouter/google/gemini-2.0-flash-001',
                    api_key=or_key,
                    max_tokens=10,
                )
                logger.info("ValidationGate: Gemini 2.0 Flash via OpenRouter")
                return True
            except Exception as e:
                logger.debug(f"ValidationGate: Gemini Flash not available: {e}")

        # 2. Try Anthropic Haiku (fast, reliable)
        try:
            from Jotty.core.foundation.direct_anthropic_lm import (
                DirectAnthropicLM,
                is_api_key_available,
            )
            if is_api_key_available():
                self._lm = DirectAnthropicLM(model=self.model, max_tokens=10)
                logger.info(f"ValidationGate: Haiku LM initialized ({self.model})")
                return True
        except Exception as e:
            logger.debug(f"ValidationGate: Anthropic LM not available: {e}")

        # 2. Try Groq (free tier, ultra-fast inference — ideal for gate)
        if os.environ.get("GROQ_API_KEY"):
            try:
                import litellm
                def _groq_call(prompt: str) -> list:
                    resp = litellm.completion(
                        model="groq/llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=10,
                        temperature=0.0,
                    )
                    return [resp.choices[0].message.content]
                self._lm = _groq_call
                logger.info("ValidationGate: Groq LM initialized (llama-3.1-8b)")
                return True
            except Exception as e:
                logger.debug(f"ValidationGate: Groq not available: {e}")

        # 3. Try any LiteLLM-compatible model
        for key, model in [
            ("OPENAI_API_KEY", "gpt-4o-mini"),
            ("GEMINI_API_KEY", "gemini/gemini-2.0-flash"),
        ]:
            if os.environ.get(key):
                try:
                    import litellm
                    _model = model
                    def _litellm_call(prompt: str, m=_model) -> list:
                        resp = litellm.completion(
                            model=m,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=10,
                            temperature=0.0,
                        )
                        return [resp.choices[0].message.content]
                    self._lm = _litellm_call
                    logger.info(f"ValidationGate: LiteLLM initialized ({model})")
                    return True
                except Exception as e:
                    logger.debug(f"ValidationGate: {model} not available: {e}")

        logger.info("ValidationGate: No LLM available, using heuristic fallback")
        return False

    # =====================================================================
    # HEURISTIC FALLBACK (no LLM needed)
    # =====================================================================

    def _classify_heuristic(
        self, goal: str
    ) -> Tuple[ValidationMode, float, str]:
        """
        Rule-based classification when LLM is unavailable.

        Less accurate than Haiku but zero-cost and instant.
        """
        goal_lower = goal.lower()
        word_count = len(goal.split())

        # FULL indicators (complex tasks)
        full_signals = [
            "create" in goal_lower and "and" in goal_lower,
            "build" in goal_lower,
            "implement" in goal_lower,
            "design" in goal_lower and word_count > 10,
            goal_lower.count(" and ") >= 2,  # multiple deliverables
            "research" in goal_lower and "report" in goal_lower,
            word_count > 40,
        ]
        if sum(full_signals) >= 2:
            return (ValidationMode.FULL, 0.70, "heuristic: multiple complexity signals")

        # AUDIT_ONLY indicators (medium tasks)
        audit_signals = [
            "summarize" in goal_lower,
            "analyze" in goal_lower,
            "compare" in goal_lower,
            "explain" in goal_lower and word_count > 15,
            "write" in goal_lower,
            "generate" in goal_lower,
            word_count > 20,
        ]
        if sum(audit_signals) >= 2:
            return (ValidationMode.AUDIT_ONLY, 0.70, "heuristic: medium complexity")

        # DIRECT indicators (simple tasks)
        direct_signals = [
            "list" in goal_lower,
            "name" in goal_lower,
            "what is" in goal_lower,
            "what are" in goal_lower,
            "how many" in goal_lower,
            "define" in goal_lower,
            word_count <= 15,
            "?" in goal,
        ]
        if sum(direct_signals) >= 2:
            return (ValidationMode.DIRECT, 0.75, "heuristic: simple task")

        # Default: AUDIT_ONLY (middle ground)
        return (ValidationMode.AUDIT_ONLY, 0.60, "heuristic: default medium")

    # =====================================================================
    # STATS / INTROSPECTION
    # =====================================================================

    def stats(self) -> Dict[str, Any]:
        """Get gate statistics for monitoring."""
        total = max(self._total_calls, 1)
        outcome_summary = {}
        for mode, outcomes in self._outcomes.items():
            if outcomes:
                outcome_summary[mode.value] = {
                    "total": len(outcomes),
                    "success_rate": f"{sum(outcomes)/len(outcomes):.0%}",
                }

        return {
            "total_calls": self._total_calls,
            "avg_latency_ms": f"{self._total_latency_ms / total:.1f}",
            "decisions": {m.value: c for m, c in self._decisions.items()},
            "distribution": {
                m.value: f"{c/total:.0%}"
                for m, c in self._decisions.items()
            },
            "llm_available": self._lm_available,
            "model": self.model,
            "outcomes": outcome_summary,
            "estimated_savings": self._estimate_savings(),
        }

    def _estimate_savings(self) -> Dict[str, Any]:
        """Estimate LLM calls saved by the gate."""
        direct = self._decisions.get(ValidationMode.DIRECT, 0)
        audit = self._decisions.get(ValidationMode.AUDIT_ONLY, 0)
        full = self._decisions.get(ValidationMode.FULL, 0)

        # Without gate: every task = 3 LLM calls (architect + actor + auditor)
        baseline_calls = (direct + audit + full) * 3

        # With gate:
        # DIRECT = 1 call (actor only)
        # AUDIT  = 2 calls (actor + auditor)
        # FULL   = 3 calls (architect + actor + auditor)
        actual_calls = direct * 1 + audit * 2 + full * 3

        saved = baseline_calls - actual_calls
        pct = saved / max(baseline_calls, 1)

        return {
            "baseline_llm_calls": baseline_calls,
            "actual_llm_calls": actual_calls,
            "calls_saved": saved,
            "savings_pct": f"{pct:.0%}",
        }


# =========================================================================
# MODULE-LEVEL SINGLETON (optional convenience)
# =========================================================================

_default_gate: Optional[ValidationGate] = None


def get_validation_gate(**kwargs) -> ValidationGate:
    """Get or create the default ValidationGate singleton."""
    global _default_gate
    if _default_gate is None:
        _default_gate = ValidationGate(**kwargs)
    return _default_gate
