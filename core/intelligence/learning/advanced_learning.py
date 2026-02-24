"""
Advanced Learning Components
============================

Research-backed techniques integrated into Jotty's learning system:

1. Reflexion        — Natural language failure reflection (Shinn et al. 2023)
2. FewShotCurator   — Auto-curate best episodes as DSPy few-shot examples
3. DomainDSPyOptimizer — BootstrapFewShotWithRandomSearch per crystallized domain
4. VoyagerSkillLib  — Auto-extract reusable skill patterns from successes

Infrastructure integration:
- LearningStore (SQLite) for persistence
- LearningService for episode recording
- TDLambdaLearner for value updates
- DSPy for structured LLM interaction

Note: LLM-as-Judge lives in LearningService.llm_judge_quality_with_feedback()
(Sonnet, 5-dimension rubric, structural digest) — not duplicated here.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# 1. REFLEXION — Natural language failure reflection (Shinn et al.)
# =============================================================================


class Reflexion:
    """Reflect on execution failures, store reflections, and retrieve on retry.

    Implements the Reflexion paper's key insight: when an agent fails, it
    generates a natural language "reflection" on what went wrong. On the
    next attempt at a similar task, the reflection is injected into the
    prompt, preventing repeated mistakes.

    Integration:
    - _post_execute_learning calls reflect_on_failure() for failed episodes
    - _pre_execute_learning calls get_relevant_reflections() to inject context
    - ReflectionRecord (already in LearningStore) persists reflections
    """

    _instance: Optional["Reflexion"] = None

    def __init__(self) -> None:
        self._lm: Any = None
        self._reflect_module: Any = None
        self._init_attempts = 0

    @classmethod
    def get_instance(cls) -> "Reflexion":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_init(self) -> bool:
        if self._reflect_module is not None:
            return True
        if self._init_attempts > 2:
            return False
        self._init_attempts += 1
        try:
            import dspy

            from Jotty.core.infrastructure.foundation.unified_lm_provider import (
                get_fast_lm,
            )

            self._lm = get_fast_lm()

            class ReflectionSignature(dspy.Signature):
                """Reflect on a failed AI execution to learn what went wrong.
                Produce a concise, actionable reflection that will help on retry."""

                goal: str = dspy.InputField(desc="What the agent was trying to do")
                output_excerpt: str = dspy.InputField(
                    desc="What the agent actually produced (truncated)"
                )
                error_info: str = dspy.InputField(desc="Error type and message, if any")
                observation: str = dspy.OutputField(desc="What happened (factual, 1-2 sentences)")
                analysis: str = dspy.OutputField(desc="Why it failed (root cause, 1-2 sentences)")
                adjustment: str = dspy.OutputField(
                    desc="What to do differently next time (actionable, 1-2 sentences)"
                )

            self._reflect_module = dspy.ChainOfThought(ReflectionSignature)
            return True
        except Exception as e:
            logger.debug(f"Reflexion init failed: {e}")
            return False

    def reflect_on_failure(
        self,
        episode_id: str,
        unit_name: str,
        goal: str,
        output: str,
        error_type: str = "",
        error_message: str = "",
        step: int = 0,
    ) -> Optional[Dict[str, str]]:
        """Generate a reflection on a failed episode and persist it.

        Args:
            episode_id: The failed episode's ID
            unit_name: Name of the agent/swarm that failed
            goal: What was being attempted
            output: What was actually produced
            error_type: Classification of the error
            error_message: Error details

        Returns:
            Dict with observation/analysis/adjustment, or None on failure
        """
        if not self._ensure_init():
            return None

        try:
            import dspy

            from .learning_store import LearningStore, ReflectionRecord

            error_info = (
                f"{error_type}: {error_message}"
                if error_type
                else "No explicit error; output was low quality"
            )

            with dspy.context(lm=self._lm):
                result = self._reflect_module(
                    goal=goal,
                    output_excerpt=output[:1500],
                    error_info=error_info[:500],
                )

            reflection = ReflectionRecord(
                reflection_id=f"refl_{uuid.uuid4().hex[:12]}",
                episode_id=episode_id,
                step=step,
                unit_name=unit_name,
                observation=str(result.observation)[:500],
                analysis=str(result.analysis)[:500],
                adjustment=str(result.adjustment)[:500],
                applied=False,
            )

            store = LearningStore.get_instance()
            store.save_reflection(reflection)
            logger.info(
                f"Reflexion: stored reflection for {unit_name} "
                f"(episode {episode_id[:12]}): {reflection.adjustment[:80]}"
            )
            return {
                "observation": reflection.observation,
                "analysis": reflection.analysis,
                "adjustment": reflection.adjustment,
            }
        except Exception as e:
            logger.debug(f"Reflexion generation failed: {e}")
            return None

    def get_relevant_reflections(self, unit_name: str, limit: int = 3) -> List[str]:
        """Retrieve recent reflections for a unit, formatted for prompt injection.

        Returns actionable adjustment strings ready to inject into the agent's
        system prompt as "lessons learned from past failures."
        """
        try:
            from .learning_store import LearningStore

            store = LearningStore.get_instance()
            reflections = store.get_reflections(unit_name=unit_name, limit=limit)
            if not reflections:
                return []
            return [
                f"[Past failure] {r.observation} → Fix: {r.adjustment}"
                for r in reflections
                if r.adjustment
            ]
        except Exception as e:
            logger.debug(f"Reflexion retrieval failed: {e}")
            return []


# =============================================================================
# 2. FEW-SHOT CURATOR — Auto-curate best episodes as DSPy examples
# =============================================================================


class FewShotCurator:
    """Auto-curate successful episodes from LearningStore as DSPy examples.

    Bridges the gap between Jotty's episode recording and DSPy's few-shot
    optimization. Queries the best episodes (high quality + success) and
    converts them into dspy.Example objects for BootstrapFewShot or MIPRO.

    Integration:
    - Agents can call get_examples() to seed their DSPy modules
    - BootstrapFewShot can use curated examples as the training set
    - MIPRO can use them for prompt optimization
    """

    _instance: Optional["FewShotCurator"] = None

    @classmethod
    def get_instance(cls) -> "FewShotCurator":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_examples(
        self,
        domain: str = "",
        task_type: str = "",
        n: int = 5,
        min_quality: float = 0.7,
    ) -> List[Any]:
        """Query best episodes and convert to dspy.Example objects.

        Each Example has fields: goal, approach, output — matching the
        typical agent signature pattern.

        Args:
            domain: Filter by domain (e.g., "coding", "research")
            task_type: Filter by task type
            n: Number of examples to return
            min_quality: Minimum quality threshold

        Returns:
            List of dspy.Example objects ready for BootstrapFewShot
        """
        try:
            import dspy

            from .learning_store import LearningStore

            store = LearningStore.get_instance()
            episodes = store.query_episodes(
                domain=domain or None,
                task_type=task_type or None,
                success_only=True,
                limit=n * 3,  # Over-fetch then filter by quality
            )

            # Filter by quality and sort descending
            episodes = [ep for ep in episodes if ep.quality >= min_quality]
            episodes.sort(key=lambda e: e.quality, reverse=True)
            episodes = episodes[:n]

            examples = []
            for ep in episodes:
                goal = ep.context.get("task", ep.context.get("goal", ep.task_type))
                approach = ep.action.get("paradigm", ep.action.get("model", ""))
                output_text = ep.outcome.get("content", ep.outcome.get("summary", ""))

                if not goal or not output_text:
                    continue

                ex = dspy.Example(
                    goal=str(goal)[:500],
                    approach=str(approach)[:200],
                    output=str(output_text)[:2000],
                ).with_inputs("goal")
                examples.append(ex)

            logger.debug(
                f"FewShotCurator: curated {len(examples)} examples "
                f"(domain={domain}, task_type={task_type})"
            )
            return examples
        except Exception as e:
            logger.debug(f"FewShotCurator failed: {e}")
            return []

    def get_distilled_examples(
        self,
        domain: str = "",
        agent_name: str = "",
        n: int = 10,
    ) -> List[Any]:
        """Get DSPy examples from distilled lessons (more concise than episodes).

        Uses the DistilledLesson table for compact, LLM-extracted facts.
        """
        try:
            import dspy

            from .learning_store import LearningStore

            store = LearningStore.get_instance()
            lessons = store.get_distilled_lessons(
                domain=domain or None,
                agent_name=agent_name or None,
                min_confidence=0.5,
                limit=n,
            )

            return [
                dspy.Example(
                    context=lesson.applicability,
                    lesson=lesson.lesson,
                    confidence=str(lesson.confidence),
                ).with_inputs("context")
                for lesson in lessons
                if lesson.lesson
            ]
        except Exception as e:
            logger.debug(f"FewShotCurator distilled failed: {e}")
            return []

    def optimize_module(
        self,
        module: Any,
        domain: str = "",
        task_type: str = "",
        n_examples: int = 10,
        max_bootstrapped: int = 4,
    ) -> Any:
        """Run DSPy BootstrapFewShot on a module using curated episodes.

        This is the key integration: episodes recorded by LearningService
        are used to optimize DSPy agent signatures automatically.

        Args:
            module: DSPy module to optimize
            domain: Episode domain filter
            task_type: Episode task type filter
            n_examples: Number of training examples
            max_bootstrapped: Max bootstrapped demos

        Returns:
            Optimized DSPy module (or original if optimization fails)
        """
        try:
            import dspy

            from Jotty.core.infrastructure.foundation.unified_lm_provider import (
                get_fast_lm,
            )

            examples = self.get_examples(domain, task_type, n=n_examples, min_quality=0.75)
            if len(examples) < 3:
                logger.debug("FewShotCurator: not enough examples for optimization")
                return module

            optimizer = dspy.BootstrapFewShot(
                max_bootstrapped_demos=max_bootstrapped,
                max_labeled_demos=min(len(examples), 8),
            )

            fast_lm = get_fast_lm()
            with dspy.context(lm=fast_lm):
                optimized = optimizer.compile(module, trainset=examples)

            logger.info(f"FewShotCurator: optimized module with {len(examples)} examples")
            return optimized
        except Exception as e:
            logger.debug(f"FewShotCurator optimization failed: {e}")
            return module


# =============================================================================
# 3. DOMAIN DSPy OPTIMIZER — BootstrapFewShotWithRandomSearch per domain
# =============================================================================


def _get_domain_task_classes():
    """Lazy-create DSPy signature and module classes (avoids import-time dspy dependency)."""
    import dspy

    class DomainTaskSignature(dspy.Signature):
        """Execute a domain-specific task and produce high-quality output."""

        task_description: str = dspy.InputField(desc="What to generate or do")
        domain: str = dspy.InputField(desc="Domain (e.g. plantuml, mermaid, coding)")
        output: str = dspy.OutputField(desc="Complete, high-quality output for the task")

    class ValidationSignature(dspy.Signature):
        """Validate domain output and identify specific errors to fix."""

        original_task: str = dspy.InputField(desc="The original task description")
        generated_output: str = dspy.InputField(desc="The generated output to validate")
        domain: str = dspy.InputField(desc="Domain (e.g. plantuml, mermaid)")
        is_valid: bool = dspy.OutputField(desc="Whether the output is structurally valid")
        errors: str = dspy.OutputField(desc="List of specific errors found, or 'none'")

    class RefinementSignature(dspy.Signature):
        """Fix errors in generated output based on validation feedback."""

        original_task: str = dspy.InputField(desc="The original task description")
        draft_output: str = dspy.InputField(desc="The draft output with errors")
        validation_errors: str = dspy.InputField(desc="Specific errors to fix")
        domain: str = dspy.InputField(desc="Domain (e.g. plantuml, mermaid)")
        output: str = dspy.OutputField(desc="Corrected output with all errors fixed")

    class DomainTaskModule(dspy.Module):
        """Single-stage: generate only. Optimized via BootstrapFewShot."""

        def __init__(self):
            super().__init__()
            self.generate = dspy.ChainOfThought(DomainTaskSignature)

        def forward(self, task_description: str, domain: str):
            return self.generate(task_description=task_description, domain=domain)

    class DomainTaskPipeline(dspy.Module):
        """Multi-stage: Generate -> Validate -> Refine (self-healing).

        Stage 1: Generate initial output (ChainOfThought)
        Stage 2: Validate the output for structural/syntax errors
        Stage 3: If errors found, refine and fix them

        Each stage is independently optimizable by DSPy.
        """

        def __init__(self):
            super().__init__()
            self.generate = dspy.ChainOfThought(DomainTaskSignature)
            self.validate = dspy.Predict(ValidationSignature)
            self.refine = dspy.ChainOfThought(RefinementSignature)

        def forward(self, task_description: str, domain: str):
            # Stage 1: Generate
            gen_result = self.generate(task_description=task_description, domain=domain)
            draft = gen_result.output

            # Stage 2: Validate
            val_result = self.validate(
                original_task=task_description,
                generated_output=draft,
                domain=domain,
            )

            # Stage 3: Refine if errors found
            if not val_result.is_valid or (
                val_result.errors and val_result.errors.lower() != "none"
            ):
                refined = self.refine(
                    original_task=task_description,
                    draft_output=draft,
                    validation_errors=val_result.errors,
                    domain=domain,
                )
                return refined

            return gen_result

    return DomainTaskSignature, DomainTaskModule, DomainTaskPipeline


class DomainDSPyOptimizer:
    """Optimize a DSPy module for a specific domain using gold examples.

    Combines three data sources:
    1. Successful episodes from LearningStore (probation runs)
    2. Distilled lessons (compact, high-signal)
    3. External gold data (web research examples, teacher-crafted)

    The optimized module is saved per domain and loaded by the
    crystallized agent at inference time.

    Usage:
        optimizer = DomainDSPyOptimizer()

        # Add external gold data (from web research, teacher)
        optimizer.add_gold_examples("plantuml", [
            {"task": "Generate sequence diagram for auth flow",
             "output": "@startuml\nparticipant User\n...@enduml"},
        ])

        # Optimize (uses episodes + gold data + distilled lessons)
        module = optimizer.optimize("plantuml")

        # Use at inference
        result = module(task_description="Generate class diagram", domain="plantuml")
    """

    _instance: Optional["DomainDSPyOptimizer"] = None
    _gold_data: Dict[str, List[Dict[str, str]]]  # domain → [{task, output}]

    def __init__(self):
        from pathlib import Path

        self._save_dir = Path.home() / "jotty" / "learning" / "dspy_optimized"
        self._gold_data = {}
        self._save_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_instance(cls) -> "DomainDSPyOptimizer":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def add_gold_examples(self, domain: str, examples: List[Dict[str, str]]) -> None:
        """Register external gold examples for a domain.

        Args:
            domain: Domain key (e.g. "plantuml", "mermaid")
            examples: List of {task: str, output: str} dicts
        """
        if domain not in self._gold_data:
            self._gold_data[domain] = []
        self._gold_data[domain].extend(examples)
        logger.info(
            f"DomainDSPyOptimizer: added {len(examples)} gold examples "
            f"for {domain} (total={len(self._gold_data[domain])})"
        )

    def _gather_training_data(self, domain: str, min_quality: float = 0.6) -> List:
        """Gather DSPy Examples from all sources for a domain."""
        import dspy

        examples = []

        # Source 1: Successful episodes from LearningStore
        try:
            from .learning_store import LearningStore

            store = LearningStore.get_instance()
            episodes = store.query_episodes(domain=domain, success_only=True, limit=30)
            episodes = [ep for ep in episodes if ep.quality >= min_quality]
            episodes.sort(key=lambda e: e.quality, reverse=True)

            for ep in episodes[:15]:
                goal = ep.context.get("task", ep.context.get("goal", ""))
                output = ep.outcome.get("content", ep.outcome.get("summary", ""))
                if goal and output and len(output) > 30:
                    examples.append(
                        dspy.Example(
                            task_description=str(goal)[:500],
                            domain=domain,
                            output=str(output)[:3000],
                        ).with_inputs("task_description", "domain")
                    )
        except Exception as e:
            logger.debug(f"DomainDSPyOptimizer: episode fetch failed: {e}")

        # Source 2: External gold data (web research + teacher)
        for item in self._gold_data.get(domain, []):
            task = item.get("task", item.get("description", ""))
            output = item.get("output", item.get("plantuml_code", ""))
            if task and output:
                examples.append(
                    dspy.Example(
                        task_description=str(task)[:500],
                        domain=domain,
                        output=str(output)[:3000],
                    ).with_inputs("task_description", "domain")
                )

        # Source 3: Distilled lessons as extra context
        try:
            from .learning_store import LearningStore

            store = LearningStore.get_instance()
            lessons = store.get_distilled_lessons(
                domain=domain, min_confidence=0.5, limit=10, hierarchical=True
            )
            for lesson in lessons:
                if lesson.lesson and lesson.applicability:
                    examples.append(
                        dspy.Example(
                            task_description=f"[Lesson] {lesson.applicability}",
                            domain=domain,
                            output=lesson.lesson,
                        ).with_inputs("task_description", "domain")
                    )
        except Exception as e:
            logger.debug(f"DomainDSPyOptimizer: lesson fetch failed: {e}")

        logger.info(
            f"DomainDSPyOptimizer: gathered {len(examples)} training examples " f"for {domain}"
        )
        return examples

    def optimize(
        self,
        domain: str,
        max_bootstrapped: int = 4,
        max_labeled: int = 8,
        num_candidate_programs: int = 6,
        metric=None,
    ):
        """Run BootstrapFewShotWithRandomSearch optimization for a domain.

        Uses the 3-stage DomainTaskPipeline (generate→validate→refine)
        and tries `num_candidate_programs` random configurations, keeping
        the best. Sonnet acts as teacher, Haiku as student at inference.
        """
        import dspy

        _, _, Pipeline = _get_domain_task_classes()
        module = Pipeline()
        examples = self._gather_training_data(domain)

        if len(examples) < 3:
            logger.warning(
                f"DomainDSPyOptimizer: only {len(examples)} examples for "
                f"{domain}, skipping optimization"
            )
            return module

        if metric is None:

            def metric(example, prediction, trace=None):
                pred = getattr(prediction, "output", "")
                if not pred or len(pred) < 20:
                    return 0.0
                gold = getattr(example, "output", "")
                score = 0.3
                if len(pred) > 50:
                    score += 0.2
                gold_words = set(gold.lower().split())
                pred_words = set(pred.lower().split())
                if gold_words:
                    overlap = len(gold_words & pred_words) / len(gold_words)
                    score += 0.5 * min(overlap, 1.0)
                return min(score, 1.0)

        try:
            from Jotty.core.infrastructure.foundation.unified_lm_provider import (
                UnifiedLMProvider,
            )

            teacher_lm = UnifiedLMProvider.create_lm(provider="anthropic", model="sonnet")

            optimizer = dspy.BootstrapFewShotWithRandomSearch(
                metric=metric,
                max_bootstrapped_demos=max_bootstrapped,
                max_labeled_demos=min(len(examples), max_labeled),
                num_candidate_programs=num_candidate_programs,
            )

            with dspy.context(lm=teacher_lm):
                optimized = optimizer.compile(module, trainset=examples[:12])

            save_path = str(self._save_dir / f"{domain}_task_pipeline.json")
            optimized.save(save_path)
            logger.info(
                f"DomainDSPyOptimizer: optimized {domain} pipeline with "
                f"{len(examples)} examples ({num_candidate_programs} candidates), "
                f"saved to {save_path}"
            )
            return optimized

        except Exception as e:
            logger.warning(f"DomainDSPyOptimizer: optimization failed: {e}")
            return module

    def load_optimized(self, domain: str):
        """Load a previously optimized module from disk."""
        # Try pipeline first (new format), fall back to legacy module
        pipeline_path = self._save_dir / f"{domain}_task_pipeline.json"
        legacy_path = self._save_dir / f"{domain}_task_module.json"

        if pipeline_path.exists():
            path, use_pipeline = pipeline_path, True
        elif legacy_path.exists():
            path, use_pipeline = legacy_path, False
        else:
            return None

        try:
            _, Module, Pipeline = _get_domain_task_classes()
            module = Pipeline() if use_pipeline else Module()
            module.load(str(path))
            logger.debug(
                f"DomainDSPyOptimizer: loaded optimized {'pipeline' if use_pipeline else 'module'} for {domain}"
            )
            return module
        except Exception as e:
            logger.debug(f"DomainDSPyOptimizer: load failed for {domain}: {e}")
            return None

    def has_optimized(self, domain: str) -> bool:
        """Check if an optimized module exists for this domain."""
        return (self._save_dir / f"{domain}_task_pipeline.json").exists() or (
            self._save_dir / f"{domain}_task_module.json"
        ).exists()


# =============================================================================
# 4. VOYAGER-STYLE SKILL LIBRARY — Auto-extract reusable patterns
# =============================================================================


class VoyagerSkillLib:
    """Auto-extract and grow a library of reusable skill patterns.

    Inspired by Voyager (NVIDIA): when a task succeeds with high quality,
    extract the approach as a reusable "skill pattern" in PatternRecord.
    Future tasks can look up relevant patterns and reuse proven strategies.

    Integration:
    - Triggered by _post_execute_learning when quality > threshold
    - Patterns stored in LearningStore's patterns table
    - _pre_execute_learning retrieves patterns via get_applicable_patterns()
    """

    _instance: Optional["VoyagerSkillLib"] = None

    @classmethod
    def get_instance(cls) -> "VoyagerSkillLib":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def extract_skill_pattern(
        self,
        episode_id: str,
        domain: str,
        task_type: str,
        goal: str,
        approach: str,
        quality: float,
    ) -> Optional[str]:
        """Extract a reusable skill pattern from a successful high-quality episode.

        Only extracts if quality >= 0.8 to ensure patterns are reliable.

        Returns:
            Pattern ID if extracted, None otherwise
        """
        if quality < 0.8:
            return None

        try:
            from .learning_store import LearningStore, PatternRecord

            store = LearningStore.get_instance()

            # Check for existing similar pattern to avoid duplicates
            existing = store.get_patterns(domain=domain, pattern_type="success_strategy", limit=20)
            for p in existing:
                if p.description == goal[:200]:
                    # Boost confidence of existing pattern
                    p.confidence = min(1.0, p.confidence + 0.05)
                    p.evidence_count += 1
                    store.save_pattern(p)
                    return p.pattern_id

            pattern = PatternRecord(
                pattern_id=f"skill_{uuid.uuid4().hex[:12]}",
                source_domain=domain,
                pattern_type="success_strategy",
                description=goal[:200],
                conditions={"task_type": task_type, "min_quality": 0.7},
                recommendation=approach[:500],
                confidence=min(1.0, quality),
                evidence_count=1,
                applicable_domains=[domain],
                metadata={"episode_id": episode_id, "extracted_quality": quality},
            )
            store.save_pattern(pattern)
            logger.info(f"VoyagerSkillLib: extracted pattern '{goal[:60]}' (q={quality:.2f})")
            return pattern.pattern_id
        except Exception as e:
            logger.debug(f"VoyagerSkillLib extraction failed: {e}")
            return None

    def get_applicable_patterns(
        self,
        domain: str,
        task_type: str = "",
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get proven skill patterns applicable to current task.

        Returns patterns sorted by confidence, ready for prompt injection.
        """
        try:
            from .learning_store import LearningStore

            store = LearningStore.get_instance()
            patterns = store.get_patterns(
                domain=domain,
                pattern_type="success_strategy",
                min_confidence=0.5,
                limit=limit,
            )
            return [
                {
                    "strategy": p.recommendation,
                    "confidence": p.confidence,
                    "evidence": p.evidence_count,
                    "description": p.description,
                }
                for p in patterns
            ]
        except Exception as e:
            logger.debug(f"VoyagerSkillLib retrieval failed: {e}")
            return []
