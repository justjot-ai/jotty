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

import dataclasses
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
    """Lazy-create DSPy signature and module (avoids import-time dspy dependency)."""
    import dspy

    class DomainTaskSignature(dspy.Signature):
        """Execute a domain-specific task and produce high-quality output."""

        task_description: str = dspy.InputField(desc="What to generate or do")
        domain: str = dspy.InputField(desc="Domain (e.g. plantuml, mermaid, coding)")
        output: str = dspy.OutputField(desc="Complete, high-quality output for the task")

    class DomainTaskModule(dspy.Module):
        """ChainOfThought generation + cheap programmatic validation + one retry.

        No extra LLM calls for validation — just checks structural markers
        (e.g. @startuml/@enduml for PlantUML) and retries once with a hint.
        """

        def __init__(self):
            super().__init__()
            self.generate = dspy.ChainOfThought(DomainTaskSignature)

        def forward(self, task_description: str, domain: str):
            result = self.generate(task_description=task_description, domain=domain)
            output = getattr(result, "output", "")

            vr = validate_output(output, domain)
            if not vr.valid:
                hint = vr.issue or "Output failed validation."
                retry_desc = f"{task_description}\n\nCRITICAL: {hint}"
                result = self.generate(task_description=retry_desc, domain=domain)

            return result

    return DomainTaskSignature, DomainTaskModule


_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could of in to for on with at by from as into "
    "through during before after above below between under again further then "
    "once here there when where why how all both each few more most other some "
    "such no nor not only own same so than too very and but or if".split()
)


def _gold_metric(example, prediction, trace=None) -> float:
    """Quality-aware DSPy metric: structural quality + key-term F1 + coherence.

    Replaces naive word-overlap with a 4-dimension score:
      1. Key-term F1 (content preservation, stop-words excluded)
      2. Structural quality (formatting, sentence structure)
      3. Length appropriateness (within reasonable range of gold)
      4. Coherence (lexical diversity, no degenerate repetition)
    """
    pred = getattr(prediction, "output", "")
    if not pred or len(pred) < 20:
        return 0.0
    gold = getattr(example, "output", "")
    if not gold:
        return 0.3 if len(pred) > 50 else 0.1

    # 1. Key-term F1 (stop-words removed, lowercased)
    gold_terms = {w for w in gold.lower().split() if w not in _STOPWORDS and len(w) > 2}
    pred_terms = {w for w in pred.lower().split() if w not in _STOPWORDS and len(w) > 2}
    if gold_terms and pred_terms:
        recall = len(gold_terms & pred_terms) / len(gold_terms)
        precision = len(gold_terms & pred_terms) / len(pred_terms)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    elif gold_terms:
        f1 = 0.0
    else:
        f1 = 0.5

    # 2. Structural quality (formatting signals present in output)
    structure = 0.0
    if any(marker in pred for marker in ("\n", "- ", "* ", "1.", "## ", "```")):
        structure += 0.5
    sentences = [s.strip() for s in pred.split(".") if len(s.strip()) > 10]
    if len(sentences) >= 2:
        structure += 0.3
    # Domain-specific validation (library → structural cascade)
    domain_str = getattr(example, "domain", "")
    if domain_str:
        vr = validate_output(pred, domain_str)
        if vr.valid:
            structure += 0.2 * vr.confidence  # library=1.0 → full credit, structural=0.5 → half
        elif vr.method == "library" and vr.confidence >= 0.9:
            return 0.0  # definitive parser says invalid → zero score
    structure = min(structure, 1.0)

    # 3. Length appropriateness (0.3x-3x of gold is acceptable range)
    ratio = len(pred) / max(len(gold), 1)
    if 0.3 <= ratio <= 3.0:
        length_score = 1.0 - 0.3 * abs(ratio - 1.0)
    else:
        length_score = max(0.0, 0.5 - 0.2 * abs(ratio - 1.0))
    length_score = max(0.0, min(1.0, length_score))

    # 4. Coherence (lexical diversity — catches degenerate repetition)
    words = pred.lower().split()
    uniqueness = len(set(words)) / max(len(words), 1) if words else 0
    coherence = min(1.0, uniqueness / 0.4) if uniqueness < 0.4 else 1.0

    score = 0.35 * f1 + 0.25 * structure + 0.20 * length_score + 0.20 * coherence
    return round(min(score, 1.0), 4)


@dataclasses.dataclass
class ValidationResult:
    """Structured result from output validation cascade.

    Used by DSPy metric, DomainTaskModule retry, and episode recording.
    """

    valid: bool
    method: str  # "library", "structural", "none"
    confidence: float  # 0.0-1.0: how certain we are (1.0 = definitive parser)
    errors: List[str] = dataclasses.field(default_factory=list)

    @property
    def issue(self) -> str:
        """Backward-compat: single error string (or empty)."""
        return "; ".join(self.errors) if self.errors else ""


# ---------------------------------------------------------------------------
# Per-domain library validators (Level 1)
# Each returns ValidationResult. Only called when the library is available.
# ---------------------------------------------------------------------------


def _validate_python(code: str) -> ValidationResult:
    """ast.parse() — definitive syntax check."""
    import ast as _ast

    try:
        _ast.parse(code)
        return ValidationResult(valid=True, method="library", confidence=1.0)
    except SyntaxError as e:
        return ValidationResult(
            valid=False,
            method="library",
            confidence=1.0,
            errors=[f"Python syntax error: {e.msg} (line {e.lineno})"],
        )


def _validate_sql(code: str) -> ValidationResult:
    """sqlglot.transpile() — dialect-aware SQL parsing."""
    try:
        import sqlglot

        sqlglot.transpile(code)
        return ValidationResult(valid=True, method="library", confidence=0.95)
    except Exception as e:
        return ValidationResult(
            valid=False,
            method="library",
            confidence=0.95,
            errors=[f"SQL syntax error: {str(e)[:200]}"],
        )


def _validate_json(text: str) -> ValidationResult:
    """json.loads() — definitive JSON check."""
    import json as _json

    try:
        _json.loads(text)
        return ValidationResult(valid=True, method="library", confidence=1.0)
    except _json.JSONDecodeError as e:
        return ValidationResult(
            valid=False,
            method="library",
            confidence=1.0,
            errors=[f"JSON error: {e.msg} (pos {e.pos})"],
        )


def _validate_yaml(text: str) -> ValidationResult:
    """yaml.safe_load() — definitive YAML check."""
    try:
        import yaml as _yaml

        _yaml.safe_load(text)
        return ValidationResult(valid=True, method="library", confidence=1.0)
    except Exception as e:
        return ValidationResult(
            valid=False,
            method="library",
            confidence=1.0,
            errors=[f"YAML error: {str(e)[:200]}"],
        )


def _validate_html(text: str) -> ValidationResult:
    """html.parser — permissive but catches gross errors."""
    from html.parser import HTMLParser

    class _StrictParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.errors: List[str] = []
            self._stack: List[str] = []
            _VOID = {
                "br",
                "hr",
                "img",
                "input",
                "meta",
                "link",
                "area",
                "base",
                "col",
                "embed",
                "source",
                "track",
                "wbr",
            }
            self._void = _VOID

        def handle_starttag(self, tag, attrs):
            if tag.lower() not in self._void:
                self._stack.append(tag.lower())

        def handle_endtag(self, tag):
            if self._stack and self._stack[-1] == tag.lower():
                self._stack.pop()
            elif tag.lower() not in self._void:
                self.errors.append(f"Unexpected closing </{tag}>")

    parser = _StrictParser()
    try:
        parser.feed(text)
        if parser.errors:
            return ValidationResult(
                valid=False,
                method="library",
                confidence=0.7,
                errors=parser.errors[:5],
            )
        return ValidationResult(valid=True, method="library", confidence=0.7)
    except Exception as e:
        return ValidationResult(
            valid=False,
            method="library",
            confidence=0.7,
            errors=[f"HTML parse error: {str(e)[:200]}"],
        )


def _validate_plantuml_server(code: str) -> Optional[ValidationResult]:
    """Validate PlantUML by rendering via the public HTTP server.

    Uses /svg endpoint (text-based, so we can detect error messages).
    Returns None if the server is unreachable (fall back to structural).
    """
    import urllib.error
    import urllib.request
    import zlib

    def _plantuml_encode(text: str) -> str:
        compressed = zlib.compress(text.encode("utf-8"))[2:-4]
        _ENCODING = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
        result = []
        for i in range(0, len(compressed), 3):
            b1 = compressed[i]
            b2 = compressed[i + 1] if i + 1 < len(compressed) else 0
            b3 = compressed[i + 2] if i + 2 < len(compressed) else 0
            result.append(_ENCODING[b1 >> 2])
            result.append(_ENCODING[((b1 & 0x3) << 4) | (b2 >> 4)])
            result.append(_ENCODING[((b2 & 0xF) << 2) | (b3 >> 6)])
            result.append(_ENCODING[b3 & 0x3F])
        return "".join(result)

    try:
        encoded = _plantuml_encode(code)
        url = f"http://www.plantuml.com/plantuml/svg/{encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "Jotty/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        if "Syntax Error" in body or "ErrorURLEncoder" in body:
            return ValidationResult(
                valid=False,
                method="library",
                confidence=0.95,
                errors=["PlantUML server reported syntax error in diagram."],
            )
        if "<svg" in body:
            return ValidationResult(valid=True, method="library", confidence=0.95)
        return None
    except urllib.error.HTTPError as e:
        if 400 <= e.code < 500:
            return ValidationResult(
                valid=False,
                method="library",
                confidence=0.95,
                errors=[f"PlantUML server rejected diagram (HTTP {e.code})."],
            )
        return None  # 5xx = server issue, fall back
    except Exception:
        return None  # network error, fall back


# ---------------------------------------------------------------------------
# Validator registry: domain → library validator function
# ---------------------------------------------------------------------------
_LIBRARY_VALIDATORS: Dict[str, Any] = {
    "python": _validate_python,
    "sql": _validate_sql,
    "json": _validate_json,
    "yaml": _validate_yaml,
    "html": _validate_html,
}


def validate_output(output: str, domain: str) -> ValidationResult:
    """Verification cascade: library (L1) → structural heuristic (L2).

    L1: Library parser (definitive, free, <100ms)
    L2: Keyword/structure checks (heuristic, free, <1ms)

    LLM judge (L3) is intentionally separate — lives in
    LearningService.llm_judge_quality_with_feedback() and is only called
    for episode recording / graduation decisions (not in hot loops).
    """
    if not output or len(output) < 20:
        return ValidationResult(
            valid=False,
            method="structural",
            confidence=0.8,
            errors=["Output is too short. Generate complete output."],
        )

    d = domain.lower().split(":")[0]

    # --- Level 1: Library validator (when available) ---
    # Extract code blocks for mixed-content outputs
    code = _extract_code_block(output, d)

    library_result: Optional[ValidationResult] = None

    if d in _LIBRARY_VALIDATORS:
        library_result = _LIBRARY_VALIDATORS[d](code)
        if not library_result.valid:
            return library_result

    if d == "plantuml":
        server_result = _validate_plantuml_server(code)
        if server_result is not None:
            return server_result  # server is authoritative (valid or invalid)
        # server unreachable — fall through to structural

    # If a library already validated OK, return that (preserves its confidence)
    if library_result and library_result.valid:
        return library_result

    # --- Level 2: Structural heuristic (fallback when no library available) ---
    if d == "plantuml" and "@startuml" not in output:
        return ValidationResult(
            valid=False,
            method="structural",
            confidence=0.5,
            errors=["Missing @startuml/@enduml wrapper. Wrap code in @startuml...@enduml."],
        )
    if d == "mermaid" and not any(
        kw in output
        for kw in (
            "graph ",
            "sequenceDiagram",
            "classDiagram",
            "flowchart",
            "erDiagram",
            "gantt",
            "pie",
            "stateDiagram",
        )
    ):
        return ValidationResult(
            valid=False,
            method="structural",
            confidence=0.5,
            errors=["Missing Mermaid diagram type declaration."],
        )
    if d == "sql" and not any(
        kw in output.upper()
        for kw in ("SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "WITH")
    ):
        return ValidationResult(
            valid=False,
            method="structural",
            confidence=0.4,
            errors=["No SQL keywords found."],
        )

    return ValidationResult(valid=True, method="structural", confidence=0.5)


def _extract_code_block(output: str, domain: str) -> str:
    """Extract fenced code block from output if present, else return raw output.

    Handles ```lang ... ``` wrapping common in LLM responses.
    """
    import re as _re

    patterns = [
        _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL),
    ]
    # For PlantUML, also try @startuml...@enduml extraction
    if domain == "plantuml":
        m = _re.search(r"(@startuml.*?@enduml)", output, _re.DOTALL)
        if m:
            return m.group(1)

    for pat in patterns:
        m = pat.search(output)
        if m:
            return m.group(1).strip()

    return output


def _check_output(output: str, domain: str) -> str:
    """Backward-compatible wrapper: returns issue string or empty.

    Delegates to validate_output() cascade (library → structural).
    """
    result = validate_output(output, domain)
    return result.issue


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
        """Register external gold examples for a domain."""
        if domain not in self._gold_data:
            self._gold_data[domain] = []
        self._gold_data[domain].extend(examples)
        logger.info(
            f"DomainDSPyOptimizer: +{len(examples)} gold for {domain} "
            f"(total={len(self._gold_data[domain])})"
        )

    def generate_gold_from_llm(
        self,
        domain: str,
        tasks: List[str],
        model: str = "claude-sonnet-4-20250514",
    ) -> int:
        """Generate gold examples by calling a strong LLM.

        The single highest-ROI way to get training data: have Sonnet/Opus
        produce perfect outputs, then use those to train Haiku via DSPy.

        Args:
            domain: Domain key (e.g. "plantuml", "mermaid")
            tasks: List of task description strings
            model: Strong model to use as teacher

        Returns:
            Number of valid examples added
        """
        import anthropic

        client = anthropic.Anthropic()
        added = 0

        for task_desc in tasks:
            try:
                resp = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    messages=[
                        {
                            "role": "user",
                            "content": f"{task_desc}\n\nOutput ONLY the code/content, no explanation.",
                        }
                    ],
                )
                output = getattr(resp.content[0], "text", "").strip()
                vr = validate_output(output, domain) if output else None
                if vr and vr.valid:
                    self.add_gold_examples(domain, [{"task": task_desc, "output": output}])
                    added += 1
                    logger.debug(
                        f"Gold gen OK ({vr.method}, conf={vr.confidence:.2f}): {task_desc[:60]}"
                    )
                else:
                    errs = vr.errors if vr else ["empty output"]
                    logger.debug(f"Gold gen REJECTED ({errs}): {task_desc[:60]}")
            except Exception as e:
                logger.debug(f"Gold gen failed: {e}")

        logger.info(f"generate_gold_from_llm: {added}/{len(tasks)} valid for {domain}")
        return added

    def _gather_training_data(self, domain: str, min_quality: float = 0.7) -> List:
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
        strategy: str = "auto",
    ):
        """Optimize a DSPy module for a domain using gold episodes.

        Args:
            strategy: Which DSPy optimizer to use:
                - "auto": MIPROv2 if >=8 examples (rewrites instructions + demos),
                  falls back to BootstrapFewShotWithRandomSearch otherwise.
                - "mipro": Force MIPROv2 (instruction + demo optimization).
                - "bootstrap": Force BootstrapFewShotWithRandomSearch (demo-only).
        """
        import dspy

        _, DomainTaskModule = _get_domain_task_classes()
        module = DomainTaskModule()
        examples = self._gather_training_data(domain)

        if len(examples) < 3:
            logger.warning(
                f"DomainDSPyOptimizer: only {len(examples)} examples for "
                f"{domain}, skipping optimization"
            )
            return module

        if metric is None:
            metric = _gold_metric

        # 80/20 train/dev split
        import random as _rng

        shuffled = list(examples)
        _rng.shuffle(shuffled)
        split = max(3, int(len(shuffled) * 0.8))
        trainset = shuffled[:split]
        devset = shuffled[split:] if split < len(shuffled) else shuffled[-1:]

        try:
            from Jotty.core.infrastructure.foundation.unified_lm_provider import (
                UnifiedLMProvider,
            )

            teacher_lm = UnifiedLMProvider.create_lm(provider="anthropic", model="sonnet")
            student_lm = UnifiedLMProvider.create_lm(provider="anthropic", model="haiku")

            # Pick optimizer strategy
            use_mipro = strategy == "mipro" or (strategy == "auto" and len(trainset) >= 8)

            if use_mipro:
                optimizer = dspy.MIPROv2(
                    metric=metric,
                    prompt_model=teacher_lm,
                    task_model=student_lm,
                    max_bootstrapped_demos=max_bootstrapped,
                    max_labeled_demos=min(len(trainset), max_labeled),
                    auto="light",
                    verbose=False,
                )
                optimized = optimizer.compile(
                    module,
                    trainset=trainset[:16],
                    valset=devset,
                )
                opt_name = "MIPROv2"
            else:
                optimizer = dspy.BootstrapFewShotWithRandomSearch(
                    metric=metric,
                    max_bootstrapped_demos=max_bootstrapped,
                    max_labeled_demos=min(len(trainset), max_labeled),
                    num_candidate_programs=num_candidate_programs,
                )
                with dspy.context(lm=teacher_lm):
                    optimized = optimizer.compile(module, trainset=trainset[:12])
                opt_name = "BootstrapRS"

            # Evaluate on held-out dev set
            dev_scores = []
            with dspy.context(lm=student_lm):
                for ex in devset:
                    try:
                        pred = optimized(
                            task_description=ex.task_description,
                            domain=ex.domain,
                        )
                        dev_scores.append(metric(ex, pred))
                    except Exception:
                        dev_scores.append(0.0)
            avg_dev = sum(dev_scores) / max(len(dev_scores), 1)

            save_path = str(self._save_dir / f"{domain}_task_module.json")
            optimized.save(save_path)
            logger.info(
                f"DomainDSPyOptimizer: {opt_name} optimized {domain} — "
                f"train={len(trainset)}, dev={len(devset)}, "
                f"dev_score={avg_dev:.2f}, saved to {save_path}"
            )
            return optimized

        except Exception as e:
            logger.warning(f"DomainDSPyOptimizer: optimization failed: {e}")
            return module

    def load_optimized(self, domain: str):
        """Load a previously optimized module from disk."""
        path = self._save_dir / f"{domain}_task_module.json"
        if not path.exists():
            return None
        try:
            _, DomainTaskModule = _get_domain_task_classes()
            module = DomainTaskModule()
            module.load(str(path))
            logger.debug(f"DomainDSPyOptimizer: loaded {path.name}")
            return module
        except Exception as e:
            logger.debug(f"DomainDSPyOptimizer: load failed: {e}")
            return None

    def has_optimized(self, domain: str) -> bool:
        """Check if an optimized module exists for this domain."""
        return (self._save_dir / f"{domain}_task_module.json").exists()


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
