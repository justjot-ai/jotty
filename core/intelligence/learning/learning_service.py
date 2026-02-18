"""
LearningService - Unified Learning for All Execution Units
===========================================================

Single service shared by agents, swarms, pipelines, and orchestrator.
Replaces scattered learning mixins with a clean service interface.

Responsibilities:
    record()   — Record any execution outcome
    query()    — Get recommendations for a given context
    reflect()  — Mid-execution reflection and adjustment
    transfer() — Transfer learning patterns between domains
    start/end_episode() — Episode lifecycle for multi-step tracking

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                   LearningService (singleton)                │
    │  ┌───────────┐ ┌───────────┐ ┌──────────┐ ┌────────────┐  │
    │  │ TD-Lambda │ │ Patterns  │ │ Reflect  │ │  Transfer  │  │
    │  │ (values)  │ │ (extract) │ │ (adjust) │ │  (cross)   │  │
    │  └─────┬─────┘ └─────┬─────┘ └────┬─────┘ └──────┬─────┘  │
    │        └──────────────┴────────────┴──────────────┘        │
    │                        LearningStore                        │
    │                    (SQLite, persistent)                      │
    └─────────────────────────────────────────────────────────────┘

    Used by:  Agent.execute()  |  Swarm.execute()  |  Orchestrator.run()

Author: Jotty Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .learning_store import (
    EpisodeRecord,
    LearningStore,
    PatternRecord,
    ReflectionRecord,
    ValueEstimate,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DOMAIN CLASSIFIER — Detects actual domain from task text
# =============================================================================

# Keyword sets per domain, ordered by specificity (most specific first)
_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "coding": [
        "implement",
        "function",
        "class ",
        "code",
        "python",
        "javascript",
        "typescript",
        "algorithm",
        "data structure",
        "api",
        "rest ",
        "refactor",
        "debug",
        "unit test",
        "integration test",
        "lru",
        "cache",
        "rate limit",
        "database",
        "sql",
        "redis",
        "docker",
        "kubernetes",
        "deploy",
        "ci/cd",
        "git",
        "compile",
        "runtime",
        "bug",
        "exception",
        "stack trace",
        "leetcode",
        "linked list",
        "binary tree",
        "hash map",
        "thread",
        "async",
        "concurren",
    ],
    "research": [
        "research",
        "analyze",
        "analysis",
        "study",
        "survey",
        "paper",
        "literature",
        "findings",
        "evidence",
        "hypothesis",
        "methodology",
        "experiment",
        "dataset",
        "statistical",
        "quantitative",
        "qualitative",
        "peer review",
        "citation",
        "journal",
        "academic",
        "scholar",
    ],
    "system_design": [
        "system design",
        "architect",
        "scalab",
        "distributed",
        "microservice",
        "load balanc",
        "failover",
        "availability",
        "latency",
        "throughput",
        "kafka",
        "message queue",
        "event driven",
        "cqrs",
        "cap theorem",
        "consensus",
        "replication",
        "sharding",
        "partition",
        "p99",
    ],
    "data_science": [
        "machine learning",
        "neural network",
        "deep learning",
        "model",
        "training",
        "inference",
        "xgboost",
        "random forest",
        "regression",
        "classification",
        "clustering",
        "feature engineer",
        "backpropagation",
        "gradient",
        "loss function",
        "optimizer",
        "epoch",
        "batch",
        "transformer",
        "attention",
        "embedding",
        "fine-tun",
        "rl ",
        "reinforcement learning",
        "continual learning",
        "catastrophic forgetting",
    ],
    "economics": [
        "economic",
        "gdp",
        "inflation",
        "labor market",
        "monetary",
        "fiscal",
        "trade",
        "tariff",
        "supply chain",
        "market",
        "investment",
        "portfolio",
        "stock",
        "bond",
        "interest rate",
        "unemployment",
        "productivity",
        "inequality",
        "policy",
    ],
    "writing": [
        "write",
        "essay",
        "article",
        "blog",
        "content",
        "copywriting",
        "narrative",
        "storytelling",
        "creative writing",
        "proofread",
        "grammar",
        "tone",
        "audience",
        "persuasive",
        "report",
    ],
    "math": [
        "prove",
        "theorem",
        "lemma",
        "equation",
        "integral",
        "derivative",
        "matrix",
        "eigenvalue",
        "probability",
        "combinatorics",
        "topology",
        "group theory",
        "number theory",
        "optimization",
        "convex",
    ],
}

# Response quality signal detectors
_STRUCTURE_MARKERS = re.compile(
    r"(?:^|\n)\s*(?:"
    r"#{1,4}\s|"  # Markdown headings
    r"\d+\.\s|"  # Numbered lists
    r"[A-Z]\.\s|"  # Lettered sections
    r"\*\*[A-Z]|"  # Bold section headers
    r"```|"  # Code blocks
    r"\|.*\|.*\|"  # Tables
    r")",
    re.MULTILINE,
)

_CODE_BLOCK_RE = re.compile(r"```\w*\n.*?```", re.DOTALL)
_CITATION_RE = re.compile(r"\b(?:et al\.?|(?:19|20)\d{2}[a-z]?)\b")


def classify_domain(text: str) -> Tuple[str, str]:
    """
    Classify task text into (domain, task_type) using keyword matching.

    Returns the most specific matching domain, with ties broken by
    match count. Falls back to 'general' if no strong match.
    """
    text_lower = text.lower()
    scores: Dict[str, int] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[domain] = score

    if not scores:
        return "general", "general"

    # Pick the domain with the highest keyword hit count
    best_domain = max(scores, key=scores.get)  # type: ignore[arg-type]

    # Derive task_type from the top keywords that matched
    keywords_hit = [kw for kw in _DOMAIN_KEYWORDS[best_domain] if kw in text_lower]
    task_type = keywords_hit[0].strip() if keywords_hit else best_domain

    # Cross-domain detection: if 2+ domains score highly, mark as synthesis
    high_scorers = [d for d, s in scores.items() if s >= 3]
    if len(high_scorers) >= 3:
        return "synthesis", "cross_domain"
    if len(high_scorers) == 2:
        return best_domain, f"{high_scorers[0]}+{high_scorers[1]}"

    return best_domain, task_type


def analyze_response(content: str, goal: str) -> Dict[str, Any]:
    """
    Analyze LLM response to extract quality signals. Pure heuristics,
    no LLM calls. Returns a dict of features for learning records.
    """
    if not content:
        return {"empty": True, "quality_score": 0.0}

    content_lower = content.lower()
    goal_lower = goal.lower()

    # Structure analysis
    structure_hits = len(_STRUCTURE_MARKERS.findall(content))
    has_headings = bool(re.search(r"(?:^|\n)#{1,4}\s", content))
    has_numbered_list = bool(re.search(r"(?:^|\n)\s*\d+\.\s", content))
    has_code_blocks = bool(_CODE_BLOCK_RE.search(content))
    code_block_count = len(_CODE_BLOCK_RE.findall(content))
    has_table = bool(re.search(r"\|.*\|.*\|", content))

    # Depth signals
    word_count = len(content.split())
    paragraph_count = len([p for p in content.split("\n\n") if p.strip()])
    has_citations = bool(_CITATION_RE.search(content))
    has_math = bool(re.search(r"[=∑∫∂∇λαβγ]|\\frac|O\(n", content))

    # Goal coverage: how many goal keywords appear in the response
    goal_keywords = set(re.findall(r"\b[a-z]{4,}\b", goal_lower))
    response_keywords = set(re.findall(r"\b[a-z]{4,}\b", content_lower))
    if goal_keywords:
        coverage = len(goal_keywords & response_keywords) / len(goal_keywords)
    else:
        coverage = 0.5

    # Compute composite quality score
    quality = 0.0
    quality += min(0.20, structure_hits * 0.02)  # Structure: up to 0.20
    quality += min(0.15, word_count / 5000 * 0.15)  # Length depth: up to 0.15
    quality += min(0.15, paragraph_count / 10 * 0.15)  # Paragraphs: up to 0.15
    quality += coverage * 0.25  # Goal coverage: up to 0.25
    quality += 0.05 if has_code_blocks else 0.0  # Code: 0.05
    quality += 0.05 if has_citations else 0.0  # Citations: 0.05
    quality += 0.05 if has_math else 0.0  # Math: 0.05
    quality += 0.05 if has_table else 0.0  # Tables: 0.05
    quality += 0.05 if has_headings else 0.0  # Headings: 0.05
    quality = min(1.0, quality)

    return {
        "quality_score": round(quality, 3),
        "word_count": word_count,
        "paragraph_count": paragraph_count,
        "structure_score": min(1.0, structure_hits / 10),
        "goal_coverage": round(coverage, 3),
        "has_code": has_code_blocks,
        "code_block_count": code_block_count,
        "has_headings": has_headings,
        "has_numbered_list": has_numbered_list,
        "has_table": has_table,
        "has_citations": has_citations,
        "has_math": has_math,
        "content_length": len(content),
    }


# =============================================================================
# EPISODE TRACKER (for multi-step execution tracking)
# =============================================================================


class ActiveEpisode:
    """Tracks a running execution episode with its steps."""

    def __init__(
        self,
        episode_id: str,
        unit_type: str,
        unit_name: str,
        domain: str,
        task_type: str,
        context: Dict[str, Any],
    ) -> None:
        self.episode_id = episode_id
        self.unit_type = unit_type
        self.unit_name = unit_name
        self.domain = domain
        self.task_type = task_type
        self.context = context
        self.start_time = time.time()
        self.steps: List[Dict[str, Any]] = []
        self.reflections: List[Dict[str, Any]] = []
        self.parent_episode_id: Optional[str] = None

    def add_step(
        self, step_name: str, action: Dict[str, Any], outcome: Dict[str, Any], success: bool
    ) -> None:
        self.steps.append(
            {
                "step": len(self.steps) + 1,
                "name": step_name,
                "action": action,
                "outcome": outcome,
                "success": success,
                "timestamp": time.time(),
            }
        )


# =============================================================================
# LEARNING SERVICE
# =============================================================================


class LearningService:
    """
    Unified learning service for all Jotty execution units.

    Singleton per process. All agents, swarms, pipelines, and the orchestrator
    share this service. Learning data persists in SQLite via LearningStore.

    Key design principles:
    1. ANY execution unit can record outcomes and query for guidance
    2. Learning happens at multiple timescales (intra-step, post-episode, cross-episode)
    3. Patterns transfer across domains automatically
    4. Self-evolving: extracts patterns from accumulated data

    Usage:
        service = LearningService.get_instance()

        # Record an outcome
        service.record("CodingSwarm", "swarm", "coding", "code_generation",
                       context={"task": "REST API"}, action={"paradigm": "pipeline"},
                       outcome={"code": "...", "tests_pass": True},
                       success=True, quality=0.85)

        # Query for guidance
        guidance = service.query("coding", "code_generation",
                                 context={"task": "REST API"})

        # Mid-execution reflection
        adjustment = service.reflect(episode_id, step=3,
                                      observation="Test failures on edge cases",
                                      unit_name="TestWriter")

        # Transfer learning
        patterns = service.transfer("coding", "devops")
    """

    _instance: Optional["LearningService"] = None

    def __init__(self, store: Optional[LearningStore] = None) -> None:
        self._store = store or LearningStore.get_instance()
        self._active_episodes: Dict[str, ActiveEpisode] = {}

        # In-memory caches for hot-path queries (avoid DB hits)
        self._success_rate_cache: Dict[str, Tuple[float, float]] = {}  # key -> (rate, timestamp)
        self._pattern_cache: Dict[str, Tuple[List[PatternRecord], float]] = {}
        self._cache_ttl = 60.0  # Cache TTL in seconds

        # Pattern extraction thresholds
        self._min_episodes_for_pattern = 3
        self._pattern_extraction_interval = 5  # Extract patterns every N records
        self._record_count = 0

        logger.info("LearningService initialized")

    @classmethod
    def get_instance(cls, store: Optional[LearningStore] = None) -> "LearningService":
        """Get or create the singleton LearningService."""
        if cls._instance is None:
            cls._instance = cls(store)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    # =========================================================================
    # RECORD — Record any execution outcome
    # =========================================================================

    def record(
        self,
        unit_name: str,
        unit_type: str,
        domain: str,
        task_type: str,
        context: Dict[str, Any],
        action: Dict[str, Any],
        outcome: Dict[str, Any],
        success: bool,
        quality: float = 0.0,
        execution_time: float = 0.0,
        cost: float = 0.0,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        parent_episode_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record an execution outcome from any unit (agent, swarm, pipeline).

        Args:
            unit_name: Name of the execution unit (e.g., "CodingSwarm", "ResearchAgent")
            unit_type: Type: "agent", "swarm", "pipeline", "orchestrator"
            domain: Domain (e.g., "coding", "research", "devops")
            task_type: Specific task type (e.g., "code_generation", "stock_analysis")
            context: Execution context (task description, inputs)
            action: What was done (paradigm, tools used, model chosen)
            outcome: Results produced
            success: Whether the execution succeeded
            quality: Quality score 0.0-1.0
            execution_time: Seconds taken
            cost: USD cost of LLM calls
            error_type: Classified error category (if failed)
            error_message: Error details (if failed)
            parent_episode_id: Parent episode ID (for nested episodes)
            metadata: Additional metadata

        Returns:
            Episode ID
        """
        episode_id = f"ep_{unit_name}_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        episode = EpisodeRecord(
            episode_id=episode_id,
            unit_type=unit_type,
            unit_name=unit_name,
            domain=domain,
            task_type=task_type,
            context=self._truncate_dict(context),
            action=self._truncate_dict(action),
            outcome=self._truncate_dict(outcome),
            success=success,
            quality=quality,
            execution_time=execution_time,
            cost=cost,
            error_type=error_type,
            error_message=error_message[:500] if error_message else None,
            parent_episode_id=parent_episode_id,
            metadata=metadata or {},
        )

        try:
            self._store.save_episode(episode)
        except Exception as e:
            logger.warning(f"Failed to save episode: {e}")
            return episode_id

        # Update value estimates via TD learning
        self._update_values(domain, task_type, action, success, quality)

        # Invalidate caches
        self._invalidate_cache(domain, task_type)

        # Periodic pattern extraction — run for triggering domain AND
        # sweep all domains that have accumulated enough episodes
        self._record_count += 1
        if self._record_count % self._pattern_extraction_interval == 0:
            try:
                self._extract_patterns(domain)
            except Exception as e:
                logger.debug(f"Pattern extraction failed for {domain}: {e}")

            # Sweep: extract patterns for other domains that have enough data
            try:
                conn = self._store._get_conn()
                rows = conn.execute(
                    "SELECT DISTINCT domain FROM episodes WHERE domain != ? AND domain != ''",
                    (domain,),
                ).fetchall()
                for row in rows:
                    other_domain = row["domain"]
                    try:
                        self._extract_patterns(other_domain)
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"Pattern sweep failed: {e}")

        logger.debug(
            f"Recorded: {unit_name} ({unit_type}) domain={domain} "
            f"task={task_type} success={success} quality={quality:.2f}"
        )
        return episode_id

    # =========================================================================
    # QUERY — Get recommendations for a given context
    # =========================================================================

    def query(
        self,
        domain: str,
        task_type: str = "",
        context: Optional[Dict[str, Any]] = None,
        unit_name: str = "",
    ) -> Dict[str, Any]:
        """
        Query for guidance based on domain and task context.

        Returns recommendations, success rates, known patterns, and
        failure analysis to help the calling unit make better decisions.

        Args:
            domain: Execution domain
            task_type: Specific task type
            context: Current task context
            unit_name: Name of the querying unit

        Returns:
            Dict with:
                - success_rate: Historical success rate
                - patterns: Applicable behavioral patterns
                - recommendations: Specific action recommendations
                - failure_analysis: Recent failure patterns to avoid
                - best_action: Recommended action based on value estimates
        """
        result: Dict[str, Any] = {
            "has_learning": False,
            "success_rate": 0.0,
            "total_episodes": 0,
            "patterns": [],
            "recommendations": [],
            "failure_analysis": [],
            "best_action": None,
            "improving": False,
        }

        try:
            # 1. Success rate
            rate, total = self._get_cached_success_rate(domain, task_type)
            result["success_rate"] = rate
            result["total_episodes"] = total
            result["has_learning"] = total > 0

            # 2. Applicable patterns
            patterns = self._get_cached_patterns(domain)
            result["patterns"] = [
                {
                    "description": p.description,
                    "recommendation": p.recommendation,
                    "confidence": p.confidence,
                    "type": p.pattern_type,
                }
                for p in patterns[:5]
            ]

            # 3. Build recommendations from patterns + value estimates
            recommendations = []
            for p in patterns[:3]:
                if p.confidence >= 0.6:
                    recommendations.append(p.recommendation)
            result["recommendations"] = recommendations

            # 4. Recent failure analysis
            if rate < 0.8:
                failures = self._store.get_failure_analysis(domain, task_type, limit=5)
                result["failure_analysis"] = [
                    {
                        "error_type": f.get("error_type", "unknown"),
                        "error_message": f.get("error_message", "")[:200],
                        "task_type": f.get("task_type", ""),
                    }
                    for f in failures
                ]

            # 5. Best action from value estimates
            best = self._get_best_action(domain, task_type)
            if best:
                result["best_action"] = best

            # 6. Improvement trend
            report = self._store.get_improvement_report(domain)
            result["improving"] = report.get("improving", False)

        except Exception as e:
            logger.debug(f"Query failed: {e}")

        return result

    # =========================================================================
    # REFLECT — Mid-execution reflection and adjustment
    # =========================================================================

    def reflect(
        self,
        episode_id: str,
        step: int,
        observation: str,
        unit_name: str,
        analysis: str = "",
    ) -> Dict[str, Any]:
        """
        Mid-execution reflection. Called when a step fails or produces
        unexpected results, enabling the system to adjust its approach
        for subsequent steps.

        Args:
            episode_id: Current episode ID
            step: Step number that triggered reflection
            observation: What happened (e.g., "Test failures on edge cases")
            unit_name: Which unit is reflecting
            analysis: Optional analysis of why it happened

        Returns:
            Dict with:
                - adjustment: Recommended adjustment for next steps
                - similar_failures: Past failures with similar patterns
                - recovery_strategies: Strategies that worked for similar issues
        """
        result: Dict[str, Any] = {
            "adjustment": "",
            "similar_failures": [],
            "recovery_strategies": [],
        }

        try:
            # Find similar past failures
            failures = self._store.get_failure_analysis(limit=20)
            similar = []
            obs_lower = observation.lower()
            for f in failures:
                err = (f.get("error_message", "") or "").lower()
                if any(word in err for word in obs_lower.split() if len(word) > 3):
                    similar.append(f)

            result["similar_failures"] = similar[:3]

            # Find patterns that succeeded in similar contexts
            patterns = self._store.get_patterns(
                pattern_type="failure_avoidance", min_confidence=0.5
            )
            recovery = [
                p.recommendation
                for p in patterns
                if any(word in p.description.lower() for word in obs_lower.split() if len(word) > 3)
            ]
            result["recovery_strategies"] = recovery[:3]

            # Build adjustment recommendation
            if recovery:
                result["adjustment"] = f"Based on {len(recovery)} past recoveries: {recovery[0]}"
            elif similar:
                # Learn from past: what was different about successes after similar failures
                result["adjustment"] = (
                    f"Similar failure seen {len(similar)} times before. "
                    f"Common error: {similar[0].get('error_type', 'unknown')}. "
                    f"Consider adjusting approach."
                )
            else:
                result["adjustment"] = "No prior data. Recording for future learning."

            # Save reflection for future use
            reflection = ReflectionRecord(
                reflection_id=f"ref_{uuid.uuid4().hex[:8]}",
                episode_id=episode_id,
                step=step,
                unit_name=unit_name,
                observation=observation[:500],
                analysis=analysis[:500] or result["adjustment"],
                adjustment=result["adjustment"][:500],
                applied=False,
            )
            self._store.save_reflection(reflection)

        except Exception as e:
            logger.debug(f"Reflect failed: {e}")
            result["adjustment"] = "Reflection unavailable, proceeding with default approach."

        return result

    def post_execution_reflect(
        self,
        episode_id: str,
        goal: str,
        content: str,
        domain: str,
        quality_score: float,
        execution_time: float,
    ) -> None:
        """
        Post-execution reflection: analyze what worked and record insights.

        Called after every successful execution to build strategic knowledge
        about what response patterns lead to quality outcomes.
        """
        analysis = analyze_response(content, goal)

        # Build observation from analysis
        obs_parts = [f"domain={domain}, quality={quality_score:.2f}, time={execution_time:.1f}s"]
        if analysis.get("has_code"):
            obs_parts.append(f"included {analysis.get('code_block_count', 0)} code blocks")
        if analysis.get("has_headings"):
            obs_parts.append("used section headings")
        if analysis.get("has_citations"):
            obs_parts.append("cited sources")
        if analysis.get("has_math"):
            obs_parts.append("included math formulations")
        obs_parts.append(f"word_count={analysis.get('word_count', 0)}")
        obs_parts.append(f"goal_coverage={analysis.get('goal_coverage', 0):.0%}")
        observation = "; ".join(obs_parts)

        # Build analysis string
        if quality_score >= 0.85:
            analysis_str = (
                f"HIGH QUALITY response in {domain}. Key factors: "
                f"structure={analysis.get('structure_score', 0):.2f}, "
                f"coverage={analysis.get('goal_coverage', 0):.2f}. "
                f"This approach should be reinforced."
            )
        elif quality_score >= 0.5:
            analysis_str = f"ADEQUATE response in {domain}. " f"Improvement areas: "
            if analysis.get("goal_coverage", 0) < 0.7:
                analysis_str += "increase goal coverage; "
            if analysis.get("structure_score", 0) < 0.3:
                analysis_str += "add more structure (headings, lists); "
            if analysis.get("word_count", 0) < 500:
                analysis_str += "increase depth; "
        else:
            analysis_str = (
                f"LOW QUALITY response in {domain}. "
                f"Significant improvement needed in coverage and depth."
            )

        try:
            self.reflect(
                episode_id=episode_id,
                step=0,
                observation=observation,
                unit_name="Orchestrator",
                analysis=analysis_str,
            )
        except Exception as e:
            logger.debug(f"Post-execution reflection failed: {e}")

    def mark_reflection_applied(
        self, episode_id: str, step: int, improvement: Optional[float] = None
    ) -> None:
        """Mark a reflection as applied and record improvement delta."""
        reflections = self._store.get_reflections(episode_id=episode_id)
        for r in reflections:
            if r.step == step:
                r.applied = True
                r.improvement = improvement
                self._store.save_reflection(r)
                break

    # =========================================================================
    # TRANSFER — Cross-domain pattern transfer
    # =========================================================================

    def transfer(self, source_domain: str, target_domain: str) -> List[Dict[str, Any]]:
        """
        Find patterns from source_domain that may apply to target_domain.

        Args:
            source_domain: Domain to transfer FROM
            target_domain: Domain to transfer TO

        Returns:
            List of transferable patterns with recommendations
        """
        patterns = self._store.get_patterns(domain=source_domain, min_confidence=0.6)

        transferable = []
        for p in patterns:
            if target_domain in p.applicable_domains or p.pattern_type in (
                "success_strategy",
                "tool_preference",
            ):
                transferable.append(
                    {
                        "pattern": p.description,
                        "recommendation": p.recommendation,
                        "confidence": p.confidence,
                        "source": p.source_domain,
                        "type": p.pattern_type,
                    }
                )

                # Auto-register pattern for target domain if not already
                if target_domain not in p.applicable_domains:
                    p.applicable_domains.append(target_domain)
                    self._store.save_pattern(p)

        return transferable

    # =========================================================================
    # EPISODE LIFECYCLE — For multi-step execution tracking
    # =========================================================================

    def start_episode(
        self,
        unit_name: str,
        unit_type: str,
        domain: str,
        task_type: str,
        context: Dict[str, Any],
        parent_episode_id: Optional[str] = None,
    ) -> str:
        """
        Start tracking a multi-step execution episode.

        Returns episode_id for use in subsequent step tracking and reflection.
        """
        episode_id = f"ep_{unit_name}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        episode = ActiveEpisode(
            episode_id=episode_id,
            unit_type=unit_type,
            unit_name=unit_name,
            domain=domain,
            task_type=task_type,
            context=context,
        )
        episode.parent_episode_id = parent_episode_id
        self._active_episodes[episode_id] = episode
        return episode_id

    def record_step(
        self,
        episode_id: str,
        step_name: str,
        action: Dict[str, Any],
        outcome: Dict[str, Any],
        success: bool,
    ) -> None:
        """Record a step within an active episode."""
        episode = self._active_episodes.get(episode_id)
        if episode:
            episode.add_step(step_name, action, outcome, success)

    def end_episode(
        self,
        episode_id: str,
        success: bool,
        quality: float = 0.0,
        cost: float = 0.0,
        outcome: Optional[Dict[str, Any]] = None,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        action_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        End an active episode and persist it.

        Args:
            action_metadata: Rich metadata from the caller (provider, model, domain, etc.)
                             Merged with step-based action summary. This is the primary
                             source of action data — step counts are supplementary.

        Returns the episode_id for reference.
        """
        episode = self._active_episodes.pop(episode_id, None)
        if not episode:
            return self.record(
                unit_name="unknown",
                unit_type="unknown",
                domain="",
                task_type="",
                context={},
                action=action_metadata or {},
                outcome=outcome or {},
                success=success,
                quality=quality,
                cost=cost,
            )

        execution_time = time.time() - episode.start_time

        # Build action: merge caller metadata (primary) with step summary (supplementary)
        action_summary: Dict[str, Any] = {}
        if action_metadata:
            action_summary.update(action_metadata)
        action_summary["steps"] = len(episode.steps)
        if episode.steps:
            action_summary["step_names"] = [s["name"] for s in episode.steps]
            action_summary["step_successes"] = [s["success"] for s in episode.steps]

        return self.record(
            unit_name=episode.unit_name,
            unit_type=episode.unit_type,
            domain=episode.domain,
            task_type=episode.task_type,
            context=episode.context,
            action=action_summary,
            outcome=outcome or {"steps": episode.steps},
            success=success,
            quality=quality,
            execution_time=execution_time,
            cost=cost,
            error_type=error_type,
            error_message=error_message,
            parent_episode_id=episode.parent_episode_id,
        )

    # =========================================================================
    # BUILD CONTEXT — Generate learning context string for agent prompts
    # =========================================================================

    def build_context_string(self, domain: str, task_type: str = "", unit_name: str = "") -> str:
        """
        Build a rich, actionable learning context string for injection into
        agent system prompts or swarm context.

        Includes: success rate, quality benchmarks, best strategies,
        structural expectations, domain-specific patterns, and failure warnings.
        """
        guidance = self.query(domain, task_type, unit_name=unit_name)
        parts: List[str] = []
        parts.append(f"[LEARNING CONTEXT — {domain}]")

        total = guidance.get("total_episodes", 0)
        rate = guidance.get("success_rate", 0.0)

        if not guidance.get("has_learning"):
            # Even without domain-specific data, try cross-domain transfer
            xfer_domains = ["general", "coding", "research", "system_design"]
            for xd in xfer_domains:
                if xd == domain:
                    continue
                xg = self.query(xd, "")
                if xg.get("has_learning") and xg.get("total_episodes", 0) >= 5:
                    xfer_patterns = self.transfer(xd, domain)
                    if xfer_patterns:
                        parts.append(f"Cross-domain insight from {xd}:")
                        for xp in xfer_patterns[:2]:
                            parts.append(f"  - {xp['recommendation']}")
                        break
            if len(parts) == 1:
                return ""
            return "\n".join(parts)

        parts.append(f"Performance: {rate:.0%} success across {total} episodes")

        # Improvement trend with direction
        report = self._store.get_improvement_report(domain)
        trend = report.get("trend", 0)
        if trend > 0.05:
            parts.append(f"Trend: IMPROVING (+{trend:.0%} vs historical)")
        elif trend < -0.05:
            parts.append(f"Trend: DECLINING ({trend:.0%} vs historical) — increase rigor")
        elif total > 10:
            parts.append("Trend: STABLE")

        # Quality benchmarks from recent episodes
        try:
            recent = self._store.query_episodes(domain=domain, limit=20)
            if recent:
                qualities = [e.quality for e in recent if e.quality > 0]
                if qualities:
                    avg_q = sum(qualities) / len(qualities)
                    best_q = max(qualities)
                    parts.append(f"Quality benchmark: avg={avg_q:.2f}, best={best_q:.2f}")

                # Extract what made best episodes good
                best_episodes = sorted(
                    [e for e in recent if e.success],
                    key=lambda e: e.quality,
                    reverse=True,
                )[:3]
                if best_episodes:
                    structural_signals = set()
                    for ep in best_episodes:
                        out = ep.outcome or {}
                        if out.get("has_code"):
                            structural_signals.add("include code examples")
                        if out.get("has_headings"):
                            structural_signals.add("use clear section headings")
                        if out.get("has_numbered_list"):
                            structural_signals.add("use numbered lists for steps")
                        if out.get("has_citations"):
                            structural_signals.add("cite sources and papers")
                        if out.get("has_math"):
                            structural_signals.add("include mathematical formulations")
                        if out.get("has_table"):
                            structural_signals.add("use tables for comparisons")
                        wc = out.get("word_count", 0)
                        if wc > 2000:
                            structural_signals.add(f"aim for {wc//500*500}+ word depth")
                    if structural_signals:
                        parts.append(
                            "Best responses tend to: " + "; ".join(sorted(structural_signals)[:4])
                        )
        except Exception as e:
            logger.debug(f"Quality benchmark extraction failed: {e}")

        # Best action from TD-Lambda value estimates
        best = guidance.get("best_action")
        if best and best.get("confidence", 0) > 0.3:
            parts.append(
                f"Recommended approach: {best['action']} (value={best['expected_value']:.2f})"
            )

        # High-confidence patterns (>= 0.5)
        for p in guidance.get("patterns", []):
            if p.get("confidence", 0) >= 0.5:
                parts.append(f"Pattern: {p['recommendation']}")

        # Failure warnings
        failures = guidance.get("failure_analysis", [])
        if failures:
            error_types = set(f.get("error_type", "") for f in failures if f.get("error_type"))
            if error_types:
                parts.append(f"Known failure modes to avoid: {', '.join(error_types)}")

        # Best approach template — concrete example of what worked best
        best = self.get_best_approach_for_domain(domain, task_type)
        if best and best.get("structural_features"):
            parts.append(
                f"Best prior response (quality={best['quality']:.2f}) featured: "
                + "; ".join(best["structural_features"][:5])
            )

        # Cross-domain transfer (always check for insights from related domains)
        related_domains = {
            "coding": ["system_design", "data_science"],
            "research": ["data_science", "writing"],
            "system_design": ["coding"],
            "data_science": ["coding", "math"],
            "synthesis": ["research", "data_science", "coding"],
        }
        for rd in related_domains.get(domain, []):
            xfer = self.transfer(rd, domain)
            if xfer:
                best_xfer = max(xfer, key=lambda x: x.get("confidence", 0))
                if best_xfer.get("confidence", 0) >= 0.6:
                    parts.append(f"From {rd} experience: {best_xfer['recommendation']}")
                    break

        return "\n".join(parts)

    # =========================================================================
    # IMPROVEMENT REPORT
    # =========================================================================

    def get_best_approach_for_domain(
        self, domain: str, task_type: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Return the best historical approach for a domain: what action metadata,
        structural features, and quality score characterized the top episode.

        This gives the system a concrete template to follow, not just averages.
        """
        try:
            episodes = self._store.query_episodes(domain=domain, limit=50)
            successful = [e for e in episodes if e.success and e.quality > 0]
            if not successful:
                return None

            best = max(successful, key=lambda e: e.quality)
            outcome = best.outcome or {}

            approach = {
                "quality": best.quality,
                "execution_time": best.execution_time,
                "structural_features": [],
                "action": best.action,
            }

            if outcome.get("has_code"):
                approach["structural_features"].append(
                    f"included {outcome.get('code_block_count', 'multiple')} code blocks"
                )
            if outcome.get("has_headings"):
                approach["structural_features"].append("organized with section headings")
            if outcome.get("has_numbered_list"):
                approach["structural_features"].append("used numbered lists")
            if outcome.get("has_citations"):
                approach["structural_features"].append("cited references")
            if outcome.get("has_math"):
                approach["structural_features"].append("included mathematical formulations")
            if outcome.get("has_table"):
                approach["structural_features"].append("used comparison tables")
            wc = outcome.get("word_count", 0)
            if wc:
                approach["structural_features"].append(f"~{wc} words depth")
            gc = outcome.get("goal_coverage", 0)
            if gc:
                approach["goal_coverage"] = gc

            return approach
        except Exception as e:
            logger.debug(f"get_best_approach_for_domain failed: {e}")
            return None

    async def llm_judge_quality(
        self,
        goal: str,
        content: str,
        domain: str,
        heuristic_score: float,
    ) -> float:
        """
        Use a cheap LLM call to judge response quality. Returns a score 0.0-1.0.

        Only called for complex tasks (>500 words) where heuristic scoring
        is insufficient. Uses a fast model to minimize cost.

        Falls back to heuristic_score on any failure.
        """
        if len(content) < 500:
            return heuristic_score

        try:
            from Jotty.core.intelligence.orchestration.execution.unified_executor import (
                ChatExecutor,
            )

            judge = ChatExecutor(
                provider="anthropic", model="claude-sonnet-4-20250514", max_steps=1
            )

            prompt = (
                f"Rate this response on a scale of 0.0 to 1.0. Consider:\n"
                f"- Correctness and accuracy\n"
                f"- Completeness (does it address all parts of the task?)\n"
                f"- Depth and specificity (concrete details, not vague)\n"
                f"- Structure and clarity\n\n"
                f"TASK: {goal[:500]}\n\n"
                f"RESPONSE (first 2000 chars): {content[:2000]}\n\n"
                f'Reply with ONLY a JSON object: {{"score": 0.XX, "reason": "brief explanation"}}'
            )

            result = await judge.execute(prompt)
            text = getattr(result, "content", "")

            import json as _json

            # Extract JSON from response
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        parsed = _json.loads(line)
                        score = float(parsed.get("score", heuristic_score))
                        return max(0.0, min(1.0, score))
                    except (ValueError, _json.JSONDecodeError):
                        continue

            return heuristic_score

        except Exception as e:
            logger.debug(f"LLM judge failed, using heuristic: {e}")
            return heuristic_score

    def improvement_report(self, domain: str = "") -> Dict[str, Any]:
        """Get improvement report for monitoring."""
        return self._store.get_improvement_report(domain)

    # =========================================================================
    # INTERNAL: Value updates, caching, pattern extraction
    # =========================================================================

    def _update_values(
        self, domain: str, task_type: str, action: Dict[str, Any], success: bool, quality: float
    ) -> None:
        """Update TD value estimates based on outcome."""
        state_key = self._make_state_key(domain, task_type)
        action_key = self._make_action_key(action)
        reward = quality if success else -0.1

        existing = self._store.get_value(state_key, action_key, domain)

        if existing:
            # TD(0) update: V(s) = V(s) + alpha * (reward - V(s))
            alpha = max(0.01, 0.1 / (1 + existing.update_count * 0.01))
            new_value = existing.value + alpha * (reward - existing.value)
            td_error = reward - existing.value

            updated = ValueEstimate(
                state_key=state_key,
                action_key=action_key,
                domain=domain,
                value=new_value,
                td_error=td_error,
                update_count=existing.update_count + 1,
            )
        else:
            updated = ValueEstimate(
                state_key=state_key,
                action_key=action_key,
                domain=domain,
                value=reward,
                td_error=reward,
                update_count=1,
            )

        self._store.save_value(updated)

    def _extract_patterns(self, domain: str) -> None:
        """
        Auto-extract behavioral patterns from accumulated episode data.

        Extracts:
        1. Success strategies (what actions/approaches lead to high quality)
        2. Quality drivers (structural features that correlate with quality)
        3. Speed patterns (what's fast vs slow)
        4. Domain-specific insights (coding needs code, research needs citations)
        5. Failure avoidance patterns
        """
        episodes = self._store.query_episodes(domain=domain, limit=100)
        if len(episodes) < self._min_episodes_for_pattern:
            return

        successes = [e for e in episodes if e.success and e.quality >= 0.7]
        failures = [e for e in episodes if not e.success]
        high_quality = [e for e in episodes if e.quality >= 0.85]
        low_quality = [e for e in episodes if e.success and e.quality < 0.5]

        # 1. Action-based success strategies (original logic, improved)
        if len(successes) >= 3:
            action_counts: Dict[str, int] = defaultdict(int)
            for e in successes:
                for key, val in e.action.items():
                    if isinstance(val, (str, int, float, bool)):
                        action_counts[f"{key}={val}"] += 1

            for action_str, count in action_counts.items():
                if count >= 3:
                    confidence = count / len(successes)
                    pattern_id = hashlib.md5(f"success_{domain}_{action_str}".encode()).hexdigest()[
                        :12
                    ]

                    self._store.save_pattern(
                        PatternRecord(
                            pattern_id=pattern_id,
                            source_domain=domain,
                            pattern_type="success_strategy",
                            description=(
                                f"In {domain}, {action_str} correlates with success "
                                f"({count}/{len(successes)} episodes)"
                            ),
                            conditions={"domain": domain},
                            recommendation=f"When working on {domain} tasks, prefer {action_str}",
                            confidence=confidence,
                            evidence_count=count,
                            applicable_domains=[domain],
                        )
                    )

        # 2. Quality driver patterns — what structural features appear in best responses
        if len(high_quality) >= 2:
            quality_signals: Dict[str, int] = defaultdict(int)
            for e in high_quality:
                out = e.outcome or {}
                if out.get("has_code"):
                    quality_signals["include_code"] += 1
                if out.get("has_headings"):
                    quality_signals["use_headings"] += 1
                if out.get("has_numbered_list"):
                    quality_signals["use_numbered_lists"] += 1
                if out.get("has_citations"):
                    quality_signals["cite_sources"] += 1
                if out.get("has_math"):
                    quality_signals["include_math"] += 1
                if out.get("has_table"):
                    quality_signals["use_tables"] += 1
                if out.get("word_count", 0) > 2000:
                    quality_signals["depth_over_2000_words"] += 1
                ss = out.get("structure_score", 0)
                if ss > 0.5:
                    quality_signals["high_structure"] += 1
                gc = out.get("goal_coverage", 0)
                if gc > 0.7:
                    quality_signals["high_goal_coverage"] += 1

            readable_names = {
                "include_code": "include code examples with implementations",
                "use_headings": "organize with clear section headings",
                "use_numbered_lists": "use numbered lists for sequential steps",
                "cite_sources": "cite sources, papers, and references",
                "include_math": "include mathematical formulations where relevant",
                "use_tables": "use tables for data comparisons",
                "depth_over_2000_words": "provide in-depth responses (2000+ words)",
                "high_structure": "use well-structured formatting (headings, lists, code blocks)",
                "high_goal_coverage": "address all key aspects of the task",
            }

            for signal, count in quality_signals.items():
                if count >= 2:
                    confidence = count / len(high_quality)
                    pattern_id = hashlib.md5(f"quality_{domain}_{signal}".encode()).hexdigest()[:12]

                    desc = readable_names.get(signal, signal)
                    self._store.save_pattern(
                        PatternRecord(
                            pattern_id=pattern_id,
                            source_domain=domain,
                            pattern_type="quality_driver",
                            description=(
                                f"High-quality {domain} responses tend to {desc} "
                                f"({count}/{len(high_quality)} top episodes)"
                            ),
                            conditions={"domain": domain},
                            recommendation=f"For {domain} tasks: {desc}",
                            confidence=confidence,
                            evidence_count=count,
                            applicable_domains=[domain, "general"],
                        )
                    )

        # 3. Speed patterns — what's efficient
        if len(successes) >= 5:
            times = [e.execution_time for e in successes if e.execution_time > 0]
            if times:
                avg_time = sum(times) / len(times)
                fast = [e for e in successes if 0 < e.execution_time < avg_time * 0.7]
                if len(fast) >= 2:
                    fast_actions = defaultdict(int)
                    for e in fast:
                        for k, v in e.action.items():
                            if isinstance(v, (str, int, float, bool)):
                                fast_actions[f"{k}={v}"] += 1
                    for action_str, count in fast_actions.items():
                        if count >= 2:
                            pattern_id = hashlib.md5(
                                f"speed_{domain}_{action_str}".encode()
                            ).hexdigest()[:12]
                            self._store.save_pattern(
                                PatternRecord(
                                    pattern_id=pattern_id,
                                    source_domain=domain,
                                    pattern_type="speed_optimization",
                                    description=(
                                        f"In {domain}, {action_str} tends to be faster "
                                        f"(avg {avg_time:.0f}s, fast episodes <{avg_time*0.7:.0f}s)"
                                    ),
                                    conditions={"domain": domain},
                                    recommendation=(
                                        f"For faster {domain} execution, prefer {action_str}"
                                    ),
                                    confidence=count / len(fast),
                                    evidence_count=count,
                                    applicable_domains=[domain],
                                )
                            )

        # 4. Quality contrast — what distinguishes high from low quality
        if len(high_quality) >= 2 and len(low_quality) >= 2:
            high_features: Dict[str, float] = defaultdict(float)
            low_features: Dict[str, float] = defaultdict(float)
            for e in high_quality:
                out = e.outcome or {}
                for k in ["word_count", "structure_score", "goal_coverage", "code_block_count"]:
                    high_features[k] += float(out.get(k, 0))
            for e in low_quality:
                out = e.outcome or {}
                for k in ["word_count", "structure_score", "goal_coverage", "code_block_count"]:
                    low_features[k] += float(out.get(k, 0))

            for k in high_features:
                h_avg = high_features[k] / len(high_quality)
                l_avg = low_features[k] / len(low_quality)
                if h_avg > l_avg * 1.5 and h_avg > 0:
                    pattern_id = hashlib.md5(f"contrast_{domain}_{k}".encode()).hexdigest()[:12]
                    self._store.save_pattern(
                        PatternRecord(
                            pattern_id=pattern_id,
                            source_domain=domain,
                            pattern_type="quality_contrast",
                            description=(
                                f"In {domain}, high-quality responses have {h_avg:.0f} avg {k} "
                                f"vs {l_avg:.0f} in low-quality — {h_avg/max(l_avg,1):.1f}x difference"
                            ),
                            conditions={"domain": domain},
                            recommendation=(
                                f"For {domain} tasks, aim for higher {k} " f"(target: {h_avg:.0f}+)"
                            ),
                            confidence=min(0.9, 0.5 + (h_avg - l_avg) / max(h_avg, 1) * 0.4),
                            evidence_count=len(high_quality) + len(low_quality),
                            applicable_domains=[domain],
                        )
                    )

        # 5. Failure avoidance patterns (original, kept)
        if len(failures) >= 3:
            error_counts: Dict[str, int] = defaultdict(int)
            for e in failures:
                if e.error_type:
                    error_counts[e.error_type] += 1

            for error_type, count in error_counts.items():
                if count >= 2:
                    pattern_id = hashlib.md5(f"failure_{domain}_{error_type}".encode()).hexdigest()[
                        :12
                    ]

                    self._store.save_pattern(
                        PatternRecord(
                            pattern_id=pattern_id,
                            source_domain=domain,
                            pattern_type="failure_avoidance",
                            description=(
                                f"In {domain}, {error_type} errors occur frequently ({count} times)"
                            ),
                            conditions={"domain": domain, "error_type": error_type},
                            recommendation=f"Add error handling for {error_type} in {domain} tasks",
                            confidence=min(0.9, count / len(failures)),
                            evidence_count=count,
                            applicable_domains=[domain],
                        )
                    )

        logger.debug(
            f"Pattern extraction complete for domain={domain}: "
            f"{len(successes)} successes, {len(high_quality)} high-quality, "
            f"{len(failures)} failures"
        )

    def _get_best_action(self, domain: str, task_type: str) -> Optional[Dict[str, Any]]:
        """Find the highest-value action for this state."""
        state_key = self._make_state_key(domain, task_type)
        conn = self._store._get_conn()
        row = conn.execute(
            """SELECT action_key, value, update_count FROM value_estimates
               WHERE state_key = ? AND domain = ?
               ORDER BY value DESC LIMIT 1""",
            (state_key, domain),
        ).fetchone()

        if row and row["update_count"] >= 3:
            return {
                "action": row["action_key"],
                "expected_value": round(row["value"], 3),
                "confidence": min(1.0, row["update_count"] / 10),
            }
        return None

    def _get_cached_success_rate(self, domain: str, task_type: str) -> Tuple[float, int]:
        """Get success rate with caching."""
        key = f"{domain}:{task_type}"
        cached = self._success_rate_cache.get(key)
        if cached and (time.time() - cached[1]) < self._cache_ttl:
            return cached[0], int(cached[1])  # Using cached timestamp slot for total

        rate, total = self._store.get_success_rate(domain, task_type)
        self._success_rate_cache[key] = (rate, time.time())
        return rate, total

    def _get_cached_patterns(self, domain: str) -> List[PatternRecord]:
        """Get patterns with caching."""
        cached = self._pattern_cache.get(domain)
        if cached and (time.time() - cached[1]) < self._cache_ttl:
            return cached[0]

        patterns = self._store.get_patterns(domain=domain, min_confidence=0.3)
        self._pattern_cache[domain] = (patterns, time.time())
        return patterns

    def _invalidate_cache(self, domain: str, task_type: str) -> None:
        """Invalidate relevant caches."""
        key = f"{domain}:{task_type}"
        self._success_rate_cache.pop(key, None)
        self._pattern_cache.pop(domain, None)

    @staticmethod
    def _make_state_key(domain: str, task_type: str) -> str:
        return f"{domain}:{task_type}" if task_type else domain

    @staticmethod
    def _make_action_key(action: Dict[str, Any]) -> str:
        parts = sorted(
            f"{k}={v}" for k, v in action.items() if isinstance(v, (str, int, float, bool))
        )
        return "|".join(parts[:5]) or "default"

    @staticmethod
    def _truncate_dict(d: Dict[str, Any], max_str_len: int = 500) -> Dict[str, Any]:
        """Truncate string values in a dict to avoid DB bloat."""
        result = {}
        for k, v in list(d.items())[:20]:
            if isinstance(v, str) and len(v) > max_str_len:
                result[k] = v[:max_str_len] + "..."
            elif isinstance(v, dict):
                result[k] = LearningService._truncate_dict(v, max_str_len)
            else:
                result[k] = v
        return result


# =============================================================================
# MODULE-LEVEL CONVENIENCE
# =============================================================================


def get_learning_service(store: Optional[LearningStore] = None) -> LearningService:
    """Get the singleton LearningService instance."""
    return LearningService.get_instance(store)
