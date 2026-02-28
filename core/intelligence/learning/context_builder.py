"""
LearningContextBuilder — Builds learning context for agent prompts.

Extracted from LearningService to keep responsibilities focused.
Collects signals (lessons, reflexion, failures, transfer) and
assembles them within a token budget.

Used by: LearningService.build_context_string() (delegates here)
"""

import logging
import math as _math
import re
import time
from collections import Counter as _Counter
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from .domain_classifier import _get_related_domains
from .learning_store import EpisodeRecord, LearningStore

logger = logging.getLogger(__name__)

_embeddings_instance: Any = None


def _get_embeddings() -> Any:
    global _embeddings_instance
    if _embeddings_instance is None:
        from .embeddings import EmbeddingService

        _embeddings_instance = EmbeddingService()
    return _embeddings_instance


_DOMAIN_BOOTSTRAP: Dict[str, str] = {
    "coding": (
        "[Quality guidance for coding tasks]\n"
        "  - Wrap ALL code in markdown fenced blocks (```python)\n"
        "  - Include complete, runnable code with class/function definitions\n"
        "  - Add test functions (def test_*) with 5+ assertions\n"
        "  - Use section headings to organize (## Implementation, ## Tests)"
    ),
    "algorithms": (
        "[Quality guidance for algorithm tasks]\n"
        "  - Include complete implementations in code blocks (```python)\n"
        "  - Add test functions (def test_*) with assertions verifying correctness\n"
        "  - Analyze time/space complexity with O() notation\n"
        "  - Use section headings (## Algorithm, ## Analysis, ## Tests)"
    ),
    "compiler_design": (
        "[Quality guidance for compiler/interpreter tasks]\n"
        "  - Include complete implementations: tokenizer, parser, evaluator\n"
        "  - Define proper classes (Token, AST nodes, Environment)\n"
        "  - Add test functions covering arithmetic, functions, closures\n"
        "  - Include 5+ assertions validating each language feature"
    ),
    "data_science": (
        "[Quality guidance for data science tasks]\n"
        "  - Include complete class implementations with fit/predict methods\n"
        "  - Add mathematical formulations (equations, proofs)\n"
        "  - Write test functions with assertions verifying correctness\n"
        "  - Use code blocks (```python) for all implementations"
    ),
    "system_design": (
        "[Quality guidance for system design tasks]\n"
        "  - Wrap code examples in markdown fenced blocks (```python)\n"
        "  - Include architecture diagrams or structured descriptions\n"
        "  - Analyze trade-offs (latency, throughput, consistency)\n"
        "  - Use tables for comparisons where appropriate"
    ),
    "math": (
        "[Quality guidance for math tasks]\n"
        "  - Include formal definitions and theorem statements\n"
        "  - Provide step-by-step proofs with justification\n"
        "  - Use mathematical notation and formulas\n"
        "  - Add concrete examples to illustrate abstract concepts"
    ),
    "research": (
        "[Quality guidance for research tasks]\n"
        "  - Structure with clear sections (Introduction, Analysis, Conclusion)\n"
        "  - Cite sources and provide evidence for claims\n"
        "  - Compare multiple perspectives or approaches\n"
        "  - Include a summary/conclusion section"
    ),
}

_STOP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "and",
    "or",
    "not",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "it",
    "this",
    "that",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "will",
    "would",
    "could",
    "should",
    "may",
    "can",
    "must",
    "shall",
    "from",
    "by",
    "as",
    "if",
    "but",
    "so",
    "all",
    "each",
    "every",
    "any",
    "no",
    "your",
    "my",
    "their",
    "our",
    "its",
    "use",
    "using",
}


class LearningContextBuilder:
    """Builds learning context strings for agent prompts.

    Dependencies: LearningStore (read-only) + a query callback from LearningService.
    """

    HOLDOUT_RATE = 0.10

    def __init__(
        self,
        store: LearningStore,
        config: Any,
        query_fn: Callable[..., Dict[str, Any]],
    ) -> None:
        self._store = store
        self._config = config
        self._query = query_fn
        self.last_holdout = False

    # =====================================================================
    # PUBLIC: build_context_string
    # =====================================================================

    def build_context_string(
        self, domain: str, task_type: str = "", unit_name: str = "", goal: str = ""
    ) -> str:
        """Build learning context for an agent's prompt.

        Simple pipeline: collect signals -> assemble within budget.
        Online holdout: ~10% of calls return empty for continuous validation.
        """
        import random

        if random.random() < self.HOLDOUT_RATE:
            self.last_holdout = True
            return ""
        self.last_holdout = False

        MAX_BUDGET = self._config.context_max_total or 2000
        sections: List[str] = []

        guidance = self._query(domain, task_type, unit_name=unit_name)
        has_learning = guidance.get("has_learning", False)
        total = guidance.get("total_episodes", 0)
        rate = guidance.get("success_rate", 0.0)
        has_failures = rate < 0.90 and total >= 3

        # -- Signal 1: Distilled lessons --
        try:
            dist_lessons = self.retrieve_distilled_lessons(
                domain, goal=goal, agent_name=unit_name, top_k=3
            )
            if dist_lessons:
                lines = [f"- {dl['lesson']}" for dl in dist_lessons[:3] if dl.get("lesson")]
                if lines:
                    sections.append(f"[Learned patterns for {domain}]\n" + "\n".join(lines))
        except Exception:
            pass

        # -- Signal 2: Reflexion (past failure avoidance) --
        try:
            from .advanced_learning import Reflexion

            past = Reflexion.get_instance().get_relevant_reflections(
                unit_name=unit_name or domain, limit=2
            )
            if past:
                sections.append("[Avoid past mistakes]\n" + "\n".join(f"  - {r}" for r in past[:2]))
        except Exception:
            pass

        # -- Signal 3: Failure hints (only when struggling) --
        if has_failures:
            failures = guidance.get("failure_analysis", [])
            seen: set = set()
            hints = []
            for f in failures:
                desc = f.get("description", f.get("error_type", ""))
                if desc and desc not in seen:
                    seen.add(desc)
                    hints.append(f"  - {desc[:100]}")
            if hints:
                sections.append(f"Gaps to address ({rate:.0%} success):\n" + "\n".join(hints[:2]))

        # -- Signal 4: Cross-domain transfer (only when no local learning) --
        if not has_learning:
            for related in _get_related_domains(domain)[:2]:
                rel = self._query(related, "")
                if rel.get("has_learning") and rel.get("total_episodes", 0) >= 2:
                    pats = rel.get("patterns", [])
                    recs = [
                        p.get("recommendation", "") for p in pats[:3] if p.get("recommendation")
                    ]
                    if recs:
                        sections.append(
                            f"[Adapted from {related}]\n" + "\n".join(f"  - {r}" for r in recs)
                        )
                    break

        # -- Fallback: bootstrap guidance for cold-start domains --
        if not sections:
            bootstrap = _DOMAIN_BOOTSTRAP.get(domain, "")
            if bootstrap:
                return bootstrap

        result = "\n\n".join(s[:500] for s in sections)
        return result[:MAX_BUDGET] if len(result) > MAX_BUDGET else result

    # =====================================================================
    # PUBLIC: build_retrieval_context (RAG-style few-shot)
    # =====================================================================

    def build_retrieval_context(
        self,
        domain: str,
        task_type: str = "",
        goal: str = "",
        agent_name: str = "",
    ) -> str:
        """Build few-shot learning context from best prior responses."""
        similar = self.retrieve_similar_responses(
            domain,
            task_type,
            goal,
            top_k=3,
            agent_name=agent_name,
        )
        if not similar:
            return ""

        good = [r for r in similar if r["quality"] >= 0.5]
        if not good:
            return ""

        best_resp = max(good, key=lambda r: r.get("relevance_score", r["quality"]))
        excerpt = best_resp.get("excerpt", "")
        if not excerpt or len(excerpt) < 50:
            return ""

        relevance = best_resp.get("relevance_score", 0.0)
        best_quality = best_resp["quality"]

        # Same-task detection: inject structural guidance instead of answer
        if relevance > 0.92:
            content_len = best_resp.get("full_content_len", 0)
            if not content_len:
                actual_content = best_resp.get("actual_content", "")
                content_len = len(actual_content) if actual_content else len(excerpt)
            return "\n".join(
                [
                    f"[QUALITY BASELINE — your previous response scored Q={best_quality:.2f} "
                    f"and was {content_len} chars]",
                    f"Your response MUST be at least {content_len} characters.",
                    "Cover ALL aspects requested. Do NOT summarize or abbreviate.",
                ]
            )

        # Different-task: inject as few-shot example
        parts: List[str] = [f"[REFERENCE FROM SIMILAR TASK (Q={best_quality:.2f})]"]
        if best_resp.get("goal_preview"):
            parts.append(f"Task: {best_resp['goal_preview']}")

        actual_content = best_resp.get("actual_content", "")
        if actual_content and len(actual_content) > len(excerpt):
            parts.append(f"Response:\n{actual_content[:1000]}")
        else:
            parts.append(f"Excerpt:\n{excerpt[:1000]}")

        return "\n".join(parts)

    # =====================================================================
    # PUBLIC: retrieve_similar_responses
    # =====================================================================

    def retrieve_similar_responses(
        self,
        domain: str,
        task_type: str = "",
        goal: str = "",
        top_k: int = 2,
        agent_name: str = "",
    ) -> List[Dict[str, Any]]:
        """Retrieve excerpts from best prior responses via embeddings or TF-IDF."""
        emb = _get_embeddings()
        now = time.time()

        if goal and emb.available:
            goal_vec = emb.embed(goal)
            if goal_vec is not None:
                scored = self._embedding_retrieval(domain, goal_vec, agent_name, now)
            else:
                scored = self._tfidf_retrieval(domain, goal, agent_name, now)
        else:
            scored = self._tfidf_retrieval(domain, goal, agent_name, now)

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, ep in scored[:top_k]:
            outcome = ep.outcome or {}
            actual_content = ""
            full_content_len = 0
            for _ck in ("content", "response", "result"):
                _cv = outcome.get(_ck, "")
                if isinstance(_cv, str) and len(_cv) > 200:
                    full_content_len = len(_cv)
                    actual_content = _cv[:1200]
                    break
            stored_len = outcome.get("content_length", 0)
            if stored_len and stored_len > full_content_len:
                full_content_len = stored_len

            excerpt = outcome.get("response_excerpt", "") or actual_content
            if isinstance(excerpt, str) and len(excerpt) > 1000:
                excerpt = excerpt[:1000] + "..."

            if not excerpt:
                goal_text = str(ep.context.get("goal", ep.context.get("message", "")))
                if not goal_text:
                    continue
                excerpt = f"[Task: {goal_text[:200]}] Quality={ep.quality:.2f}"

            results.append(
                {
                    "domain": ep.domain,
                    "task_type": ep.task_type,
                    "quality": ep.quality,
                    "relevance_score": round(score, 3),
                    "excerpt": excerpt,
                    "actual_content": actual_content[:1200] if actual_content else "",
                    "full_content_len": full_content_len,
                    "goal_preview": str(ep.context.get("goal", ep.context.get("message", "")))[
                        :100
                    ],
                    "agent_name": ep.unit_name,
                }
            )

        return results

    # =====================================================================
    # PUBLIC: retrieve_distilled_lessons
    # =====================================================================

    def retrieve_distilled_lessons(
        self,
        domain: str,
        goal: str = "",
        agent_name: str = "",
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve most relevant distilled lessons via embeddings or confidence."""
        emb_svc = _get_embeddings()

        if goal and emb_svc.available:
            goal_vec = emb_svc.embed(goal)
            if goal_vec is not None:
                return self._embedding_lesson_retrieval(domain, goal_vec, agent_name, top_k)

        lessons = self._store.get_distilled_lessons(
            domain=domain,
            agent_name=agent_name or None,
            limit=top_k * 3,
            hierarchical=True,
        )

        goal_lower = goal.lower() if goal else ""
        scored = []
        for l in lessons:
            score = l.confidence
            if ":" in l.domain:
                sub = l.domain.split(":", 1)[1]
                if sub and sub in goal_lower:
                    score += 0.4
                elif sub and sub not in goal_lower:
                    score -= 0.3
            scored.append((score, l))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                "lesson": l.lesson,
                "type": l.context_type,
                "applies_to": l.applicability,
                "confidence": l.confidence,
                "agent": l.agent_name,
                "domain": l.domain,
            }
            for _, l in scored[:top_k]
        ]

    # =====================================================================
    # PUBLIC: get_best_approach_for_domain
    # =====================================================================

    def get_best_approach_for_domain(
        self, domain: str, task_type: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Return the best historical approach for a domain."""
        try:
            episodes = self._store.query_episodes(domain=domain, limit=50)
            successful = [e for e in episodes if e.success and e.quality > 0]
            if not successful:
                return None

            best = max(successful, key=lambda e: e.quality)
            outcome = best.outcome or {}

            approach: Dict[str, Any] = {
                "quality": best.quality,
                "execution_time": best.execution_time,
                "structural_features": [],
                "outline": None,
                "action": best.action,
            }

            excerpt = outcome.get("response_excerpt", "")
            if "Outline:" in excerpt:
                for line in excerpt.split("\n"):
                    if line.startswith("Outline:"):
                        approach["outline"] = line[len("Outline:") :].strip()
                        break

            if outcome.get("has_code"):
                approach["structural_features"].append(
                    f"{outcome.get('code_block_count', 'multiple')} code blocks"
                )
            if outcome.get("has_headings"):
                approach["structural_features"].append("section headings")
            if outcome.get("has_citations"):
                approach["structural_features"].append("cited references")
            if outcome.get("has_math") and domain not in (
                "coding",
                "algorithms",
                "compiler_design",
            ):
                approach["structural_features"].append("math formulations")
            if outcome.get("has_table"):
                approach["structural_features"].append("comparison tables")
            wc = outcome.get("word_count", 0)
            if wc:
                approach["structural_features"].append(f"~{wc} words")

            return approach
        except Exception as e:
            logger.debug(f"get_best_approach_for_domain failed: {e}")
            return None

    # =====================================================================
    # PRIVATE: retrieval internals
    # =====================================================================

    def _embedding_retrieval(
        self,
        domain: str,
        goal_vec: Any,
        agent_name: str,
        now: float,
    ) -> List[Tuple[float, EpisodeRecord]]:
        """Retrieve episodes using embedding cosine similarity."""
        import numpy as np

        from .embeddings import EmbeddingService

        scored: List[Tuple[float, EpisodeRecord]] = []

        ep_embs = self._store.get_episodes_with_embeddings(domain=domain, limit=200)
        if not ep_embs:
            for alt_domain in _get_related_domains(domain):
                ep_embs = self._store.get_episodes_with_embeddings(domain=alt_domain, limit=100)
                if ep_embs:
                    break
        if not ep_embs:
            ep_embs = self._store.get_episodes_with_embeddings(domain=None, limit=100)

        for ep, emb_blob in ep_embs:
            if ep.quality <= 0:
                continue
            ep_vec = EmbeddingService.deserialize(emb_blob)
            similarity = float(np.dot(goal_vec, ep_vec))

            age_h = max((now - ep.timestamp) / 3600, 0.01)
            recency = 0.15 * _math.exp(-age_h / 6)

            has_content = any(
                isinstance(ep.outcome.get(k, ""), str) and len(ep.outcome.get(k, "")) > 200
                for k in ("content", "response_excerpt")
            )
            content_bonus = 0.1 if has_content else 0.0
            agent_bonus = 0.1 if agent_name and ep.unit_name == agent_name else 0.0

            score = similarity * 0.45 + ep.quality * 0.30 + recency + content_bonus + agent_bonus
            scored.append((score, ep))

        return scored

    def _tfidf_retrieval(
        self,
        domain: str,
        goal: str,
        agent_name: str,
        now: float,
    ) -> List[Tuple[float, EpisodeRecord]]:
        """Fallback TF-IDF retrieval when embeddings unavailable."""
        episodes = self._store.query_episodes(domain=domain, success_only=True, limit=50)
        if not episodes:
            for alt_domain in _get_related_domains(domain):
                episodes = self._store.query_episodes(
                    domain=alt_domain, success_only=True, limit=20
                )
                if episodes:
                    break
            if not episodes:
                episodes = self._store.query_episodes(domain="general", success_only=True, limit=20)

        if not episodes:
            return []

        episodes = [
            ep
            for ep in episodes
            if ep.outcome
            and (
                any(
                    isinstance(ep.outcome.get(k, ""), str) and len(ep.outcome.get(k, "")) > 100
                    for k in ("content", "response", "response_excerpt")
                )
                or (ep.context.get("goal") or ep.context.get("message"))
            )
        ]
        if not episodes:
            return []

        scored: List[Tuple[float, EpisodeRecord]] = []

        def _tokenize(text: str) -> List[str]:
            return [w for w in re.findall(r"\b[a-z]{3,}\b", text.lower()) if w not in _STOP_WORDS]

        if goal:
            goal_tokens = _tokenize(goal)
            goal_tf = _Counter(goal_tokens)
            doc_tokens_list = []
            for ep in episodes:
                ctx_text = str(ep.context.get("goal", ep.context.get("message", "")))
                doc_tokens_list.append(_tokenize(ctx_text))

            n_docs = len(doc_tokens_list) + 1
            df: Dict[str, int] = defaultdict(int)
            for dtoks in doc_tokens_list:
                for w in set(dtoks):
                    df[w] += 1
            for w in set(goal_tokens):
                df[w] += 1

            def _tfidf_vec(tf: Dict[str, int]) -> Dict[str, float]:
                return {w: c * _math.log(n_docs / max(df.get(w, 1), 1)) for w, c in tf.items()}

            def _cosine(v1: Dict[str, float], v2: Dict[str, float]) -> float:
                shared = set(v1) & set(v2)
                if not shared:
                    return 0.0
                dot = sum(v1[w] * v2[w] for w in shared)
                mag1 = _math.sqrt(sum(x * x for x in v1.values()))
                mag2 = _math.sqrt(sum(x * x for x in v2.values()))
                return dot / max(mag1 * mag2, 1e-10)

            goal_vec = _tfidf_vec(goal_tf)

            for i, ep in enumerate(episodes):
                if ep.quality <= 0:
                    continue
                doc_tf = _Counter(doc_tokens_list[i])
                doc_vec = _tfidf_vec(doc_tf)
                relevance = _cosine(goal_vec, doc_vec)
                age_h = max((now - ep.timestamp) / 3600, 0.01)
                recency = 0.15 * _math.exp(-age_h / 6)
                has_content = any(
                    isinstance(ep.outcome.get(k, ""), str) and len(ep.outcome.get(k, "")) > 200
                    for k in ("content", "response_excerpt")
                )
                content_bonus = 0.1 if has_content else 0.0
                agent_bonus = 0.1 if agent_name and ep.unit_name == agent_name else 0.0
                score = ep.quality * 0.4 + relevance * 0.4 + recency + content_bonus + agent_bonus
                scored.append((score, ep))
        else:
            for ep in episodes:
                if ep.quality <= 0:
                    continue
                age_h = max((now - ep.timestamp) / 3600, 0.01)
                recency = 0.15 * _math.exp(-age_h / 6)
                has_content = any(
                    isinstance(ep.outcome.get(k, ""), str) and len(ep.outcome.get(k, "")) > 200
                    for k in ("content", "response_excerpt")
                )
                content_bonus = 0.1 if has_content else 0.0
                agent_bonus = 0.1 if agent_name and ep.unit_name == agent_name else 0.0
                scored.append((ep.quality * 0.75 + recency + content_bonus + agent_bonus, ep))

        return scored

    def _embedding_lesson_retrieval(
        self,
        domain: str,
        goal_vec: Any,
        agent_name: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Retrieve distilled lessons using embedding cosine similarity."""
        import numpy as np

        from .embeddings import EmbeddingService

        lesson_embs = self._store.get_distilled_lessons_with_embeddings(
            domain=domain,
            limit=100,
            hierarchical=True,
        )
        if not lesson_embs:
            lesson_embs = self._store.get_distilled_lessons_with_embeddings(domain=None, limit=100)
        if not lesson_embs:
            return []

        scored = []
        for lesson, emb_blob in lesson_embs:
            lesson_vec = EmbeddingService.deserialize(emb_blob)
            similarity = float(np.dot(goal_vec, lesson_vec))
            agent_bonus = 0.1 if agent_name and lesson.agent_name == agent_name else 0.0
            score = similarity * 0.6 + lesson.confidence * 0.3 + agent_bonus
            scored.append((score, lesson))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                "lesson": lesson.lesson,
                "type": lesson.context_type,
                "applies_to": lesson.applicability,
                "confidence": lesson.confidence,
                "agent": lesson.agent_name,
                "domain": lesson.domain,
                "relevance": round(score, 3),
            }
            for score, lesson in scored[:top_k]
        ]
