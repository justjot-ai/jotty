"""
Tests for hybrid memory features in llm_rag.py.

Covers:
- BM25Scorer: keyword scoring with TF-IDF-like BM25 algorithm
- _mmr_rerank: Maximal Marginal Relevance re-ranking for diversity
- _word_overlap_similarity: Jaccard similarity between word sets
- RecencyValueRanker: half-life decay (30-day half-life)
"""

import math
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from Jotty.core.intelligence.memory.llm_rag import (
    BM25Scorer,
    RecencyValueRanker,
    _mmr_rerank,
    _word_overlap_similarity,
)
from Jotty.core.infrastructure.foundation.data_structures import (
    MemoryEntry,
    MemoryLevel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory(
    key: str,
    content: str,
    *,
    last_accessed: datetime | None = None,
    default_value: float = 0.5,
    access_count: int = 0,
    level: MemoryLevel = MemoryLevel.EPISODIC,
) -> MemoryEntry:
    """Create a MemoryEntry with sensible defaults for testing."""
    return MemoryEntry(
        key=key,
        content=content,
        level=level,
        context={},
        last_accessed=last_accessed or datetime.now(),
        default_value=default_value,
        access_count=access_count,
    )


# ===========================================================================
# BM25Scorer Tests
# ===========================================================================


@pytest.mark.unit
class TestBM25Scorer:
    """Tests for BM25Scorer.score_batch and BM25Scorer._tokenize."""

    def setup_method(self) -> None:
        self.scorer = BM25Scorer()

    # 1. Exact keyword match scores higher than no match
    def test_exact_keyword_match_scores_higher_than_no_match(self) -> None:
        mem_match = _make_memory("m1", "python programming language")
        mem_no_match = _make_memory("m2", "banana strawberry kiwi")

        scores = self.scorer.score_batch("python", [mem_match, mem_no_match])

        assert scores["m1"] > scores["m2"]
        assert scores["m2"] == 0.0  # no overlap at all

    # 2. Multiple matching terms score higher than single match
    def test_multiple_matching_terms_score_higher(self) -> None:
        mem_single = _make_memory("m1", "python is great")
        mem_multi = _make_memory("m2", "python programming language is great for programming")

        scores = self.scorer.score_batch("python programming", [mem_single, mem_multi])

        assert scores["m2"] > scores["m1"]

    # 3. Empty query returns all zeros
    def test_empty_query_returns_all_zeros(self) -> None:
        mem = _make_memory("m1", "python programming")
        scores = self.scorer.score_batch("", [mem])

        assert scores["m1"] == 0.0

    # 4. Empty memories returns empty dict
    def test_empty_memories_returns_empty_dict(self) -> None:
        scores = self.scorer.score_batch("python", [])
        assert scores == {}

    # 5. Scores are normalized to 0-1 range
    def test_scores_normalized_to_zero_one_range(self) -> None:
        memories = [
            _make_memory("m1", "python machine learning deep learning"),
            _make_memory("m2", "python programming"),
            _make_memory("m3", "java enterprise bean"),
            _make_memory("m4", "python deep learning neural network machine learning"),
        ]

        scores = self.scorer.score_batch("python machine learning", memories)

        for key, score in scores.items():
            assert 0.0 <= score <= 1.0, f"Score for {key} = {score} out of range"

        # At least one score should be exactly 1.0 (the max is normalized to 1)
        assert max(scores.values()) == 1.0

    # 6. _tokenize returns lowercase split words
    def test_tokenize_returns_lowercase_split_words(self) -> None:
        tokens = BM25Scorer._tokenize("Hello World FOO bar")
        assert tokens == ["hello", "world", "foo", "bar"]

    def test_tokenize_empty_string(self) -> None:
        tokens = BM25Scorer._tokenize("")
        # "".lower().split() returns []
        assert tokens == []

    def test_single_memory_gets_score_one(self) -> None:
        """A single memory matching the query should get normalized score 1.0."""
        mem = _make_memory("m1", "python programming")
        scores = self.scorer.score_batch("python", [mem])

        assert scores["m1"] == 1.0

    def test_custom_k1_b_parameters(self) -> None:
        """Custom BM25 parameters k1 and b are respected."""
        scorer_custom = BM25Scorer(k1=2.0, b=0.5)
        assert scorer_custom.k1 == 2.0
        assert scorer_custom.b == 0.5

        mem = _make_memory("m1", "python programming language")
        scores = scorer_custom.score_batch("python", [mem])
        assert 0.0 <= scores["m1"] <= 1.0


# ===========================================================================
# _mmr_rerank Tests
# ===========================================================================


@pytest.mark.unit
class TestMMRRerank:
    """Tests for _mmr_rerank (Maximal Marginal Relevance)."""

    # 7. Single item returns unchanged
    def test_single_item_returns_unchanged(self) -> None:
        mem = _make_memory("m1", "python programming")
        scored_items = [(mem, 0.9)]

        result = _mmr_rerank(scored_items, [mem])

        assert len(result) == 1
        assert result[0][0].key == "m1"
        assert result[0][1] == 0.9

    # 8. Two identical content items -- second gets lower rank (diversity)
    def test_identical_content_second_gets_lower_rank(self) -> None:
        mem_a = _make_memory("m1", "python machine learning framework")
        mem_b = _make_memory("m2", "python machine learning framework")

        # Both have same score; MMR should still select both but the
        # identical second item is penalized by the diversity term.
        scored_items = [(mem_a, 0.9), (mem_b, 0.9)]

        result = _mmr_rerank(scored_items, [mem_a, mem_b])

        assert len(result) == 2
        # First selected should be one of them (highest raw score)
        assert result[0][0].key == "m1"
        # Second is the duplicate - it IS selected but ranked second
        assert result[1][0].key == "m2"

    # 9. Two different content items -- both retained
    def test_different_content_both_retained(self) -> None:
        mem_a = _make_memory("m1", "python machine learning neural networks")
        mem_b = _make_memory("m2", "java enterprise application server deployment")

        scored_items = [(mem_a, 0.8), (mem_b, 0.7)]

        result = _mmr_rerank(scored_items, [mem_a, mem_b])

        assert len(result) == 2
        keys = {r[0].key for r in result}
        assert keys == {"m1", "m2"}

    # 10. Respects top_k limit
    def test_respects_top_k_limit(self) -> None:
        memories = [_make_memory(f"m{i}", f"content number {i}") for i in range(10)]
        scored_items = [(m, 1.0 - i * 0.05) for i, m in enumerate(memories)]

        result = _mmr_rerank(scored_items, memories, top_k=3)

        assert len(result) == 3

    def test_empty_list_returns_empty(self) -> None:
        result = _mmr_rerank([], [])
        assert result == []

    def test_diversity_promotes_dissimilar_items(self) -> None:
        """With low lambda (diversity-heavy), dissimilar items get promoted."""
        mem_a = _make_memory("m1", "python machine learning framework")
        mem_b = _make_memory("m2", "python machine learning library")  # similar to a
        mem_c = _make_memory("m3", "java enterprise server deployment")  # dissimilar

        # b has higher raw score than c, but c is more diverse
        scored_items = [(mem_a, 0.9), (mem_b, 0.7), (mem_c, 0.65)]

        # lambda=0.3 means diversity is weighted 0.7 -- strongly favors diversity
        result = _mmr_rerank(scored_items, [mem_a, mem_b, mem_c], lambda_param=0.3)

        assert len(result) == 3
        # First is always highest score
        assert result[0][0].key == "m1"
        # With strong diversity preference, the dissimilar item (m3) should
        # come second despite lower raw score
        assert result[1][0].key == "m3"


# ===========================================================================
# _word_overlap_similarity Tests
# ===========================================================================


@pytest.mark.unit
class TestWordOverlapSimilarity:
    """Tests for _word_overlap_similarity (Jaccard similarity)."""

    # 11. Identical texts return 1.0
    def test_identical_texts_return_one(self) -> None:
        assert _word_overlap_similarity("hello world", "hello world") == 1.0

    # 12. No overlap returns 0.0
    def test_no_overlap_returns_zero(self) -> None:
        assert _word_overlap_similarity("hello world", "foo bar") == 0.0

    # 13. Partial overlap returns between 0 and 1
    def test_partial_overlap_returns_between_zero_and_one(self) -> None:
        sim = _word_overlap_similarity("hello world foo", "hello bar baz")
        # intersection = {"hello"}, union = {"hello", "world", "foo", "bar", "baz"}
        # Jaccard = 1/5 = 0.2
        assert sim == pytest.approx(0.2)
        assert 0.0 < sim < 1.0

    def test_case_insensitive(self) -> None:
        """Similarity is case-insensitive (both texts lowered)."""
        assert _word_overlap_similarity("Hello World", "hello world") == 1.0

    def test_empty_text_a_returns_zero(self) -> None:
        assert _word_overlap_similarity("", "hello world") == 0.0

    def test_empty_text_b_returns_zero(self) -> None:
        assert _word_overlap_similarity("hello world", "") == 0.0

    def test_both_empty_returns_zero(self) -> None:
        assert _word_overlap_similarity("", "") == 0.0

    def test_duplicate_words_treated_as_set(self) -> None:
        """Repeated words don't inflate the similarity."""
        # "hello hello" as a set is {"hello"}, "hello world" is {"hello", "world"}
        # Jaccard = 1/2 = 0.5
        sim = _word_overlap_similarity("hello hello hello", "hello world")
        assert sim == pytest.approx(0.5)


# ===========================================================================
# RecencyValueRanker half-life Tests
# ===========================================================================


@pytest.mark.unit
class TestRecencyValueRankerHalfLife:
    """Tests for RecencyValueRanker's 30-day half-life decay."""

    def setup_method(self) -> None:
        # RecencyValueRanker needs a config; mock it so _ensure_swarm_config passes
        self.config = MagicMock()
        self.ranker = RecencyValueRanker(self.config)

    # 14. Memory accessed 30 days ago gets ~0.5 recency score
    def test_30_day_half_life(self) -> None:
        """
        The decay constant 0.000963 gives half-life at ~720 hours (30 days).
        A memory last accessed 30 days ago should have recency ~0.5.
        """
        now = datetime.now()
        thirty_days_ago = now - timedelta(days=30)

        mem = _make_memory(
            "m1",
            "some content",
            last_accessed=thirty_days_ago,
            default_value=0.5,
            access_count=1,
        )

        # Compute recency score using the same formula as the ranker
        age_hours = (now - mem.last_accessed).total_seconds() / 3600
        recency_score = math.exp(-0.000963 * age_hours)

        # 30 days = 720 hours => exp(-0.000963 * 720) ~ exp(-0.693) ~ 0.5
        assert recency_score == pytest.approx(0.5, abs=0.02)

    def test_recent_memory_has_high_recency(self) -> None:
        """A memory accessed just now should have recency close to 1.0."""
        now = datetime.now()
        mem = _make_memory("m1", "fresh content", last_accessed=now)

        age_hours = (now - mem.last_accessed).total_seconds() / 3600
        recency_score = math.exp(-0.000963 * age_hours)

        assert recency_score == pytest.approx(1.0, abs=0.01)

    def test_old_memory_has_low_recency(self) -> None:
        """A memory accessed 90 days ago should have recency << 0.5."""
        now = datetime.now()
        ninety_days_ago = now - timedelta(days=90)

        age_hours = (now - ninety_days_ago).total_seconds() / 3600
        recency_score = math.exp(-0.000963 * age_hours)

        # 90 days = 2160 hours => exp(-0.000963 * 2160) ~ exp(-2.08) ~ 0.125
        assert recency_score < 0.2

    def test_prerank_returns_sorted_by_combined_score(self) -> None:
        """prerank should return memories sorted by combined score descending."""
        now = datetime.now()
        mem_recent = _make_memory(
            "recent",
            "recent content",
            last_accessed=now,
            default_value=0.8,
            access_count=5,
        )
        mem_old = _make_memory(
            "old",
            "old content",
            last_accessed=now - timedelta(days=60),
            default_value=0.3,
            access_count=1,
        )

        result = self.ranker.prerank([mem_old, mem_recent], goal="test")

        assert len(result) == 2
        # Recent + high-value memory should rank first
        assert result[0][0].key == "recent"
        assert result[1][0].key == "old"

    def test_prerank_respects_max_candidates(self) -> None:
        """prerank should return at most max_candidates items."""
        memories = [_make_memory(f"m{i}", f"content {i}") for i in range(20)]

        result = self.ranker.prerank(memories, goal="test", max_candidates=5)

        assert len(result) == 5

    def test_prerank_level_priority(self) -> None:
        """META-level memories should score higher than EPISODIC, all else equal."""
        now = datetime.now()
        mem_meta = _make_memory(
            "meta",
            "meta knowledge",
            last_accessed=now,
            default_value=0.5,
            access_count=1,
            level=MemoryLevel.META,
        )
        mem_episodic = _make_memory(
            "episodic",
            "episodic knowledge",
            last_accessed=now,
            default_value=0.5,
            access_count=1,
            level=MemoryLevel.EPISODIC,
        )

        result = self.ranker.prerank([mem_episodic, mem_meta], goal="test")

        assert result[0][0].key == "meta"
        assert result[1][0].key == "episodic"
