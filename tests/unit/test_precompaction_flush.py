"""
Tests for AgentRunner._extract_and_flush_to_memory — pre-compaction memory flush.

This feature saves key facts to episodic memory right before context compression,
ensuring important information is not permanently lost when parts are trimmed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runner():
    """
    Create an AgentRunner with fully mocked dependencies.

    We patch the __init__ to avoid pulling in heavy real dependencies
    (ValidatorAgent, SwarmMemory, TDLambdaLearner, etc.) and only
    exercise the pure-logic _extract_and_flush_to_memory method.
    """
    from Jotty.core.intelligence.orchestration.execution.agent_runner import AgentRunner

    with patch.object(AgentRunner, "__init__", lambda self, *a, **kw: None):
        runner = AgentRunner.__new__(AgentRunner)
    return runner


def _make_agent(*, has_memory: bool = True, memory_is_none: bool = False) -> MagicMock:
    """
    Create a mock agent with configurable memory attribute.

    Args:
        has_memory: If False, the agent has no ``memory`` attribute at all.
        memory_is_none: If True, ``agent.memory`` is ``None``.
    """
    agent = MagicMock()
    if not has_memory:
        # Remove the auto-created memory attribute so getattr returns None
        del agent.memory
    elif memory_is_none:
        agent.memory = None
    else:
        # Normal case: memory with a .store() method
        agent.memory = MagicMock()
        agent.memory.store = MagicMock()
    return agent


def _large_parts(n: int = 5, chars_each: int = 3000) -> list[str]:
    """Return a list of n parts whose total chars exceed the 8000-char threshold."""
    return [f"content-{i} " + "x" * (chars_each - len(f"content-{i} ")) for i in range(n)]


def _small_parts() -> list[str]:
    """Return parts whose total chars are well below the 8000-char threshold."""
    return ["short part one", "short part two"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPreCompactionFlush:
    """Tests for AgentRunner._extract_and_flush_to_memory."""

    # 1. Flush is called and stores to memory when parts > 8000 chars total
    def test_stores_to_memory_when_parts_exceed_threshold(self):
        runner = _make_runner()
        agent = _make_agent(has_memory=True)
        parts = _large_parts(n=5, chars_each=3000)  # 15000 chars total

        runner._extract_and_flush_to_memory(parts, agent)

        agent.memory.store.assert_called_once()
        call_kwargs = agent.memory.store.call_args
        # Positional arg (content) should start with the flush prefix
        stored_content = call_kwargs[0][0]
        assert stored_content.startswith("[Pre-compaction flush]")
        # Level should be episodic
        assert call_kwargs[1]["level"] == "episodic"
        # Metadata should tag the source
        assert call_kwargs[1]["metadata"] == {"source": "pre_compaction_flush"}

    # 2. Flush skips when parts total < 8000 chars (too small)
    def test_skips_when_total_chars_below_threshold(self):
        runner = _make_runner()
        agent = _make_agent(has_memory=True)
        parts = _small_parts()

        runner._extract_and_flush_to_memory(parts, agent)

        agent.memory.store.assert_not_called()

    # 3. Flush skips when agent has no memory attribute
    def test_skips_when_agent_has_no_memory_attribute(self):
        runner = _make_runner()
        agent = _make_agent(has_memory=False)
        parts = _large_parts()

        # Should not raise
        runner._extract_and_flush_to_memory(parts, agent)
        # No memory attribute means nothing to call — just verify no exception

    # 4. Flush skips when agent.memory is None
    def test_skips_when_agent_memory_is_none(self):
        runner = _make_runner()
        agent = _make_agent(memory_is_none=True)
        parts = _large_parts()

        # Should not raise
        runner._extract_and_flush_to_memory(parts, agent)

    # 5. Exception safety: if memory.store() raises, no exception propagates
    def test_exception_in_memory_store_does_not_propagate(self):
        runner = _make_runner()
        agent = _make_agent(has_memory=True)
        agent.memory.store.side_effect = RuntimeError("disk full")
        parts = _large_parts()

        # Must not raise
        runner._extract_and_flush_to_memory(parts, agent)

    # 6. Content stored is truncated to 2000 chars
    def test_stored_content_truncated_to_2000_chars(self):
        runner = _make_runner()
        agent = _make_agent(has_memory=True)
        # Each part is 5000 chars; 3 parts = 15000 chars total.
        # The flush_content before truncation will be > 2000 chars.
        parts = _large_parts(n=3, chars_each=5000)

        runner._extract_and_flush_to_memory(parts, agent)

        agent.memory.store.assert_called_once()
        stored_content = agent.memory.store.call_args[0][0]
        assert len(stored_content) <= 2000

    # 7. Key facts extracted from last 3 parts (reversed)
    def test_key_facts_from_last_three_parts(self):
        runner = _make_runner()
        agent = _make_agent(has_memory=True)

        # Build 5 parts, each large enough to pass threshold
        parts = [
            "A" * 2000,  # part 0 — should NOT appear in key_facts
            "B" * 2000,  # part 1 — should NOT appear in key_facts
            "FACT_C " + "c" * 2000,  # part 2 — last 3 includes this
            "FACT_D " + "d" * 2000,  # part 3
            "FACT_E " + "e" * 2000,  # part 4
        ]

        runner._extract_and_flush_to_memory(parts, agent)

        agent.memory.store.assert_called_once()
        stored_content = agent.memory.store.call_args[0][0]

        # Key facts come from last 3 parts (indices 2, 3, 4), reversed
        assert "FACT_E" in stored_content
        assert "FACT_D" in stored_content
        assert "FACT_C" in stored_content
        # Parts 0 and 1 are pure "A"/"B" filler, not in last 3
        # (they won't appear as key_facts since we only take parts[-3:])

    # 8. Each snippet is capped at 500 chars
    def test_snippets_capped_at_500_chars(self):
        runner = _make_runner()
        agent = _make_agent(has_memory=True)

        marker = "UNIQUE_MARKER_"
        # Place a marker at position 510+ so it would be cut off by the 500-char cap
        parts = [
            "x" * 3000,
            "y" * 3000,
            "z" * 505 + marker + "z" * 2000,  # marker starts at char 505
        ]

        runner._extract_and_flush_to_memory(parts, agent)

        agent.memory.store.assert_called_once()
        stored_content = agent.memory.store.call_args[0][0]
        # The marker is beyond the 500-char snippet boundary
        assert marker not in stored_content

    # 9. Empty parts list — no crash, no store
    def test_empty_parts_list(self):
        runner = _make_runner()
        agent = _make_agent(has_memory=True)

        runner._extract_and_flush_to_memory([], agent)

        agent.memory.store.assert_not_called()

    # 10. All parts are whitespace-only — key_facts ends up empty, no store
    def test_whitespace_only_parts_skipped(self):
        runner = _make_runner()
        agent = _make_agent(has_memory=True)
        # Total chars exceed 8000 but all snippets strip to empty
        parts = [" " * 3000, " " * 3000, " " * 3000]

        runner._extract_and_flush_to_memory(parts, agent)

        agent.memory.store.assert_not_called()

    # 11. Fewer than 3 parts still works (takes all available)
    def test_fewer_than_three_parts(self):
        runner = _make_runner()
        agent = _make_agent(has_memory=True)
        parts = ["ONLY_PART " + "a" * 9000]  # 1 part, > 8000 chars

        runner._extract_and_flush_to_memory(parts, agent)

        agent.memory.store.assert_called_once()
        stored_content = agent.memory.store.call_args[0][0]
        assert "ONLY_PART" in stored_content

    # 12. Separator "---" present between multiple key facts
    def test_separator_between_key_facts(self):
        runner = _make_runner()
        agent = _make_agent(has_memory=True)
        parts = [
            "ALPHA " + "a" * 3000,
            "BETA " + "b" * 3000,
            "GAMMA " + "c" * 3000,
        ]

        runner._extract_and_flush_to_memory(parts, agent)

        agent.memory.store.assert_called_once()
        stored_content = agent.memory.store.call_args[0][0]
        assert "---" in stored_content
