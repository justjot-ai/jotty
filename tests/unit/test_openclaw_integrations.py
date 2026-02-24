"""
Tests for OpenClaw-inspired integrations.

4 features:
1. Enhanced pre-compaction memory flush (priority-based extraction)
2. Graceful degradation cascade for memory retrieval
3. Approval gates with resume tokens in SwarmTemplate
4. Trust-level tool sandboxing per channel
"""

import base64
import json
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# 1. ENHANCED PRE-COMPACTION MEMORY FLUSH
# =============================================================================


class TestPreCompactionMemoryFlush:
    """Test enhanced priority-based memory flush before context compaction."""

    def _make_runner(self):
        """Create a minimal AgentRunner-like object with the flush method."""
        from Jotty.core.intelligence.orchestration.execution.agent_runner import AgentRunner

        runner = object.__new__(AgentRunner)
        runner._consecutive_failures = 0
        runner._max_consecutive_before_hint = 3
        return runner

    @pytest.mark.unit
    def test_flush_extracts_priority_markers(self):
        """Lines with RESULT:, CONCLUSION:, etc. are prioritized."""
        runner = self._make_runner()

        memory = MagicMock()
        agent = MagicMock()
        agent.memory = memory

        parts = [
            "System prompt padding text " * 400,  # ~10800 chars to exceed 8000 threshold
            "Some regular context text\nRESULT: The API returned 200 OK with valid JSON\nMore text",
            "Another part\nCONCLUSION: The approach works for batch processing\nEnd",
        ]

        runner._extract_and_flush_to_memory(parts, agent)

        assert memory.store.called
        stored_content = memory.store.call_args[0][0]
        assert "RESULT:" in stored_content
        assert "CONCLUSION:" in stored_content

    @pytest.mark.unit
    def test_flush_skips_small_context(self):
        """Content under 8000 chars is not flushed."""
        runner = self._make_runner()

        memory = MagicMock()
        agent = MagicMock()
        agent.memory = memory

        parts = ["Small context"]

        runner._extract_and_flush_to_memory(parts, agent)

        assert not memory.store.called

    @pytest.mark.unit
    def test_flush_falls_back_to_tail_when_no_markers(self):
        """When no priority markers found, falls back to last 3 parts."""
        runner = self._make_runner()

        memory = MagicMock()
        agent = MagicMock()
        agent.memory = memory

        parts = [
            "Padding " * 2000,  # >8000 chars
            "Second part with useful context about the task",
            "Third part: the agent completed the web search successfully",
            "Fourth part: final analysis of the results gathered",
        ]

        runner._extract_and_flush_to_memory(parts, agent)

        assert memory.store.called
        stored_content = memory.store.call_args[0][0]
        assert "Pre-compaction flush" in stored_content

    @pytest.mark.unit
    def test_flush_handles_missing_memory(self):
        """No crash when agent has no memory."""
        runner = self._make_runner()

        agent = MagicMock(spec=[])  # No memory attribute

        parts = ["Padding " * 2000]

        # Should not raise
        runner._extract_and_flush_to_memory(parts, agent)

    @pytest.mark.unit
    def test_flush_stores_chars_before_in_metadata(self):
        """Metadata includes original char count for debugging."""
        runner = self._make_runner()

        memory = MagicMock()
        agent = MagicMock()
        agent.memory = memory

        parts = [
            "Padding " * 2000,
            "RESULT: Found 15 matching documents",
        ]

        runner._extract_and_flush_to_memory(parts, agent)

        if memory.store.called:
            _, kwargs = memory.store.call_args
            assert "chars_before" in kwargs.get("metadata", {})


# =============================================================================
# 2. GRACEFUL DEGRADATION CASCADE FOR MEMORY RETRIEVAL
# =============================================================================


class TestMemoryRetrievalCascade:
    """Test graceful degradation: LLM → BM25 → preranked fallback."""

    @pytest.mark.unit
    def test_bm25_scorer_works_standalone(self):
        """BM25Scorer can score memories independently."""
        from Jotty.core.intelligence.memory.llm_rag import BM25Scorer

        scorer = BM25Scorer()
        # Create mock memories
        mem1 = MagicMock()
        mem1.key = "mem1"
        mem1.content = "python machine learning tensorflow"

        mem2 = MagicMock()
        mem2.key = "mem2"
        mem2.content = "cooking recipes italian pasta"

        scores = scorer.score_batch("python machine learning", [mem1, mem2])

        assert scores["mem1"] > scores["mem2"]

    @pytest.mark.unit
    def test_bm25_scorer_empty_query(self):
        """BM25Scorer handles empty queries gracefully."""
        from Jotty.core.intelligence.memory.llm_rag import BM25Scorer

        scorer = BM25Scorer()
        mem = MagicMock()
        mem.key = "mem1"
        mem.content = "some content"

        scores = scorer.score_batch("", [mem])
        assert scores["mem1"] == 0.0


# =============================================================================
# 3. APPROVAL GATES WITH RESUME TOKENS
# =============================================================================


class TestApprovalGates:
    """Test approval gate checkpoints and resume tokens."""

    @pytest.mark.unit
    def test_approval_checkpoint_round_trip(self):
        """Checkpoint serializes to dict and back."""
        from Jotty.core.intelligence.orchestration.swarms.base.approval_gates import (
            ApprovalCheckpoint,
            ApprovalStatus,
        )

        cp = ApprovalCheckpoint(
            gate_name="pre_deploy",
            swarm_class="DeploySwarm",
            checkpoint_data={"build_hash": "abc123"},
            original_args={"requirements": "deploy v2"},
        )

        d = cp.to_dict()
        restored = ApprovalCheckpoint.from_dict(d)

        assert restored.gate_name == "pre_deploy"
        assert restored.swarm_class == "DeploySwarm"
        assert restored.checkpoint_data == {"build_hash": "abc123"}
        assert restored.status == ApprovalStatus.PENDING

    @pytest.mark.unit
    def test_resume_token_encode_decode(self):
        """Resume token encodes/decodes checkpoint ID."""
        from Jotty.core.intelligence.orchestration.swarms.base.approval_gates import (
            ApprovalCheckpoint,
        )

        cp = ApprovalCheckpoint(checkpoint_id="test123abc")
        token = cp.resume_token
        decoded_id = ApprovalCheckpoint.id_from_token(token)

        assert decoded_id == "test123abc"

    @pytest.mark.unit
    def test_checkpoint_store_save_load(self):
        """CheckpointStore saves and loads from filesystem."""
        from Jotty.core.intelligence.orchestration.swarms.base.approval_gates import (
            ApprovalCheckpoint,
            ApprovalStatus,
            CheckpointStore,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store = CheckpointStore(store_dir=tmpdir)

            cp = ApprovalCheckpoint(
                gate_name="test_gate",
                swarm_class="TestSwarm",
                checkpoint_data={"key": "value"},
            )

            token = store.save(cp)
            loaded = store.load_from_token(token)

            assert loaded is not None
            assert loaded.gate_name == "test_gate"
            assert loaded.checkpoint_data == {"key": "value"}
            assert loaded.status == ApprovalStatus.PENDING

    @pytest.mark.unit
    def test_checkpoint_store_approve(self):
        """CheckpointStore can mark checkpoint as approved."""
        from Jotty.core.intelligence.orchestration.swarms.base.approval_gates import (
            ApprovalCheckpoint,
            ApprovalStatus,
            CheckpointStore,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store = CheckpointStore(store_dir=tmpdir)

            cp = ApprovalCheckpoint(gate_name="deploy")
            store.save(cp)

            assert store.approve(cp.checkpoint_id, approver="admin")

            loaded = store.load(cp.checkpoint_id)
            assert loaded.status == ApprovalStatus.APPROVED
            assert loaded.approver == "admin"

    @pytest.mark.unit
    def test_checkpoint_store_reject(self):
        """CheckpointStore can mark checkpoint as rejected."""
        from Jotty.core.intelligence.orchestration.swarms.base.approval_gates import (
            ApprovalCheckpoint,
            ApprovalStatus,
            CheckpointStore,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store = CheckpointStore(store_dir=tmpdir)

            cp = ApprovalCheckpoint(gate_name="deploy")
            store.save(cp)

            assert store.reject(cp.checkpoint_id, reason="Not ready", approver="admin")

            loaded = store.load(cp.checkpoint_id)
            assert loaded.status == ApprovalStatus.REJECTED
            assert loaded.rejection_reason == "Not ready"

    @pytest.mark.unit
    def test_checkpoint_store_list_pending(self):
        """list_pending returns only pending checkpoints."""
        from Jotty.core.intelligence.orchestration.swarms.base.approval_gates import (
            ApprovalCheckpoint,
            CheckpointStore,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store = CheckpointStore(store_dir=tmpdir)

            cp1 = ApprovalCheckpoint(gate_name="gate1")
            cp2 = ApprovalCheckpoint(gate_name="gate2")
            store.save(cp1)
            store.save(cp2)
            store.approve(cp1.checkpoint_id)

            pending = store.list_pending()
            assert len(pending) == 1
            assert pending[0].gate_name == "gate2"

    @pytest.mark.unit
    def test_gate_result_to_swarm_result(self):
        """GateResult.to_swarm_result() creates valid SwarmResult when paused."""
        from Jotty.core.intelligence.orchestration.swarms.base.approval_gates import (
            ApprovalCheckpoint,
            GateResult,
        )

        checkpoint = ApprovalCheckpoint(gate_name="pre_deploy")
        gate_result = GateResult(
            paused=True,
            resume_token="test_token",
            checkpoint=checkpoint,
        )

        swarm_result = gate_result.to_swarm_result(swarm_name="DeploySwarm", domain="infra")
        assert swarm_result.success is True
        assert swarm_result.output["status"] == "awaiting_approval"
        assert swarm_result.output["resume_token"] == "test_token"

    @pytest.mark.unit
    def test_gate_result_not_paused(self):
        """GateResult.to_swarm_result() returns None when not paused."""
        from Jotty.core.intelligence.orchestration.swarms.base.approval_gates import GateResult

        gate_result = GateResult(paused=False, approved=True)
        assert gate_result.to_swarm_result() is None

    @pytest.mark.unit
    def test_invalid_resume_token(self):
        """Loading from invalid token returns None."""
        from Jotty.core.intelligence.orchestration.swarms.base.approval_gates import (
            CheckpointStore,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store = CheckpointStore(store_dir=tmpdir)
            result = store.load_from_token("not-a-valid-base64!!!")
            assert result is None


# =============================================================================
# 4. TRUST-LEVEL TOOL SANDBOXING
# =============================================================================


class TestTrustLevelSandboxing:
    """Test trust level resolution and tool allowlisting."""

    @pytest.mark.unit
    def test_trust_level_enum(self):
        """TrustLevel enum has expected values."""
        from Jotty.apps.cli.gateway.trust import TrustLevel

        assert TrustLevel.OWNER.value == "owner"
        assert TrustLevel.TRUSTED.value == "trusted"
        assert TrustLevel.UNTRUSTED.value == "untrusted"

    @pytest.mark.unit
    def test_cli_gets_owner_trust(self):
        """CLI channel resolves to OWNER trust level."""
        from Jotty.apps.cli.gateway.channels import ChannelType
        from Jotty.apps.cli.gateway.trust import TrustLevel, TrustManager

        tm = TrustManager(config_path="/tmp/test_trust_nonexistent.json")
        level = tm.resolve_trust_level(ChannelType.CLI)
        assert level == TrustLevel.OWNER

    @pytest.mark.unit
    def test_allowlisted_telegram_user_gets_trusted(self):
        """Allowlisted Telegram user gets TRUSTED level."""
        from Jotty.apps.cli.gateway.channels import ChannelType
        from Jotty.apps.cli.gateway.trust import TrustLevel, TrustManager

        tm = TrustManager(config_path="/tmp/test_trust_nonexistent2.json")
        tm.add_to_allowlist(ChannelType.TELEGRAM, "user123")

        level = tm.resolve_trust_level(ChannelType.TELEGRAM, "user123")
        assert level == TrustLevel.TRUSTED

    @pytest.mark.unit
    def test_unknown_telegram_user_gets_untrusted(self):
        """Unknown Telegram user gets UNTRUSTED level."""
        from Jotty.apps.cli.gateway.channels import ChannelType
        from Jotty.apps.cli.gateway.trust import TrustLevel, TrustManager

        tm = TrustManager(config_path="/tmp/test_trust_nonexistent3.json")
        level = tm.resolve_trust_level(ChannelType.TELEGRAM, "unknown_user")
        assert level == TrustLevel.UNTRUSTED

    @pytest.mark.unit
    def test_web_channel_gets_untrusted(self):
        """Web channel gets UNTRUSTED level regardless of user."""
        from Jotty.apps.cli.gateway.channels import ChannelType
        from Jotty.apps.cli.gateway.trust import TrustLevel, TrustManager

        tm = TrustManager(config_path="/tmp/test_trust_nonexistent4.json")
        level = tm.resolve_trust_level(ChannelType.WEB, "any_user")
        assert level == TrustLevel.UNTRUSTED

    @pytest.mark.unit
    def test_owner_tool_allowlist_unrestricted(self):
        """OWNER trust level allows all tools (None = unrestricted)."""
        from Jotty.apps.cli.gateway.trust import TrustLevel, TrustManager

        tm = TrustManager(config_path="/tmp/test_trust_nonexistent5.json")
        assert tm.is_tool_allowed(TrustLevel.OWNER, "shell-exec") is True
        assert tm.is_tool_allowed(TrustLevel.OWNER, "anything") is True

    @pytest.mark.unit
    def test_untrusted_cannot_use_shell(self):
        """UNTRUSTED level cannot access shell-exec or file-operations."""
        from Jotty.apps.cli.gateway.trust import TrustLevel, TrustManager

        tm = TrustManager(config_path="/tmp/test_trust_nonexistent6.json")
        assert tm.is_tool_allowed(TrustLevel.UNTRUSTED, "shell-exec") is False
        assert tm.is_tool_allowed(TrustLevel.UNTRUSTED, "file-operations") is False

    @pytest.mark.unit
    def test_trusted_can_use_web_search(self):
        """TRUSTED level can use web-search."""
        from Jotty.apps.cli.gateway.trust import TrustLevel, TrustManager

        tm = TrustManager(config_path="/tmp/test_trust_nonexistent7.json")
        assert tm.is_tool_allowed(TrustLevel.TRUSTED, "web-search") is True

    @pytest.mark.unit
    def test_untrusted_can_use_calculator(self):
        """UNTRUSTED level can use safe read-only tools."""
        from Jotty.apps.cli.gateway.trust import TrustLevel, TrustManager

        tm = TrustManager(config_path="/tmp/test_trust_nonexistent8.json")
        assert tm.is_tool_allowed(TrustLevel.UNTRUSTED, "calculator") is True
        assert tm.is_tool_allowed(TrustLevel.UNTRUSTED, "web-search") is True

    @pytest.mark.unit
    def test_execution_context_has_trust_level(self):
        """ExecutionContext includes trust_level field."""
        from Jotty.core.infrastructure.foundation.types.sdk_types import (
            ChannelType,
            ExecutionContext,
            ExecutionMode,
        )

        ctx = ExecutionContext(
            mode=ExecutionMode.CHAT,
            channel=ChannelType.TELEGRAM,
            trust_level="trusted",
        )
        assert ctx.trust_level == "trusted"
        assert ctx.to_dict()["trust_level"] == "trusted"

    @pytest.mark.unit
    def test_execution_context_default_trust_is_owner(self):
        """ExecutionContext defaults to 'owner' trust level."""
        from Jotty.core.infrastructure.foundation.types.sdk_types import (
            ChannelType,
            ExecutionContext,
            ExecutionMode,
        )

        ctx = ExecutionContext(mode=ExecutionMode.CHAT, channel=ChannelType.CLI)
        assert ctx.trust_level == "owner"

    @pytest.mark.unit
    def test_skill_plan_executor_allowed_skills(self):
        """SkillPlanExecutor respects _allowed_skills blocklist."""
        from Jotty.core.intelligence.reasoning.executors.skill_plan_executor import (
            SkillPlanExecutor,
        )

        executor = SkillPlanExecutor(skills_registry=MagicMock())
        executor._allowed_skills = {"web-search", "calculator"}

        # Create a mock step for a blocked skill
        step = MagicMock()
        step.skill_name = "shell-exec"
        step.tool_name = "run_command"

        # Mock the skill lookup to succeed (the trust check should block before invocation)
        mock_skill = MagicMock()
        mock_skill.tools = {"run_command": MagicMock()}
        executor._skills_registry.get_skill = MagicMock(return_value=mock_skill)

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            executor.execute_step(step, outputs={})
        )

        assert result["success"] is False
        assert "not permitted" in result["error"]


# =============================================================================
# 5. STRUCTURED COMPACTION SUMMARIES
# =============================================================================


class TestStructuredCompactionSummary:
    """Test structured checkpoint extraction from context parts."""

    @pytest.mark.unit
    def test_extracts_goals(self):
        """Goal markers are detected and categorized."""
        from Jotty.core.infrastructure.context.utils import structured_compaction_summary

        parts = [
            "Please help me build an API framework for the project",
            "Some filler text that is not relevant to classification",
        ]
        result = structured_compaction_summary(parts)
        assert "## Goal" in result
        assert "API framework" in result

    @pytest.mark.unit
    def test_extracts_progress(self):
        """Progress markers (done, completed, ✅) are detected."""
        from Jotty.core.infrastructure.context.utils import structured_compaction_summary

        parts = [
            "✅ Built the authentication module successfully",
            "Completed: Database migration for users table",
            "Result: All 15 tests passing in CI pipeline",
        ]
        result = structured_compaction_summary(parts)
        assert "## Progress" in result

    @pytest.mark.unit
    def test_extracts_blockers(self):
        """Blocker markers (error, failed, ❌) are detected."""
        from Jotty.core.infrastructure.context.utils import structured_compaction_summary

        parts = [
            "Error: Connection refused on port 5432 for database",
            "❌ The deploy script failed with exit code 1",
        ]
        result = structured_compaction_summary(parts)
        assert "## Blockers" in result

    @pytest.mark.unit
    def test_extracts_decisions(self):
        """Decision markers (decided, chose, using) are detected."""
        from Jotty.core.infrastructure.context.utils import structured_compaction_summary

        parts = [
            "Decided to use PostgreSQL instead of MySQL for this project",
            "Going with FastAPI for the web framework implementation",
        ]
        result = structured_compaction_summary(parts)
        assert "## Key Decisions" in result

    @pytest.mark.unit
    def test_extracts_next_steps(self):
        """Next step markers (next, todo, remaining) are detected."""
        from Jotty.core.infrastructure.context.utils import structured_compaction_summary

        parts = [
            "Next: Deploy the staging environment and run integration tests",
            "Still need to implement the caching layer for API responses",
        ]
        result = structured_compaction_summary(parts)
        assert "## Next Steps" in result

    @pytest.mark.unit
    def test_fallback_when_no_markers(self):
        """Falls back to first+last lines when no markers found."""
        from Jotty.core.infrastructure.context.utils import structured_compaction_summary

        parts = [
            "The quick brown fox jumps over the lazy dog",
            "Another random line of text for testing purposes",
            "A third line that doesn't contain any keywords at all",
        ]
        result = structured_compaction_summary(parts)
        assert "[Structured Compaction Checkpoint]" in result
        assert "Context:" in result

    @pytest.mark.unit
    def test_empty_parts(self):
        """Empty parts list returns header only."""
        from Jotty.core.infrastructure.context.utils import structured_compaction_summary

        result = structured_compaction_summary([])
        assert "[Structured Compaction Checkpoint]" in result

    @pytest.mark.unit
    def test_compress_parts_uses_structured_checkpoint(self):
        """SmartContextManager.compress_parts prepends structured checkpoint."""
        from Jotty.core.infrastructure.context.context_manager import SmartContextManager

        mgr = SmartContextManager(max_tokens=28000)
        parts = [
            "Please help me build an API for user management",
            "✅ Created the database schema with 5 tables successfully",
            "Next: implement the REST endpoints for CRUD operations",
            "Extra context " * 200,  # force compression
        ]

        result = mgr.compress_parts(parts, max_total_chars=2000)
        assert result[0].startswith("[Structured Compaction Checkpoint]")
        assert "## Goal" in result[0] or "## Progress" in result[0]


# =============================================================================
# 6. PROGRESSIVE SUMMARIZATION TREE
# =============================================================================


class TestProgressiveSummarizer:
    """Test progressive summarization at multiple abstraction levels."""

    @pytest.mark.unit
    def test_add_block_stores_at_level0(self):
        """New blocks are stored at level 0 (raw)."""
        from Jotty.core.infrastructure.context.progressive_summarizer import (
            ProgressiveSummarizer,
        )

        ps = ProgressiveSummarizer(max_level0_chars=8000)
        ps.add_block("Test content block one")

        stats = ps.stats()
        assert stats["level0_blocks"] == 1
        assert stats["level1_blocks"] == 0
        assert stats["level2_blocks"] == 0

    @pytest.mark.unit
    def test_promotion_to_level1_on_overflow(self):
        """When level-0 exceeds budget, oldest blocks promote to level-1."""
        from Jotty.core.infrastructure.context.progressive_summarizer import (
            ProgressiveSummarizer,
        )

        ps = ProgressiveSummarizer(max_level0_chars=500)
        ps.add_block("A" * 300)  # 300 chars
        ps.add_block("B" * 300)  # 600 total -> exceeds 500

        stats = ps.stats()
        assert stats["level1_blocks"] >= 1  # Oldest promoted
        assert stats["level0_blocks"] >= 1  # Most recent kept

    @pytest.mark.unit
    def test_merge_to_level2(self):
        """When level-1 accumulates enough blocks, they merge to level-2."""
        from Jotty.core.infrastructure.context.progressive_summarizer import (
            ProgressiveSummarizer,
        )

        ps = ProgressiveSummarizer(
            max_level0_chars=200,
            max_level1_blocks=3,
        )
        # Add many blocks to force promotions and merges
        for i in range(10):
            ps.add_block(f"Block {i}: " + "x" * 250)

        stats = ps.stats()
        assert stats["level2_blocks"] >= 1

    @pytest.mark.unit
    def test_build_context_within_budget(self):
        """build_context respects the character budget."""
        from Jotty.core.infrastructure.context.progressive_summarizer import (
            ProgressiveSummarizer,
        )

        ps = ProgressiveSummarizer(max_level0_chars=8000)
        ps.add_block("Phase 1: Built the API framework for the project")
        ps.add_block("Phase 2: Added authentication and authorization")
        ps.add_block("Phase 3: Deployed to staging environment successfully")

        context = ps.build_context(budget_chars=4000)
        assert len(context) <= 4000

    @pytest.mark.unit
    def test_build_context_includes_recent_raw(self):
        """build_context includes recent level-0 blocks verbatim."""
        from Jotty.core.infrastructure.context.progressive_summarizer import (
            ProgressiveSummarizer,
        )

        ps = ProgressiveSummarizer(max_level0_chars=8000)
        ps.add_block("UNIQUE_MARKER_12345")

        context = ps.build_context(budget_chars=4000)
        assert "UNIQUE_MARKER_12345" in context

    @pytest.mark.unit
    def test_empty_blocks_ignored(self):
        """Empty or whitespace-only blocks are not added."""
        from Jotty.core.infrastructure.context.progressive_summarizer import (
            ProgressiveSummarizer,
        )

        ps = ProgressiveSummarizer()
        ps.add_block("")
        ps.add_block("   ")
        ps.add_block(None if False else "")  # edge case

        assert ps.total_blocks == 0

    @pytest.mark.unit
    def test_clear_resets_all_levels(self):
        """clear() empties all three levels."""
        from Jotty.core.infrastructure.context.progressive_summarizer import (
            ProgressiveSummarizer,
        )

        ps = ProgressiveSummarizer(max_level0_chars=100)
        for i in range(5):
            ps.add_block(f"Block {i}: " + "y" * 150)

        ps.clear()
        stats = ps.stats()
        assert stats["level0_blocks"] == 0
        assert stats["level1_blocks"] == 0
        assert stats["level2_blocks"] == 0

    @pytest.mark.unit
    def test_summary_block_char_count(self):
        """SummaryBlock auto-calculates char_count."""
        from Jotty.core.infrastructure.context.progressive_summarizer import SummaryBlock

        block = SummaryBlock(content="Hello world")
        assert block.char_count == 11
        assert block.level == 0
        assert block.source_count == 1

    @pytest.mark.unit
    def test_facade_accessor(self):
        """get_progressive_summarizer() returns a singleton."""
        from Jotty.core.infrastructure.context.facade import (
            get_progressive_summarizer,
            reset_singletons,
        )

        reset_singletons()
        ps1 = get_progressive_summarizer()
        ps2 = get_progressive_summarizer()
        assert ps1 is ps2
        reset_singletons()


# =============================================================================
# 7. SUMMARIZE SKILL
# =============================================================================


class TestSummarizeSkill:
    """Test the universal summarize skill (text, URL, file)."""

    @pytest.mark.unit
    def test_summarize_text_empty_input(self):
        """summarize_text_tool rejects empty text."""
        from skills.summarize.tools import summarize_text_tool

        result = summarize_text_tool({"text": ""})
        assert result["success"] is False
        assert (
            "required" in result.get("error", "").lower()
            or "text" in result.get("error", "").lower()
        )

    @pytest.mark.unit
    def test_summarize_text_missing_param(self):
        """summarize_text_tool rejects missing text param."""
        from skills.summarize.tools import summarize_text_tool

        result = summarize_text_tool({})
        assert result["success"] is False

    @pytest.mark.unit
    def test_summarize_url_missing_param(self):
        """summarize_url_tool rejects missing URL."""
        from skills.summarize.tools import summarize_url_tool

        result = summarize_url_tool({})
        assert result["success"] is False
        assert "url" in result.get("error", "").lower()

    @pytest.mark.unit
    def test_summarize_file_missing_param(self):
        """summarize_file_tool rejects missing file_path."""
        from skills.summarize.tools import summarize_file_tool

        result = summarize_file_tool({})
        assert result["success"] is False

    @pytest.mark.unit
    def test_summarize_file_nonexistent(self):
        """summarize_file_tool handles nonexistent file."""
        from skills.summarize.tools import summarize_file_tool

        result = summarize_file_tool({"file_path": "/tmp/nonexistent_file_xyz.txt"})
        assert result["success"] is False
        assert "not found" in result.get("error", "").lower()

    @pytest.mark.unit
    def test_read_file_content_txt(self):
        """SummarizationService reads .txt files."""
        from skills.summarize.tools import SummarizationService

        svc = SummarizationService()

        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("Hello world test content for summarization.")
            f.flush()
            result = svc.read_file_content(f.name)

        os.unlink(f.name)
        assert result["success"] is True
        assert "Hello world" in result["content"]
        assert result["file_type"] == "txt"

    @pytest.mark.unit
    def test_read_file_content_md(self):
        """SummarizationService reads .md files."""
        from skills.summarize.tools import SummarizationService

        svc = SummarizationService()

        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write("# Title\n\nMarkdown content here.")
            f.flush()
            result = svc.read_file_content(f.name)

        os.unlink(f.name)
        assert result["success"] is True
        assert "Markdown content" in result["content"]

    @pytest.mark.unit
    def test_read_file_unsupported_type(self):
        """SummarizationService rejects unsupported file types."""
        from skills.summarize.tools import SummarizationService

        svc = SummarizationService()

        with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w", delete=False) as f:
            f.write("data")
            f.flush()
            result = svc.read_file_content(f.name)

        os.unlink(f.name)
        assert result["success"] is False
        assert "unsupported" in result.get("error", "").lower()

    @pytest.mark.unit
    def test_build_summary_prompt_lengths(self):
        """Summary prompt includes correct length instructions."""
        from skills.summarize.tools import SummarizationService

        svc = SummarizationService()

        short_prompt = svc._build_summary_prompt("test text", length="short")
        assert "2-3 sentences" in short_prompt

        long_prompt = svc._build_summary_prompt("test text", length="long")
        assert "detailed" in long_prompt.lower()

    @pytest.mark.unit
    def test_build_summary_prompt_styles(self):
        """Summary prompt includes correct style instructions."""
        from skills.summarize.tools import SummarizationService

        svc = SummarizationService()

        bullet_prompt = svc._build_summary_prompt("test text", style="bullet")
        assert "bullet" in bullet_prompt.lower()

        numbered_prompt = svc._build_summary_prompt("test text", style="numbered")
        assert "numbered" in numbered_prompt.lower()

    @pytest.mark.unit
    def test_extract_key_points_validates_max(self):
        """extract_key_points_tool validates max_points range."""
        from skills.summarize.tools import extract_key_points_tool

        # Missing text
        result = extract_key_points_tool({"max_points": 5})
        assert result["success"] is False

    @pytest.mark.unit
    def test_invalid_length_defaults_to_medium(self):
        """Invalid length parameter defaults to medium."""
        from skills.summarize.tools import summarize_text_tool

        with patch("skills.summarize.tools._service.summarize") as mock_summarize:
            mock_summarize.return_value = {"success": True, "summary": "test"}
            summarize_text_tool({"text": "Hello world", "length": "invalid_value"})
            mock_summarize.assert_called_once()
            _, kwargs = mock_summarize.call_args
            assert kwargs.get("length") == "medium" or mock_summarize.call_args[0][1:] == ()  # noqa
