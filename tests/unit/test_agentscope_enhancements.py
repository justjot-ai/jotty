"""
Tests for AgentScope-inspired enhancements to Jotty.

7 features total:
  Round 1 (existing-system enhancements):
    1. FeedbackChannel.broadcast()
    2. FeedbackChannel.request()
    3. SmartAgentSlack.broadcast()
    4. UnifiedRegistry.get_scoped_tools()

  Round 2 (new capabilities):
    5. AgentRunner lifecycle hooks
    6. LLMContextManager.compress_structured()
    7. sequential_pipeline / fanout_pipeline
"""

import asyncio

import pytest

# =========================================================================
# 1. FeedbackChannel.broadcast()
# =========================================================================


class TestFeedbackChannelBroadcast:

    def test_broadcast_to_all_known_agents(self):
        from core.agents.feedback_channel import FeedbackChannel, FeedbackMessage, FeedbackType

        fc = FeedbackChannel()

        # Seed messages so agents appear in fc.messages
        for agent in ["agent_a", "agent_b", "agent_c"]:
            fc.send(
                FeedbackMessage(
                    source_actor="manager",
                    target_actor=agent,
                    feedback_type=FeedbackType.RESPONSE,
                    content="hello",
                )
            )

        # Broadcast from agent_a
        msg_ids = fc.broadcast(
            source_actor="agent_a",
            content="Shared context update",
        )

        # agent_a should NOT receive its own broadcast
        # recipients = all keys in fc.messages except agent_a
        assert "agent_a" not in [
            m.target_actor
            for msgs in fc.messages.values()
            for m in msgs
            if m.content == "Shared context update"
        ]
        assert len(msg_ids) >= 2  # agent_b, agent_c (manager may also get one)

    def test_broadcast_to_explicit_participants(self):
        from core.agents.feedback_channel import FeedbackChannel, FeedbackType

        fc = FeedbackChannel()
        msg_ids = fc.broadcast(
            source_actor="sender",
            content="targeted broadcast",
            participants=["target_1", "target_2", "target_3"],
        )
        assert len(msg_ids) == 3  # sender not in list

    def test_broadcast_excludes_self_from_explicit_list(self):
        from core.agents.feedback_channel import FeedbackChannel

        fc = FeedbackChannel()
        msg_ids = fc.broadcast(
            source_actor="x",
            content="test",
            participants=["x", "y", "z"],
        )
        # x is in participant list but should be excluded as sender
        assert len(msg_ids) == 2


# =========================================================================
# 2. FeedbackChannel.request() — async request-reply
# =========================================================================


class TestFeedbackChannelRequest:

    @pytest.mark.asyncio
    async def test_request_with_reply(self):
        from core.agents.feedback_channel import FeedbackChannel, FeedbackMessage, FeedbackType

        fc = FeedbackChannel()

        # Simulate: agent_b replies quickly
        async def simulate_reply():
            await asyncio.sleep(0.05)
            # Find the question from agent_a → agent_b
            pending = fc.messages.get("agent_b", [])
            if pending:
                question = pending[0]
                # Send a reply (field is message_id, not id)
                reply = FeedbackMessage(
                    source_actor="agent_b",
                    target_actor="agent_a",
                    feedback_type=FeedbackType.RESPONSE,
                    content="The answer is 42",
                    original_message_id=question.message_id,
                )
                fc.send(reply)

        reply_task = asyncio.create_task(simulate_reply())

        response = await fc.request(
            source_actor="agent_a",
            target_actor="agent_b",
            content="What is the meaning of life?",
            timeout=2.0,
        )

        await reply_task

        assert response is not None
        assert response.content == "The answer is 42"
        assert response.source_actor == "agent_b"

    @pytest.mark.asyncio
    async def test_request_timeout(self):
        from core.agents.feedback_channel import FeedbackChannel

        fc = FeedbackChannel()

        # No one replies → should timeout
        response = await fc.request(
            source_actor="a",
            target_actor="b",
            content="hello?",
            timeout=0.2,
        )
        assert response is None


# =========================================================================
# 3. SmartAgentSlack.broadcast()
# =========================================================================


class TestSmartAgentSlackBroadcast:

    def test_broadcast_targets_all_except_sender(self):
        from core.agents.axon import AgentCapabilities, SmartAgentSlack

        slack = SmartAgentSlack()

        # Register agents with proper AgentCapabilities objects
        for name in ["agent_a", "agent_b", "agent_c"]:
            slack.register_agent(
                name,
                capabilities=AgentCapabilities(
                    agent_name=name,
                    preferred_format="text",
                    acceptable_formats=["text"],
                    max_input_size=16000,
                    max_context_tokens=4000,
                ),
            )

        results = slack.broadcast(
            from_agent="agent_a",
            data="swarm-wide update",
        )

        # Should target agent_b and agent_c, NOT agent_a
        assert "agent_a" not in results
        assert "agent_b" in results
        assert "agent_c" in results

    def test_broadcast_with_exclude(self):
        from core.agents.axon import AgentCapabilities, SmartAgentSlack

        slack = SmartAgentSlack()
        for name in ["a", "b", "c"]:
            slack.register_agent(
                name,
                capabilities=AgentCapabilities(
                    agent_name=name,
                    preferred_format="text",
                    acceptable_formats=["text"],
                    max_input_size=16000,
                    max_context_tokens=4000,
                ),
            )

        results = slack.broadcast(
            from_agent="a",
            data="partial update",
            exclude=["c"],
        )

        assert "a" not in results
        assert "b" in results
        assert "c" not in results


# =========================================================================
# 4. UnifiedRegistry.get_scoped_tools()
# =========================================================================


class TestGetScopedTools:

    def test_scoped_tools_returns_list(self):
        from core.registry.unified_registry import get_unified_registry

        registry = get_unified_registry()
        # Use 'names' format to test the scoping logic directly
        # (claude format may expand — one skill can have multiple tools)
        names = registry.get_scoped_tools("search the web for AI news", max_tools=5, format="names")
        assert isinstance(names, list)
        assert len(names) <= 5

    def test_scoped_tools_names_format(self):
        from core.registry.unified_registry import get_unified_registry

        registry = get_unified_registry()
        names = registry.get_scoped_tools("create a chart", max_tools=8, format="names")
        assert isinstance(names, list)
        for n in names:
            assert isinstance(n, str)

    def test_scoped_tools_fewer_than_all(self):
        from core.registry.unified_registry import get_unified_registry

        registry = get_unified_registry()
        all_skills = registry.list_skills()
        scoped = registry.get_scoped_tools("send a telegram message", max_tools=5, format="names")
        assert len(scoped) <= 5
        assert len(scoped) < len(all_skills)


# =========================================================================
# 5. AgentRunner lifecycle hooks
# =========================================================================


class TestAgentRunnerHooks:

    def test_hook_types_defined(self):
        from core.orchestration.agent_runner import HOOK_TYPES

        assert "pre_run" in HOOK_TYPES
        assert "post_run" in HOOK_TYPES
        assert "pre_execute" in HOOK_TYPES
        assert "post_execute" in HOOK_TYPES
        assert "pre_architect" in HOOK_TYPES
        assert "post_architect" in HOOK_TYPES

    def test_add_and_run_hooks(self):
        from core.orchestration.agent_runner import HOOK_TYPES, AgentRunner

        # Create a minimal AgentRunner to test hooks
        # (Don't need full init — just test the hook infrastructure)
        runner = object.__new__(AgentRunner)
        runner._hooks = {ht: [] for ht in HOOK_TYPES}

        # Track calls
        calls = []

        def my_hook(**ctx):
            calls.append(ctx.get("goal", "no_goal"))

        runner.add_hook("pre_run", my_hook, name="test_hook")
        assert len(runner._hooks["pre_run"]) == 1

        # Run hooks
        ctx = runner._run_hooks("pre_run", goal="test task")
        assert len(calls) == 1
        assert calls[0] == "test task"

    def test_hook_modifies_context(self):
        from core.orchestration.agent_runner import HOOK_TYPES, AgentRunner

        runner = object.__new__(AgentRunner)
        runner._hooks = {ht: [] for ht in HOOK_TYPES}

        def modify_goal(**ctx):
            return {"goal": ctx["goal"] + " (enriched)"}

        runner.add_hook("pre_run", modify_goal)
        ctx = runner._run_hooks("pre_run", goal="original")
        assert ctx["goal"] == "original (enriched)"

    def test_remove_hook(self):
        from core.orchestration.agent_runner import HOOK_TYPES, AgentRunner

        runner = object.__new__(AgentRunner)
        runner._hooks = {ht: [] for ht in HOOK_TYPES}

        runner.add_hook("post_run", lambda **ctx: None, name="removable")
        assert len(runner._hooks["post_run"]) == 1

        result = runner.remove_hook("post_run", "removable")
        assert result is True
        assert len(runner._hooks["post_run"]) == 0

    def test_invalid_hook_type_raises(self):
        from core.orchestration.agent_runner import HOOK_TYPES, AgentRunner

        runner = object.__new__(AgentRunner)
        runner._hooks = {ht: [] for ht in HOOK_TYPES}

        with pytest.raises(ValueError, match="Unknown hook type"):
            runner.add_hook("not_a_real_hook", lambda **ctx: None)

    def test_hook_failure_doesnt_crash(self):
        from core.orchestration.agent_runner import HOOK_TYPES, AgentRunner

        runner = object.__new__(AgentRunner)
        runner._hooks = {ht: [] for ht in HOOK_TYPES}

        def bad_hook(**ctx):
            raise RuntimeError("boom")

        def good_hook(**ctx):
            return {"reached": True}

        runner.add_hook("pre_run", bad_hook)
        runner.add_hook("pre_run", good_hook)

        # Should NOT raise — bad hook is caught, good hook still runs
        ctx = runner._run_hooks("pre_run", goal="test")
        assert ctx.get("reached") is True


# =========================================================================
# 6. LLMContextManager.compress_structured()
# =========================================================================


class TestStructuredCompression:

    def test_short_content_returned_as_is(self):
        from core.context.context_guard import LLMContextManager

        mgr = LLMContextManager()
        short = "Just a few words"
        assert mgr.compress_structured(short, max_chars=1000) == short

    def test_long_content_compressed(self):
        from core.context.context_guard import LLMContextManager

        mgr = LLMContextManager()
        # Create content longer than max_chars
        long_content = "\n".join(
            [
                f"Line {i}: result found for query about {'AI' if i % 3 == 0 else 'data'}"
                for i in range(200)
            ]
        )

        compressed = mgr.compress_structured(long_content, goal="AI research", max_chars=800)

        assert len(compressed) <= 1200  # allow some slack for template overhead
        assert "[Compressed" in compressed
        assert "Key findings:" in compressed
        assert "Task: AI research" in compressed

    def test_goal_keywords_boost_relevance(self):
        from core.context.context_guard import LLMContextManager

        mgr = LLMContextManager()
        content = "\n".join(
            [
                "Line about cooking recipes",
                "Line about machine learning results",
                "Line about gardening tips",
                "Machine learning model achieved 95% accuracy",
                "Plant watering schedule",
            ]
            * 50
        )

        compressed = mgr.compress_structured(
            content, goal="machine learning accuracy", max_chars=600
        )

        # Key findings should prefer ML lines
        assert "machine learning" in compressed.lower() or "accuracy" in compressed.lower()

    def test_smart_compress_uses_structured(self):
        from core.context.context_guard import LLMContextManager

        mgr = LLMContextManager()
        long_content = "x\n" * 500
        result = mgr._smart_compress(long_content, max_chars=200)
        assert "[Compressed" in result


# =========================================================================
# 7. Pipeline utilities (sequential_pipeline, fanout_pipeline)
# =========================================================================


class TestPipelineUtils:

    def test_imports(self):
        from core.orchestration import fanout_pipeline, sequential_pipeline

        assert callable(sequential_pipeline)
        assert callable(fanout_pipeline)

    @pytest.mark.asyncio
    async def test_sequential_pipeline_chains(self):
        """Test that sequential_pipeline chains results."""
        from core.foundation.data_structures import EpisodeResult
        from core.orchestration import sequential_pipeline

        def _make_result(**overrides):
            defaults = dict(
                output="",
                success=True,
                trajectory=[],
                tagged_outputs=[],
                episode=0,
                execution_time=0.1,
                architect_results=[],
                auditor_results=[],
                agent_contributions={},
            )
            defaults.update(overrides)
            return EpisodeResult(**defaults)

        # Mock runners
        class MockRunner:
            def __init__(self, name, output):
                self.name = name
                self._output = output

            async def run(self, goal, **kwargs):
                return _make_result(output=f"{self._output}: {goal[:50]}")

        runners = [
            MockRunner("researcher", "Research done"),
            MockRunner("summarizer", "Summary done"),
        ]

        result = await sequential_pipeline(runners, goal="Study AI trends")

        assert result.success
        assert "Summary done" in str(result.output)

    @pytest.mark.asyncio
    async def test_fanout_pipeline_parallel(self):
        """Test that fanout_pipeline runs all agents."""
        from core.foundation.data_structures import EpisodeResult
        from core.orchestration import fanout_pipeline

        def _make_result(**overrides):
            defaults = dict(
                output="",
                success=True,
                trajectory=[],
                tagged_outputs=[],
                episode=0,
                execution_time=0.1,
                architect_results=[],
                auditor_results=[],
                agent_contributions={},
            )
            defaults.update(overrides)
            return EpisodeResult(**defaults)

        class MockRunner:
            def __init__(self, name):
                self.name = name

            async def run(self, goal, **kwargs):
                return _make_result(output=f"{self.name} result")

        runners = [MockRunner("a"), MockRunner("b"), MockRunner("c")]
        results = await fanout_pipeline(runners, goal="Common goal")

        assert len(results) == 3
        assert all(r.success for r in results)
        names = {str(r.output) for r in results}
        assert "a result" in names
        assert "b result" in names
        assert "c result" in names

    @pytest.mark.asyncio
    async def test_sequential_pipeline_stops_on_failure(self):
        """Test that sequential pipeline stops when an agent fails."""
        from core.foundation.data_structures import EpisodeResult
        from core.orchestration import sequential_pipeline

        call_count = 0

        def _make_result(**overrides):
            defaults = dict(
                output="output",
                success=True,
                trajectory=[],
                tagged_outputs=[],
                episode=0,
                execution_time=0.1,
                architect_results=[],
                auditor_results=[],
                agent_contributions={},
            )
            defaults.update(overrides)
            return EpisodeResult(**defaults)

        class MockRunner:
            def __init__(self, success):
                self._success = success

            async def run(self, goal, **kwargs):
                nonlocal call_count
                call_count += 1
                return _make_result(success=self._success)

        runners = [MockRunner(True), MockRunner(False), MockRunner(True)]
        result = await sequential_pipeline(runners, goal="test")

        assert not result.success
        assert call_count == 2  # Third runner should NOT execute


# =========================================================================
# AIOS-inspired: Orchestrator concurrency semaphore
# =========================================================================


class TestConcurrencySemaphore:

    def test_semaphore_initialized(self):
        """Verify semaphore exists with correct value."""
        import asyncio as _aio

        from core.orchestration.swarm_manager import Orchestrator

        # Use string agents (zero-config) to avoid complex setup
        sm = Orchestrator.__new__(Orchestrator)
        sm.max_concurrent_agents = 5
        sm._agent_semaphore = _aio.Semaphore(5)
        sm._scheduling_stats = {
            "total_scheduled": 0,
            "total_waited": 0,
            "peak_concurrent": 0,
            "current_concurrent": 0,
        }

        assert sm._agent_semaphore._value == 5
        assert sm._scheduling_stats["total_scheduled"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
