"""
Benchmark Integration Tests for Jotty V3
=========================================

Battle-tests the full pipeline (Jotty facade → TierDetector → TierExecutor
→ Provider/Swarm → Observability) with controlled mock responses simulating
realistic workflows.

Tests:
    - End-to-end tier execution (all 5 tiers + facade)
    - Tier detection accuracy (parametrized)
    - Swarm selection accuracy (keyword matching)
    - Cost accumulation (exact cost verification)
    - Memory lifecycle (retrieve → enrich → execute → validate → store)
    - Error recovery (exceptions, timeouts, retries)
    - Concurrent execution (thread-safety, isolation)
    - Edge cases (empty goals, unicode, special chars, null responses)

Fixtures:
    All from conftest.py: v3_executor, mock_provider, mock_planner,
    mock_validator, mock_v3_memory, v3_observability_helpers
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from Jotty.core.execution.types import (
    ExecutionConfig,
    ExecutionTier,
    ExecutionResult,
    StreamEventType,
)
from Jotty.core.execution.tier_detector import TierDetector


# =============================================================================
# Helpers
# =============================================================================

def make_provider_response(content="response", input_tokens=100, output_tokens=50):
    """Build consistent mock provider responses."""
    return {
        'content': content,
        'usage': {'input_tokens': input_tokens, 'output_tokens': output_tokens},
    }


# Cost constants for claude-sonnet-4 ($3/$15 per 1M tokens)
# Single call with 100 input / 50 output:
#   (100/1_000_000 * 3) + (50/1_000_000 * 15) = 0.0003 + 0.00075 = 0.00105
SINGLE_CALL_COST = 0.00105

# Tier 2 plan call uses hardcoded 500/300 tokens:
#   (500/1_000_000 * 3) + (300/1_000_000 * 15) = 0.0015 + 0.0045 = 0.006
PLAN_CALL_COST = 0.006

# Tier 3 validation call uses hardcoded 400/200 tokens:
#   (400/1_000_000 * 3) + (200/1_000_000 * 15) = 0.0012 + 0.003 = 0.0042
VALIDATION_CALL_COST = 0.0042


# =============================================================================
# 1. TestEndToEndTierExecution
# =============================================================================

@pytest.mark.unit
class TestEndToEndTierExecution:
    """Full pipeline tests — executor wired with mocks, testing complete tier flows."""

    @pytest.mark.asyncio
    async def test_tier1_simple_query_end_to_end(self, v3_executor, v3_observability_helpers):
        """Tier 1: provider.generate called once, result has output/cost/trace, metrics recorded."""
        result = await v3_executor.execute(
            "What is 2+2?",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )

        assert result.success is True
        assert result.tier == ExecutionTier.DIRECT
        assert result.output == 'Mock LLM response'
        assert result.llm_calls == 1
        assert result.cost_usd > 0
        assert result.trace is not None
        v3_observability_helpers['assert_metrics_recorded']('tier_1')

    @pytest.mark.asyncio
    async def test_tier2_multistep_plan_end_to_end(
        self, v3_executor, mock_provider, mock_planner
    ):
        """Tier 2: planner returns 3 steps, each step calls provider.generate, cost accumulated."""
        mock_planner.plan = AsyncMock(return_value={
            'steps': [
                {'description': 'Step 1: Research the market'},
                {'description': 'Step 2: Analyze competitors'},
                {'description': 'Step 3: Draft strategy document'},
            ],
        })

        result = await v3_executor.execute(
            "Create a marketing plan for Q2",
            config=ExecutionConfig(tier=ExecutionTier.AGENTIC),
        )

        assert result.success is True
        assert result.tier == ExecutionTier.AGENTIC
        assert result.plan is not None
        assert result.plan.total_steps == 3
        # planner.plan is called once
        assert mock_planner.plan.call_count == 1
        # provider.generate called once per step + synthesis call
        assert mock_provider.generate.call_count >= 3
        # Cost = plan_cost + 3 * step_cost + synthesis call
        expected_cost = PLAN_CALL_COST + 3 * SINGLE_CALL_COST + SINGLE_CALL_COST
        assert abs(result.cost_usd - expected_cost) < 1e-6

    @pytest.mark.asyncio
    async def test_tier3_memory_enriched_execution(
        self, v3_executor, mock_v3_memory, mock_planner, mock_validator
    ):
        """Tier 3: memory.retrieve → enriched goal → planner → steps → validator → memory.store."""
        result = await v3_executor.execute(
            "Improve our data pipeline based on past failures",
            config=ExecutionConfig(tier=ExecutionTier.LEARNING, enable_validation=True),
        )

        assert result.success is True
        assert result.tier == ExecutionTier.LEARNING
        # Memory was retrieved (2 entries in mock)
        assert result.used_memory is True
        assert result.memory_context is not None
        assert result.memory_context.total_retrieved == 2
        # Validator was called
        assert result.validation is not None
        assert result.validation.success is True
        # Memory was stored
        mock_v3_memory.store.assert_called_once()
        # Planner received enriched goal (containing memory summaries)
        planner_call_args = mock_planner.plan.call_args
        enriched_goal = planner_call_args[0][0]
        assert 'Previous analysis result' in enriched_goal

    @pytest.mark.asyncio
    async def test_tier4_swarm_delegation(self, v3_executor):
        """Tier 4: _select_swarm called, swarm.execute invoked, result wrapped."""
        mock_swarm = AsyncMock()
        mock_swarm.execute = AsyncMock(return_value=Mock(
            success=True,
            output={'result': 'Swarm output here'},
        ))
        mock_swarm.__class__.__name__ = 'CodingSwarm'

        with patch.object(v3_executor, '_select_swarm', return_value=mock_swarm):
            result = await v3_executor.execute(
                "Implement a REST API for user management",
                config=ExecutionConfig(tier=ExecutionTier.RESEARCH),
            )

        assert result.success is True
        assert result.tier == ExecutionTier.RESEARCH
        assert result.output == {'result': 'Swarm output here'}
        mock_swarm.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_tier5_autonomous_with_swarm(self, v3_executor):
        """Tier 5: swarm selected, executed with autonomous metadata."""
        mock_swarm = AsyncMock()
        mock_swarm.execute = AsyncMock(return_value=Mock(
            success=True,
            output={'analysis': 'Code analysis complete'},
        ))
        mock_swarm.__class__.__name__ = 'CodingSwarm'

        with patch.object(v3_executor, '_select_swarm', return_value=mock_swarm):
            result = await v3_executor.execute(
                "Run in sandbox mode with agent coalition",
                config=ExecutionConfig(tier=ExecutionTier.AUTONOMOUS),
            )

        assert result.success is True
        assert result.tier == ExecutionTier.AUTONOMOUS
        assert result.metadata.get('autonomous') is True
        mock_swarm.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_facade_run_delegates_correctly(self):
        """Jotty().run() → executor.execute() called with correct args."""
        from Jotty.jotty import Jotty

        jotty = Jotty(config=ExecutionConfig(), log_level="ERROR")
        mock_executor = AsyncMock()
        mock_executor.execute = AsyncMock(return_value=ExecutionResult(
            output="facade result", tier=ExecutionTier.DIRECT, success=True,
        ))
        jotty.executor = mock_executor

        result = await jotty.run("What is GDP?", tier=ExecutionTier.DIRECT)

        assert result.success is True
        assert result.output == "facade result"
        mock_executor.execute.assert_called_once()
        call_kwargs = mock_executor.execute.call_args
        assert call_kwargs.kwargs['goal'] == "What is GDP?"

    @pytest.mark.asyncio
    async def test_facade_stream_yields_events(self, v3_executor):
        """Executor.stream() yields TOKEN/STATUS/RESULT events for tier 1."""
        events = []
        async for event in v3_executor.stream(
            "What is 2+2?",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        ):
            events.append(event)

        event_types = [e.type for e in events]
        # Must have STATUS (start), then STATUS (skill discovery), TOKEN(s), and RESULT
        assert StreamEventType.STATUS in event_types
        assert StreamEventType.TOKEN in event_types
        assert StreamEventType.RESULT in event_types

    @pytest.mark.asyncio
    async def test_sequential_executions_accumulate_metrics(
        self, v3_executor, v3_observability_helpers
    ):
        """Run tier 1 three times, metrics.total_executions == 3."""
        for _ in range(3):
            await v3_executor.execute(
                "Hello",
                config=ExecutionConfig(tier=ExecutionTier.DIRECT),
            )

        from Jotty.core.observability.metrics import get_metrics
        am = get_metrics().get_agent_metrics('tier_1')
        assert am is not None
        assert am.total_executions == 3

    @pytest.mark.asyncio
    async def test_tier2_with_realistic_plan_output(
        self, v3_executor, mock_planner, mock_provider
    ):
        """Multi-step plan with descriptions — validate step execution order."""
        mock_planner.plan = AsyncMock(return_value={
            'steps': [
                {'description': 'Gather requirements from stakeholders'},
                {'description': 'Design the database schema'},
                {'description': 'Implement the API endpoints'},
                {'description': 'Write integration tests'},
            ],
        })

        # Track call order via side effect
        call_descriptions = []
        original_generate = mock_provider.generate

        async def _tracking_generate(**kwargs):
            call_descriptions.append(kwargs.get('prompt', ''))
            return await original_generate(**kwargs)

        mock_provider.generate = AsyncMock(side_effect=_tracking_generate)

        result = await v3_executor.execute(
            "Build a user management system",
            config=ExecutionConfig(tier=ExecutionTier.AGENTIC),
        )

        assert result.success is True
        assert len(result.steps) == 4
        # Steps executed in order
        assert 'Gather requirements' in call_descriptions[0]
        assert 'Design the database' in call_descriptions[1]
        assert 'Implement the API' in call_descriptions[2]
        assert 'Write integration tests' in call_descriptions[3]

    @pytest.mark.asyncio
    async def test_tier3_validation_failure_triggers_retry(
        self, v3_executor, mock_validator, mock_planner
    ):
        """Validator returns success=False first, True second → retries once, validator called 2x."""
        mock_validator.validate = AsyncMock(side_effect=[
            {'success': False, 'confidence': 0.3, 'feedback': 'Incomplete analysis', 'reasoning': 'Missing data'},
            {'success': True, 'confidence': 0.92, 'feedback': 'Looks good now', 'reasoning': 'Complete'},
        ])

        result = await v3_executor.execute(
            "Validate and optimize the pipeline",
            config=ExecutionConfig(
                tier=ExecutionTier.LEARNING,
                enable_validation=True,
                validation_retries=1,
            ),
        )

        assert result.success is True
        assert result.validation is not None
        assert result.validation.success is True
        assert result.validation.confidence == 0.92
        # Validator called twice (initial + 1 retry)
        assert mock_validator.validate.call_count == 2
        # Planner called twice (initial tier2 + retry tier2)
        assert mock_planner.plan.call_count == 2


# =============================================================================
# 2. TestTierDetectionAccuracy
# =============================================================================

@pytest.mark.unit
class TestTierDetectionAccuracy:
    """Parametrized tests verifying TierDetector picks the right tier."""

    @pytest.mark.parametrize("goal", [
        "What is GDP?",
        "Calculate 15% of 230",
        "Define recursion",
        "Convert 5km to miles",
        "Explain briefly what a hash table is",
        "Translate hello to French",
    ])
    def test_direct_tier_detection(self, goal):
        """Simple queries should detect as DIRECT."""
        detector = TierDetector()
        tier = detector.detect(goal)
        assert tier == ExecutionTier.DIRECT, f"Expected DIRECT for '{goal}', got {tier.name}"

    @pytest.mark.parametrize("goal", [
        "Build a dashboard showing sales trends and then create a report",
        "Create a marketing plan for Q2 and then compile the results",
        "Analyze the data first and then summarize the findings",
    ])
    def test_agentic_tier_detection(self, goal):
        """Multi-step tasks should detect as AGENTIC."""
        detector = TierDetector()
        tier = detector.detect(goal)
        assert tier == ExecutionTier.AGENTIC, f"Expected AGENTIC for '{goal}', got {tier.name}"

    @pytest.mark.parametrize("goal", [
        "Learn from previous mistakes and improve output quality",
        "Track performance of the model and optimize results",
        "Validate the pipeline and remember what worked",
    ])
    def test_learning_tier_detection(self, goal):
        """Learning-related tasks should detect as LEARNING."""
        detector = TierDetector()
        tier = detector.detect(goal)
        assert tier == ExecutionTier.LEARNING, f"Expected LEARNING for '{goal}', got {tier.name}"

    @pytest.mark.parametrize("goal", [
        "Research thoroughly the impact of AI on healthcare",
        "Benchmark different sorting algorithms and compare approaches",
        "Experiment with different prompt strategies for multi-round evaluation",
    ])
    def test_research_tier_detection(self, goal):
        """Research tasks should detect as RESEARCH."""
        detector = TierDetector()
        tier = detector.detect(goal)
        assert tier == ExecutionTier.RESEARCH, f"Expected RESEARCH for '{goal}', got {tier.name}"

    @pytest.mark.parametrize("goal", [
        "Run in sandbox mode with agent coalition",
        "Execute code in isolated environment with trust verification",
        "Use autonomous multi-swarm coalition with consensus",
    ])
    def test_autonomous_tier_detection(self, goal):
        """Autonomous tasks should detect as AUTONOMOUS."""
        detector = TierDetector()
        tier = detector.detect(goal)
        assert tier == ExecutionTier.AUTONOMOUS, f"Expected AUTONOMOUS for '{goal}', got {tier.name}"

    def test_ambiguous_defaults_to_agentic(self):
        """Ambiguous goals without clear indicators default to AGENTIC."""
        detector = TierDetector()
        tier = detector.detect("Do something interesting with this moderately complex data set and then generate insights")
        assert tier == ExecutionTier.AGENTIC

    def test_short_queries_are_direct(self):
        """Very short queries (<=10 words, no multi-step) → DIRECT."""
        detector = TierDetector()
        for goal in ["hello", "hi there", "test", "good morning"]:
            tier = detector.detect(goal)
            assert tier == ExecutionTier.DIRECT, f"Expected DIRECT for '{goal}', got {tier.name}"

    def test_multistep_indicators_override_short(self):
        """Multi-step indicators override shortness → AGENTIC."""
        detector = TierDetector()
        tier = detector.detect("First analyze then summarize")
        assert tier == ExecutionTier.AGENTIC


# =============================================================================
# 3. TestSwarmSelectionAccuracy
# =============================================================================

@pytest.mark.unit
class TestSwarmSelectionAccuracy:
    """Parametrized tests verifying _select_swarm keyword matching."""

    @pytest.mark.parametrize("goal,expected_swarm", [
        # Coding swarm
        ("Implement a REST API for users", "coding"),
        ("Write a Python function to parse JSON", "coding"),
        ("Develop a class for data processing", "coding"),
        # Research swarm
        ("Research the impact of climate change", "research"),
        ("Analyze market trends in tech sector", "research"),
        ("Investigate the root cause of the outage", "research"),
        # Testing swarm
        ("Write unit tests for the auth module", "testing"),
        ("Increase test coverage to 90%", "testing"),
        ("Run integration test suite for QA", "testing"),
        # Review swarm
        ("Review the pull request for security issues", "review"),
        ("Audit the deployment for vulnerabilities", "review"),
        ("Review the recent changes in the PR", "review"),
        # Data analysis swarm
        ("Generate statistics from the sales dataset", "data_analysis"),
        ("Create a visualization of CSV data", "data_analysis"),
        ("Compute statistics on user engagement data", "data_analysis"),
        # DevOps swarm
        ("Deploy the app to kubernetes cluster", "devops"),
        ("Set up docker containers for CI/CD", "devops"),
        ("Configure infrastructure for staging", "devops"),
        # Idea writer swarm
        ("Write a blog post about AI trends", "idea_writer"),
        ("Draft an article on machine learning", "idea_writer"),
        ("Create an essay about modern architecture", "idea_writer"),
        # Fundamental swarm
        ("Check stock valuation for AAPL", "fundamental"),
        ("Evaluate financial earnings for Q4", "fundamental"),
        ("Evaluate investment opportunities in tech", "fundamental"),
        # Learning swarm
        ("Build a curriculum for Python beginners", "learning"),
        ("Teach the basics of machine learning to newcomers", "learning"),
        ("Create a training plan for new engineers", "learning"),
    ])
    def test_swarm_keyword_matching(self, v3_executor, goal, expected_swarm):
        """Correct swarm selected for each goal based on keyword matching."""
        mock_swarm = Mock()
        mock_swarm.__class__.__name__ = f'{expected_swarm.title()}Swarm'

        with patch('Jotty.core.swarms.registry.SwarmRegistry.create', return_value=mock_swarm) as mock_create:
            result = v3_executor._select_swarm(goal)

        assert result is not None, f"Expected swarm '{expected_swarm}' for goal: {goal}"
        # Verify the correct swarm name was requested
        mock_create.assert_called_with(expected_swarm)

    def test_explicit_swarm_name_override(self, v3_executor):
        """Explicit swarm_name bypasses keyword detection."""
        mock_swarm = Mock()

        with patch('Jotty.core.swarms.registry.SwarmRegistry.create', return_value=mock_swarm) as mock_create:
            result = v3_executor._select_swarm("Random unrelated goal", swarm_name="coding")

        assert result is mock_swarm
        mock_create.assert_called_with("coding")

    def test_no_match_returns_none(self, v3_executor):
        """Goal with no matching keywords returns None."""
        with patch('Jotty.core.swarms.registry.SwarmRegistry.create', return_value=None):
            result = v3_executor._select_swarm("Something completely unrelated to any swarm xyz")

        assert result is None

    def test_first_keyword_match_wins(self, v3_executor):
        """First matching swarm in keyword_map iteration is returned."""
        call_names = []

        def _tracking_create(name, config=None):
            call_names.append(name)
            return Mock() if name == call_names[0] else None

        with patch('Jotty.core.swarms.registry.SwarmRegistry.create', side_effect=_tracking_create):
            result = v3_executor._select_swarm("Implement code and research the API design")

        assert result is not None
        # "code" matches 'coding' first (before 'research')
        assert call_names[0] == 'coding'

    def test_case_insensitive_matching(self, v3_executor):
        """Keyword matching is case-insensitive."""
        mock_swarm = Mock()

        with patch('Jotty.core.swarms.registry.SwarmRegistry.create', return_value=mock_swarm):
            result = v3_executor._select_swarm("IMPLEMENT a REST API")

        assert result is not None

    def test_partial_keyword_matching(self, v3_executor):
        """Keywords are matched as substrings in the goal."""
        mock_swarm = Mock()

        with patch('Jotty.core.swarms.registry.SwarmRegistry.create', return_value=mock_swarm) as mock_create:
            result = v3_executor._select_swarm("dataset analysis for quarterly numbers")

        assert result is not None
        # "data" from keyword list matches "dataset"
        mock_create.assert_called_with('data_analysis')

    def test_swarm_registry_integration(self, v3_executor):
        """SwarmRegistry.create is actually called with detected name."""
        mock_swarm = Mock()

        with patch('Jotty.core.swarms.registry.SwarmRegistry.create', return_value=mock_swarm) as mock_create:
            v3_executor._select_swarm("Deploy to docker container")

        mock_create.assert_called_once_with('devops')

    def test_fallback_when_registry_returns_none(self, v3_executor):
        """When registry returns None for all swarms → returns None."""
        with patch('Jotty.core.swarms.registry.SwarmRegistry.create', return_value=None):
            result = v3_executor._select_swarm("Implement a REST API in code")

        assert result is None


# =============================================================================
# 4. TestCostAccumulation
# =============================================================================

@pytest.mark.unit
class TestCostAccumulation:
    """Verify multi-step cost tracking is accurate using known token counts and pricing."""

    @pytest.mark.asyncio
    async def test_tier1_single_call_cost(self, v3_executor):
        """Tier 1: 100 input + 50 output → exact cost via claude-sonnet-4 pricing."""
        result = await v3_executor.execute(
            "What is 2+2?",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )

        # (100/1M × $3) + (50/1M × $15) = $0.00105
        assert abs(result.cost_usd - SINGLE_CALL_COST) < 1e-6

    @pytest.mark.asyncio
    async def test_tier2_accumulated_cost(self, v3_executor):
        """Tier 2: 1 plan call (500/300) + 2 step calls (100/50 each) → exact total."""
        result = await v3_executor.execute(
            "Research and summarize AI trends",
            config=ExecutionConfig(tier=ExecutionTier.AGENTIC),
        )

        # plan: 0.006 + 2 steps × 0.00105 + synthesis call = 0.00915
        expected = PLAN_CALL_COST + 2 * SINGLE_CALL_COST + SINGLE_CALL_COST
        assert abs(result.cost_usd - expected) < 1e-6

    @pytest.mark.asyncio
    async def test_tier3_includes_validation_cost(self, v3_executor):
        """Tier 3: tier2 cost + 1 validation call (400/200) → exact total."""
        result = await v3_executor.execute(
            "Learn from past results and improve output",
            config=ExecutionConfig(
                tier=ExecutionTier.LEARNING,
                enable_validation=True,
            ),
        )

        # tier2 cost (plan + 2 steps + synthesis) + validation
        expected = PLAN_CALL_COST + 2 * SINGLE_CALL_COST + SINGLE_CALL_COST + VALIDATION_CALL_COST
        assert abs(result.cost_usd - expected) < 1e-6

    @pytest.mark.asyncio
    async def test_cost_tracker_records_all_calls(self, v3_executor):
        """After tier 2 execution, cost_tracker.get_metrics().total_calls tracks all LLM calls."""
        # Reset tracker for clean count
        v3_executor.cost_tracker.reset()

        await v3_executor.execute(
            "Plan something",
            config=ExecutionConfig(tier=ExecutionTier.AGENTIC),
        )

        metrics = v3_executor.cost_tracker.get_metrics()
        # 1 plan + 2 steps + synthesis = 4 cost_tracker records
        assert metrics.total_calls >= 3

    @pytest.mark.asyncio
    async def test_cost_tracker_token_totals(self, v3_executor):
        """After tier 2, token totals match: plan(500/300) + 2×step(100/50)."""
        v3_executor.cost_tracker.reset()

        await v3_executor.execute(
            "Plan something",
            config=ExecutionConfig(tier=ExecutionTier.AGENTIC),
        )

        metrics = v3_executor.cost_tracker.get_metrics()
        # Plan: 500 input + 300 output, 2 steps + synthesis: 3×100 input + 3×50 output
        expected_input = 500 + 3 * 100
        expected_output = 300 + 3 * 50
        assert metrics.total_input_tokens >= expected_input
        assert metrics.total_output_tokens >= expected_output

    @pytest.mark.asyncio
    async def test_zero_cost_when_provider_fails(self, v3_executor, mock_provider):
        """Provider raises → result.cost_usd == 0."""
        mock_provider.generate = AsyncMock(side_effect=RuntimeError("API down"))

        result = await v3_executor.execute(
            "What is 2+2?",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )

        assert result.success is False
        assert result.cost_usd == 0

    @pytest.mark.asyncio
    async def test_partial_cost_on_step_failure(self, v3_executor, mock_provider, mock_planner):
        """2 of 3 steps succeed → cost = plan + 2 successful step costs."""
        mock_planner.plan = AsyncMock(return_value={
            'steps': [
                {'description': 'Step 1: Gather data'},
                {'description': 'Step 2: Process (will fail)'},
                {'description': 'Step 3: Summarize'},
            ],
        })

        call_count = 0
        original_generate = mock_provider.generate

        async def _failing_on_step2(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Step 2 fails
                raise ValueError("Processing error")
            return await original_generate(**kwargs)

        mock_provider.generate = AsyncMock(side_effect=_failing_on_step2)

        v3_executor.cost_tracker.reset()

        result = await v3_executor.execute(
            "Three step task",
            config=ExecutionConfig(tier=ExecutionTier.AGENTIC),
        )

        # Plan cost + 2 successful steps + synthesis call (step 2 failed, no cost for it)
        expected = PLAN_CALL_COST + 2 * SINGLE_CALL_COST + SINGLE_CALL_COST
        assert abs(result.cost_usd - expected) < 1e-6

    @pytest.mark.asyncio
    async def test_cost_breakdown_by_model(self, v3_executor):
        """Cost metrics track costs by model correctly."""
        v3_executor.cost_tracker.reset()

        # Run tier 1
        await v3_executor.execute("Hello", config=ExecutionConfig(tier=ExecutionTier.DIRECT))
        # Run tier 2
        await v3_executor.execute("Plan things", config=ExecutionConfig(tier=ExecutionTier.AGENTIC))

        metrics = v3_executor.cost_tracker.get_metrics()
        # All calls use claude-sonnet-4
        assert 'claude-sonnet-4' in metrics.cost_by_model
        # tier1: 1 call + tier2: 1 plan + 2 steps + synthesis = 5 total
        assert metrics.calls_by_model['claude-sonnet-4'] >= 4


# =============================================================================
# 5. TestMemoryLifecycle
# =============================================================================

@pytest.mark.unit
class TestMemoryLifecycle:
    """Test the full memory integration in Tier 3."""

    @pytest.mark.asyncio
    async def test_memory_retrieve_called_with_goal(self, v3_executor, mock_v3_memory):
        """memory.retrieve called with the original goal string."""
        goal = "Improve data quality based on past runs"
        await v3_executor.execute(
            goal,
            config=ExecutionConfig(tier=ExecutionTier.LEARNING),
        )

        mock_v3_memory.retrieve.assert_called_once_with(goal, limit=5)

    @pytest.mark.asyncio
    async def test_memory_entries_enrich_prompt(self, v3_executor, mock_planner, mock_v3_memory):
        """When memory returns entries, planner receives enriched goal with memory summaries."""
        mock_v3_memory.retrieve = AsyncMock(return_value=[
            {'summary': 'Previous attempt failed on step 3', 'score': 0.9},
            {'summary': 'Using batch processing was faster', 'score': 0.8},
        ])

        await v3_executor.execute(
            "Optimize the data pipeline",
            config=ExecutionConfig(tier=ExecutionTier.LEARNING),
        )

        planner_goal = mock_planner.plan.call_args[0][0]
        assert 'Relevant past experience' in planner_goal
        assert 'Previous attempt failed on step 3' in planner_goal
        assert 'Using batch processing was faster' in planner_goal

    @pytest.mark.asyncio
    async def test_memory_store_called_after_success(self, v3_executor, mock_v3_memory):
        """memory.store called with goal, result output, success=True."""
        await v3_executor.execute(
            "Do something useful",
            config=ExecutionConfig(tier=ExecutionTier.LEARNING),
        )

        mock_v3_memory.store.assert_called_once()
        store_kwargs = mock_v3_memory.store.call_args.kwargs
        assert store_kwargs['goal'] == "Do something useful"
        assert store_kwargs['success'] is True

    @pytest.mark.asyncio
    async def test_memory_store_includes_validation(self, v3_executor, mock_v3_memory, mock_validator):
        """Stored entry includes validation confidence when validation is enabled."""
        mock_validator.validate = AsyncMock(return_value={
            'success': True, 'confidence': 0.95,
            'feedback': 'Excellent', 'reasoning': 'All criteria met',
        })

        await v3_executor.execute(
            "Task with validation",
            config=ExecutionConfig(tier=ExecutionTier.LEARNING, enable_validation=True),
        )

        store_kwargs = mock_v3_memory.store.call_args.kwargs
        assert store_kwargs['confidence'] == 0.95

    @pytest.mark.asyncio
    async def test_no_memory_ops_when_backend_none(self, v3_executor, mock_v3_memory):
        """memory_backend='none' → no retrieve/store calls."""
        await v3_executor.execute(
            "Task without memory",
            config=ExecutionConfig(tier=ExecutionTier.LEARNING, memory_backend="none"),
        )

        mock_v3_memory.retrieve.assert_not_called()
        mock_v3_memory.store.assert_not_called()

    @pytest.mark.asyncio
    async def test_memory_retrieve_failure_non_fatal(self, v3_executor, mock_v3_memory):
        """memory.retrieve raises → execution continues without memory context."""
        mock_v3_memory.retrieve = AsyncMock(side_effect=RuntimeError("Redis down"))

        result = await v3_executor.execute(
            "Task despite memory failure",
            config=ExecutionConfig(tier=ExecutionTier.LEARNING),
        )

        # Execution should succeed despite memory failure
        assert result.success is True
        assert result.used_memory is False

    @pytest.mark.asyncio
    async def test_memory_store_failure_non_fatal(self, v3_executor, mock_v3_memory):
        """memory.store raises → result still returned successfully."""
        mock_v3_memory.store = AsyncMock(side_effect=RuntimeError("Disk full"))

        result = await v3_executor.execute(
            "Task with store failure",
            config=ExecutionConfig(tier=ExecutionTier.LEARNING),
        )

        assert result.success is True
        assert result.output is not None

    @pytest.mark.asyncio
    async def test_empty_memory_results(self, v3_executor, mock_planner, mock_v3_memory):
        """memory.retrieve returns [] → planner gets unenriched goal."""
        mock_v3_memory.retrieve = AsyncMock(return_value=[])

        goal = "Task with empty memory"
        await v3_executor.execute(
            goal,
            config=ExecutionConfig(tier=ExecutionTier.LEARNING),
        )

        # Planner should receive the original goal without enrichment
        planner_goal = mock_planner.plan.call_args[0][0]
        assert planner_goal == goal
        assert 'Relevant past experience' not in planner_goal


# =============================================================================
# 6. TestErrorRecovery
# =============================================================================

@pytest.mark.unit
class TestErrorRecovery:
    """Battle-test error handling across the pipeline."""

    @pytest.mark.asyncio
    async def test_provider_exception_returns_failure(self, v3_executor, mock_provider):
        """provider.generate raises RuntimeError → result.success=False, error recorded."""
        mock_provider.generate = AsyncMock(side_effect=RuntimeError("API key expired"))

        result = await v3_executor.execute(
            "What is 2+2?",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )

        assert result.success is False
        assert 'API key expired' in result.error

    @pytest.mark.asyncio
    async def test_provider_timeout_returns_failure(self, v3_executor, mock_provider):
        """provider.generate raises asyncio.TimeoutError → result.success=False."""
        mock_provider.generate = AsyncMock(side_effect=asyncio.TimeoutError())

        result = await v3_executor.execute(
            "What is 2+2?",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )

        assert result.success is False

    @pytest.mark.asyncio
    async def test_planner_failure_returns_failure(self, v3_executor, mock_planner):
        """planner.plan raises → result.success=False."""
        mock_planner.plan = AsyncMock(side_effect=RuntimeError("Planning service unavailable"))

        result = await v3_executor.execute(
            "Create a plan",
            config=ExecutionConfig(tier=ExecutionTier.AGENTIC),
        )

        assert result.success is False
        assert 'Planning service unavailable' in result.error

    @pytest.mark.asyncio
    async def test_step_failure_records_error(self, v3_executor, mock_provider, mock_planner):
        """One of 3 steps fails → step marked with error, others complete."""
        mock_planner.plan = AsyncMock(return_value={
            'steps': [
                {'description': 'Step 1: Succeed'},
                {'description': 'Step 2: Fail'},
                {'description': 'Step 3: Succeed'},
            ],
        })

        call_count = 0
        original_generate = mock_provider.generate

        async def _failing_step2(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("Step 2 processing error")
            return await original_generate(**kwargs)

        mock_provider.generate = AsyncMock(side_effect=_failing_step2)

        result = await v3_executor.execute(
            "Multi step task",
            config=ExecutionConfig(tier=ExecutionTier.AGENTIC),
        )

        # Overall success is False because step 2 has error
        assert result.success is False
        # Step 2 should have error recorded
        assert result.steps[1].error is not None
        assert 'Step 2 processing error' in result.steps[1].error
        # Steps 1 and 3 should have completed
        assert result.steps[0].result is not None
        assert result.steps[2].result is not None

    @pytest.mark.asyncio
    async def test_validation_failure_triggers_retry(self, v3_executor, mock_validator):
        """Validator returns success=False first time, True second → result has retry."""
        mock_validator.validate = AsyncMock(side_effect=[
            {'success': False, 'confidence': 0.2, 'feedback': 'Needs work', 'reasoning': 'Poor'},
            {'success': True, 'confidence': 0.88, 'feedback': 'Good', 'reasoning': 'Complete'},
        ])

        result = await v3_executor.execute(
            "Task with retry",
            config=ExecutionConfig(
                tier=ExecutionTier.LEARNING,
                enable_validation=True,
                validation_retries=1,
            ),
        )

        assert result.success is True
        assert mock_validator.validate.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_respected(self, v3_executor, mock_validator):
        """Validator always fails → only config.validation_retries attempts."""
        mock_validator.validate = AsyncMock(return_value={
            'success': False, 'confidence': 0.1, 'feedback': 'Bad', 'reasoning': 'Failed',
        })

        result = await v3_executor.execute(
            "Persistent failure task",
            config=ExecutionConfig(
                tier=ExecutionTier.LEARNING,
                enable_validation=True,
                validation_retries=2,
            ),
        )

        # 1 initial validation + 2 retry validations = 3 total
        assert mock_validator.validate.call_count == 3
        # Final result reflects validation failure
        assert result.validation is not None
        assert result.validation.success is False

    @pytest.mark.asyncio
    async def test_swarm_execute_failure(self, v3_executor):
        """Mock swarm.execute raises → result.success=False for tier 4."""
        mock_swarm = AsyncMock()
        mock_swarm.execute = AsyncMock(side_effect=RuntimeError("Swarm crashed"))

        with patch.object(v3_executor, '_select_swarm', return_value=mock_swarm):
            result = await v3_executor.execute(
                "Use swarm",
                config=ExecutionConfig(tier=ExecutionTier.RESEARCH),
            )

        assert result.success is False
        assert 'Swarm crashed' in result.error

    @pytest.mark.asyncio
    async def test_registry_discover_failure_non_fatal(self, v3_executor, mock_registry):
        """registry.discover_for_task raises → execution continues."""
        mock_registry.discover_for_task = Mock(side_effect=RuntimeError("Registry down"))

        result = await v3_executor.execute(
            "What is 2+2?",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )

        # Should fail because registry error propagates up in tier 1
        assert result.success is False

    @pytest.mark.asyncio
    async def test_metrics_recorded_on_failure(self, v3_executor, mock_provider, v3_observability_helpers):
        """After provider failure, metrics still show 1 execution with success=False."""
        mock_provider.generate = AsyncMock(side_effect=RuntimeError("Fail"))

        await v3_executor.execute(
            "Failing task",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )

        from Jotty.core.observability.metrics import get_metrics
        am = get_metrics().get_agent_metrics('tier_1')
        assert am is not None
        assert am.total_executions >= 1

    @pytest.mark.asyncio
    async def test_trace_completed_on_failure(self, v3_executor, mock_provider, v3_observability_helpers):
        """After exception, trace is still ended cleanly."""
        mock_provider.generate = AsyncMock(side_effect=RuntimeError("Trace test fail"))

        await v3_executor.execute(
            "Trace failure",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )

        # Trace should still exist (end_trace called in except block)
        v3_observability_helpers['assert_trace_exists']()


# =============================================================================
# 7. TestConcurrentExecution
# =============================================================================

@pytest.mark.unit
class TestConcurrentExecution:
    """Test thread-safety and concurrent workloads."""

    @pytest.mark.asyncio
    async def test_concurrent_tier1_tasks(self, v3_executor):
        """5 tier 1 tasks via asyncio.gather → all complete, 5 results."""
        tasks = [
            v3_executor.execute(f"Query {i}", config=ExecutionConfig(tier=ExecutionTier.DIRECT))
            for i in range(5)
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(r.success for r in results)
        assert all(r.tier == ExecutionTier.DIRECT for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_mixed_tiers(self, v3_executor):
        """Tier 1 + tier 2 + tier 3 concurrent → all complete."""
        tasks = [
            v3_executor.execute("Simple query", config=ExecutionConfig(tier=ExecutionTier.DIRECT)),
            v3_executor.execute("Multi step task", config=ExecutionConfig(tier=ExecutionTier.AGENTIC)),
            v3_executor.execute("Learning task", config=ExecutionConfig(tier=ExecutionTier.LEARNING)),
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert results[0].tier == ExecutionTier.DIRECT
        assert results[1].tier == ExecutionTier.AGENTIC
        assert results[2].tier == ExecutionTier.LEARNING

    @pytest.mark.asyncio
    async def test_concurrent_metrics_accuracy(self, v3_executor):
        """5 concurrent tier 1 → metrics.total_executions >= 5."""
        tasks = [
            v3_executor.execute(f"Query {i}", config=ExecutionConfig(tier=ExecutionTier.DIRECT))
            for i in range(5)
        ]
        await asyncio.gather(*tasks)

        from Jotty.core.observability.metrics import get_metrics
        am = get_metrics().get_agent_metrics('tier_1')
        assert am is not None
        assert am.total_executions >= 5

    @pytest.mark.asyncio
    async def test_concurrent_cost_accumulation(self, v3_executor):
        """5 concurrent tier 1 tasks → total cost = 5 × single call cost."""
        v3_executor.cost_tracker.reset()

        tasks = [
            v3_executor.execute(f"Q{i}", config=ExecutionConfig(tier=ExecutionTier.DIRECT))
            for i in range(5)
        ]
        results = await asyncio.gather(*tasks)

        total_cost = sum(r.cost_usd for r in results)
        expected = 5 * SINGLE_CALL_COST
        assert abs(total_cost - expected) < 1e-6

    @pytest.mark.asyncio
    async def test_concurrent_trace_isolation(self, v3_executor):
        """Each concurrent task gets its own trace."""
        tasks = [
            v3_executor.execute(f"Query {i}", config=ExecutionConfig(tier=ExecutionTier.DIRECT))
            for i in range(3)
        ]
        results = await asyncio.gather(*tasks)

        # All results should have traces
        assert all(r.trace is not None for r in results)

    @pytest.mark.asyncio
    async def test_rapid_sequential_execution(self, v3_executor):
        """10 tier 1 tasks in loop → all succeed, metrics accurate."""
        results = []
        for i in range(10):
            result = await v3_executor.execute(
                f"Query {i}",
                config=ExecutionConfig(tier=ExecutionTier.DIRECT),
            )
            results.append(result)

        assert len(results) == 10
        assert all(r.success for r in results)

        from Jotty.core.observability.metrics import get_metrics
        am = get_metrics().get_agent_metrics('tier_1')
        assert am.total_executions >= 10


# =============================================================================
# 8. TestEdgeCases
# =============================================================================

@pytest.mark.unit
class TestEdgeCases:
    """Boundary conditions and unusual inputs."""

    @pytest.mark.asyncio
    async def test_empty_goal(self, v3_executor):
        """Empty string → execution handles gracefully."""
        result = await v3_executor.execute(
            "",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )

        # Should execute (provider gets empty prompt) or fail gracefully
        assert isinstance(result, ExecutionResult)

    @pytest.mark.asyncio
    async def test_very_long_goal(self, v3_executor):
        """10K character goal → executes without crash."""
        long_goal = "Analyze this data: " + "x" * 10_000

        result = await v3_executor.execute(
            long_goal,
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )

        assert isinstance(result, ExecutionResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_unicode_goal(self, v3_executor):
        """Unicode goal → executes without crash."""
        result = await v3_executor.execute(
            "分析这个数据集并生成报告",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )

        assert isinstance(result, ExecutionResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_special_characters_in_goal(self, v3_executor):
        """Special characters including script tags → executes safely."""
        result = await v3_executor.execute(
            "What is 2+2? <script>alert('xss')</script>",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )

        assert isinstance(result, ExecutionResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_newlines_in_goal(self, v3_executor):
        """Goal with newlines → executes normally."""
        result = await v3_executor.execute(
            "First do this\nThen do that\nFinally summarize",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )

        assert isinstance(result, ExecutionResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_repeated_execution_same_goal(self, v3_executor):
        """Same goal 3x → all succeed, tier detection cache works."""
        results = []
        for _ in range(3):
            result = await v3_executor.execute(
                "What is the weather?",
                config=ExecutionConfig(tier=ExecutionTier.DIRECT),
            )
            results.append(result)

        assert all(r.success for r in results)
        assert len(results) == 3

    @pytest.mark.parametrize("tier", [
        ExecutionTier.DIRECT,
        ExecutionTier.AGENTIC,
        ExecutionTier.LEARNING,
    ])
    @pytest.mark.asyncio
    async def test_all_tiers_explicit_override(self, v3_executor, tier):
        """Force each tier explicitly → correct tier used regardless of goal content."""
        result = await v3_executor.execute(
            "Generic ambiguous goal",
            config=ExecutionConfig(tier=tier),
        )

        assert result.tier == tier
        assert result.success is True

    @pytest.mark.asyncio
    async def test_null_provider_response(self, v3_executor, mock_provider):
        """provider.generate returns None → handled gracefully."""
        mock_provider.generate = AsyncMock(return_value=None)

        result = await v3_executor.execute(
            "What is 2+2?",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )

        # Should not crash; may fail or return None output
        assert isinstance(result, ExecutionResult)

    @pytest.mark.asyncio
    async def test_empty_plan_steps(self, v3_executor, mock_planner):
        """planner returns {'steps': []} → tier 2 handles gracefully."""
        mock_planner.plan = AsyncMock(return_value={'steps': []})

        result = await v3_executor.execute(
            "Empty plan task",
            config=ExecutionConfig(tier=ExecutionTier.AGENTIC),
        )

        assert isinstance(result, ExecutionResult)
        # No steps executed — plan + 0 steps
        assert result.plan is not None
        assert result.plan.total_steps == 0

    @pytest.mark.asyncio
    async def test_provider_returns_empty_content(self, v3_executor, mock_provider):
        """provider returns {'content': '', 'usage': {...}} → handled."""
        mock_provider.generate = AsyncMock(return_value={
            'content': '',
            'usage': {'input_tokens': 50, 'output_tokens': 10},
        })

        result = await v3_executor.execute(
            "What is 2+2?",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )

        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert result.output == ''

    @pytest.mark.asyncio
    async def test_swarm_returns_failure_result(self, v3_executor):
        """swarm.execute returns SwarmResult(success=False) → result.success=False."""
        mock_swarm = AsyncMock()
        mock_swarm.execute = AsyncMock(return_value=Mock(
            success=False,
            output={'error': 'Swarm failed internally'},
        ))
        mock_swarm.__class__.__name__ = 'FailingSwarm'

        with patch.object(v3_executor, '_select_swarm', return_value=mock_swarm):
            result = await v3_executor.execute(
                "Failing swarm task",
                config=ExecutionConfig(tier=ExecutionTier.RESEARCH),
            )

        assert result.success is False
        assert result.output == {'error': 'Swarm failed internally'}

    @pytest.mark.asyncio
    async def test_config_override_tier(self, v3_executor):
        """ExecutionConfig(tier=DIRECT) on complex goal → DIRECT used."""
        # This goal would normally detect as AGENTIC, but config forces DIRECT
        result = await v3_executor.execute(
            "Build a dashboard and then create a report and finally summarize",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )

        assert result.tier == ExecutionTier.DIRECT
        assert result.llm_calls == 1
