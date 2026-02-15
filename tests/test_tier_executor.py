"""
TierExecutor Unit Tests
=======================

Tests TierExecutor initialization, LLMProvider, TierValidationResult,
ComplexityGate logic, tier routing, cost tracking, and helper methods.

Focuses on areas NOT covered by test_v3_execution.py:
- TierExecutor init and config wiring
- LLMProvider class: init, provider detection, generate method
- TierValidationResult dataclass
- ComplexityGate internal prompt and error handling
- Tier routing dispatch logic
- Cost tracking integration
- _FallbackValidator
- _parse_plan and _enrich_with_memory helpers
- _fallback_aggregate
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from Jotty.core.modes.execution.types import (
    ExecutionConfig,
    ExecutionTier,
    ExecutionResult,
    ExecutionPlan,
    ExecutionStep,
    TierValidationResult,
    MemoryContext,
)
from Jotty.core.modes.execution.executor import (
    TierExecutor,
    LLMProvider,
    ComplexityGate,
    _FallbackValidator,
)


# =============================================================================
# LLMProvider
# =============================================================================

@pytest.mark.unit
class TestLLMProvider:
    """Test LLMProvider initialization, provider detection, and generate."""

    def test_default_provider_is_anthropic(self):
        """LLMProvider defaults to anthropic provider and claude-sonnet model."""
        provider = LLMProvider()
        assert provider._provider_name == 'anthropic'
        assert 'claude' in provider._model
        assert provider._client is None

    def test_custom_provider_and_model(self):
        """LLMProvider stores custom provider and model names."""
        provider = LLMProvider(provider='openai', model='gpt-4o')
        assert provider._provider_name == 'openai'
        assert provider._model == 'gpt-4o'

    def test_none_provider_defaults_to_anthropic(self):
        """Passing provider=None defaults to 'anthropic'."""
        provider = LLMProvider(provider=None, model=None)
        assert provider._provider_name == 'anthropic'
        assert 'claude' in provider._model

    @pytest.mark.asyncio
    async def test_generate_anthropic_extracts_content(self):
        """generate() with anthropic provider extracts text blocks from response."""
        provider = LLMProvider(provider='anthropic', model='claude-sonnet-4')

        # Mock the anthropic client
        mock_block = Mock()
        mock_block.text = "Hello world"
        mock_response = Mock()
        mock_response.content = [mock_block]
        mock_response.usage = Mock(input_tokens=10, output_tokens=5)

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        result = await provider.generate("Say hello")
        assert result['content'] == "Hello world"
        assert result['usage']['input_tokens'] == 10
        assert result['usage']['output_tokens'] == 5

    @pytest.mark.asyncio
    async def test_generate_non_anthropic_fallback(self):
        """generate() with non-anthropic provider uses DSPy LM fallback."""
        provider = LLMProvider(provider='openai', model='gpt-4o')

        # Mock the client as a callable (DSPy LM style)
        mock_client = Mock(return_value=["DSPy response text"])
        provider._client = mock_client

        result = await provider.generate("Test prompt")
        assert result['content'] == "DSPy response text"
        assert result['usage']['input_tokens'] == 250
        assert result['usage']['output_tokens'] == 250


# =============================================================================
# TierValidationResult
# =============================================================================

@pytest.mark.unit
class TestTierValidationResult:
    """Test TierValidationResult dataclass fields and defaults."""

    def test_fields_assigned_correctly(self):
        """All fields are stored correctly on construction."""
        vr = TierValidationResult(
            success=True,
            confidence=0.95,
            feedback="Looks good",
            reasoning="Output matches expected criteria",
        )
        assert vr.success is True
        assert vr.confidence == 0.95
        assert vr.feedback == "Looks good"
        assert vr.reasoning == "Output matches expected criteria"
        assert isinstance(vr.timestamp, datetime)

    def test_failed_validation_result(self):
        """A failing TierValidationResult has success=False."""
        vr = TierValidationResult(
            success=False,
            confidence=0.3,
            feedback="Output is incorrect",
            reasoning="Missing key information",
        )
        assert vr.success is False
        assert vr.confidence == 0.3


# =============================================================================
# ComplexityGate
# =============================================================================

@pytest.mark.unit
class TestComplexityGateInternal:
    """Test ComplexityGate prompt format and error handling."""

    def test_prompt_contains_goal_placeholder(self):
        """ComplexityGate._PROMPT has {goal} placeholder."""
        assert '{goal}' in ComplexityGate._PROMPT

    def test_prompt_asks_for_direct_or_tools(self):
        """Prompt instructs the LLM to respond DIRECT or TOOLS."""
        assert 'DIRECT' in ComplexityGate._PROMPT
        assert 'TOOLS' in ComplexityGate._PROMPT

    @pytest.mark.asyncio
    async def test_should_skip_planning_returns_true_on_direct(self):
        """should_skip_planning returns True when LLM says DIRECT."""
        gate = ComplexityGate()
        mock_response = Mock()
        mock_response.content = [Mock(text="DIRECT")]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        gate._client = mock_client

        result = await gate.should_skip_planning("What is 2+2?")
        assert result is True

    @pytest.mark.asyncio
    async def test_should_skip_planning_returns_false_on_tools(self):
        """should_skip_planning returns False when LLM says TOOLS."""
        gate = ComplexityGate()
        mock_response = Mock()
        mock_response.content = [Mock(text="TOOLS")]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        gate._client = mock_client

        result = await gate.should_skip_planning("Research AI and build a report")
        assert result is False

    @pytest.mark.asyncio
    async def test_should_skip_planning_defaults_false_on_error(self):
        """should_skip_planning returns False (safe default) on exception."""
        gate = ComplexityGate()

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=RuntimeError("API unavailable")
        )
        gate._client = mock_client

        result = await gate.should_skip_planning("Any task")
        assert result is False


# =============================================================================
# TierExecutor Initialization and Config
# =============================================================================

@pytest.mark.unit
class TestTierExecutorInit:
    """Test TierExecutor initialization, config wiring, and lazy properties."""

    def test_default_config_when_none_provided(self):
        """TierExecutor creates default ExecutionConfig when none given."""
        executor = TierExecutor()
        assert executor.config is not None
        assert isinstance(executor.config, ExecutionConfig)
        assert executor.config.tier is None  # auto-detect

    def test_custom_config_stored(self):
        """TierExecutor stores provided ExecutionConfig."""
        cfg = ExecutionConfig(tier=ExecutionTier.LEARNING, temperature=0.5)
        executor = TierExecutor(config=cfg)
        assert executor.config.tier == ExecutionTier.LEARNING
        assert executor.config.temperature == 0.5

    def test_lazy_components_initially_none(self):
        """All lazy-loaded components start as None."""
        executor = TierExecutor()
        assert executor._provider is None
        assert executor._planner is None
        assert executor._memory is None
        assert executor._validator is None
        assert executor._complexity_gate is None
        assert executor._metrics is None
        assert executor._tracer is None
        assert executor._cost_tracker is None

    def test_injected_provider_used(self):
        """Injected provider is returned by the provider property."""
        mock_prov = Mock()
        executor = TierExecutor(provider=mock_prov)
        assert executor.provider is mock_prov

    def test_injected_registry_used(self):
        """Injected registry is returned by the registry property."""
        mock_reg = Mock()
        executor = TierExecutor(registry=mock_reg)
        assert executor.registry is mock_reg

    def test_complexity_gate_lazy_created(self):
        """complexity_gate property creates ComplexityGate on first access."""
        executor = TierExecutor()
        gate = executor.complexity_gate
        assert isinstance(gate, ComplexityGate)
        # Second access returns same instance
        assert executor.complexity_gate is gate


# =============================================================================
# Tier Routing Dispatch
# =============================================================================

@pytest.mark.unit
class TestTierRouting:
    """Test that execute() dispatches to the correct tier method."""

    @pytest.mark.asyncio
    async def test_direct_tier_routes_to_tier1(self, v3_executor):
        """ExecutionTier.DIRECT routes to _execute_tier1."""
        result = await v3_executor.execute(
            "Hello", config=ExecutionConfig(tier=ExecutionTier.DIRECT)
        )
        assert result.tier == ExecutionTier.DIRECT

    @pytest.mark.asyncio
    async def test_agentic_tier_routes_to_tier2(self, v3_executor):
        """ExecutionTier.AGENTIC routes to _execute_tier2."""
        result = await v3_executor.execute(
            "Plan and execute", config=ExecutionConfig(tier=ExecutionTier.AGENTIC)
        )
        assert result.tier == ExecutionTier.AGENTIC

    @pytest.mark.asyncio
    async def test_learning_tier_routes_to_tier3(self, v3_executor):
        """ExecutionTier.LEARNING routes to _execute_tier3."""
        result = await v3_executor.execute(
            "Learn and improve", config=ExecutionConfig(tier=ExecutionTier.LEARNING)
        )
        assert result.tier == ExecutionTier.LEARNING

    @pytest.mark.asyncio
    async def test_auto_detect_tier_when_none(self, v3_executor):
        """When tier is None, execute() auto-detects via TierDetector."""
        with patch.object(
            v3_executor._detector, 'adetect',
            new_callable=AsyncMock, return_value=ExecutionTier.DIRECT
        ):
            result = await v3_executor.execute(
                "What is 2+2?", config=ExecutionConfig(tier=None)
            )
            assert result.tier == ExecutionTier.DIRECT


# =============================================================================
# Cost Tracking
# =============================================================================

@pytest.mark.unit
class TestCostTracking:
    """Test cost tracking integration in TierExecutor."""

    @pytest.mark.asyncio
    async def test_tier1_cost_from_cost_tracker(self, v3_executor):
        """Tier 1 cost_usd comes from CostTracker, is positive."""
        result = await v3_executor.execute(
            "What is 2+2?",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )
        assert result.cost_usd > 0.0

    @pytest.mark.asyncio
    async def test_cost_tracker_lazy_loads(self):
        """cost_tracker property lazy-loads a CostTracker instance."""
        executor = TierExecutor()
        tracker = executor.cost_tracker
        assert tracker is not None
        # CostTracker should have record_llm_call method
        assert hasattr(tracker, 'record_llm_call')


# =============================================================================
# _FallbackValidator
# =============================================================================

@pytest.mark.unit
class TestFallbackValidator:
    """Test _FallbackValidator for when ValidatorAgent is unavailable."""

    @pytest.mark.asyncio
    async def test_validate_returns_parsed_json(self):
        """_FallbackValidator parses JSON from LLM response."""
        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(return_value={
            'content': '{"success": true, "confidence": 0.85, "feedback": "Good", "reasoning": "Correct"}',
        })
        validator = _FallbackValidator(mock_provider)
        result = await validator.validate("Is this correct?")
        assert result['success'] is True
        assert result['confidence'] == 0.85
        assert result['feedback'] == "Good"

    @pytest.mark.asyncio
    async def test_validate_handles_non_json_response(self):
        """_FallbackValidator returns default when LLM gives non-JSON."""
        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(return_value={
            'content': 'This looks correct to me.',
        })
        validator = _FallbackValidator(mock_provider)
        result = await validator.validate("Is this correct?")
        # Falls back to default values
        assert result['success'] is True
        assert result['confidence'] == 0.7

    @pytest.mark.asyncio
    async def test_validate_handles_provider_error(self):
        """_FallbackValidator returns safe defaults on provider exception."""
        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(
            side_effect=RuntimeError("LLM down")
        )
        validator = _FallbackValidator(mock_provider)
        result = await validator.validate("Validate this")
        assert result['success'] is True
        assert result['confidence'] == 0.5
        assert 'skipped' in result['feedback'].lower()


# =============================================================================
# Helper Methods: _parse_plan, _enrich_with_memory, _fallback_aggregate
# =============================================================================

@pytest.mark.unit
class TestHelperMethods:
    """Test TierExecutor helper methods for plan parsing, memory enrichment, aggregation."""

    def test_parse_plan_from_dict_steps(self):
        """_parse_plan converts dict steps into ExecutionPlan."""
        executor = TierExecutor()
        plan_result = {
            'steps': [
                {'description': 'Research topic', 'skill': 'web-search'},
                {'description': 'Write summary', 'skill': None},
            ],
            'estimated_cost': 0.05,
        }
        plan = executor._parse_plan("Test goal", plan_result)
        assert isinstance(plan, ExecutionPlan)
        assert plan.goal == "Test goal"
        assert plan.total_steps == 2
        assert plan.steps[0].description == 'Research topic'
        assert plan.steps[0].skill == 'web-search'
        assert plan.steps[1].step_num == 2
        assert plan.estimated_cost == 0.05

    def test_parse_plan_empty_steps(self):
        """_parse_plan handles empty step list gracefully."""
        executor = TierExecutor()
        plan = executor._parse_plan("Goal", {'steps': []})
        assert plan.total_steps == 0

    def test_enrich_with_memory_adds_context(self):
        """_enrich_with_memory appends memory entries to goal."""
        executor = TierExecutor()
        context = MemoryContext(
            entries=[
                {'summary': 'Past finding A'},
                {'summary': 'Past finding B'},
            ],
            relevance_scores=[0.9, 0.8],
            total_retrieved=2,
            retrieval_time_ms=5.0,
        )
        enriched = executor._enrich_with_memory("Analyze data", context)
        assert "Analyze data" in enriched
        assert "Relevant past experience" in enriched
        assert "Past finding A" in enriched
        assert "Past finding B" in enriched

    def test_enrich_with_memory_no_context(self):
        """_enrich_with_memory returns original goal when no context."""
        executor = TierExecutor()
        assert executor._enrich_with_memory("My goal", None) == "My goal"

    def test_fallback_aggregate_empty(self):
        """_fallback_aggregate with empty results returns message."""
        executor = TierExecutor()
        assert executor._fallback_aggregate([], "goal") == "No results generated."

    def test_fallback_aggregate_single(self):
        """_fallback_aggregate with single result returns its output."""
        executor = TierExecutor()
        result = executor._fallback_aggregate([{'output': 'Only one'}], "goal")
        assert result == 'Only one'

    def test_fallback_aggregate_multiple(self):
        """_fallback_aggregate with multiple results concatenates them."""
        executor = TierExecutor()
        results = [
            {'output': 'Result A'},
            {'output': 'Result B'},
        ]
        aggregated = executor._fallback_aggregate(results, "My goal")
        assert "My goal" in aggregated
        assert "Step 1" in aggregated
        assert "Result A" in aggregated
        assert "Step 2" in aggregated
        assert "Result B" in aggregated


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
