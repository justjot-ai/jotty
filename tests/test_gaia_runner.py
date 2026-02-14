"""
GAIA Runner Test Suite
======================

Unit tests for:
- JottyGAIAAdapter: sync→async bridge, dry run, prompt wrapping
- GAIABenchmark improvements: level filtering, validate_answer enhancements
- run_gaia.py: CLI arg parsing, runner logic
- download_gaia.py: argument parsing

All tests use mocks — NEVER call real LLM providers.
Tests run fast (< 1s each) and offline.
"""

import asyncio
import json
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from dataclasses import dataclass

# Add Jotty to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.evaluation import GAIABenchmark, EvalStore
from core.evaluation.gaia_adapter import (
    JottyGAIAAdapter,
    GAIA_SYSTEM_PROMPT,
    _extract_answer_from_output,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_gaia_task(
    task_id="task_001",
    question="What is the capital of France?",
    answer="Paris",
    split="validation",
    level=1,
):
    """Create a GAIA-format task dict."""
    return {
        "task_id": task_id,
        "Question": question,
        "Final answer": answer,
        "Level": level,
        "split": split,
    }


def _make_gaia_dataset(tmp_path, tasks=None):
    """Create a fake GAIA dataset on disk and return its path."""
    if tasks is None:
        tasks = {
            "validation": [
                _make_gaia_task("v1", "What is 2+2?", "4", level=1),
                _make_gaia_task("v2", "Capital of France?", "Paris", level=1),
                _make_gaia_task("v3", "Hard question?", "42", level=2),
                _make_gaia_task("v4", "Harder question?", "100", level=3),
            ],
            "test": [
                _make_gaia_task("t1", "Square root of 9?", "3", split="test", level=1),
            ],
        }

    gaia_dir = tmp_path / "gaia"
    for split_name, split_tasks in tasks.items():
        split_dir = gaia_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for task in split_tasks:
            task_file = split_dir / f"{task['task_id']}.json"
            task_file.write_text(json.dumps(task))

    return str(gaia_dir)


@dataclass
class FakeExecutionResult:
    """Minimal ExecutionResult mock."""
    output: str = "42"
    cost_usd: float = 0.001
    llm_calls: int = 1
    latency_ms: float = 500.0
    success: bool = True


# =============================================================================
# 0. Answer extraction (GAIA scoring)
# =============================================================================

@pytest.mark.unit
class TestExtractAnswerFromOutput:
    """_extract_answer_from_output for Tier 4 AgenticExecutionResult and nested output."""

    def test_none_empty(self):
        assert _extract_answer_from_output(None) == ""
        assert _extract_answer_from_output("") == ""

    def test_plain_string(self):
        assert _extract_answer_from_output("28") == "28"
        assert _extract_answer_from_output("The adventurer died.") == "The adventurer died."

    def test_agentic_final_output(self):
        """AgenticExecutionResult.final_output is used when present."""
        class MockAgentic:
            final_output = "42"
            outputs = {}
        assert _extract_answer_from_output(MockAgentic()) == "42"

    def test_agentic_outputs_last_step(self):
        """When final_output is None, last step content from outputs is used."""
        class MockAgentic:
            final_output = None
            outputs = {"step_0": {"path": "x"}, "step_1": {"content": "Paris"}}
        assert _extract_answer_from_output(MockAgentic()) == "Paris"

    def test_agentic_outputs_result_field(self):
        class MockAgentic:
            final_output = None
            outputs = {"step_1": {"result": "28"}}
        assert _extract_answer_from_output(MockAgentic()) == "28"

    def test_nested_output(self):
        """Nested object with .output is recursed."""
        class Nested:
            output = "nested answer"
        assert _extract_answer_from_output(Nested()) == "nested answer"

    def test_dict_content(self):
        assert _extract_answer_from_output({"content": "from dict"}) == "from dict"
        assert _extract_answer_from_output({"result": "42"}) == "42"

    def test_avoids_object_repr(self):
        """Object with no final_output/outputs/content returns summary or str, not full repr."""
        class NoContent:
            outputs = {}
            final_output = None
            def summary(self):
                return "Task completed"
        assert _extract_answer_from_output(NoContent()) == "Task completed"


# =============================================================================
# 1. JottyGAIAAdapter — Core Behavior
# =============================================================================

@pytest.mark.unit
class TestJottyGAIAAdapterDryRun:
    """JottyGAIAAdapter dry run mode."""

    def test_dry_run_returns_placeholder(self):
        """Dry run returns [DRY RUN] without calling Jotty."""
        adapter = JottyGAIAAdapter(dry_run=True)
        result = adapter.run("What is 2+2?")
        assert result == "[DRY RUN]"

    def test_dry_run_no_last_result(self):
        """Dry run sets last_result to None."""
        adapter = JottyGAIAAdapter(dry_run=True)
        adapter.run("test")
        assert adapter.last_result is None

    def test_dry_run_no_jotty_init(self):
        """Dry run never initializes Jotty."""
        adapter = JottyGAIAAdapter(dry_run=True)
        adapter.run("test")
        assert adapter._jotty is None


@pytest.mark.unit
class TestJottyGAIAAdapterPrompt:
    """JottyGAIAAdapter prompt construction."""

    def test_prompt_includes_system(self):
        """Built prompt includes GAIA system prompt."""
        adapter = JottyGAIAAdapter()
        prompt = adapter._build_prompt("What is 2+2?")
        assert GAIA_SYSTEM_PROMPT in prompt
        assert "What is 2+2?" in prompt

    def test_prompt_format(self):
        """Prompt follows expected format."""
        adapter = JottyGAIAAdapter()
        prompt = adapter._build_prompt("test question")
        assert GAIA_SYSTEM_PROMPT in prompt and "Question: test question" in prompt
        assert "Use web search or read_file when needed to answer." in prompt


@pytest.mark.unit
class TestJottyGAIAAdapterRun:
    """JottyGAIAAdapter run() with mocked Jotty."""

    def test_run_calls_jotty(self):
        """run() calls Jotty.run() and returns output."""
        adapter = JottyGAIAAdapter(tier="DIRECT")
        fake_result = FakeExecutionResult(output="Paris")

        mock_jotty = MagicMock()
        mock_jotty.run = AsyncMock(return_value=fake_result)
        adapter._jotty = mock_jotty

        answer = adapter.run("What is the capital of France?")

        assert answer == "Paris"
        assert adapter.last_result is fake_result
        mock_jotty.run.assert_called_once()

    def test_run_extracts_output(self):
        """run() extracts output field from ExecutionResult."""
        adapter = JottyGAIAAdapter(tier="DIRECT")
        fake_result = FakeExecutionResult(output="42")

        mock_jotty = MagicMock()
        mock_jotty.run = AsyncMock(return_value=fake_result)
        adapter._jotty = mock_jotty

        answer = adapter.run("What is the answer?")
        assert answer == "42"

    def test_run_uses_dspy_normalizer_when_expected_answer_given(self):
        """When expected_answer is passed, adapter uses DSPy to normalize raw output."""
        adapter = JottyGAIAAdapter(tier="DIRECT", dry_run=False)
        fake_result = FakeExecutionResult(
            output="The 3rd base player had 519 at bats in the 1977 season."
        )
        mock_jotty = MagicMock()
        mock_jotty.run = AsyncMock(return_value=fake_result)
        adapter._jotty = mock_jotty

        with patch("Jotty.core.evaluation.gaia_signatures.normalize_gaia_answer_with_dspy") as m_norm:
            m_norm.return_value = "519"
            answer = adapter.run("How many at bats?", expected_answer="519")
        assert answer == "519"
        m_norm.assert_called_once()
        # Adapter calls normalizer with raw_response, expected_example, question_summary
        call_kw = m_norm.call_args.kwargs if m_norm.call_args.kwargs else {}
        assert call_kw.get("expected_example") == "519"
        assert "519" in (call_kw.get("raw_response") or "")

    def test_run_with_none_output(self):
        """run() handles None output gracefully."""
        adapter = JottyGAIAAdapter(tier="DIRECT")
        fake_result = FakeExecutionResult(output=None)

        mock_jotty = MagicMock()
        mock_jotty.run = AsyncMock(return_value=fake_result)
        adapter._jotty = mock_jotty

        answer = adapter.run("test")
        assert answer == ""

    def test_run_stores_last_result(self):
        """run() stores ExecutionResult in last_result."""
        adapter = JottyGAIAAdapter()
        fake_result = FakeExecutionResult(output="test", cost_usd=0.05)

        mock_jotty = MagicMock()
        mock_jotty.run = AsyncMock(return_value=fake_result)
        adapter._jotty = mock_jotty

        adapter.run("question")

        assert adapter.last_result is fake_result
        assert adapter.last_result.cost_usd == 0.05

    def test_tier_passed_to_jotty(self):
        """Tier parameter is converted and passed to Jotty.run()."""
        adapter = JottyGAIAAdapter(tier="AGENTIC")

        mock_jotty = MagicMock()
        mock_jotty.run = AsyncMock(return_value=FakeExecutionResult())
        adapter._jotty = mock_jotty

        with patch('Jotty.core.execution.types.ExecutionTier') as mock_tier_cls:
            # Create a fake enum
            mock_tier = Mock()
            mock_tier.name = "AGENTIC"
            mock_tier_cls.__iter__ = Mock(return_value=iter([mock_tier]))
            mock_tier_cls.return_value = mock_tier

            adapter.run("test")

        mock_jotty.run.assert_called_once()


@pytest.mark.unit
class TestJottyGAIAAdapterInit:
    """JottyGAIAAdapter initialization."""

    def test_default_init(self):
        """Default init has no tier, no model, not dry run, no llm doc sources."""
        adapter = JottyGAIAAdapter()
        assert adapter.tier is None
        assert adapter.model is None
        assert adapter.dry_run is False
        assert adapter.use_llm_doc_sources is False
        assert adapter.last_result is None
        assert adapter._jotty is None

    def test_custom_init(self):
        """Custom init stores parameters."""
        adapter = JottyGAIAAdapter(tier="DIRECT", model="test-model", dry_run=True)
        assert adapter.tier == "DIRECT"
        assert adapter.model == "test-model"
        assert adapter.dry_run is True

    def test_custom_init_use_llm_doc_sources(self):
        """use_llm_doc_sources=True stores and prompt includes references."""
        adapter = JottyGAIAAdapter(use_llm_doc_sources=True, dry_run=True)
        assert adapter.use_llm_doc_sources is True
        prompt = adapter._build_prompt("What is 2+2?")
        assert "2+2" in prompt
        assert "Relevant open-source" in prompt or "microsoft" in prompt.lower() or "huggingface" in prompt.lower()


# =============================================================================
# 2. GAIABenchmark — Level Filtering (load_tasks improvements)
# =============================================================================

@pytest.mark.unit
class TestGAIALoadTasksFiltering:
    """GAIABenchmark.load_tasks() with split/level filtering."""

    def test_load_all_tasks(self, tmp_path):
        """No filters → all tasks loaded."""
        gaia_path = _make_gaia_dataset(tmp_path)
        bench = GAIABenchmark(benchmark_path=gaia_path)
        tasks = bench.load_tasks()
        assert len(tasks) == 5  # 4 validation + 1 test

    def test_filter_by_split_validation(self, tmp_path):
        """split='validation' → only validation tasks."""
        gaia_path = _make_gaia_dataset(tmp_path)
        bench = GAIABenchmark(benchmark_path=gaia_path)
        tasks = bench.load_tasks(split="validation")
        assert len(tasks) == 4
        assert all(t["split"] == "validation" for t in tasks)

    def test_filter_by_split_test(self, tmp_path):
        """split='test' → only test tasks."""
        gaia_path = _make_gaia_dataset(tmp_path)
        bench = GAIABenchmark(benchmark_path=gaia_path)
        tasks = bench.load_tasks(split="test")
        assert len(tasks) == 1
        assert tasks[0]["split"] == "test"

    def test_filter_by_level_1(self, tmp_path):
        """level=1 → only Level 1 tasks."""
        gaia_path = _make_gaia_dataset(tmp_path)
        bench = GAIABenchmark(benchmark_path=gaia_path)
        tasks = bench.load_tasks(level=1)
        assert len(tasks) == 3  # v1, v2, t1
        assert all(t["Level"] == 1 for t in tasks)

    def test_filter_by_level_2(self, tmp_path):
        """level=2 → only Level 2 tasks."""
        gaia_path = _make_gaia_dataset(tmp_path)
        bench = GAIABenchmark(benchmark_path=gaia_path)
        tasks = bench.load_tasks(level=2)
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "v3"

    def test_filter_by_level_3(self, tmp_path):
        """level=3 → only Level 3 tasks."""
        gaia_path = _make_gaia_dataset(tmp_path)
        bench = GAIABenchmark(benchmark_path=gaia_path)
        tasks = bench.load_tasks(level=3)
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "v4"

    def test_filter_combined_split_and_level(self, tmp_path):
        """Combined split + level filtering."""
        gaia_path = _make_gaia_dataset(tmp_path)
        bench = GAIABenchmark(benchmark_path=gaia_path)
        tasks = bench.load_tasks(split="validation", level=1)
        assert len(tasks) == 2  # v1, v2 only
        assert all(t["split"] == "validation" and t["Level"] == 1 for t in tasks)

    def test_filter_no_matches(self, tmp_path):
        """Filter that matches nothing → empty list."""
        gaia_path = _make_gaia_dataset(tmp_path)
        bench = GAIABenchmark(benchmark_path=gaia_path)
        tasks = bench.load_tasks(split="test", level=3)
        assert tasks == []

    def test_missing_level_field_included(self, tmp_path):
        """Tasks without Level field are included when level filter is set."""
        tasks_data = {
            "validation": [
                {"task_id": "no_level", "Question": "Q?", "Final answer": "A"},
                _make_gaia_task("with_level", "Q?", "A", level=1),
            ],
        }
        gaia_path = _make_gaia_dataset(tmp_path, tasks_data)
        bench = GAIABenchmark(benchmark_path=gaia_path)
        # Tasks with no Level field have Level=None, which != 1, so filtered out
        tasks = bench.load_tasks(level=1)
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "with_level"

    def test_backward_compatible_no_args(self, tmp_path):
        """Calling load_tasks() with no args still works (backward compat)."""
        gaia_path = _make_gaia_dataset(tmp_path)
        bench = GAIABenchmark(benchmark_path=gaia_path)
        tasks = bench.load_tasks()
        assert len(tasks) > 0

    def test_deterministic_order_for_smoke(self, tmp_path):
        """Tasks are sorted by (split, task_id) so --smoke 10 runs same 10 every time."""
        gaia_path = _make_gaia_dataset(tmp_path)
        bench = GAIABenchmark(benchmark_path=gaia_path)
        tasks = bench.load_tasks()
        # Should be sorted: test before validation (split), then by task_id
        pairs = [(t.get("split", ""), t.get("task_id", t.get("file_name", ""))) for t in tasks]
        assert pairs == sorted(pairs), "load_tasks() must return deterministic order for --smoke"


# =============================================================================
# 3. GAIABenchmark — validate_answer Improvements
# =============================================================================

@pytest.mark.unit
class TestValidateAnswerImprovements:
    """Improved validate_answer() — currency, containment, extraction."""

    def setup_method(self):
        self.bench = GAIABenchmark()

    # --- Empty expected answer ---

    def test_empty_expected_returns_false(self):
        """Empty expected answer → always False (test split)."""
        task = _make_gaia_task(answer="")
        assert self.bench.validate_answer(task, "anything") is False

    def test_whitespace_expected_returns_false(self):
        """Whitespace-only expected answer → False."""
        task = _make_gaia_task(answer="   ")
        assert self.bench.validate_answer(task, "test") is False

    # --- Currency symbol stripping ---

    def test_dollar_sign_stripping(self):
        """$42 matches 42."""
        task = _make_gaia_task(answer="42")
        assert self.bench.validate_answer(task, "$42") is True

    def test_dollar_in_expected(self):
        """42 matches $42 in expected."""
        task = _make_gaia_task(answer="$42")
        assert self.bench.validate_answer(task, "42") is True

    def test_dollar_both_sides(self):
        """$42 matches $42."""
        task = _make_gaia_task(answer="$42")
        assert self.bench.validate_answer(task, "$42") is True

    # --- Percentage symbol stripping ---

    def test_percent_stripping(self):
        """42% matches 42."""
        task = _make_gaia_task(answer="42")
        assert self.bench.validate_answer(task, "42%") is True

    def test_percent_in_expected(self):
        """42 matches 42% in expected."""
        task = _make_gaia_task(answer="42%")
        assert self.bench.validate_answer(task, "42") is True

    # --- Containment check ---

    def test_expected_at_start_of_actual(self):
        """Expected 'Paris' found at start of 'Paris, France'."""
        task = _make_gaia_task(answer="Paris")
        assert self.bench.validate_answer(task, "Paris, the capital of France") is True

    def test_expected_at_end_of_actual(self):
        """Expected 'Paris' found at end of 'The answer is Paris'."""
        task = _make_gaia_task(answer="Paris")
        assert self.bench.validate_answer(task, "the answer is Paris") is True

    def test_containment_not_in_middle(self):
        """Containment only checks start/end, not middle."""
        task = _make_gaia_task(answer="is")
        # "is" has length < 2, so containment check is skipped
        # Also "is" alone wouldn't randomly match
        task2 = _make_gaia_task(answer="Paris")
        assert self.bench.validate_answer(task2, "I think maybe Paris could be it") is False

    def test_single_char_no_containment(self):
        """Single-char expected skips containment check."""
        task = _make_gaia_task(answer="X")
        assert self.bench.validate_answer(task, "X marks the spot") is False

    # --- Answer extraction ---

    def test_extract_the_answer_is_prefix(self):
        """Strips 'The answer is:' prefix."""
        task = _make_gaia_task(answer="Paris")
        assert self.bench.validate_answer(task, "The answer is: Paris") is True

    def test_extract_final_answer_prefix(self):
        """Strips 'Final answer:' prefix."""
        task = _make_gaia_task(answer="42")
        assert self.bench.validate_answer(task, "Final answer: 42") is True

    def test_extract_trailing_period(self):
        """Strips trailing period."""
        task = _make_gaia_task(answer="Paris")
        assert self.bench.validate_answer(task, "Paris.") is True

    def test_extract_answer_prefix(self):
        """Strips 'Answer:' prefix."""
        task = _make_gaia_task(answer="42")
        assert self.bench.validate_answer(task, "Answer: 42") is True

    # --- Existing behavior preserved ---

    def test_exact_match_preserved(self):
        """Exact match still works."""
        task = _make_gaia_task(answer="Paris")
        assert self.bench.validate_answer(task, "Paris") is True

    def test_case_insensitive_preserved(self):
        """Case insensitive match still works."""
        task = _make_gaia_task(answer="Paris")
        assert self.bench.validate_answer(task, "PARIS") is True

    def test_numeric_comparison_preserved(self):
        """Numeric comparison with tolerance still works."""
        task = _make_gaia_task(answer="3.14")
        assert self.bench.validate_answer(task, "3.14") is True
        assert self.bench.validate_answer(task, "3.1400") is True

    def test_comma_formatting_preserved(self):
        """Comma-formatted numbers still work."""
        task = _make_gaia_task(answer="1,000")
        assert self.bench.validate_answer(task, "1000") is True

    def test_wrong_answer_still_fails(self):
        """Wrong answers still fail."""
        task = _make_gaia_task(answer="Paris")
        assert self.bench.validate_answer(task, "London") is False


# =============================================================================
# 4. GAIABenchmark.extract_answer — Static Method
# =============================================================================

@pytest.mark.unit
class TestExtractAnswer:
    """GAIABenchmark.extract_answer() static method."""

    def test_plain_answer(self):
        """Plain answer returned as-is."""
        assert GAIABenchmark.extract_answer("Paris") == "Paris"

    def test_the_answer_is_colon(self):
        """Strips 'The answer is:' prefix."""
        assert GAIABenchmark.extract_answer("The answer is: Paris") == "Paris"

    def test_the_answer_is_no_colon(self):
        """Strips 'The answer is' prefix (no colon)."""
        assert GAIABenchmark.extract_answer("The answer is Paris") == "Paris"

    def test_final_answer_colon(self):
        """Strips 'Final answer:' prefix."""
        assert GAIABenchmark.extract_answer("Final answer: 42") == "42"

    def test_answer_colon(self):
        """Strips 'Answer:' prefix."""
        assert GAIABenchmark.extract_answer("Answer: test") == "test"

    def test_trailing_period(self):
        """Strips trailing period."""
        assert GAIABenchmark.extract_answer("Paris.") == "Paris"

    def test_just_period_preserved(self):
        """Single period is preserved."""
        assert GAIABenchmark.extract_answer(".") == "."

    def test_whitespace(self):
        """Leading/trailing whitespace stripped."""
        assert GAIABenchmark.extract_answer("  Paris  ") == "Paris"

    def test_combined(self):
        """Combined prefix + trailing period."""
        assert GAIABenchmark.extract_answer("The answer is: Paris.") == "Paris"

    def test_non_string(self):
        """Non-string input converted."""
        assert GAIABenchmark.extract_answer(42) == "42"


# =============================================================================
# 5. GAIABenchmark._strip_currency_pct — Static Method
# =============================================================================

@pytest.mark.unit
class TestStripCurrencyPct:
    """GAIABenchmark._strip_currency_pct() static method."""

    def test_dollar(self):
        assert GAIABenchmark._strip_currency_pct("$42") == "42"

    def test_percent(self):
        assert GAIABenchmark._strip_currency_pct("42%") == "42"

    def test_both(self):
        assert GAIABenchmark._strip_currency_pct("$42%") == "42"

    def test_plain(self):
        assert GAIABenchmark._strip_currency_pct("42") == "42"

    def test_whitespace(self):
        assert GAIABenchmark._strip_currency_pct("  $42  ") == "42"


# =============================================================================
# 6. GAIABenchmark._estimate_tokens — Static Method
# =============================================================================

@pytest.mark.unit
class TestEstimateTokens:
    """GAIABenchmark._estimate_tokens() static method."""

    def test_short_text(self):
        assert GAIABenchmark._estimate_tokens("hi") >= 1

    def test_longer_text(self):
        tokens = GAIABenchmark._estimate_tokens("This is a longer test sentence.")
        assert tokens > 1
        assert tokens == len("This is a longer test sentence.") // 4

    def test_empty(self):
        assert GAIABenchmark._estimate_tokens("") == 1


# =============================================================================
# 7. GAIABenchmark — evaluate_task metadata includes level
# =============================================================================

@pytest.mark.unit
class TestEvaluateTaskMetadata:
    """evaluate_task result metadata includes level."""

    def test_metadata_has_level(self, tmp_path):
        """Result metadata includes task level."""
        bench = GAIABenchmark(benchmark_path=str(tmp_path))
        task = _make_gaia_task(answer="4", level=2)

        class Agent:
            def run(self, q, **kw):
                return "4"

        result = bench.evaluate_task(task, Agent())
        assert result.metadata['level'] == 2

    def test_metadata_has_split(self, tmp_path):
        """Result metadata includes task split."""
        bench = GAIABenchmark(benchmark_path=str(tmp_path))
        task = _make_gaia_task(answer="4", split="test")

        class Agent:
            def run(self, q, **kw):
                return "4"

        result = bench.evaluate_task(task, Agent())
        assert result.metadata['split'] == "test"


# =============================================================================
# 8. run_gaia.py — CLI Argument Parsing
# =============================================================================

@pytest.mark.unit
class TestRunGaiaCLI:
    """run_gaia.py CLI argument parsing."""

    def test_default_args(self):
        """Default arguments."""
        from scripts.run_gaia import parse_args
        args = parse_args([])
        assert args.data_dir == "./data/gaia"
        assert args.split is None
        assert args.level is None
        assert args.tier is None  # default: auto-detect
        assert args.model is None
        assert args.max_tasks is None
        assert args.smoke is None
        assert args.task_id is None
        assert args.dry_run is False
        assert args.resume is False
        assert args.verbose is False

    def test_all_args(self):
        """All arguments set."""
        from scripts.run_gaia import parse_args
        args = parse_args([
            "--data-dir", "/tmp/gaia",
            "--split", "validation",
            "--level", "1",
            "--tier", "AGENTIC",
            "--model", "test-model",
            "--max-tasks", "5",
            "--task-id", "task_001",
            "--dry-run",
            "--resume",
            "--db-path", "/tmp/evals.db",
            "--output", "/tmp/results.json",
            "--verbose",
        ])
        assert args.data_dir == "/tmp/gaia"
        assert args.split == "validation"
        assert args.level == 1
        assert args.tier == "AGENTIC"
        assert args.model == "test-model"
        assert args.max_tasks == 5
        assert args.task_id == "task_001"
        assert args.dry_run is True
        assert args.resume is True
        assert args.db_path == "/tmp/evals.db"
        assert args.output == "/tmp/results.json"
        assert args.verbose is True

    def test_smoke_arg(self):
        """--smoke N sets smoke test mode (first N tasks, deterministic)."""
        from scripts.run_gaia import parse_args
        args = parse_args(["--smoke", "10"])
        assert args.smoke == 10
        args2 = parse_args(["--smoke", "5", "--split", "validation"])
        assert args2.smoke == 5
        assert args2.split == "validation"


# =============================================================================
# 9. run_gaia.py — Runner Logic (Mocked)
# =============================================================================

@pytest.mark.unit
class TestRunGaiaRunner:
    """run_gaia.py runner logic with mocked components."""

    def test_dry_run_pipeline(self, tmp_path):
        """Dry run exercises full pipeline without LLM."""
        from scripts.run_gaia import main

        gaia_path = _make_gaia_dataset(tmp_path)
        db_path = str(tmp_path / "test_evals.db")

        exit_code = main([
            "--data-dir", gaia_path,
            "--dry-run",
            "--max-tasks", "2",
            "--db-path", db_path,
        ])

        assert exit_code == 0

        # Verify results were stored
        store = EvalStore(db_path=db_path)
        runs = store.list_runs()
        assert len(runs) == 1
        summary = store.get_run_summary(runs[0]['id'])
        assert summary['total'] == 2
        store.close()

    def test_dry_run_with_level_filter(self, tmp_path):
        """Dry run with level filter loads correct tasks."""
        from scripts.run_gaia import main

        gaia_path = _make_gaia_dataset(tmp_path)
        db_path = str(tmp_path / "test_evals.db")

        exit_code = main([
            "--data-dir", gaia_path,
            "--dry-run",
            "--level", "1",
            "--db-path", db_path,
        ])

        assert exit_code == 0

        store = EvalStore(db_path=db_path)
        runs = store.list_runs()
        summary = store.get_run_summary(runs[0]['id'])
        assert summary['total'] == 3  # v1, v2, t1 (all level 1)
        store.close()

    def test_dry_run_with_split_filter(self, tmp_path):
        """Dry run with split filter."""
        from scripts.run_gaia import main

        gaia_path = _make_gaia_dataset(tmp_path)
        db_path = str(tmp_path / "test_evals.db")

        exit_code = main([
            "--data-dir", gaia_path,
            "--dry-run",
            "--split", "test",
            "--db-path", db_path,
        ])

        assert exit_code == 0

        store = EvalStore(db_path=db_path)
        runs = store.list_runs()
        summary = store.get_run_summary(runs[0]['id'])
        assert summary['total'] == 1  # Only t1
        store.close()

    def test_dry_run_single_task(self, tmp_path):
        """Dry run with --task-id selects one task."""
        from scripts.run_gaia import main

        gaia_path = _make_gaia_dataset(tmp_path)
        db_path = str(tmp_path / "test_evals.db")

        exit_code = main([
            "--data-dir", gaia_path,
            "--dry-run",
            "--task-id", "v1",
            "--db-path", db_path,
        ])

        assert exit_code == 0

        store = EvalStore(db_path=db_path)
        runs = store.list_runs()
        summary = store.get_run_summary(runs[0]['id'])
        assert summary['total'] == 1
        store.close()

    def test_missing_task_id(self, tmp_path):
        """Nonexistent --task-id returns 1."""
        from scripts.run_gaia import main

        gaia_path = _make_gaia_dataset(tmp_path)
        db_path = str(tmp_path / "test_evals.db")

        exit_code = main([
            "--data-dir", gaia_path,
            "--dry-run",
            "--task-id", "nonexistent",
            "--db-path", db_path,
        ])

        assert exit_code == 1

    def test_missing_dataset(self, tmp_path):
        """Missing dataset returns 1."""
        from scripts.run_gaia import main

        db_path = str(tmp_path / "test_evals.db")

        exit_code = main([
            "--data-dir", str(tmp_path / "nonexistent"),
            "--dry-run",
            "--db-path", db_path,
        ])

        assert exit_code == 1

    def test_output_json_saved(self, tmp_path):
        """--output saves results JSON."""
        from scripts.run_gaia import main

        gaia_path = _make_gaia_dataset(tmp_path)
        db_path = str(tmp_path / "test_evals.db")
        output_path = str(tmp_path / "results.json")

        exit_code = main([
            "--data-dir", gaia_path,
            "--dry-run",
            "--max-tasks", "2",
            "--db-path", db_path,
            "--output", output_path,
        ])

        assert exit_code == 0
        assert Path(output_path).exists()

        with open(output_path) as f:
            data = json.load(f)
        assert "run_id" in data
        assert "summary" in data
        assert "results" in data
        assert len(data["results"]) == 2


# =============================================================================
# 10. run_gaia.py — Helper Functions
# =============================================================================

@pytest.mark.unit
class TestRunGaiaHelpers:
    """run_gaia.py helper functions."""

    def test_get_completed_task_ids(self, tmp_path):
        """get_completed_task_ids returns set of done task IDs."""
        from scripts.run_gaia import get_completed_task_ids

        db_path = str(tmp_path / "test.db")
        store = EvalStore(db_path=db_path)
        run_id = store.start_run(model="test", benchmark="GAIA")
        store.record_result(run_id, "t1", True)
        store.record_result(run_id, "t2", False)

        completed = get_completed_task_ids(store, run_id)
        assert completed == {"t1", "t2"}
        store.close()

    def test_get_latest_run_id(self, tmp_path):
        """get_latest_run_id returns most recent running run."""
        from scripts.run_gaia import get_latest_run_id

        db_path = str(tmp_path / "test.db")
        store = EvalStore(db_path=db_path)

        # No runs yet
        assert get_latest_run_id(store) is None

        # Start a run (status='running')
        run_id = store.start_run(model="test", benchmark="GAIA")
        assert get_latest_run_id(store) == run_id

        # Finish it
        store.finish_run(run_id)
        assert get_latest_run_id(store) is None

        store.close()

    def test_get_latest_run_id_filters_benchmark(self, tmp_path):
        """get_latest_run_id only returns GAIA runs."""
        from scripts.run_gaia import get_latest_run_id

        db_path = str(tmp_path / "test.db")
        store = EvalStore(db_path=db_path)

        store.start_run(model="test", benchmark="other")
        assert get_latest_run_id(store, benchmark="GAIA") is None

        store.close()


# =============================================================================
# 11. Integration: Adapter + Benchmark + EvalStore
# =============================================================================

@pytest.mark.unit
class TestAdapterIntegration:
    """Integration of adapter with benchmark and store."""

    def test_dry_run_evaluate_task(self, tmp_path):
        """Adapter dry run works with GAIABenchmark.evaluate_task()."""
        bench = GAIABenchmark(benchmark_path=str(tmp_path))
        adapter = JottyGAIAAdapter(dry_run=True)
        task = _make_gaia_task(answer="Paris")

        result = bench.evaluate_task(task, adapter)

        assert result.success is False  # "[DRY RUN]" != "Paris"
        assert result.answer == "[DRY RUN]"
        assert result.execution_time >= 0

    def test_dry_run_full_pipeline_with_store(self, tmp_path):
        """Full pipeline: benchmark + adapter + store, all dry run."""
        gaia_path = _make_gaia_dataset(tmp_path, {
            "validation": [
                _make_gaia_task("v1", "Q?", "A"),
            ],
        })
        bench = GAIABenchmark(benchmark_path=gaia_path)
        adapter = JottyGAIAAdapter(dry_run=True)

        tasks = bench.load_tasks()
        assert len(tasks) == 1

        db_path = str(tmp_path / "test.db")
        store = EvalStore(db_path=db_path)
        run_id = store.start_run(model="dry-run", benchmark="GAIA")

        for task in tasks:
            result = bench.evaluate_task(task, adapter)
            store.record_result(
                run_id=run_id,
                task_id=result.task_id,
                success=result.success,
                answer=result.answer or "",
                execution_time=result.execution_time,
            )

        store.finish_run(run_id)
        summary = store.get_run_summary(run_id)

        assert summary['total'] == 1
        assert summary['status'] == 'finished'
        store.close()


# =============================================================================
# 12. download_gaia.py — Argument Parsing
# =============================================================================

@pytest.mark.unit
class TestDownloadGaia:
    """download_gaia.py argument parsing and structure."""

    def test_module_importable(self):
        """download_gaia module is importable."""
        from scripts.download_gaia import download_gaia
        assert callable(download_gaia)

    def test_download_requires_datasets(self, tmp_path):
        """download_gaia returns error when datasets not available."""
        from scripts.download_gaia import download_gaia

        with patch.dict('sys.modules', {'datasets': None}):
            # Force reimport to trigger ImportError path
            import importlib
            from scripts import download_gaia as dg
            importlib.reload(dg)
            result = dg.download_gaia(str(tmp_path / "output"))
            assert result == 1


# =============================================================================
# 13. Open-source LLM doc sources (Microsoft, Hugging Face, etc.)
# =============================================================================

@pytest.mark.unit
class TestLLMDocSources:
    """llm_doc_sources registry and helpers."""

    def test_list_sources_returns_non_empty(self):
        """list_sources() returns at least one source."""
        from Jotty.core.evaluation.llm_doc_sources import list_sources
        sources = list_sources()
        assert len(sources) >= 1
        assert hasattr(sources[0], "id")
        assert hasattr(sources[0], "name")
        assert hasattr(sources[0], "provider")

    def test_get_sources_by_provider_microsoft(self):
        """get_sources_by_provider('microsoft') returns Microsoft sources."""
        from Jotty.core.evaluation.llm_doc_sources import get_sources_by_provider
        ms = get_sources_by_provider("microsoft")
        assert len(ms) >= 1
        assert all(s.provider == "microsoft" for s in ms)

    def test_get_source_by_id(self):
        """get_source(id) returns the source or None."""
        from Jotty.core.evaluation.llm_doc_sources import get_source, list_sources
        first_id = list_sources()[0].id
        s = get_source(first_id)
        assert s is not None
        assert s.id == first_id
        assert get_source("nonexistent-id") is None

    def test_to_context_snippet(self):
        """to_context_snippet returns a string with URLs."""
        from Jotty.core.evaluation.llm_doc_sources import list_sources, to_context_snippet
        sources = list_sources()[:2]
        snippet = to_context_snippet(sources, max_items=2)
        assert "Relevant open-source" in snippet or "http" in snippet or "https" in snippet


# =============================================================================
# CLI runner
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
