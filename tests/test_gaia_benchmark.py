"""
GAIA Benchmark Test Suite
=========================

Comprehensive tests for the GAIA benchmark integration including:
- GAIABenchmark: task loading, evaluation, answer validation
- BenchmarkResult & BenchmarkMetrics: data classes and calculations
- CustomBenchmark: custom task evaluation
- EvalStore: SQLite persistence, run tracking, model comparison
- EvaluationProtocol: multi-run evaluation, aggregation
- Full pipeline: GAIABenchmark → EvalStore integration

All tests use mocks — NEVER call real LLM providers.
Tests run fast (< 1s each) and offline.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add Jotty to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.evaluation import (
    Benchmark,
    BenchmarkMetrics,
    BenchmarkResult,
    CustomBenchmark,
    EvalStore,
    EvaluationProtocol,
    EvaluationReport,
    GAIABenchmark,
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
                _make_gaia_task("v1", "What is 2+2?", "4"),
                _make_gaia_task("v2", "Capital of France?", "Paris"),
            ],
            "test": [
                _make_gaia_task("t1", "Square root of 9?", "3", split="test"),
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


class MockAgent:
    """Agent that returns a fixed answer."""

    def __init__(self, answer="4"):
        self.answer = answer
        self.calls = []

    def run(self, question, **kwargs):
        self.calls.append(question)
        return self.answer


class MockExecuteAgent:
    """Agent that uses execute() instead of run()."""

    def __init__(self, answer="4"):
        self.answer = answer

    def execute(self, question, **kwargs):
        return self.answer


class FailingAgent:
    """Agent that raises on every call."""

    def run(self, question, **kwargs):
        raise RuntimeError("Agent crashed")


class SlowAgent:
    """Agent with controllable delay (instant, but we track time)."""

    def __init__(self, answer="4"):
        self.answer = answer

    def run(self, question, **kwargs):
        return self.answer


# =============================================================================
# 1. GAIABenchmark — Initialization & Task Loading
# =============================================================================


@pytest.mark.unit
class TestGAIABenchmarkInit:
    """GAIABenchmark initialization and configuration."""

    def test_default_path(self):
        """Default benchmark_path is ./data/gaia."""
        bench = GAIABenchmark()
        assert bench.benchmark_path == Path("./data/gaia")
        assert bench.name == "GAIA"

    def test_custom_path(self):
        """Custom benchmark_path is stored correctly."""
        bench = GAIABenchmark(benchmark_path="/tmp/custom_gaia")
        assert bench.benchmark_path == Path("/tmp/custom_gaia")

    def test_inherits_benchmark(self):
        """GAIABenchmark is a Benchmark subclass."""
        bench = GAIABenchmark()
        assert isinstance(bench, Benchmark)

    def test_tasks_initially_empty(self):
        """Tasks list starts empty before load_tasks()."""
        bench = GAIABenchmark()
        assert bench.tasks == []


@pytest.mark.unit
class TestGAIABenchmarkLoadTasks:
    """GAIABenchmark task loading from disk."""

    def test_load_validation_and_test(self, tmp_path):
        """Loads tasks from both validation/ and test/ directories."""
        gaia_path = _make_gaia_dataset(tmp_path)
        bench = GAIABenchmark(benchmark_path=gaia_path)
        tasks = bench.load_tasks()

        assert len(tasks) == 3
        splits = {t["split"] for t in tasks}
        assert "validation" in splits
        assert "test" in splits

    def test_load_validation_only(self, tmp_path):
        """Loads from validation/ when test/ doesn't exist."""
        tasks_data = {
            "validation": [_make_gaia_task("v1", "Q?", "A")],
        }
        gaia_path = _make_gaia_dataset(tmp_path, tasks_data)
        bench = GAIABenchmark(benchmark_path=gaia_path)
        tasks = bench.load_tasks()

        assert len(tasks) == 1
        assert tasks[0]["split"] == "validation"

    def test_load_test_only(self, tmp_path):
        """Loads from test/ when validation/ doesn't exist."""
        tasks_data = {
            "test": [_make_gaia_task("t1", "Q?", "A", split="test")],
        }
        gaia_path = _make_gaia_dataset(tmp_path, tasks_data)
        bench = GAIABenchmark(benchmark_path=gaia_path)
        tasks = bench.load_tasks()

        assert len(tasks) == 1
        assert tasks[0]["split"] == "test"

    def test_load_empty_dataset(self, tmp_path):
        """Empty directories → empty task list."""
        gaia_dir = tmp_path / "gaia"
        (gaia_dir / "validation").mkdir(parents=True)
        (gaia_dir / "test").mkdir(parents=True)

        bench = GAIABenchmark(benchmark_path=str(gaia_dir))
        tasks = bench.load_tasks()
        assert tasks == []

    def test_missing_dataset_raises(self):
        """Missing dataset path raises FileNotFoundError."""
        bench = GAIABenchmark(benchmark_path="/nonexistent/path/gaia")
        with pytest.raises(FileNotFoundError, match="GAIA dataset not found"):
            bench.load_tasks()

    def test_malformed_json_skipped(self, tmp_path):
        """Malformed JSON files are skipped with a warning."""
        gaia_dir = tmp_path / "gaia" / "validation"
        gaia_dir.mkdir(parents=True)

        # Good task
        good_task = _make_gaia_task("good", "Q?", "A")
        (gaia_dir / "good.json").write_text(json.dumps(good_task))

        # Bad JSON
        (gaia_dir / "bad.json").write_text("{invalid json")

        bench = GAIABenchmark(benchmark_path=str(tmp_path / "gaia"))
        tasks = bench.load_tasks()

        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "good"

    def test_split_tag_added(self, tmp_path):
        """Each task gets a 'split' field based on its directory."""
        gaia_path = _make_gaia_dataset(tmp_path)
        bench = GAIABenchmark(benchmark_path=gaia_path)
        tasks = bench.load_tasks()

        for task in tasks:
            assert "split" in task
            assert task["split"] in ("validation", "test")


# =============================================================================
# 2. GAIABenchmark — Answer Validation
# =============================================================================


@pytest.mark.unit
class TestGAIAValidateAnswer:
    """GAIABenchmark answer validation logic."""

    def setup_method(self):
        self.bench = GAIABenchmark()

    def test_exact_match(self):
        """Exact case-insensitive match."""
        task = _make_gaia_task(answer="Paris")
        assert self.bench.validate_answer(task, "Paris") is True
        assert self.bench.validate_answer(task, "paris") is True
        assert self.bench.validate_answer(task, "PARIS") is True

    def test_whitespace_tolerance(self):
        """Leading/trailing whitespace is ignored."""
        task = _make_gaia_task(answer="Paris")
        assert self.bench.validate_answer(task, "  Paris  ") is True
        assert self.bench.validate_answer(task, "\tParis\n") is True

    def test_punctuation_tolerance(self):
        """Punctuation differences are tolerated."""
        task = _make_gaia_task(answer="Paris, France")
        assert self.bench.validate_answer(task, "Paris France") is True

    def test_numeric_exact(self):
        """Exact numeric match."""
        task = _make_gaia_task(answer="42")
        assert self.bench.validate_answer(task, "42") is True

    def test_numeric_float_tolerance(self):
        """Small floating-point differences are tolerated."""
        task = _make_gaia_task(answer="3.14")
        assert self.bench.validate_answer(task, "3.14") is True
        assert self.bench.validate_answer(task, "3.1400") is True

    def test_numeric_comma_formatting(self):
        """Comma-formatted numbers match."""
        task = _make_gaia_task(answer="1,000")
        assert self.bench.validate_answer(task, "1000") is True

    def test_wrong_answer(self):
        """Clearly wrong answer fails."""
        task = _make_gaia_task(answer="Paris")
        assert self.bench.validate_answer(task, "London") is False

    def test_empty_answer(self):
        """Empty answer doesn't match non-empty expected."""
        task = _make_gaia_task(answer="Paris")
        assert self.bench.validate_answer(task, "") is False

    def test_both_empty(self):
        """Empty expected → False (test split has no answers)."""
        task = _make_gaia_task(answer="")
        assert self.bench.validate_answer(task, "") is False

    def test_numeric_completely_different(self):
        """Numerically different values fail."""
        task = _make_gaia_task(answer="42")
        assert self.bench.validate_answer(task, "99") is False

    def test_non_string_answer_converted(self):
        """Non-string answers are converted via str()."""
        task = _make_gaia_task(answer="42")
        assert self.bench.validate_answer(task, 42) is True


# =============================================================================
# 3. GAIABenchmark — Task Evaluation
# =============================================================================


@pytest.mark.unit
class TestGAIAEvaluateTask:
    """GAIABenchmark single task evaluation."""

    def test_correct_answer(self, tmp_path):
        """Agent returning correct answer → success=True."""
        bench = GAIABenchmark(benchmark_path=str(tmp_path))
        task = _make_gaia_task(answer="4")
        agent = MockAgent(answer="4")

        result = bench.evaluate_task(task, agent)

        assert isinstance(result, BenchmarkResult)
        assert result.success is True
        assert result.answer == "4"
        assert result.execution_time > 0
        assert result.error is None

    def test_wrong_answer(self, tmp_path):
        """Agent returning wrong answer → success=False."""
        bench = GAIABenchmark(benchmark_path=str(tmp_path))
        task = _make_gaia_task(answer="4")
        agent = MockAgent(answer="99")

        result = bench.evaluate_task(task, agent)

        assert result.success is False
        assert result.answer == "99"

    def test_agent_exception(self, tmp_path):
        """Agent raising exception → success=False with error."""
        bench = GAIABenchmark(benchmark_path=str(tmp_path))
        task = _make_gaia_task(answer="4")
        agent = FailingAgent()

        result = bench.evaluate_task(task, agent)

        assert result.success is False
        assert "Agent crashed" in result.error

    def test_execute_method(self, tmp_path):
        """Agent with execute() instead of run() works."""
        bench = GAIABenchmark(benchmark_path=str(tmp_path))
        task = _make_gaia_task(answer="4")
        agent = MockExecuteAgent(answer="4")

        result = bench.evaluate_task(task, agent)

        assert result.success is True

    def test_no_run_or_execute_raises(self, tmp_path):
        """Agent without run() or execute() → error result."""
        bench = GAIABenchmark(benchmark_path=str(tmp_path))
        task = _make_gaia_task(answer="4")
        agent = object()  # No run() or execute()

        result = bench.evaluate_task(task, agent)

        assert result.success is False
        assert "run" in result.error or "execute" in result.error

    def test_metadata_includes_expected_answer(self, tmp_path):
        """Result metadata includes expected answer and question."""
        bench = GAIABenchmark(benchmark_path=str(tmp_path))
        task = _make_gaia_task(question="What is 2+2?", answer="4")
        agent = MockAgent(answer="4")

        result = bench.evaluate_task(task, agent)

        assert result.metadata["expected_answer"] == "4"
        assert result.metadata["question"] == "What is 2+2?"

    def test_task_id_from_task_id_field(self, tmp_path):
        """task_id comes from task dict's task_id field."""
        bench = GAIABenchmark(benchmark_path=str(tmp_path))
        task = _make_gaia_task(task_id="gaia_001")
        agent = MockAgent(answer="Paris")

        result = bench.evaluate_task(task, agent)

        assert result.task_id == "gaia_001"

    def test_task_id_fallback_to_file_name(self, tmp_path):
        """Falls back to file_name when task_id is missing."""
        bench = GAIABenchmark(benchmark_path=str(tmp_path))
        task = {"Question": "Q?", "Final answer": "A", "file_name": "test.json"}
        agent = MockAgent(answer="A")

        result = bench.evaluate_task(task, agent)

        assert result.task_id == "test.json"


# =============================================================================
# 4. Benchmark.evaluate() — Full Benchmark Run
# =============================================================================


@pytest.mark.unit
class TestBenchmarkEvaluate:
    """Benchmark base class evaluate() method."""

    def test_perfect_score(self, tmp_path):
        """All correct → 100% pass rate."""
        tasks = {
            "validation": [
                _make_gaia_task("v1", "2+2?", "4"),
                _make_gaia_task("v2", "3+3?", "6"),
            ],
        }
        gaia_path = _make_gaia_dataset(tmp_path, tasks)
        bench = GAIABenchmark(benchmark_path=gaia_path)

        class PerfectAgent:
            def run(self, q, **kw):
                if "2+2" in q:
                    return "4"
                return "6"

        metrics = bench.evaluate(PerfectAgent())

        assert isinstance(metrics, BenchmarkMetrics)
        assert metrics.total_tasks == 2
        assert metrics.successful_tasks == 2
        assert metrics.failed_tasks == 0
        assert metrics.pass_rate == 1.0

    def test_zero_score(self, tmp_path):
        """All wrong → 0% pass rate."""
        tasks = {
            "validation": [_make_gaia_task("v1", "Q?", "correct")],
        }
        gaia_path = _make_gaia_dataset(tmp_path, tasks)
        bench = GAIABenchmark(benchmark_path=gaia_path)

        metrics = bench.evaluate(MockAgent(answer="wrong"))

        assert metrics.total_tasks == 1
        assert metrics.successful_tasks == 0
        assert metrics.pass_rate == 0.0

    def test_partial_score(self, tmp_path):
        """1 of 2 correct → 50% pass rate."""
        tasks = {
            "validation": [
                _make_gaia_task("v1", "2+2?", "4"),
                _make_gaia_task("v2", "Capital?", "Paris"),
            ],
        }
        gaia_path = _make_gaia_dataset(tmp_path, tasks)
        bench = GAIABenchmark(benchmark_path=gaia_path)

        # Agent always answers "4" — correct for v1, wrong for v2
        metrics = bench.evaluate(MockAgent(answer="4"))

        assert metrics.total_tasks == 2
        assert metrics.successful_tasks == 1
        assert metrics.pass_rate == 0.5

    def test_metrics_aggregation(self, tmp_path):
        """Metrics aggregate execution times correctly."""
        tasks = {
            "validation": [
                _make_gaia_task("v1", "Q1?", "A"),
                _make_gaia_task("v2", "Q2?", "A"),
            ],
        }
        gaia_path = _make_gaia_dataset(tmp_path, tasks)
        bench = GAIABenchmark(benchmark_path=gaia_path)

        metrics = bench.evaluate(MockAgent(answer="A"))

        assert metrics.avg_execution_time >= 0
        assert metrics.total_cost == 0  # Mock agent has no cost
        assert len(metrics.results) == 2

    def test_lazy_loading(self, tmp_path):
        """Tasks are loaded on first evaluate() call."""
        tasks_data = {"validation": [_make_gaia_task("v1", "Q?", "A")]}
        gaia_path = _make_gaia_dataset(tmp_path, tasks_data)
        bench = GAIABenchmark(benchmark_path=gaia_path)

        assert bench.tasks == []
        bench.evaluate(MockAgent(answer="A"))
        assert len(bench.tasks) == 1

    def test_results_to_dict(self, tmp_path):
        """BenchmarkMetrics.to_dict() serializes correctly."""
        tasks_data = {"validation": [_make_gaia_task("v1", "Q?", "4")]}
        gaia_path = _make_gaia_dataset(tmp_path, tasks_data)
        bench = GAIABenchmark(benchmark_path=gaia_path)

        metrics = bench.evaluate(MockAgent(answer="4"))
        d = metrics.to_dict()

        assert d["total_tasks"] == 1
        assert d["pass_rate"] == 1.0
        assert len(d["results"]) == 1
        assert d["results"][0]["success"] is True


# =============================================================================
# 5. BenchmarkResult — Data Class
# =============================================================================


@pytest.mark.unit
class TestBenchmarkResult:
    """BenchmarkResult data class."""

    def test_to_dict(self):
        """to_dict serializes all fields."""
        result = BenchmarkResult(
            task_id="t1",
            success=True,
            answer="42",
            execution_time=1.5,
            cost=0.01,
            tokens_used=100,
            metadata={"key": "value"},
        )
        d = result.to_dict()

        assert d["task_id"] == "t1"
        assert d["success"] is True
        assert d["answer"] == "42"
        assert d["execution_time"] == 1.5
        assert d["cost"] == 0.01
        assert d["tokens_used"] == 100
        assert d["metadata"] == {"key": "value"}

    def test_defaults(self):
        """Default values for optional fields."""
        result = BenchmarkResult(task_id="t1", success=False)

        assert result.answer is None
        assert result.error is None
        assert result.execution_time == 0.0
        assert result.cost == 0.0
        assert result.tokens_used == 0
        assert result.metadata == {}

    def test_error_field(self):
        """Error field records failure reason."""
        result = BenchmarkResult(task_id="t1", success=False, error="Timeout")
        assert result.error == "Timeout"
        assert result.to_dict()["error"] == "Timeout"


# =============================================================================
# 6. CustomBenchmark
# =============================================================================


@pytest.mark.unit
class TestCustomBenchmark:
    """CustomBenchmark for user-defined tasks."""

    def test_basic_evaluation(self):
        """Correct answer → 100% pass rate."""
        tasks = [
            {"id": "t1", "question": "2+2?", "answer": "4"},
            {"id": "t2", "question": "3+3?", "answer": "6"},
        ]
        bench = CustomBenchmark(name="math", tasks=tasks)

        class MathAgent:
            def run(self, q, **kw):
                if "2+2" in q:
                    return "4"
                return "6"

        metrics = bench.evaluate(MathAgent())

        assert metrics.pass_rate == 1.0
        assert metrics.total_tasks == 2

    def test_custom_validate_func(self):
        """Custom validation function overrides default."""
        tasks = [{"id": "t1", "question": "Q?", "answer": "4"}]

        def always_true(task, answer):
            return True

        bench = CustomBenchmark(name="custom", tasks=tasks, validate_func=always_true)
        metrics = bench.evaluate(MockAgent(answer="wrong"))

        assert metrics.pass_rate == 1.0

    def test_prompt_field_fallback(self):
        """Falls back to 'prompt' field when 'question' is missing."""
        tasks = [{"id": "t1", "prompt": "Calculate 2+2", "answer": "4"}]
        bench = CustomBenchmark(name="test", tasks=tasks)

        metrics = bench.evaluate(MockAgent(answer="4"))
        assert metrics.pass_rate == 1.0

    def test_agent_exception_handled(self):
        """Agent exception → task fails but benchmark continues."""
        tasks = [
            {"id": "t1", "question": "Q?", "answer": "A"},
            {"id": "t2", "question": "Q2?", "answer": "A"},
        ]
        bench = CustomBenchmark(name="test", tasks=tasks)

        metrics = bench.evaluate(FailingAgent())

        assert metrics.total_tasks == 2
        assert metrics.failed_tasks == 2
        assert metrics.pass_rate == 0.0


# =============================================================================
# 7. EvalStore — SQLite Persistence
# =============================================================================


@pytest.mark.unit
class TestEvalStore:
    """EvalStore SQLite-backed evaluation results."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create an EvalStore with a temp DB."""
        db_path = str(tmp_path / "test_evals.db")
        s = EvalStore(db_path=db_path)
        yield s
        s.close()

    def test_start_run(self, store):
        """start_run returns a run_id string."""
        run_id = store.start_run(model="claude-sonnet-4", benchmark="GAIA")
        assert isinstance(run_id, str)
        assert len(run_id) > 0

    def test_record_result(self, store):
        """record_result stores a task result."""
        run_id = store.start_run(model="claude-sonnet-4", benchmark="GAIA")
        store.record_result(
            run_id=run_id,
            task_id="t1",
            success=True,
            answer="42",
            execution_time=1.5,
            cost=0.01,
            tokens_used=100,
        )
        summary = store.get_run_summary(run_id)

        assert summary["total"] == 1
        assert summary["passed"] == 1
        assert summary["pass_rate"] == 1.0

    def test_multiple_results(self, store):
        """Multiple results tracked per run."""
        run_id = store.start_run(model="claude-sonnet-4", benchmark="GAIA")
        store.record_result(run_id, "t1", True, answer="4", execution_time=1.0)
        store.record_result(run_id, "t2", False, error="Wrong answer", execution_time=2.0)
        store.record_result(run_id, "t3", True, answer="Paris", execution_time=0.5)

        summary = store.get_run_summary(run_id)

        assert summary["total"] == 3
        assert summary["passed"] == 2
        assert abs(summary["pass_rate"] - 2 / 3) < 1e-6

    def test_finish_run(self, store):
        """finish_run marks run as finished."""
        run_id = store.start_run(model="claude-sonnet-4", benchmark="GAIA")
        store.finish_run(run_id)

        summary = store.get_run_summary(run_id)
        assert summary["status"] == "finished"

    def test_get_run_summary_nonexistent(self, store):
        """Nonexistent run_id returns empty dict."""
        summary = store.get_run_summary("nonexistent")
        assert summary == {}

    def test_compare_models(self, store):
        """compare_models aggregates across runs by model."""
        # Claude run: 2/3 correct
        run1 = store.start_run(model="claude-sonnet-4", benchmark="GAIA")
        store.record_result(run1, "t1", True, cost=0.01)
        store.record_result(run1, "t2", False, cost=0.02)
        store.record_result(run1, "t3", True, cost=0.01)
        store.finish_run(run1)

        # GPT run: 1/2 correct
        run2 = store.start_run(model="gpt-4", benchmark="GAIA")
        store.record_result(run2, "t1", True, cost=0.03)
        store.record_result(run2, "t2", False, cost=0.04)
        store.finish_run(run2)

        comparison = store.compare_models(benchmark="GAIA")

        assert "claude-sonnet-4" in comparison
        assert "gpt-4" in comparison
        assert abs(comparison["claude-sonnet-4"]["pass_rate"] - 2 / 3) < 1e-6
        assert comparison["gpt-4"]["pass_rate"] == 0.5

    def test_compare_models_all_benchmarks(self, store):
        """compare_models without benchmark filter aggregates all."""
        run1 = store.start_run(model="claude-sonnet-4", benchmark="GAIA")
        store.record_result(run1, "t1", True)
        store.finish_run(run1)

        run2 = store.start_run(model="claude-sonnet-4", benchmark="custom")
        store.record_result(run2, "t1", False)
        store.finish_run(run2)

        comparison = store.compare_models()
        assert comparison["claude-sonnet-4"]["total"] == 2

    def test_list_runs(self, store):
        """list_runs returns recent runs."""
        store.start_run(model="model-a", benchmark="b1")
        store.start_run(model="model-b", benchmark="b2")

        runs = store.list_runs()
        assert len(runs) == 2

    def test_list_runs_limit(self, store):
        """list_runs respects limit."""
        for i in range(5):
            store.start_run(model=f"model-{i}", benchmark="b")

        runs = store.list_runs(limit=3)
        assert len(runs) == 3

    def test_cost_and_tokens_tracked(self, store):
        """Cost and token totals are tracked correctly."""
        run_id = store.start_run(model="claude-sonnet-4", benchmark="GAIA")
        store.record_result(run_id, "t1", True, cost=0.01, tokens_used=100)
        store.record_result(run_id, "t2", True, cost=0.02, tokens_used=200)

        summary = store.get_run_summary(run_id)

        assert abs(summary["total_cost"] - 0.03) < 1e-6
        assert summary["total_tokens"] == 300

    def test_avg_time_calculated(self, store):
        """Average execution time is calculated correctly."""
        run_id = store.start_run(model="claude-sonnet-4", benchmark="GAIA")
        store.record_result(run_id, "t1", True, execution_time=1.0)
        store.record_result(run_id, "t2", True, execution_time=3.0)

        summary = store.get_run_summary(run_id)

        assert abs(summary["avg_time"] - 2.0) < 1e-6

    def test_metadata_stored(self, store):
        """Run metadata is stored and retrievable."""
        run_id = store.start_run(
            model="claude-sonnet-4",
            benchmark="GAIA",
            metadata={"version": "1.0", "config": "default"},
        )
        runs = store.list_runs()
        run = [r for r in runs if r["id"] == run_id][0]
        meta = json.loads(run["metadata"])
        assert meta["version"] == "1.0"


# =============================================================================
# 8. EvaluationProtocol — Multi-Run Evaluation
# =============================================================================


@pytest.mark.unit
class TestEvaluationProtocol:
    """EvaluationProtocol multi-run reproducible evaluation."""

    def test_single_run(self, tmp_path):
        """Single run produces valid report."""
        tasks_data = {"validation": [_make_gaia_task("v1", "Q?", "4")]}
        gaia_path = _make_gaia_dataset(tmp_path, tasks_data)
        bench = GAIABenchmark(benchmark_path=gaia_path)

        protocol = EvaluationProtocol(benchmark=bench, n_runs=1, random_seed=42)
        report = protocol.evaluate(MockAgent(answer="4"), save_results=False)

        assert isinstance(report, EvaluationReport)
        assert report.n_runs == 1
        assert report.mean_pass_rate == 1.0
        assert len(report.runs) == 1

    def test_multi_run(self, tmp_path):
        """Multiple runs produce variance statistics."""
        tasks_data = {"validation": [_make_gaia_task("v1", "Q?", "4")]}
        gaia_path = _make_gaia_dataset(tmp_path, tasks_data)
        bench = GAIABenchmark(benchmark_path=gaia_path)

        protocol = EvaluationProtocol(benchmark=bench, n_runs=3, random_seed=42)
        report = protocol.evaluate(MockAgent(answer="4"), save_results=False)

        assert report.n_runs == 3
        assert len(report.runs) == 3
        assert report.mean_pass_rate == 1.0
        assert report.std_pass_rate == 0.0  # Deterministic agent

    def test_report_to_dict(self, tmp_path):
        """Report serializes to dict correctly."""
        tasks_data = {"validation": [_make_gaia_task("v1", "Q?", "4")]}
        gaia_path = _make_gaia_dataset(tmp_path, tasks_data)
        bench = GAIABenchmark(benchmark_path=gaia_path)

        protocol = EvaluationProtocol(benchmark=bench, n_runs=2, random_seed=42)
        report = protocol.evaluate(MockAgent(answer="4"), save_results=False)
        d = report.to_dict()

        assert d["benchmark_name"] == "GAIA"
        assert d["n_runs"] == 2
        assert len(d["runs"]) == 2

    def test_save_results(self, tmp_path):
        """Results saved to output directory."""
        tasks_data = {"validation": [_make_gaia_task("v1", "Q?", "4")]}
        gaia_path = _make_gaia_dataset(tmp_path, tasks_data)
        bench = GAIABenchmark(benchmark_path=gaia_path)

        output_dir = str(tmp_path / "eval_output")
        protocol = EvaluationProtocol(benchmark=bench, n_runs=1, random_seed=42)
        protocol.evaluate(MockAgent(answer="4"), save_results=True, output_dir=output_dir)

        output_path = Path(output_dir)
        assert (output_path / "GAIA_evaluation_report.json").exists()
        assert (output_path / "GAIA_summary.txt").exists()

        # Verify JSON is valid
        with open(output_path / "GAIA_evaluation_report.json") as f:
            data = json.load(f)
        assert data["benchmark_name"] == "GAIA"

    def test_different_seeds_per_run(self, tmp_path):
        """Each run gets a unique seed (base_seed + run_id)."""
        tasks_data = {"validation": [_make_gaia_task("v1", "Q?", "4")]}
        gaia_path = _make_gaia_dataset(tmp_path, tasks_data)
        bench = GAIABenchmark(benchmark_path=gaia_path)

        protocol = EvaluationProtocol(benchmark=bench, n_runs=3, random_seed=100)
        report = protocol.evaluate(MockAgent(answer="4"), save_results=False)

        seeds = [run.seed for run in report.runs]
        assert seeds == [100, 101, 102]


# =============================================================================
# 9. Full Pipeline Integration — GAIABenchmark → EvalStore
# =============================================================================


@pytest.mark.unit
class TestGAIAPipeline:
    """Full GAIA benchmark pipeline: load → evaluate → persist."""

    def test_full_pipeline(self, tmp_path):
        """Load tasks, evaluate agent, store results in EvalStore."""
        # Setup dataset
        tasks_data = {
            "validation": [
                _make_gaia_task("v1", "What is 2+2?", "4"),
                _make_gaia_task("v2", "Capital of France?", "Paris"),
                _make_gaia_task("v3", "Square root of 9?", "3"),
            ],
        }
        gaia_path = _make_gaia_dataset(tmp_path, tasks_data)

        # Setup benchmark
        bench = GAIABenchmark(benchmark_path=gaia_path)

        # Setup store
        db_path = str(tmp_path / "pipeline_evals.db")
        store = EvalStore(db_path=db_path)

        # Run evaluation
        metrics = bench.evaluate(MockAgent(answer="4"))

        # Store results
        run_id = store.start_run(model="jotty-v3", benchmark="GAIA")
        for result in metrics.results:
            store.record_result(
                run_id=run_id,
                task_id=result.task_id,
                success=result.success,
                answer=result.answer or "",
                error=result.error or "",
                execution_time=result.execution_time,
            )
        store.finish_run(run_id)

        # Verify
        summary = store.get_run_summary(run_id)
        assert summary["model"] == "jotty-v3"
        assert summary["benchmark"] == "GAIA"
        assert summary["total"] == 3
        assert summary["passed"] == 1  # Only "4" matches v1
        assert summary["status"] == "finished"

        store.close()

    def test_multi_model_comparison(self, tmp_path):
        """Compare multiple agents/models on same benchmark."""
        tasks_data = {
            "validation": [
                _make_gaia_task("v1", "2+2?", "4"),
                _make_gaia_task("v2", "Capital?", "Paris"),
            ],
        }
        gaia_path = _make_gaia_dataset(tmp_path, tasks_data)
        bench = GAIABenchmark(benchmark_path=gaia_path)
        db_path = str(tmp_path / "comparison_evals.db")
        store = EvalStore(db_path=db_path)

        # Model A: always answers "4" (1/2 correct)
        metrics_a = bench.evaluate(MockAgent(answer="4"))
        run_a = store.start_run(model="model-a", benchmark="GAIA")
        for r in metrics_a.results:
            store.record_result(run_a, r.task_id, r.success, answer=r.answer or "")
        store.finish_run(run_a)

        # Reload tasks for second run
        bench.tasks = []

        # Model B: always answers "Paris" (1/2 correct)
        metrics_b = bench.evaluate(MockAgent(answer="Paris"))
        run_b = store.start_run(model="model-b", benchmark="GAIA")
        for r in metrics_b.results:
            store.record_result(run_b, r.task_id, r.success, answer=r.answer or "")
        store.finish_run(run_b)

        # Compare
        comparison = store.compare_models(benchmark="GAIA")
        assert "model-a" in comparison
        assert "model-b" in comparison
        # Both get 1 out of 2
        assert comparison["model-a"]["pass_rate"] == 0.5
        assert comparison["model-b"]["pass_rate"] == 0.5

        store.close()

    def test_evaluation_protocol_with_store(self, tmp_path):
        """EvaluationProtocol + EvalStore end-to-end."""
        tasks_data = {"validation": [_make_gaia_task("v1", "Q?", "4")]}
        gaia_path = _make_gaia_dataset(tmp_path, tasks_data)
        bench = GAIABenchmark(benchmark_path=gaia_path)

        # Run protocol
        protocol = EvaluationProtocol(benchmark=bench, n_runs=3, random_seed=42)
        report = protocol.evaluate(MockAgent(answer="4"), save_results=False)

        # Store in EvalStore
        db_path = str(tmp_path / "protocol_evals.db")
        store = EvalStore(db_path=db_path)
        run_id = store.start_run(model="jotty-v3", benchmark="GAIA")

        # Store aggregated results from last run
        last_run = report.runs[-1]
        for result in last_run.metrics.results:
            store.record_result(
                run_id=run_id,
                task_id=result.task_id,
                success=result.success,
                answer=result.answer or "",
                execution_time=result.execution_time,
            )
        store.finish_run(run_id)

        summary = store.get_run_summary(run_id)
        assert summary["pass_rate"] == 1.0
        assert summary["status"] == "finished"

        store.close()


# =============================================================================
# 10. Edge Cases & Robustness
# =============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_benchmark(self):
        """Benchmark with zero tasks → zero metrics."""
        bench = CustomBenchmark(name="empty", tasks=[])
        metrics = bench.evaluate(MockAgent())

        assert metrics.total_tasks == 0
        assert metrics.pass_rate == 0.0

    def test_unicode_tasks(self, tmp_path):
        """Unicode questions and answers handled correctly."""
        tasks_data = {
            "validation": [
                _make_gaia_task("u1", "日本の首都は？", "東京"),
            ],
        }
        gaia_path = _make_gaia_dataset(tmp_path, tasks_data)
        bench = GAIABenchmark(benchmark_path=gaia_path)

        metrics = bench.evaluate(MockAgent(answer="東京"))
        assert metrics.pass_rate == 1.0

    def test_large_benchmark(self, tmp_path):
        """Benchmark with many tasks doesn't crash."""
        tasks_data = {
            "validation": [_make_gaia_task(f"v{i}", f"Q{i}?", str(i)) for i in range(50)],
        }
        gaia_path = _make_gaia_dataset(tmp_path, tasks_data)
        bench = GAIABenchmark(benchmark_path=gaia_path)

        metrics = bench.evaluate(MockAgent(answer="0"))
        assert metrics.total_tasks == 50
        # Only task v0 has answer "0"
        assert metrics.successful_tasks == 1

    def test_benchmark_result_serialization_roundtrip(self):
        """BenchmarkResult → dict → verify all fields preserved."""
        result = BenchmarkResult(
            task_id="roundtrip",
            success=True,
            answer="42",
            execution_time=2.5,
            cost=0.05,
            tokens_used=500,
            metadata={"level": 1, "split": "validation"},
        )
        d = result.to_dict()

        assert d["task_id"] == "roundtrip"
        assert d["success"] is True
        assert d["answer"] == "42"
        assert d["execution_time"] == 2.5
        assert d["cost"] == 0.05
        assert d["tokens_used"] == 500
        assert d["metadata"]["level"] == 1

    def test_metrics_with_all_failures(self, tmp_path):
        """All tasks fail → 0% pass rate, error info preserved."""
        tasks_data = {
            "validation": [
                _make_gaia_task("v1", "Q1?", "A1"),
                _make_gaia_task("v2", "Q2?", "A2"),
            ],
        }
        gaia_path = _make_gaia_dataset(tmp_path, tasks_data)
        bench = GAIABenchmark(benchmark_path=gaia_path)

        metrics = bench.evaluate(FailingAgent())

        assert metrics.total_tasks == 2
        assert metrics.pass_rate == 0.0
        for r in metrics.results:
            assert r.success is False
            assert r.error is not None

    def test_concurrent_eval_store_writes(self, tmp_path):
        """Multiple runs written to same store don't conflict."""
        db_path = str(tmp_path / "concurrent_evals.db")
        store = EvalStore(db_path=db_path)

        run_ids = []
        for i in range(5):
            run_id = store.start_run(model=f"model-{i}", benchmark="GAIA")
            store.record_result(run_id, f"t{i}", True)
            store.finish_run(run_id)
            run_ids.append(run_id)

        runs = store.list_runs()
        assert len(runs) == 5

        for run_id in run_ids:
            summary = store.get_run_summary(run_id)
            assert summary["total"] == 1
            assert summary["passed"] == 1

        store.close()


# =============================================================================
# CLI runner
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
