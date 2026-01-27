# Benchmark Testing Guide

**Date**: January 27, 2026

---

## Overview

This guide explains how to test Jotty agents on benchmarks using the evaluation framework.

---

## Quick Start

### Run Benchmark Tests

```bash
cd /var/www/sites/personal/stock_market/Jotty
python examples/benchmark_test.py
```

This will run:
1. Single agent benchmark (math reasoning)
2. Multi-agent benchmark (general reasoning)
3. Quick test (small benchmark)
4. GAIA benchmark (if dataset available)

---

## Available Benchmarks

### 1. Math Benchmark

**File**: `examples/benchmark_test.py` → `create_math_benchmark()`

**Tasks**: 10 math problems (addition, subtraction, multiplication, division)

**Example**:
```python
from examples.benchmark_test import create_math_benchmark

benchmark = create_math_benchmark()
# Tasks: 2+2, 10*5, 100/4, etc.
```

### 2. Reasoning Benchmark

**File**: `examples/benchmark_test.py` → `create_reasoning_benchmark()`

**Tasks**: 5 reasoning questions (logic, facts, common knowledge)

**Example**:
```python
from examples.benchmark_test import create_reasoning_benchmark

benchmark = create_reasoning_benchmark()
# Tasks: "What comes after Monday?", "What is the capital of France?", etc.
```

### 3. Coding Benchmark

**File**: `examples/benchmark_test.py` → `create_coding_benchmark()`

**Tasks**: 3 coding questions (Python syntax, functions)

**Example**:
```python
from examples.benchmark_test import create_coding_benchmark

benchmark = create_coding_benchmark()
# Tasks: "Write a Python function...", etc.
```

### 4. GAIA Benchmark

**File**: `core/evaluation/gaia_benchmark.py`

**Tasks**: Real-world AI assistant tasks (requires dataset download)

**Setup**:
```bash
# Download GAIA dataset
git clone https://github.com/gaia-benchmark/gaia.git
# Place in ./data/gaia/
```

**Usage**:
```python
from core.evaluation import GAIABenchmark

benchmark = GAIABenchmark(benchmark_path="./data/gaia")
tasks = benchmark.load_tasks()
```

---

## Creating Custom Benchmarks

### Simple Custom Benchmark

```python
from core.evaluation import CustomBenchmark

benchmark = CustomBenchmark(
    name="my_benchmark",
    tasks=[
        {"id": "task1", "question": "What is 2+2?", "answer": "4"},
        {"id": "task2", "question": "What is Python?", "answer": "A programming language"},
    ]
)
```

### Custom Validation Function

```python
def validate_answer(task: Dict[str, Any], answer: str) -> bool:
    """Custom validation logic."""
    expected = task.get('answer', '').lower().strip()
    actual = answer.lower().strip()
    
    # Exact match
    if actual == expected:
        return True
    
    # Fuzzy match (contains)
    if expected in actual or actual in expected:
        return True
    
    return False

benchmark = CustomBenchmark(
    name="my_benchmark",
    tasks=[...],
    validate_func=validate_answer
)
```

---

## Testing Jotty Agents

### Single Agent Mode

```python
from examples.benchmark_test import JottyBenchmarkWrapper
from core.foundation.data_structures import SwarmConfig
from core.evaluation import CustomBenchmark, EvaluationProtocol

# Create benchmark
benchmark = CustomBenchmark(name="test", tasks=[...])

# Create config
config = SwarmConfig(random_seed=42, enable_cost_tracking=True)

# Create wrapper
wrapper = JottyBenchmarkWrapper(config=config, use_multi_agent=False)

# Run evaluation
protocol = EvaluationProtocol(benchmark=benchmark, n_runs=5, random_seed=42)
report = protocol.evaluate(wrapper)

print(f"Pass rate: {report.mean_pass_rate:.2%} ± {report.std_pass_rate:.2%}")
```

### Multi-Agent Mode

```python
from examples.benchmark_test import JottyBenchmarkWrapper
from core.foundation.data_structures import SwarmConfig

# Create wrapper with multi-agent
config = SwarmConfig(random_seed=42)
wrapper = JottyBenchmarkWrapper(
    config=config,
    use_multi_agent=True,
    agent_name=None  # Use default agent
)

# Run evaluation
protocol = EvaluationProtocol(benchmark=benchmark, n_runs=3, random_seed=42)
report = protocol.evaluate(wrapper)
```

---

## Understanding Results

### EvaluationReport

```python
report = protocol.evaluate(wrapper)

# Aggregated metrics
print(f"Mean pass rate: {report.mean_pass_rate:.2%}")
print(f"Std pass rate: {report.std_pass_rate:.2%}")
print(f"Mean cost: ${report.mean_cost:.6f}")
print(f"Mean execution time: {report.mean_execution_time:.2f}s")

# Per-run details
for run in report.runs:
    print(f"Run {run.run_id}: pass_rate={run.metrics.pass_rate:.2%}")
```

### BenchmarkMetrics

```python
metrics = benchmark.evaluate(wrapper)

print(f"Total tasks: {metrics.total_tasks}")
print(f"Successful: {metrics.successful_tasks}")
print(f"Failed: {metrics.failed_tasks}")
print(f"Pass rate: {metrics.pass_rate:.2%}")

# Per-task results
for result in metrics.results:
    print(f"{result.task_id}: {'✅' if result.success else '❌'}")
    if result.error:
        print(f"  Error: {result.error}")
```

---

## Configuration Options

### Reproducibility

```python
config = SwarmConfig(
    random_seed=42,  # Fixed seed for reproducibility
    numpy_seed=42,
    torch_seed=42,
    enable_deterministic=True
)
```

### Cost Tracking

```python
config = SwarmConfig(
    enable_cost_tracking=True,
    cost_tracking_file="./costs.json"
)
```

### Monitoring

```python
config = SwarmConfig(
    enable_monitoring=True,
    monitoring_output_dir="./monitoring"
)
```

---

## Best Practices

### 1. Use Reproducibility

Always set `random_seed` for reproducible results:

```python
config = SwarmConfig(random_seed=42)
protocol = EvaluationProtocol(benchmark=benchmark, n_runs=5, random_seed=42)
```

### 2. Multiple Runs

Use multiple runs to track variance:

```python
protocol = EvaluationProtocol(benchmark=benchmark, n_runs=5, random_seed=42)
report = protocol.evaluate(wrapper)
# Check std_pass_rate for variance
```

### 3. Cost Tracking

Enable cost tracking to monitor expenses:

```python
config = SwarmConfig(enable_cost_tracking=True)
# Check report.mean_cost after evaluation
```

### 4. Start Small

Test on small benchmarks first:

```python
# Quick test
benchmark = CustomBenchmark(name="test", tasks=[...])  # 2-3 tasks
metrics = benchmark.evaluate(wrapper)
```

### 5. Save Results

Save results for later analysis:

```python
protocol = EvaluationProtocol(benchmark=benchmark, n_runs=5, random_seed=42)
report = protocol.evaluate(wrapper, save_results=True, output_dir="./results")
```

---

## Troubleshooting

### Issue: Agent returns wrong format

**Solution**: Check `JottyBenchmarkWrapper._extract_answer()` to handle your result format.

### Issue: GAIA dataset not found

**Solution**: Download GAIA dataset:
```bash
git clone https://github.com/gaia-benchmark/gaia.git
mkdir -p ./data/gaia
cp -r gaia/test ./data/gaia/
cp -r gaia/validation ./data/gaia/
```

### Issue: High cost

**Solution**: 
- Use smaller benchmarks
- Reduce `n_runs`
- Use cheaper models
- Enable cost tracking to monitor

### Issue: Slow execution

**Solution**:
- Use single agent mode (faster than multi-agent)
- Reduce `n_runs`
- Use smaller benchmarks
- Test on subset of tasks

---

## Example Output

```
============================================================
Test 1: Single Agent Benchmark
============================================================

📊 Running evaluation on math_reasoning...
   Tasks: 10
   Runs: 3

============================================================
Results
============================================================
Benchmark: math_reasoning
Runs: 3
Pass Rate: 85.00% ± 5.00%
Mean Cost: $0.001234 ± $0.000123
Mean Execution Time: 2.34s ± 0.45s

Per-Run Details:
  Run 1 (seed=42): pass_rate=80.00%, cost=$0.001200
  Run 2 (seed=43): pass_rate=90.00%, cost=$0.001300
  Run 3 (seed=44): pass_rate=85.00%, cost=$0.001202
```

---

## Next Steps

1. ✅ **Run basic benchmarks** - Test on math/reasoning
2. ✅ **Test different configs** - Compare single vs multi-agent
3. ✅ **Download GAIA** - Test on real-world tasks
4. ✅ **Run ablation studies** - Test component contributions
5. ✅ **Track costs** - Monitor expenses

---

**Last Updated**: January 27, 2026
