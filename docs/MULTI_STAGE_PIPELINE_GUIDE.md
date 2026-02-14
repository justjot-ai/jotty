# Multi-Stage Pipeline Guide

## Overview

The `MultiStagePipeline` utility provides a high-level interface for chaining multiple swarm executions with automatic context passing and result aggregation. It dramatically reduces boilerplate code for complex multi-stage workflows.

## Quick Start

```python
from Jotty.core.orchestration import (
    MultiStagePipeline,
    SwarmAdapter,
    MergeStrategy
)

# Create pipeline
pipeline = MultiStagePipeline(task="Build trading strategy for NVDA")

# Add Stage 1: Research
pipeline.add_stage(
    "research",
    swarms=SwarmAdapter.quick_swarms([
        ("Analyst", "Analyze NVDA fundamentals and technicals")
    ]),
    merge_strategy=MergeStrategy.CONCATENATE
)

# Add Stage 2: Strategy (uses Stage 1 results automatically!)
pipeline.add_stage(
    "strategy",
    swarms=SwarmAdapter.quick_swarms([
        ("Strategist", "Generate trading strategy")
    ]),
    context_from=["research"],  # ✨ Auto-passes research results
    merge_strategy=MergeStrategy.BEST_OF_N
)

# Execute
result = await pipeline.execute(auto_trace=True, verbose=True)
result.print_summary()
```

## Core Concepts

### 1. Pipeline Stages

Each stage represents a phase in your workflow:
- **Name**: Unique identifier for context passing
- **Swarms**: List of swarms to execute in parallel
- **Merge Strategy**: How to combine swarm results
- **Context From**: Which previous stages to use as context

### 2. Automatic Context Passing

The pipeline automatically passes results from previous stages:

```python
# Stage 1 produces research
pipeline.add_stage("research", swarms=[...])

# Stage 2 receives research automatically
pipeline.add_stage(
    "analysis",
    swarms=[...],
    context_from=["research"]  # ✨ Research results passed as context
)

# Stage 3 can receive multiple contexts
pipeline.add_stage(
    "synthesis",
    swarms=[...],
    context_from=["research", "analysis"]  # ✨ Both passed as context
)
```

Context is formatted as:
```
[RESEARCH]
<research output truncated to max_context_chars>

[ANALYSIS]
<analysis output truncated to max_context_chars>

<your task>
```

### 3. Merge Strategies

Each stage can use different merge strategies:

| Strategy | Use Case | When to Use |
|----------|----------|-------------|
| `VOTING` | Consensus | When you need majority agreement |
| `CONCATENATE` | All perspectives | When all viewpoints are valuable |
| `BEST_OF_N` | Best result | When you want the highest quality output |
| `ENSEMBLE` | Numeric | When averaging numeric predictions |
| `FIRST_SUCCESS` | Redundancy | When any successful result works |

## API Reference

### MultiStagePipeline

```python
class MultiStagePipeline:
    def __init__(self, task: str, coordinator: Optional[MultiSwarmCoordinator] = None)
```

**Parameters:**
- `task`: Overall task description
- `coordinator`: Optional custom coordinator instance

**Methods:**

#### add_stage()

```python
def add_stage(
    self,
    name: str,
    swarms: List[Any],
    merge_strategy: MergeStrategy = MergeStrategy.BEST_OF_N,
    context_from: Optional[List[str]] = None,
    context_template: Optional[str] = None,
    max_context_chars: int = 1500
) -> MultiStagePipeline
```

**Parameters:**
- `name`: Stage identifier (used for context passing)
- `swarms`: List of swarms to execute
- `merge_strategy`: How to merge results
- `context_from`: List of previous stage names to include as context
- `context_template`: Custom template for context formatting
- `max_context_chars`: Maximum characters of context to pass

**Returns:** Self for method chaining

#### execute()

```python
async def execute(
    self,
    auto_trace: bool = True,
    verbose: bool = True
) -> PipelineResult
```

**Parameters:**
- `auto_trace`: Automatically use distributed tracing
- `verbose`: Print progress messages

**Returns:** `PipelineResult` with all stage results

### PipelineResult

```python
@dataclass
class PipelineResult:
    task: str
    stages: List[StageResult]
    total_cost: float
    total_time: float
    final_result: StageResult
```

**Methods:**

#### get_stage()

```python
def get_stage(self, name: str) -> Optional[StageResult]
```

Get result from specific stage by name.

#### print_summary()

```python
def print_summary(self, verbose: bool = True)
```

Print formatted pipeline summary with costs, times, and results.

### Utility Functions

#### extract_code_from_markdown()

```python
def extract_code_from_markdown(text: str, language: str = "python") -> Optional[str]
```

Extract code from markdown code blocks.

**Parameters:**
- `text`: Text containing markdown code blocks
- `language`: Expected language (e.g., "python", "javascript")

**Returns:** Extracted code or None

**Example:**
```python
from Jotty.core.orchestration import extract_code_from_markdown

llm_output = """
Here's the code:
```python
def hello():
    print("world")
```
"""

code = extract_code_from_markdown(llm_output, language="python")
# Returns: 'def hello():\n    print("world")'
```

## Examples

### Example 1: Research → Strategy → Code Pipeline

```python
from Jotty.core.orchestration import (
    MultiStagePipeline,
    SwarmAdapter,
    MergeStrategy,
    extract_code_from_markdown
)

# Create pipeline
pipeline = MultiStagePipeline(task="Build trading bot for AAPL")

# Stage 1: Research
pipeline.add_stage(
    "research",
    swarms=SwarmAdapter.quick_swarms([
        ("Fundamental Analyst", "Analyze AAPL fundamentals"),
        ("Technical Analyst", "Analyze AAPL technicals"),
    ]),
    merge_strategy=MergeStrategy.CONCATENATE  # Need all perspectives
)

# Stage 2: Strategy
pipeline.add_stage(
    "strategy",
    swarms=SwarmAdapter.quick_swarms([
        ("Quant Strategist", "Generate systematic strategy"),
        ("Risk Manager", "Add risk management rules"),
    ]),
    context_from=["research"],
    merge_strategy=MergeStrategy.BEST_OF_N  # Pick best strategy
)

# Stage 3: Code Generation
pipeline.add_stage(
    "codegen",
    swarms=SwarmAdapter.quick_swarms([
        ("Python Developer", "Generate executable Python backtest code")
    ]),
    context_from=["strategy"],
    max_context_chars=2000
)

# Execute
result = await pipeline.execute()

# Extract generated code
code_stage = result.get_stage("codegen")
code = extract_code_from_markdown(code_stage.result.output)

# Save and run
Path("/tmp/strategy.py").write_text(code)
```

### Example 2: Multi-Perspective Analysis

```python
pipeline = MultiStagePipeline(task="Analyze market opportunity for EV startups")

# Stage 1: Market Research (parallel perspectives)
pipeline.add_stage(
    "market",
    swarms=SwarmAdapter.quick_swarms([
        ("Market Analyst", "Research EV market size and trends"),
        ("Competitive Analyst", "Analyze competitive landscape"),
        ("Consumer Analyst", "Study consumer preferences"),
    ]),
    merge_strategy=MergeStrategy.CONCATENATE
)

# Stage 2: Financial Modeling
pipeline.add_stage(
    "financials",
    swarms=SwarmAdapter.quick_swarms([
        ("Financial Analyst", "Build financial model"),
    ]),
    context_from=["market"],
    max_context_chars=1000
)

# Stage 3: Risk Assessment
pipeline.add_stage(
    "risks",
    swarms=SwarmAdapter.quick_swarms([
        ("Risk Analyst 1", "Identify market risks"),
        ("Risk Analyst 2", "Identify execution risks"),
    ]),
    context_from=["market", "financials"],
    merge_strategy=MergeStrategy.VOTING  # Consensus on risks
)

# Stage 4: Final Recommendation
pipeline.add_stage(
    "recommendation",
    swarms=SwarmAdapter.quick_swarms([
        ("Investment Committee", "Make go/no-go recommendation"),
    ]),
    context_from=["market", "financials", "risks"]  # All context
)

result = await pipeline.execute()
```

### Example 3: Custom Context Template

```python
pipeline = MultiStagePipeline(task="Write academic paper")

pipeline.add_stage("literature_review", swarms=[...])

pipeline.add_stage(
    "methodology",
    swarms=[...],
    context_from=["literature_review"],
    context_template="""
    Based on the following literature review:

    {context}

    Design a rigorous methodology:
    """
)
```

## Best Practices

### 1. Choose Appropriate Merge Strategies

- **CONCATENATE** for early stages where you need all perspectives
- **VOTING** for validation/consensus stages
- **BEST_OF_N** for strategy selection or code generation

### 2. Limit Context Size

Use `max_context_chars` to prevent context overflow:
```python
pipeline.add_stage(
    "codegen",
    swarms=[...],
    context_from=["research"],
    max_context_chars=1500  # Limit to 1500 chars
)
```

### 3. Use Descriptive Stage Names

Good names make context passing clear:
```python
# Good
pipeline.add_stage("market_research", ...)
pipeline.add_stage("financial_model", context_from=["market_research"])

# Less clear
pipeline.add_stage("stage1", ...)
pipeline.add_stage("stage2", context_from=["stage1"])
```

### 4. Enable Tracing for Complex Pipelines

```python
result = await pipeline.execute(
    auto_trace=True,  # Enable distributed tracing
    verbose=True      # See progress
)
```

### 5. Access Individual Stage Results

```python
result = await pipeline.execute()

# Get specific stages
research = result.get_stage("research")
strategy = result.get_stage("strategy")

# Access outputs
print(research.result.output)
print(f"Cost: ${research.cost:.6f}")
print(f"Time: {research.execution_time:.2f}s")
```

## Performance Considerations

### Parallel Execution Within Stages

Each stage executes its swarms in parallel:
```python
# These 3 swarms run in parallel
pipeline.add_stage("research", swarms=[
    analyst1,  # ─┐
    analyst2,  #  ├─ All execute simultaneously
    analyst3,  # ─┘
])
```

### Sequential Execution Between Stages

Stages execute sequentially (required for context passing):
```python
pipeline.add_stage("stage1", ...)  # Runs first
pipeline.add_stage("stage2", context_from=["stage1"])  # Waits for stage1
pipeline.add_stage("stage3", context_from=["stage2"])  # Waits for stage2
```

### Cost Tracking

```python
result = await pipeline.execute()

print(f"Total cost: ${result.total_cost:.6f}")
print(f"Total time: {result.total_time:.2f}s")

for stage in result.stages:
    print(f"{stage.stage_name}: ${stage.cost:.6f}")
```

## Comparison: Before vs After

### Before (Manual - 331 lines)

```python
# Manual stage execution
tracer = get_distributed_tracer("service")
coordinator = get_multi_swarm_coordinator()

# Stage 1
with tracer.trace("stage1") as trace_id:
    result1 = await coordinator.execute_parallel(swarms1, task1, strategy1)
    cost1 = result1.metadata.get('cost_usd', 0)
    output1 = result1.output

# Stage 2 (manual context passing)
context = f"Based on: {output1[:1500]}..."
task2_with_context = context + "\n" + task2
with tracer.trace("stage2") as trace_id:
    result2 = await coordinator.execute_parallel(swarms2, task2_with_context, strategy2)
    cost2 = result2.metadata.get('cost_usd', 0)

# Manual aggregation
total_cost = cost1 + cost2
# ... 100+ more lines of boilerplate ...
```

### After (Pipeline - ~80 lines)

```python
pipeline = MultiStagePipeline(task="Main task")

pipeline.add_stage("stage1", swarms=swarms1, merge_strategy=strategy1)
pipeline.add_stage("stage2", swarms=swarms2, context_from=["stage1"])

result = await pipeline.execute(auto_trace=True)
result.print_summary()
```

**Code Reduction: 75-80%**

## See Also

- [Multi-Strategy Benchmark Guide](MULTI_STRATEGY_BENCHMARK_GUIDE.md)
- [SwarmAdapter Guide](SWARM_ADAPTER_GUIDE.md)
- [Merge Strategies](MERGE_STRATEGIES.md)
