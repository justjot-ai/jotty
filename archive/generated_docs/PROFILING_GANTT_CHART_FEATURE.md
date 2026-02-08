# 📊 Profiling with Gantt Chart & Component Breakdown

## Overview

Jotty now generates **comprehensive profiling reports** with visual Gantt charts and detailed component breakdowns to help you optimize your multi-agent system.

## What Was Added

### 1. **Gantt Chart Visualization**
Visual timeline showing exactly when each agent and component executed

- **Mermaid Gantt Chart** - Render in GitHub, VS Code, or any Mermaid viewer
- **ASCII Timeline** - Text-based visualization in terminal/logs
- Shows parallel vs sequential execution
- Identifies bottlenecks visually

### 2. **Component Breakdown**
Detailed timing statistics for every component:

- **Agent** - Individual agent execution times
- **ParameterResolution** - Time spent resolving parameters
- **StatePersistence** - Time spent saving state
- **LLMCall** - Claude CLI subprocess times
- **Other** - Custom components

Each component shows:
- Count of operations
- Total time spent
- Average, min, max times
- Percentage of total execution
- Individual operation details

### 3. **Optimization Recommendations**
AI-generated suggestions based on your profiling data:

- Identifies slow agents (> 5s)
- Recommends Haiku vs Sonnet model
- Flags inefficient parameter resolution
- Detects large state persistence issues
- Calculates orchestration overhead

### 4. **Multiple Output Formats**

#### Files Generated (in `./outputs/run_*/profiling/`):

| File | Purpose | Format |
|------|---------|--------|
| `gantt_chart.mmd` | Mermaid Gantt chart for rendering | Mermaid |
| `execution_timeline.txt` | ASCII timeline + breakdown + recommendations | Text |
| `profiling_report.md` | Complete report with all visualizations | Markdown |
| `profiling_data.json` | Raw timing data for programmatic analysis | JSON |

## How to Use

### Enable Profiling

```python
from core import SwarmConfig, Conductor

config = SwarmConfig(
    enable_profiling=True,
    profiling_verbosity="summary",
    enable_beautified_logs=True,
    output_base_dir="./outputs",
    create_run_folder=True
)

conductor = Conductor(actors=[...], config=config)
result = await conductor.run(goal="Your task...")
```

### View Reports

After execution, find reports in:
```
./outputs/run_20260111_091414/profiling/
```

**Console Output:**
```
⏱️  Profiling reports saved:
   - Gantt Chart: ./outputs/run_20260111_091414/profiling/gantt_chart.mmd
   - Timeline Report: ./outputs/run_20260111_091414/profiling/execution_timeline.txt
   - Markdown Report: ./outputs/run_20260111_091414/profiling/profiling_report.md
   - JSON Data: ./outputs/run_20260111_091414/profiling/profiling_data.json
```

### Example Output

**ASCII Timeline:**
```
============================================================
⏱️  EXECUTION TIMELINE (Gantt Chart)
============================================================
Total Duration: 7.32s

Timeline: |------------------------------------------|
          0s      1.8s    3.7s    5.5s        7.3s

Agent:
  IssueDetector              |  ████████ 3.45s
  SolutionProvider           |            ██████ 2.87s
============================================================
```

**Component Breakdown:**
```
📦 Agent
   Count:      2
   Total:      6.32s
   Average:    3.16s
   Min:        2.87s
   Max:        3.45s
   % of Total: 86.3%
   Operations:
      - IssueDetector: 3.450s (3450ms)
      - SolutionProvider: 2.870s (2870ms)
```

**Optimization Recommendations:**
```
🤖 Agent Execution:
   Average time: 3.16s
   ✅ Normal range for Sonnet model (3-4s per agent)

📈 Overall Analysis:
   Total execution: 7.32s
   Agent time: 6.32s (86.3%)
   Overhead: 1.00s (13.7%)
   ✅ Orchestration overhead is reasonable (<20%)
```

## Use Cases

### 1. Performance Debugging
- Identify which agent is slow
- See if agents run sequentially or in parallel
- Find orchestration bottlenecks

### 2. Optimization Planning
- Compare Haiku vs Sonnet execution times
- Identify if parameter resolution is taking too long
- Decide if state persistence needs optimization

### 3. Cost Analysis
- See time breakdown per component
- Calculate cost per agent (time × model rate)
- Optimize for cost vs performance

### 4. Architecture Decisions
- Understand actual execution flow
- Identify opportunities for parallelization
- Make data-driven architecture choices

## Implementation Details

### Files Modified

1. **`core/utils/profiling_report.py`** (NEW)
   - ProfilingReport class
   - Gantt chart generation (Mermaid + ASCII)
   - Component breakdown statistics
   - Optimization recommendation engine

2. **`core/utils/profiler.py`** (ENHANCED)
   - Integrated with ProfilingReport
   - Enhanced `timed_block()` with component categorization
   - Added `set_output_dir()`, `set_overall_timing()`, `save_profiling_reports()`

3. **`core/orchestration/conductor.py`** (INTEGRATED)
   - Initialize profiling report with output directory
   - Wrap agent execution with component="Agent"
   - Set overall timing
   - Save reports before returning results

4. **`LOGGING_AND_PROFILING.md`** (UPDATED)
   - Documented new Gantt chart feature
   - Added examples of all report formats
   - Explained output file structure

## Example Workflow

```python
import asyncio
from core import SwarmConfig, AgentSpec, Conductor
import dspy

async def main():
    # Configure with profiling
    config = SwarmConfig(
        enable_profiling=True,
        profiling_verbosity="summary",
        enable_beautified_logs=True,
        enable_debug_logs=True,
        output_base_dir="./outputs",
        create_run_folder=True
    )

    # Create agents
    detector = AgentSpec(name="IssueDetector", agent=..., outputs=["issues"])
    fixer = AgentSpec(name="SolutionProvider", agent=...,
                      parameter_mappings={"issues": "IssueDetector.issues"},
                      outputs=["suggestions"])

    # Run with profiling
    conductor = Conductor(actors=[detector, fixer], config=config)
    result = await conductor.run(
        goal="Analyze and fix code",
        code="def foo(): pass"
    )

    # Profiling reports automatically saved to:
    # ./outputs/run_YYYYMMDD_HHMMSS/profiling/

asyncio.run(main())
```

## Benefits

✅ **Visual Understanding** - See execution timeline at a glance
✅ **Data-Driven Optimization** - Make decisions based on actual measurements
✅ **Multiple Formats** - Text, Markdown, Mermaid, JSON for any use case
✅ **AI Recommendations** - Get optimization suggestions automatically
✅ **Historical Tracking** - Compare runs over time using JSON data
✅ **Easy Integration** - Just set `enable_profiling=True`

## Next Steps

1. Run your multi-agent system with profiling enabled
2. Review the generated reports in `./outputs/run_*/profiling/`
3. Open `profiling_report.md` in VS Code to see Gantt chart rendered
4. Read optimization recommendations
5. Make improvements based on data
6. Profile again to measure improvement

---

**Questions or Issues?**

See `LOGGING_AND_PROFILING.md` for complete documentation.
