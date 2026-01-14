# Optimization Pipeline

## Overview

The `OptimizationPipeline` is a generic, domain-agnostic framework for iterative optimization with evaluation and learning. It works with **any domain** - markdown generation, Mermaid diagrams, PlantUML diagrams, code generation, documentation, or any other use case.

## Key Features

1. **Multi-Agent Orchestration**: Run multiple agents in sequence to solve complex tasks
2. **Iterative Optimization**: Automatically retry and improve until success or max iterations
3. **Gold Standard Evaluation**: Compare outputs against expected results
4. **Teacher Model Fallback**: Use a teacher agent when evaluation fails
5. **Knowledge Base Updates**: Learn from mistakes and update metadata
6. **Thinking Log**: Detailed logging of the optimization process
7. **Conductor Integration**: Works seamlessly with Jotty Conductor for advanced orchestration

## Architecture

```
OptimizationPipeline
├── Agent Pipeline (sequential execution)
│   ├── Agent 1 → Output 1
│   ├── Agent 2 → Output 2 (uses Output 1)
│   └── Agent N → Final Output
├── Evaluation
│   └── Compare against Gold Standard
├── Teacher Model (if evaluation fails)
│   └── Generate improved output
├── KB Updates (if teacher succeeds)
│   └── Learn from differences
└── Iteration Loop
    └── Repeat until success or max iterations
```

## Basic Usage

```python
from jotty.core.jotty import (
    OptimizationPipeline,
    OptimizationConfig,
    AgentConfig,
    create_optimization_pipeline
)

# Define your agents
agents = [
    AgentConfig(
        name="agent1",
        agent=MyAgent1(),
        outputs=["result"]
    ),
    AgentConfig(
        name="agent2",
        agent=MyAgent2(),
        parameter_mappings={"input": "result"}
    )
]

# Create pipeline
pipeline = create_optimization_pipeline(
    agents=agents,
    max_iterations=5,
    required_pass_count=2,
    output_path="./outputs"
)

# Define evaluation function
def evaluate(output, gold_standard, task, context):
    return {
        "score": 1.0 if output == gold_standard else 0.0,
        "status": "CORRECT" if output == gold_standard else "INCORRECT"
    }

pipeline.config.evaluation_function = evaluate

# Run optimization
result = await pipeline.optimize(
    task="Your task description",
    context={"additional": "context"},
    gold_standard="Expected output"
)
```

## Configuration Options

### OptimizationConfig

- `max_iterations` (int, default=5): Maximum number of optimization iterations
- `required_pass_count` (int, default=2): Number of consecutive successful evaluations required to stop
- `enable_teacher_model` (bool, default=True): Enable teacher model fallback
- `enable_kb_updates` (bool, default=True): Enable knowledge base updates
- `kb_update_requires_teacher` (bool, default=True): Only update KB if teacher was used
- `evaluation_function` (Callable, optional): Custom evaluation function
- `gold_standard_provider` (Callable, optional): Function to dynamically get gold standards
- `output_path` (Path, optional): Path for output files
- `thinking_log_path` (Path, optional): Path for thinking log file
- `enable_thinking_log` (bool, default=True): Enable thinking log

## Agent Configuration

Agents are configured using `AgentConfig`:

```python
AgentConfig(
    name="agent_name",
    agent=YourDSPyModule(),
    outputs=["output_field"],  # Fields this agent produces
    parameter_mappings={"param": "context_key"},  # Map parameters to context
    metadata={"is_teacher": True}  # Special metadata (teacher, KB updater)
)
```

### Special Agent Types

1. **Teacher Agent**: Set `metadata={"is_teacher": True}` or name contains "teacher"
   - Called when evaluation fails
   - Receives: `task`, `student_output`, `gold_standard`, `evaluation_feedback`

2. **KB Update Agent**: Set `metadata={"is_kb_updater": True}` or name contains "kb"/"knowledge"
   - Called after teacher succeeds
   - Receives: `student_output`, `teacher_output`, `evaluation_result`

## Evaluation Functions

Custom evaluation functions should have this signature:

```python
async def evaluate(
    output: Any,
    gold_standard: Any,
    task: str,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Returns:
        {
            "score": float (0.0-1.0),
            "status": str ("CORRECT", "INCORRECT", "ERROR"),
            "error_info": str (optional),
            "difference": str (optional)
        }
    """
    # Your evaluation logic
    return {"score": 1.0, "status": "CORRECT"}
```

## Integration with Jotty Conductor

The pipeline can use Jotty Conductor for advanced orchestration:

```python
from jotty.core.jotty import create_conductor

# Create Conductor
conductor = create_conductor(
    agents=agents,
    config=JottyConfig()
)

# Create pipeline with Conductor
pipeline = OptimizationPipeline(
    agents=agents,
    config=config,
    conductor=conductor  # Use Conductor for orchestration
)
```

When a Conductor is provided, the pipeline uses it for agent execution, benefiting from:
- Architect/Auditor validation
- Learning and memory
- Retry mechanisms
- Credit assignment

## Iteration Results

Each iteration returns an `IterationResult`:

```python
@dataclass
class IterationResult:
    iteration: int
    success: bool
    evaluation_score: float
    evaluation_status: str
    output: Any
    metadata: Dict[str, Any]
    teacher_output: Optional[Any]
    kb_updates: Optional[Dict[str, Any]]
    error: Optional[str]
```

## Final Result Structure

```python
{
    "status": "completed" | "stopped",
    "total_iterations": int,
    "consecutive_passes": int,
    "required_pass_count": int,
    "max_iterations": int,
    "optimization_complete": bool,
    "iterations": [
        {
            "iteration": int,
            "success": bool,
            "evaluation_score": float,
            "evaluation_status": str,
            "has_teacher_output": bool,
            "has_kb_updates": bool,
            "error": Optional[str]
        }
    ],
    "final_result": {
        "iteration": int,
        "output": Any,
        "evaluation_score": float,
        "evaluation_status": str
    }
}
```

## Thinking Log

The pipeline maintains a thinking log at `output_path/thinking.log` (if enabled):

```
[2025-01-15 10:30:45.123] Starting optimization for task: Generate markdown documentation
[2025-01-15 10:30:45.456] === Iteration 1/5 ===
[2025-01-15 10:30:46.789] Executing agent 1/3: ContentAnalyzer
[2025-01-15 10:30:47.012] Agent ContentAnalyzer completed successfully
[2025-01-15 10:30:48.345] Evaluating output against gold standard...
[2025-01-15 10:30:48.678] Iteration 1: Evaluation FAILED (score=0.00, status=INCORRECT)
[2025-01-15 10:30:49.012] Evaluation failed, calling teacher model...
```

## Use Cases

The OptimizationPipeline works for **any domain**:

- **Markdown Generation**: Optimize documentation, README files, API docs
- **Mermaid Diagrams**: Generate and refine flowcharts, sequence diagrams, class diagrams
- **PlantUML Diagrams**: Create UML diagrams, architecture diagrams
- **Code Generation**: Generate functions, classes, modules with iterative improvement
- **Content Creation**: Blog posts, articles, documentation
- **Data Transformation**: ETL pipelines, data cleaning, format conversion
- **Any Custom Domain**: Just provide your agents and evaluation function!

## Examples by Domain

### Markdown Generation
```python
# Generate markdown documentation
result = await pipeline.optimize(
    task="Generate API documentation",
    context={"endpoints": [...]},
    gold_standard="# API Reference\n\n..."
)
```

### Mermaid Diagram
```python
# Generate Mermaid flowchart
result = await pipeline.optimize(
    task="Generate workflow diagram",
    context={"process": "user_login"},
    gold_standard="graph TD\n    A[Start] --> B[End]"
)
```

### PlantUML Diagram
```python
# Generate PlantUML class diagram
result = await pipeline.optimize(
    task="Generate class diagram",
    context={"classes": [...]},
    gold_standard="@startuml\nclass User\n@enduml"
)
```

## Migration Guide

To adapt OptimizationPipeline for your domain:

1. **Define your agents** using AgentConfig:
   ```python
   agents = [
       AgentConfig(
           name="analyzer",
           agent=YourAnalyzerAgent(...),
           outputs=["analysis"]
       ),
       AgentConfig(
           name="generator",
           agent=YourGeneratorAgent(...),
           parameter_mappings={"input": "analysis"}
       )
   ]
   ```

2. **Create evaluation function** for your domain:
   ```python
   async def evaluate(output, gold_standard, task, context):
       # Your domain-specific evaluation logic
       # For markdown: check structure, formatting
       # For diagrams: validate syntax, check elements
       # For code: syntax check, test execution
       return {"score": 1.0, "status": "CORRECT"}
   ```

3. **Add teacher agent** (optional) for learning:
   ```python
   AgentConfig(
       name="teacher",
       agent=YourTeacherAgent(...),
       metadata={"is_teacher": True}
   )
   ```

4. **Create and run pipeline**:
   ```python
   pipeline = create_optimization_pipeline(agents, ...)
   pipeline.config.evaluation_function = evaluate
   result = await pipeline.optimize(task, context, gold_standard)
   ```

## Examples

See `examples/optimization_pipeline_example.py` for complete examples:
- Simple optimization
- With teacher model
- With KB updates
- With Conductor integration

## Best Practices

1. **Evaluation Function**: Make it robust and handle edge cases
2. **Gold Standards**: Provide clear, unambiguous gold standards
3. **Agent Order**: Order agents logically (dependencies first)
4. **Parameter Mappings**: Use parameter_mappings to connect agent outputs
5. **Thinking Log**: Enable for debugging complex optimization flows
6. **Max Iterations**: Set appropriately based on task complexity
7. **Required Passes**: Use 2+ for stability, 1 for quick iterations

## Troubleshooting

**Problem**: Optimization never completes
- Check evaluation function returns correct format
- Verify gold standard is achievable
- Check agent outputs are correct format

**Problem**: Teacher model not called
- Verify `enable_teacher_model=True`
- Check teacher agent is properly configured
- Ensure evaluation actually fails

**Problem**: KB updates not happening
- Verify `enable_kb_updates=True`
- Check KB update agent is configured
- Ensure teacher model succeeds first (if `kb_update_requires_teacher=True`)

## Future Enhancements

- [ ] Parallel agent execution
- [ ] Dynamic agent selection
- [ ] Multi-objective optimization
- [ ] Adaptive iteration limits
- [ ] Distributed optimization
- [ ] Visualization tools
