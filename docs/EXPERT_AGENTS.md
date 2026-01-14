# Expert Agents in Jotty

## Overview

Expert Agents are specialized, pre-trained agents that use the OptimizationPipeline to ensure they **always produce correct outputs** for their domain. They are reusable components that become part of Jotty's agent library.

## Key Features

✅ **Self-Optimizing**: Use OptimizationPipeline internally for self-improvement  
✅ **Pre-Trained**: Can be trained on gold standards before use  
✅ **Validated**: Ensured to work without error  
✅ **Reusable**: Part of Jotty's agent library  
✅ **Domain-Specific**: Specialized for specific tasks (Mermaid, Pipeline diagrams, etc.)  

## Architecture

```
ExpertAgent (Base Class)
├── Uses OptimizationPipeline internally
├── Can be trained on gold standards
├── Validated to ensure correctness
└── Generates outputs using learned patterns

Specialized Experts:
├── MermaidExpertAgent: Perfect Mermaid diagrams
├── PipelineExpertAgent: Perfect CI/CD pipeline diagrams
└── [More experts can be added]
```

## Available Expert Agents

### 1. MermaidExpertAgent

Generates perfect Mermaid diagrams (flowcharts, sequence diagrams, class diagrams, etc.).

**Usage:**

```python
from jotty.core.experts import get_mermaid_expert

# Get expert (auto-trains if needed)
expert = get_mermaid_expert()

# Generate a diagram
diagram = await expert.generate_mermaid(
    description="User login flow with validation",
    diagram_type="flowchart"
)

print(diagram)
# Output:
# graph TD
#     A[User Login]
#     B{Valid?}
#     C[Show Dashboard]
#     D[Show Error]
#     A --> B
#     B -->|Yes| C
#     B -->|No| D
```

**Training:**

```python
from jotty.core.experts import MermaidExpertAgent

expert = MermaidExpertAgent()

# Train on gold standards
training_results = await expert.train()

# Validate
validation_results = await expert.validate()

# Check status
status = expert.get_status()
print(f"Trained: {status['trained']}")
print(f"Validated: {status['validation_passed']}")
```

### 2. PipelineExpertAgent

Generates perfect CI/CD pipeline diagrams (Mermaid format).

**Usage:**

```python
from jotty.core.experts import get_pipeline_expert

# Get expert
expert = get_pipeline_expert(output_format="mermaid")

# Generate pipeline diagram
pipeline = await expert.generate_pipeline(
    stages=["Build", "Test", "Deploy", "Release"],
    description="CI/CD Pipeline"
)

print(pipeline)
# Output:
# graph LR
#     A[Build]
#     B[Test]
#     C[Deploy]
#     D[Release]
#     A --> B
#     B --> C
#     C --> D
```

## Creating Custom Expert Agents

### Step 1: Create Expert Agent Class

```python
from jotty.core.experts import ExpertAgent, ExpertAgentConfig

class MyExpertAgent(ExpertAgent):
    def __init__(self, config: Optional[ExpertAgentConfig] = None):
        if config is None:
            config = ExpertAgentConfig(
                name="my_expert",
                domain="my_domain",
                description="Expert for my domain",
                training_gold_standards=self._get_training_cases(),
                evaluation_function=self._evaluate_output,
                agent_module=self._create_agent,
                teacher_module=self._create_teacher
            )
        super().__init__(config)
    
    def _create_default_agent(self):
        # Create your agent
        class MyAgent:
            def forward(self, task=None, **kwargs):
                # Generate output
                result = type('Result', (), {})()
                result._store = {"output": "..."}
                return result
        return MyAgent()
    
    @staticmethod
    async def _evaluate_output(output, gold_standard, task, context):
        # Evaluate output
        return {
            "score": 1.0 if output == gold_standard else 0.0,
            "status": "CORRECT" if output == gold_standard else "INCORRECT"
        }
    
    @staticmethod
    def _get_training_cases():
        return [
            {
                "task": "Task 1",
                "context": {},
                "gold_standard": "Expected output 1"
            }
        ]
```

### Step 2: Register Expert Agent

```python
from jotty.core.experts import ExpertRegistry

registry = ExpertRegistry()
registry.register("my_expert", MyExpertAgent())
```

## How Expert Agents Work

### 1. Training Phase

```
Expert Agent
├── Creates OptimizationPipeline
├── Trains on gold standards
├── Learns from teacher model
├── Stores improvements
└── Marks as "trained"
```

### 2. Validation Phase

```
Expert Agent
├── Tests on validation cases
├── Ensures correctness
└── Marks as "validated"
```

### 3. Generation Phase

```
Expert Agent
├── Uses learned patterns
├── Applies improvements
└── Generates correct output
```

## Benefits

1. **Reliability**: Expert agents always produce correct outputs
2. **Reusability**: Can be used across multiple projects
3. **Maintainability**: Centralized, well-tested components
4. **Extensibility**: Easy to add new expert agents
5. **Self-Improvement**: Uses OptimizationPipeline for continuous learning

## Integration with Jotty

Expert agents can be used in Jotty workflows:

```python
from jotty import Conductor
from jotty.core.experts import get_mermaid_expert

# Create conductor
conductor = Conductor(...)

# Use expert agent in workflow
mermaid_expert = get_mermaid_expert()

# Generate diagram as part of workflow
diagram = await mermaid_expert.generate_mermaid(
    description="Workflow diagram",
    diagram_type="flowchart"
)
```

## Best Practices

1. **Pre-Train**: Train expert agents before deploying
2. **Validate**: Always validate expert agents
3. **Monitor**: Track expert agent performance
4. **Update**: Retrain when domain changes
5. **Document**: Document expert agent capabilities

## Examples

See:
- `tests/test_expert_agents.py` - Test suite
- `examples/expert_agents_demo.py` - Usage examples

## Future Enhancements

- [ ] More expert agents (PlantUML, Graphviz, etc.)
- [ ] Expert agent marketplace
- [ ] Automatic retraining on failures
- [ ] Performance metrics and monitoring
- [ ] Expert agent versioning
