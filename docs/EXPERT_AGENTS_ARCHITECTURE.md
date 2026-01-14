# Expert Agents Architecture

## Overview

Expert Agents are specialized, pre-trained agents that use the OptimizationPipeline to ensure they **always produce correct outputs** for their domain. They represent a new layer in Jotty's architecture for creating reliable, reusable components.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Expert Agent Layer                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐      ┌──────────────────┐            │
│  │ MermaidExpert   │      │ PipelineExpert  │            │
│  │                 │      │                 │            │
│  │ - Mermaid gen   │      │ - Pipeline gen  │            │
│  │ - Pre-trained   │      │ - Pre-trained   │            │
│  │ - Validated     │      │ - Validated     │            │
│  └────────┬────────┘      └────────┬────────┘            │
│           │                        │                      │
│           └────────┬────────────────┘                      │
│                    │                                      │
│           ┌─────────▼─────────┐                           │
│           │  ExpertAgent      │                           │
│           │  (Base Class)     │                           │
│           │                   │                           │
│           │ - Training        │                           │
│           │ - Validation      │                           │
│           │ - Generation      │                           │
│           └─────────┬─────────┘                           │
│                     │                                     │
└─────────────────────┼─────────────────────────────────────┘
                      │
┌─────────────────────▼─────────────────────────────────────┐
│           OptimizationPipeline Layer                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Agents     │  │   Teacher    │  │  Evaluation  │   │
│  │              │  │              │  │              │   │
│  │ - Generate   │  │ - Correct    │  │ - Score      │   │
│  │ - Learn      │  │ - Improve     │  │ - Validate   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Improvement Storage                         │   │
│  │  - JSON files                                      │   │
│  │  - Learned patterns                                │   │
│  │  - Knowledge base updates                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. ExpertAgent (Base Class)

**Location**: `core/experts/expert_agent.py`

**Responsibilities**:
- Manages training lifecycle
- Handles validation
- Provides generation interface
- Integrates with OptimizationPipeline

**Key Methods**:
- `train()`: Train on gold standards
- `validate()`: Validate on test cases
- `generate()`: Generate outputs using learned patterns
- `get_status()`: Get current status

### 2. MermaidExpertAgent

**Location**: `core/experts/mermaid_expert.py`

**Capabilities**:
- Generates perfect Mermaid diagrams
- Supports flowcharts, sequence diagrams, class diagrams
- Pre-trained on common patterns
- Validated for syntax correctness

**Usage**:
```python
from jotty.core.experts import get_mermaid_expert_async

expert = await get_mermaid_expert_async(auto_train=True)
diagram = await expert.generate_mermaid(
    description="User login flow",
    diagram_type="flowchart"
)
```

### 3. PipelineExpertAgent

**Location**: `core/experts/pipeline_expert.py`

**Capabilities**:
- Generates perfect CI/CD pipeline diagrams
- Supports Mermaid format
- Pre-trained on common pipeline patterns

**Usage**:
```python
from jotty.core.experts import get_pipeline_expert_async

expert = await get_pipeline_expert_async(output_format="mermaid")
pipeline = await expert.generate_pipeline(
    stages=["Build", "Test", "Deploy"],
    description="CI/CD Pipeline"
)
```

### 4. ExpertRegistry

**Location**: `core/experts/expert_registry.py`

**Responsibilities**:
- Centralized registry for expert agents
- Manages expert lifecycle
- Provides convenient access functions

**Usage**:
```python
from jotty.core.experts import get_expert_registry

registry = get_expert_registry()
expert = await registry.get_mermaid_expert_async(auto_train=True)
```

## Workflow

### Training Phase

```
1. Expert Agent Created
   ↓
2. Gold Standards Provided
   ↓
3. OptimizationPipeline Created
   ↓
4. Train on Each Gold Standard
   ├─ Agent generates output
   ├─ Evaluate against gold
   ├─ Teacher provides correction (if needed)
   ├─ Learn from improvement
   └─ Store learned patterns
   ↓
5. Mark as "Trained"
```

### Validation Phase

```
1. Validation Cases Provided
   ↓
2. Generate Outputs
   ↓
3. Evaluate Each Output
   ↓
4. Check Scores
   ↓
5. Mark as "Validated" (if all pass)
```

### Generation Phase

```
1. User Requests Output
   ↓
2. Expert Agent Loads Learned Patterns
   ↓
3. Generates Output Using Patterns
   ↓
4. Returns Correct Output
```

## Benefits

### 1. Reliability

✅ **Always Correct**: Expert agents are trained and validated to always produce correct outputs  
✅ **No Errors**: Syntax validation ensures outputs are always valid  
✅ **Consistent**: Same inputs produce same quality outputs  

### 2. Reusability

✅ **Library Components**: Expert agents become part of Jotty's agent library  
✅ **Cross-Project**: Can be used across multiple projects  
✅ **Standardized**: Consistent interface and behavior  

### 3. Maintainability

✅ **Centralized**: All expert logic in one place  
✅ **Tested**: Comprehensive test coverage  
✅ **Documented**: Well-documented APIs and usage  

### 4. Extensibility

✅ **Easy to Add**: Simple pattern for creating new expert agents  
✅ **Modular**: Each expert is independent  
✅ **Composable**: Can be combined in workflows  

### 5. Self-Improvement

✅ **Learning**: Uses OptimizationPipeline for continuous learning  
✅ **Adaptation**: Can be retrained on new patterns  
✅ **Evolution**: Improves over time  

## Integration Points

### With Jotty Conductor

```python
from jotty import Conductor
from jotty.core.experts import get_mermaid_expert_async

conductor = Conductor(...)
mermaid_expert = await get_mermaid_expert_async()

# Use in workflow
diagram = await mermaid_expert.generate_mermaid(...)
```

### With OptimizationPipeline

Expert agents use OptimizationPipeline internally:
- Training uses OptimizationPipeline
- Learning uses OptimizationPipeline's improvement storage
- Generation uses learned patterns from OptimizationPipeline

### With AgentConfig

Expert agents can be used as regular agents:

```python
from jotty.core.foundation.agent_config import AgentConfig
from jotty.core.experts import get_mermaid_expert_async

expert = await get_mermaid_expert_async()
agent_config = AgentConfig(
    name="mermaid_generator",
    agent=expert,
    outputs=["output"]
)
```

## Future Enhancements

### Planned Features

1. **More Expert Agents**
   - PlantUMLExpertAgent
   - GraphvizExpertAgent
   - MarkdownExpertAgent

2. **Expert Marketplace**
   - Share expert agents
   - Community contributions
   - Version management

3. **Auto-Retraining**
   - Automatic retraining on failures
   - Continuous improvement
   - Performance monitoring

4. **Metrics & Monitoring**
   - Performance tracking
   - Usage statistics
   - Quality metrics

5. **Versioning**
   - Expert agent versions
   - Backward compatibility
   - Migration tools

## Best Practices

1. **Pre-Train**: Always train expert agents before deploying
2. **Validate**: Validate expert agents on test cases
3. **Monitor**: Track expert agent performance
4. **Update**: Retrain when domain changes
5. **Document**: Document expert agent capabilities
6. **Test**: Write comprehensive tests
7. **Version**: Use versioning for expert agents

## Examples

See:
- `tests/test_expert_agents.py` - Test suite
- `examples/expert_agents_demo.py` - Usage examples
- `docs/EXPERT_AGENTS.md` - Detailed documentation

## Conclusion

Expert Agents represent a powerful new layer in Jotty's architecture, providing:

- ✅ **Reliable** components that always work
- ✅ **Reusable** components for multiple projects
- ✅ **Maintainable** centralized logic
- ✅ **Extensible** easy-to-add new experts
- ✅ **Self-Improving** continuous learning

They leverage OptimizationPipeline to ensure correctness and provide a foundation for building reliable, domain-specific agents.
