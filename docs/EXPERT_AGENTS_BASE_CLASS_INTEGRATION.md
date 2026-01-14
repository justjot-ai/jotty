# Expert Agents - Base Class DSPy Integration

## Overview

DSPy support has been **integrated into the base `ExpertAgent` class**, so all expert agents automatically benefit from DSPy/LLM capabilities without extra work.

## Changes Made

### 1. Base Class (`ExpertAgent`) - DSPy Integration

Added helper methods to base class:

#### `_is_dspy_module(agent)` 
Detects if an agent is a DSPy module.

#### `_call_dspy_agent(agent, **kwargs)`
Calls DSPy modules correctly (using `agent(**kwargs)` not `agent.forward()`).

#### `_extract_dspy_output(result)`
Extracts output from DSPy Predictions or regular results.

#### `_create_default_teacher()`
Creates DSPy-based teacher if available, falls back to regular teacher.

### 2. OptimizationPipeline - DSPy Support

Updated to handle DSPy modules correctly:
- Calls DSPy modules with `agent(**inputs)` (not `.forward()`)
- Extracts outputs from DSPy Predictions
- Works seamlessly with both DSPy and regular agents

### 3. Expert Implementations - Simplified

Expert agents (like `MermaidExpertAgent`) now:
- Use base class helpers automatically
- Don't need custom DSPy handling code
- Benefit from base class improvements

## Benefits

### ✅ Automatic DSPy Support

All expert agents automatically:
- Detect DSPy modules
- Call them correctly
- Extract outputs properly

### ✅ Less Code in Experts

Expert implementations are simpler:
```python
class MyExpert(ExpertAgent):
    def _create_default_agent(self):
        # Just create DSPy module - base class handles the rest!
        return dspy.ChainOfThought(MySignature)
```

### ✅ Consistent Behavior

All experts behave the same way:
- Same DSPy handling
- Same output extraction
- Same teacher creation

## Architecture

```
ExpertAgent (Base Class)
├── _is_dspy_module()          ← Detects DSPy modules
├── _call_dspy_agent()         ← Calls DSPy correctly
├── _extract_dspy_output()     ← Extracts outputs
└── _create_default_teacher()  ← Creates DSPy teacher

    ↓ (All experts inherit)

MermaidExpertAgent
PipelineExpertAgent
[Other Experts]
    ↓ (All use base class helpers automatically)
```

## Usage

### Creating a New Expert

```python
from core.experts import ExpertAgent, ExpertAgentConfig
import dspy

class MyExpert(ExpertAgent):
    def _create_default_agent(self):
        # Create DSPy signature
        class MySignature(dspy.Signature):
            task: str = dspy.InputField(...)
            output: str = dspy.OutputField(...)
        
        # Return DSPy module - base class handles everything!
        return dspy.ChainOfThought(MySignature)

# That's it! Base class handles:
# - DSPy detection
# - Correct calling
# - Output extraction
# - Teacher creation
```

### No Extra Code Needed

The base class automatically:
- ✅ Detects DSPy modules
- ✅ Calls them correctly
- ✅ Extracts outputs
- ✅ Creates DSPy teachers

## Testing

### Base Class Tests

```bash
python tests/test_expert_base_dspy_integration.py
```

Tests:
- ✅ DSPy module detection
- ✅ Output extraction
- ✅ Teacher creation

### Integration Tests

```bash
python tests/test_expert_optimization_pipeline_integration.py
```

Tests:
- ✅ Expert + OptimizationPipeline integration
- ✅ DSPy agents in pipeline
- ✅ Output extraction from pipeline

## Files Changed

### Base Classes

1. **`core/experts/expert_agent.py`**
   - Added `_is_dspy_module()`
   - Added `_call_dspy_agent()`
   - Added `_extract_dspy_output()`
   - Updated `_create_default_teacher()` to use DSPy
   - Updated `generate()` to use base class helpers

2. **`core/orchestration/optimization_pipeline.py`**
   - Updated `_run_agent_pipeline()` to handle DSPy modules
   - Updated `_call_teacher_model()` to handle DSPy modules
   - Updated `_update_knowledge_base()` to handle DSPy modules
   - Updated `_extract_agent_output()` to prioritize DSPy Predictions

### Expert Implementations

3. **`core/experts/mermaid_expert.py`**
   - Simplified to use base class helpers
   - Teacher creation falls back to base class

## Migration Guide

### Before (Manual DSPy Handling)

```python
class MyExpert(ExpertAgent):
    def generate(self, task, context):
        agent = self._create_agents()[0].agent
        
        # Manual DSPy handling
        if isinstance(agent, dspy.Module):
            result = agent(**inputs)
            output = result.output
        else:
            result = agent.forward(**inputs)
            output = result._store.get('output')
        
        return output
```

### After (Automatic via Base Class)

```python
class MyExpert(ExpertAgent):
    def generate(self, task, context):
        # Base class handles everything!
        return await super().generate(task, context)
```

## Conclusion

✅ **DSPy support is now in base classes**  
✅ **All experts benefit automatically**  
✅ **Less code needed in expert implementations**  
✅ **Consistent behavior across all experts**  
✅ **Fully tested and working**

**Expert agents are now simpler and more powerful!** 🎉
