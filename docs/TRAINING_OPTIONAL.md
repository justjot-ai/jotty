# Expert Training is Optional

## Key Insight

**Training is OPTIONAL** - any LLM can generate Mermaid diagrams (or any output) without training.

**Training is for OPTIMIZATION** - it improves quality, correctness, and learns patterns via OptimizationPipeline.

## Architecture

```
Expert Agent
    ↓
Can generate WITHOUT training (base LLM)
    ↓
Training (OPTIONAL) → OptimizationPipeline
    ↓
Learns patterns → Better quality & correctness
```

## Usage

### Without Training (Base LLM)

```python
from Jotty.core.experts import MermaidExpertAgent

expert = MermaidExpertAgent()
# No training needed!

result = await expert.generate(
    task="Create a flowchart",
    context={"description": "A simple flow"}
)
# Uses base LLM - works immediately
```

### With Training (Optimized)

```python
from Jotty.core.experts import MermaidExpertAgent

expert = MermaidExpertAgent()

# Training is OPTIONAL - for optimization
gold_standards = [
    {
        "task": "Create flowchart",
        "context": {"description": "Simple flow"},
        "gold_standard": "graph TD; Start-->End"
    }
]
await expert.train(gold_standards=gold_standards)

# Now generates with learned patterns & optimizations
result = await expert.generate(
    task="Create a flowchart",
    context={"description": "A simple flow"}
)
# Uses optimized agent - better quality & correctness
```

## What Training Does

Training via `OptimizationPipeline`:

1. **Learns from mistakes** - Uses teacher model when evaluation fails
2. **Stores improvements** - Saves learned patterns to memory
3. **Optimizes prompts** - Updates DSPy module instructions
4. **Validates correctness** - Domain-specific validation
5. **Credit assignment** - Prioritizes most effective improvements

## Benefits of Training

- ✅ **Better quality** - Learned patterns improve output
- ✅ **Correctness** - Domain-specific validation ensures valid output
- ✅ **Consistency** - Learned patterns applied consistently
- ✅ **Efficiency** - Fewer iterations needed for correct output

## Without Training

- ✅ **Works immediately** - Base LLM can generate
- ⚠️ **May need validation** - Output may not always be correct
- ⚠️ **No learned patterns** - Doesn't benefit from past improvements

## Conclusion

**Training is OPTIONAL** - experts work without it, but training optimizes them for better quality and correctness.

The expert architecture supports both:
- **Untrained**: Base LLM generation (works immediately)
- **Trained**: Optimized generation (better quality via OptimizationPipeline)
