# Gold Standards Consolidation

## Problem

We were creating a separate `pre_training_examples` parameter when `gold_standards` can serve both purposes.

## Solution: Use `gold_standards` for Both

### Unified Approach

**Single Parameter**: `gold_standards` serves multiple purposes:

1. **Pre-Training** (Pattern Extraction)
   - Extract patterns from `gold_standards` before iterative learning
   - Controlled by `enable_pre_training=True`

2. **Iterative Learning** (Fine-Tuning)
   - Learn from mistakes via optimization pipeline
   - Controlled by `training_mode="iterative"` or `"both"`

3. **Validation**
   - Verify expert performance after training
   - Uses same format

4. **Few-Shot Learning**
   - Use as examples in context for generation
   - Can include in prompt

5. **Template Learning**
   - Extract common structures/patterns
   - Learn reusable templates

6. **Domain Adaptation**
   - Adapt to new domains using examples
   - Fine-tune on domain-specific examples

---

## Updated API

### Before (Separate Parameters):
```python
await expert.train(
    gold_standards=[...],  # For iterative learning
    pre_training_examples=[...]  # For pattern extraction
)
```

### After (Unified):
```python
await expert.train(
    gold_standards=[...],  # Used for both pre-training and iterative learning
    enable_pre_training=True,  # Extract patterns first
    training_mode="both"  # "iterative", "pattern_extraction", or "both"
)
```

---

## Implementation

### Conversion Function

`_pre_train_from_gold_standards()` converts gold_standards format to examples format:

```python
# Gold standards format
{
    "task": "Generate flowchart",
    "context": {"description": "Simple flow"},
    "gold_standard": "graph TD\nA --> B"
}

# Converts to examples format
{
    "code": "graph TD\nA --> B",
    "description": "Simple flow",
    "type": "flowchart",
    "source": "gold_standard",
    "task": "Generate flowchart"
}
```

---

## Benefits

1. ✅ **Single Source of Truth**: One parameter, one format
2. ✅ **Flexible**: Can use same examples for both purposes
3. ✅ **Simpler API**: No need for separate `pre_training_examples`
4. ✅ **Backward Compatible**: Existing code still works
5. ✅ **Multiple Use Cases**: Supports 6+ use cases

---

## Use Cases Summary

| Use Case | How `gold_standards` is Used |
|----------|------------------------------|
| **Pre-Training** | Extract patterns before iterative learning |
| **Iterative Learning** | Learn from mistakes via optimization pipeline |
| **Validation** | Verify expert performance |
| **Few-Shot Learning** | Use as examples in context |
| **Template Learning** | Extract common structures |
| **Domain Adaptation** | Adapt to new domains |

---

## Conclusion

**Consolidated**: `gold_standards` now serves all purposes, eliminating need for separate `pre_training_examples` parameter.

**Benefits**: Simpler API, single source of truth, flexible training modes.
