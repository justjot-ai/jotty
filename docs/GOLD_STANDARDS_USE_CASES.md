# Gold Standards: Use Cases and Consolidation

## Answer to Your Question

**Why not use `gold_standards` instead of creating `pre_training_examples`?**

**You're absolutely right!** We've now consolidated to use `gold_standards` for all purposes.

---

## Use Cases for `gold_standards`

### 1. **Iterative Learning** (Primary Use)
**How**: Passed to `OptimizationPipeline.optimize()`
- Generate output from `task`
- Evaluate against `gold_standard`
- Learn from mistakes via teacher model
- Store improvements

**Purpose**: Fine-tuning through iterative optimization

---

### 2. **Pattern Extraction** (Pre-Training)
**How**: Extract patterns before iterative learning
- Convert `gold_standards` format to examples format
- Extract patterns from examples
- Store as initial improvements
- No iterative learning involved

**Purpose**: Pre-training before fine-tuning

**Implementation**: `_pre_train_from_gold_standards()` converts format automatically

---

### 3. **Validation**
**How**: Used in `ExpertAgent.validate()`
- Generate output for each validation case
- Compare against `gold_standard`
- Calculate scores
- Determine if expert passes

**Purpose**: Verify expert performance after training

---

### 4. **Teacher Model Input**
**How**: Passed to teacher agent
- Teacher receives `gold_standard` as input
- Teacher should return `gold_standard` exactly
- Used for correction when student fails

**Purpose**: Provide correct output for learning

---

### 5. **Few-Shot Learning** (Potential)
**How**: Use as examples in context
```python
# Include gold_standards as examples in prompt
examples = gold_standards[:3]  # Use first 3 as examples
context["examples"] = examples
```

**Purpose**: Provide examples for few-shot generation

---

### 6. **Template Learning** (Potential)
**How**: Extract templates from `gold_standards`
```python
# Learn common structures
templates = extract_templates(gold_standards)
```

**Purpose**: Learn reusable templates/patterns

---

### 7. **Domain Adaptation** (Potential)
**How**: Fine-tune on domain-specific examples
```python
# Adapt to new domain
expert.train(gold_standards=domain_examples)
```

**Purpose**: Adapt expert to new domains

---

## Consolidation: Unified Approach

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
    gold_standards=[...],  # Used for BOTH pre-training and iterative learning
    enable_pre_training=True,  # Extract patterns first
    training_mode="both"  # "iterative", "pattern_extraction", or "both"
)
```

---

## Implementation

### Conversion Function

`_pre_train_from_gold_standards()` automatically converts format:

```python
# Input: gold_standards format
{
    "task": "Generate flowchart",
    "context": {"description": "Simple flow"},
    "gold_standard": "graph TD\nA --> B"
}

# Converts to: examples format (internally)
{
    "code": "graph TD\nA --> B",
    "description": "Simple flow",
    "type": "flowchart",
    "source": "gold_standard",
    "task": "Generate flowchart"
}
```

---

## Benefits of Consolidation

1. ✅ **Single Source of Truth**: One parameter, one format
2. ✅ **Flexible**: Can use same examples for both purposes
3. ✅ **Simpler API**: No need for separate `pre_training_examples`
4. ✅ **Backward Compatible**: Existing code still works
5. ✅ **Multiple Use Cases**: Supports 7+ use cases

---

## Summary

**Question**: Why create `pre_training_examples` when `gold_standards` exists?

**Answer**: You're right - we've consolidated! Now `gold_standards` serves:
- ✅ Pre-training (pattern extraction)
- ✅ Iterative learning (fine-tuning)
- ✅ Validation
- ✅ Teacher input
- ✅ Few-shot learning (potential)
- ✅ Template learning (potential)
- ✅ Domain adaptation (potential)

**No need for separate `pre_training_examples` parameter!**
