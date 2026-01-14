# Gold Standards Analysis: Use Cases and Consolidation

## Current Usage of `gold_standards`

### 1. **Fine-Tuning via Optimization Pipeline** (Primary Use)
**Location**: `ExpertAgent.train()`

**Format**:
```python
gold_standards = [
    {
        "task": "Generate flowchart",
        "context": {"description": "..."},
        "gold_standard": "graph TD\nA --> B"
    }
]
```

**Process**:
1. For each gold standard:
   - Generate output from `task`
   - Evaluate against `gold_standard`
   - If fails, use teacher model
   - Learn from mistakes (iterative optimization)
   - Store improvements

**Purpose**: Iterative learning from mistakes

---

### 2. **Validation** (Secondary Use)
**Location**: `ExpertAgent.validate()`

**Format**: Same as training

**Process**:
1. Generate output for each validation case
2. Compare against `gold_standard`
3. Calculate scores
4. Determine if expert passes validation

**Purpose**: Verify expert performance after training

---

### 3. **Teacher Model Input** (Tertiary Use)
**Location**: `OptimizationPipeline._run_teacher_model()`

**Process**:
- Teacher receives `gold_standard` as input
- Teacher should return `gold_standard` exactly
- Used for correction when student fails

**Purpose**: Provide correct output for learning

---

## Current `pre_training_examples` Usage

### Format:
```python
pre_training_examples = [
    {
        "code": "graph TD\nA --> B",
        "description": "Simple flowchart",
        "type": "flowchart",
        "source": "curated"
    }
]
```

### Process:
1. Extract patterns from examples
2. Store as improvements (no iterative learning)
3. No optimization pipeline involved
4. No teacher model involved

**Purpose**: Pattern extraction before fine-tuning

---

## The Problem

**Issue**: We have two separate concepts doing similar things:
- `gold_standards`: Used for iterative learning
- `pre_training_examples`: Used for pattern extraction

**Question**: Why not use `gold_standards` for both?

---

## Use Cases for `gold_standards`

### Current Use Cases:
1. ✅ **Iterative Learning** - Learn from mistakes via optimization pipeline
2. ✅ **Validation** - Verify expert performance
3. ✅ **Teacher Input** - Provide correct output for correction

### Potential Additional Use Cases:
4. **Pre-Training** - Extract patterns before iterative learning
5. **Few-Shot Learning** - Use examples as context for generation
6. **Template Learning** - Learn common structures/patterns
7. **Domain Adaptation** - Adapt to new domains using examples

---

## Consolidation Proposal

### Option 1: Unified Format with Training Mode Flag

**Unified Format**:
```python
gold_standards = [
    {
        "task": "Generate flowchart",
        "context": {"description": "..."},
        "gold_standard": "graph TD\nA --> B",
        "training_mode": "iterative" | "pattern_extraction" | "both"
    }
]
```

**Benefits**:
- Single source of truth
- Flexible training modes
- Can use same examples for both purposes

**Drawbacks**:
- More complex logic
- Need to handle different modes

---

### Option 2: Use `gold_standards` for Pre-Training Too

**Approach**: Use `gold_standards` for both, with optional flag

```python
async def train(
    self,
    gold_standards: Optional[List[Dict[str, Any]]] = None,
    force_retrain: bool = False,
    enable_pre_training: bool = True,  # Extract patterns first
    training_mode: str = "both"  # "iterative", "pattern_extraction", "both"
):
    # Phase 1: Pre-training (if enabled)
    if enable_pre_training and gold_standards:
        # Extract patterns from gold_standards
        pre_training_results = await self._pre_train_from_gold_standards(gold_standards)
    
    # Phase 2: Iterative learning (if enabled)
    if training_mode in ["iterative", "both"]:
        # Use optimization pipeline
        ...
```

**Benefits**:
- No separate parameter needed
- Can reuse same examples
- Simpler API

**Drawbacks**:
- Need to convert format (gold_standards → pre_training format)

---

### Option 3: Keep Separate but Use Same Source

**Approach**: Keep separate but allow `pre_training_examples` to be derived from `gold_standards`

```python
async def train(
    self,
    gold_standards: Optional[List[Dict[str, Any]]] = None,
    force_retrain: bool = False,
    enable_pre_training: bool = True,
    pre_training_examples: Optional[List[Dict[str, Any]]] = None
):
    # If pre_training_examples not provided, derive from gold_standards
    if enable_pre_training and not pre_training_examples and gold_standards:
        pre_training_examples = self._convert_gold_standards_to_examples(gold_standards)
    
    # Use pre_training_examples for pre-training
    # Use gold_standards for iterative learning
```

**Benefits**:
- Clear separation of concerns
- Can use different examples for each phase
- Flexible

**Drawbacks**:
- Still two concepts
- Need conversion function

---

## Recommendation: **Option 2** (Use `gold_standards` for Both)

**Rationale**:
1. **Single Source of Truth**: One parameter, one format
2. **Flexible**: Can use same examples for both purposes
3. **Simpler API**: No need for separate `pre_training_examples`
4. **Backward Compatible**: Existing code still works

**Implementation**:
- Add `training_mode` parameter to control behavior
- Extract patterns from `gold_standards` if `enable_pre_training=True`
- Use `gold_standards` for iterative learning if `training_mode` includes "iterative"

---

## Other Use Cases for `gold_standards`

### 1. **Few-Shot Learning**
Use `gold_standards` as examples in context for generation:
```python
# Include gold_standards as examples in prompt
examples = gold_standards[:3]  # Use first 3 as examples
context["examples"] = examples
```

### 2. **Template Learning**
Extract templates from `gold_standards`:
```python
# Learn common structures
templates = extract_templates(gold_standards)
```

### 3. **Domain Adaptation**
Use `gold_standards` to adapt to new domain:
```python
# Fine-tune on domain-specific examples
expert.train(gold_standards=domain_examples)
```

### 4. **Continuous Learning**
Add new `gold_standards` over time:
```python
# Add new examples
expert.train(gold_standards=new_examples, force_retrain=False)
```

---

## Conclusion

**Recommendation**: Consolidate `pre_training_examples` into `gold_standards` with a `training_mode` flag.

**Benefits**:
- Simpler API
- Single source of truth
- Flexible training modes
- Supports multiple use cases

**Implementation**: Use `gold_standards` for both pre-training pattern extraction and iterative learning, controlled by flags.
