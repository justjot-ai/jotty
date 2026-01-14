# Gold Standard: What It Is and How It's Used

## What is `gold_standard`?

**`gold_standard`** is the **correct/expected output** for a given task. It's called "gold" because it represents the ideal, perfect answer that the expert should learn to produce.

---

## How `gold_standard` is Used

### 1. **Training** (Primary Use)
**Location**: `ExpertAgent.train()` → `OptimizationPipeline.optimize()`

**Process**:
1. Expert generates output from `task`
2. Output is **evaluated against `gold_standard`**
3. If output doesn't match → teacher model provides correction
4. Expert learns from the difference
5. Improvements are stored

**Purpose**: Teach expert what the correct output should be

**Example**:
```python
gold_standards = [
    {
        "task": "Generate PlantUML sequence diagram",
        "context": {"description": "User login flow"},
        "gold_standard": "@startuml\nUser -> System: Login\n@enduml"  # ← Correct output
    }
]
```

---

### 2. **Evaluation/Scoring**
**Location**: `OptimizationPipeline._evaluate_output()`

**Process**:
- Compare generated output with `gold_standard`
- Calculate similarity score (0.0 to 1.0)
- Determine if output is correct

**Purpose**: Measure how close output is to correct answer

---

### 3. **Teacher Model Input**
**Location**: `OptimizationPipeline._run_teacher_model()`

**Process**:
- Teacher receives `gold_standard` as input
- Teacher should return `gold_standard` exactly
- Used to correct student's mistakes

**Purpose**: Provide correct answer when student fails

---

### 4. **Validation**
**Location**: `ExpertAgent.validate()`

**Process**:
- Generate output for validation case
- Compare against `gold_standard`
- Calculate pass/fail score

**Purpose**: Verify expert performance after training

---

## Why "Gold Standard"?

The term comes from:
- **Gold** = Highest quality, perfect standard
- **Standard** = Benchmark to measure against

In machine learning, "gold standard" refers to the ground truth - the correct answer that all outputs are compared to.

---

## Format

**In Training Cases**:
```python
{
    "task": "Generate diagram",
    "context": {"description": "..."},
    "gold_standard": "correct_output_code_here"  # ← The correct answer
}
```

**In Optimization Pipeline**:
```python
await pipeline.optimize(
    task="Generate diagram",
    context={...},
    gold_standard="correct_output_code_here"  # ← Used for evaluation
)
```

---

## Summary

**`gold_standard`** = **The correct answer** that expert should learn to produce.

**Used for**:
1. ✅ Training (learning from mistakes)
2. ✅ Evaluation (scoring outputs)
3. ✅ Teacher input (providing corrections)
4. ✅ Validation (verifying performance)

**Not used for**:
- ❌ Pre-training pattern extraction (uses `code` field instead)
- ❌ Few-shot examples (could be, but not currently)

---

## Current Implementation

**Training Flow**:
1. Expert generates output from `task`
2. Output compared to `gold_standard` → Score calculated
3. If score < 1.0 → Teacher provides `gold_standard` as correction
4. Expert learns: "For this task, output should be `gold_standard`"
5. Improvement stored: "When task is X, use format Y"

**Key Point**: `gold_standard` is the **target** that expert learns to match.
