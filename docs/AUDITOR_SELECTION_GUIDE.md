# Auditor Selection Guide - For End Users

**Date**: January 27, 2026

---

## Quick Start

### Use the Helper Function

```python
from core.orchestration.auditor_selector import recommend_auditor

# Get recommendation
rec = recommend_auditor(
    use_case="code_generation",
    priority="quality",
    has_multiple_results=True
)

# Apply config
from core.foundation.data_structures import SwarmConfig
config = SwarmConfig(**rec['config'])
```

---

## When to Use Each Auditor Type

### 1. Single Auditor (Default) ✅

**Use when**:
- ✅ You only have **one result** to validate
- ✅ Speed is critical
- ✅ Cost is a concern
- ✅ Quality is acceptable (baseline)

**Example**:
```python
config = SwarmConfig()  # Default: single auditor
```

**Trade-offs**:
- ✅ **Cost**: Lowest
- ✅ **Speed**: Fastest
- ⚠️ **Quality**: Baseline

---

### 2. List-Wise Auditor (Best Quality) ⭐⭐⭐⭐⭐

**Use when**:
- ✅ You have **multiple results** to verify
- ✅ **Quality is critical** (code generation, data extraction)
- ✅ You can afford higher cost
- ✅ Latency is acceptable

**Example**:
```python
config = SwarmConfig(
    enable_list_wise_verification=True,
    list_wise_min_results=3,
    list_wise_max_results=5,
    list_wise_merge_strategy="best_score"
)
```

**Trade-offs**:
- ⚠️ **Cost**: Higher (verifies multiple results)
- ⚠️ **Speed**: Slower (verifies all results)
- ✅ **Quality**: **Best** (OAgents research)

**Best for**:
- Code generation
- Data extraction
- Analysis tasks
- Quality-critical tasks

---

### 3. Pair-Wise Auditor (Balanced) ⭐⭐⭐⭐

**Use when**:
- ✅ You have **multiple results**
- ✅ You want **balanced** quality/speed/cost
- ✅ Cost is a concern (but want better than single)

**Example**:
```python
config = SwarmConfig(auditor_type="pair_wise")
```

**Trade-offs**:
- ⚠️ **Cost**: Medium
- ⚠️ **Speed**: Medium
- ✅ **Quality**: Good (better than single)

**Best for**:
- General tasks
- Balanced approach
- When list-wise is too expensive

---

### 4. Confidence-Based Auditor (Fastest) ⭐⭐⭐

**Use when**:
- ✅ You have **multiple results**
- ✅ **Speed is critical**
- ✅ Results have confidence scores
- ✅ Quality is acceptable

**Example**:
```python
config = SwarmConfig(auditor_type="confidence_based")
```

**Trade-offs**:
- ✅ **Cost**: Low (no verification calls)
- ✅ **Speed**: Fastest
- ⚠️ **Quality**: Good (if confidence scores accurate)

**Best for**:
- Creative writing
- General tasks
- Speed-critical tasks

---

## Decision Tree

```
Do you have multiple results?
│
├─ NO → Use Single Auditor (default)
│
└─ YES → What's most important?
    │
    ├─ Quality → Use List-Wise Auditor
    │   └─ Best quality (OAgents research)
    │
    ├─ Speed → Use Confidence-Based Auditor
    │   └─ Fastest selection
    │
    ├─ Cost → Use Pair-Wise Auditor
    │   └─ Balanced cost/quality
    │
    └─ Balanced → Use List-Wise Auditor
        └─ Best overall quality
```

---

## Use Case Recommendations

### Code Generation

**Recommendation**: **List-Wise Auditor**

**Why**: Quality is critical for code. List-wise verification ensures best result.

```python
rec = recommend_auditor(
    use_case="code_generation",
    priority="quality",
    has_multiple_results=True
)
config = SwarmConfig(**rec['config'])
```

**Config**:
- `enable_list_wise_verification=True`
- `list_wise_min_results=3`
- `list_wise_max_results=5`
- `list_wise_merge_strategy="best_score"`

---

### Data Extraction

**Recommendation**: **List-Wise Auditor**

**Why**: Accuracy is critical. List-wise verification with consensus strategy.

```python
rec = recommend_auditor(
    use_case="data_extraction",
    priority="quality",
    has_multiple_results=True
)
config = SwarmConfig(**rec['config'])
```

**Config**:
- `enable_list_wise_verification=True`
- `list_wise_merge_strategy="consensus"`

---

### Creative Writing

**Recommendation**: **Confidence-Based Auditor**

**Why**: Speed matters more than perfect quality.

```python
rec = recommend_auditor(
    use_case="creative_writing",
    priority="speed",
    has_multiple_results=True
)
config = SwarmConfig(**rec['config'])
```

---

### General Tasks

**Recommendation**: **List-Wise Auditor** (if multiple results)

**Why**: Best overall quality.

```python
rec = recommend_auditor(
    use_case="general",
    priority="balanced",
    has_multiple_results=True
)
config = SwarmConfig(**rec['config'])
```

---

## Comparison Table

| Auditor Type     | Quality  | Speed    | Cost     | Best For            |
|------------------|----------|----------|----------|---------------------|
| **Single**       | Baseline | Fastest  | Lowest   | Default, single result |
| **List-wise**    | **Best** | Slower   | Higher   | Quality-critical tasks |
| **Pair-wise**    | Good     | Medium   | Medium   | Balanced approach |
| **Confidence-based** | Good     | Fast     | Low      | Speed-critical tasks |

**Research**: List-wise verification performs best in OAgents benchmarks.

---

## Examples

### Example 1: Code Generation (Quality Critical)

```python
from core.orchestration.auditor_selector import recommend_auditor
from core.foundation.data_structures import SwarmConfig

# Get recommendation
rec = recommend_auditor(
    use_case="code_generation",
    priority="quality",
    has_multiple_results=True
)

# Apply config
config = SwarmConfig(**rec['config'])

print(f"Using: {rec['auditor_type']}")
print(f"Reason: {rec['reason']}")
```

**Output**:
```
Using: list_wise
Reason: Quality is critical - list-wise verification provides best results
```

---

### Example 2: Creative Writing (Speed Critical)

```python
rec = recommend_auditor(
    use_case="creative_writing",
    priority="speed",
    has_multiple_results=True
)

config = SwarmConfig(**rec['config'])
```

**Output**:
```
Using: confidence_based
Reason: Speed is critical - confidence-based selection is fastest
```

---

### Example 3: Single Result (Default)

```python
rec = recommend_auditor(
    use_case="general",
    has_multiple_results=False  # Only one result
)

config = SwarmConfig(**rec['config'])
```

**Output**:
```
Using: single
Reason: Only single result available - use default single validation
```

---

## Advanced Usage

### Custom Verification Function

```python
from core.orchestration.auditor_types import ListWiseAuditor, VerificationResult

def my_verification_func(result, context=None):
    """Custom verification logic."""
    # Your domain-specific verification
    score = calculate_score(result)
    return VerificationResult(
        result=result,
        score=score,
        confidence=0.8,
        reasoning="Custom verification",
        passed=score > 0.5
    )

auditor = ListWiseAuditor(
    verification_func=my_verification_func,
    merge_strategy="best_score"
)

merged = auditor.verify_and_merge(results)
```

---

## FAQ

### Q: When should I use list-wise verification?

**A**: When:
- You have multiple results
- Quality is critical
- You can afford higher cost
- Examples: Code generation, data extraction

### Q: Is list-wise always better?

**A**: Yes, for quality. But:
- Higher cost (verifies multiple results)
- Slower (takes more time)
- Use single if you only have one result

### Q: Can I use different auditors for different tasks?

**A**: Yes! Configure per task:

```python
# Code generation: list-wise
code_config = SwarmConfig(enable_list_wise_verification=True)

# Creative writing: confidence-based
creative_config = SwarmConfig(auditor_type="confidence_based")
```

### Q: What if I'm not sure?

**A**: Use the helper:

```python
from core.orchestration.auditor_selector import recommend_auditor

rec = recommend_auditor(
    use_case="your_use_case",
    priority="balanced",
    has_multiple_results=True
)
```

---

## Summary

### Quick Decision Guide

1. **Only one result?** → Use Single (default)
2. **Quality critical?** → Use List-Wise
3. **Speed critical?** → Use Confidence-Based
4. **Cost sensitive?** → Use Pair-Wise
5. **Not sure?** → Use `recommend_auditor()` helper

---

**Last Updated**: January 27, 2026
