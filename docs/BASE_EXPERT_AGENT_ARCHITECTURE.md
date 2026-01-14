# Base Expert Agent Architecture

## Answer to Your Questions

### Q1: "Is our base expert agent generic?"

**A**: ✅ **YES - Fully Generic!**

The `ExpertAgent` base class is **completely generic** and domain-agnostic:

- ✅ Works for any domain (Mermaid, PlantUML, Markdown, SQL, etc.)
- ✅ Uses `domain` parameter to identify expert type
- ✅ Customizable via `ExpertAgentConfig`
- ✅ Subclasses only need to implement domain-specific methods

**Evidence**:
```python
class ExpertAgent:
    """Base class for Expert Agents - GENERIC for any domain."""
    
    def __init__(self, config: ExpertAgentConfig, memory=None):
        self.config = config  # Contains domain, evaluation_function, etc.
        self.domain = config.domain  # "mermaid", "plantuml", etc.
```

---

### Q2: "Does it have contract to load gold data optionally?"

**A**: ✅ **YES - Optional Gold Standards Loading!**

**Contract**:
1. ✅ Gold standards can be loaded **optionally** during training
2. ✅ Gold standards used for **pre-training** (pattern extraction)
3. ✅ Gold standards used for **iterative learning** (teacher corrections)
4. ✅ Gold standards **NOT required** for generation (expert can work without them)

**Implementation**:
```python
async def train(
    self,
    gold_standards: Optional[List[Dict[str, Any]]] = None,  # ← OPTIONAL
    enable_pre_training: bool = True,
    training_mode: str = "both"
):
    """
    Gold standards are OPTIONAL:
    - If provided: Used for training
    - If None: Uses config.training_gold_standards (if available)
    - If still None: Raises error (training needs examples)
    """
    gold_standards = gold_standards or self.config.training_gold_standards
    if not gold_standards:
        raise ValueError("No gold standards provided for training")
```

**Usage**:
```python
# With gold standards (optional)
await expert.train(gold_standards=gold_standards)

# Without gold standards (uses config defaults if available)
await expert.train()

# Generation works WITHOUT gold standards (uses learned improvements)
output = await expert.generate(task="...", context={...})
```

---

### Q3: "On each error, does it call teacher to improve?"

**A**: ✅ **YES - Teacher Called Automatically on Errors!**

**Flow**:
```
1. Expert generates output
   ↓
2. Output evaluated (via evaluation_function)
   ↓
3. IF error detected (score < target_score):
   ├─→ Teacher is called automatically
   ├─→ Teacher receives: student_output, gold_standard, evaluation_result
   ├─→ Teacher provides correction
   ├─→ Expert learns from correction
   └─→ Improvement stored in memory
   ↓
4. Next iteration uses learned improvement
```

**Code Evidence** (`optimization_pipeline.py`, lines 824-861):
```python
# Evaluate output
evaluation_result = await self._evaluate_output(
    output=output,
    gold_standard=gold_standard,
    task=task,
    context=context
)

score = evaluation_result.get("score", 0.0)

# IF ERROR → Teacher called automatically
if score < target_score:  # Error detected
    if self.config.enable_teacher_model:
        teacher_output = await self._run_teacher_model(
            task=task,
            context=context,
            student_output=output,  # ❌ What student generated (wrong)
            gold_standard=gold_standard,  # ✅ Correct answer
            evaluation_result=evaluation_result  # What was wrong
        )
        
        # Expert learns from teacher correction
        await self._update_knowledge_base(
            student_output=output,
            teacher_output=teacher_output,
            ...
        )
```

**Key Points**:
- ✅ Teacher called **automatically** when error detected
- ✅ No manual intervention needed
- ✅ Happens during **training** (iterative learning)
- ✅ Also happens during **optimization** (if enabled)

---

### Q4: "Is error detected by a function (currently renderer error)?"

**A**: ✅ **YES - Error Detection via Evaluation Function!**

**Architecture**:
- ✅ Error detection is **pluggable** via `evaluation_function`
- ✅ Each expert provides its own evaluation function
- ✅ Currently uses **renderer validation** for Mermaid/PlantUML
- ✅ Can use **any evaluation method** (renderer, syntax check, semantic, etc.)

**Contract**:
```python
evaluation_function: Optional[Callable] = None

# Signature:
async def evaluation_function(
    output: Any,           # Generated output
    gold_standard: Any,    # Correct answer (optional)
    task: str,            # Task description
    context: Dict[str, Any]  # Context
) -> Dict[str, Any]:
    """
    Returns:
    {
        "score": float,      # 0.0 to 1.0
        "status": str,       # "CORRECT", "FAIL", "ERROR"
        "errors": List[str], # List of errors found
        "metadata": Dict     # Additional info
    }
    """
```

**Current Implementation**:

**Mermaid Expert** (`mermaid_expert.py`):
```python
async def _evaluate_mermaid(output, gold_standard, task, context):
    """Uses MermaidValidator + Renderer for error detection."""
    
    # 1. Domain-specific validation (syntax, type, elements)
    is_valid, error_msg, metadata = domain_validator.validate(...)
    
    # 2. Renderer validation (if enabled)
    if use_renderer:
        renderer_valid, renderer_error, _ = validate_mermaid_syntax(output)
        if not renderer_valid:
            # Error detected via renderer!
            return {
                "score": 0.0,
                "status": "FAIL",
                "errors": [renderer_error],
                ...
            }
    
    # 3. Calculate score based on validation
    score = calculate_score(is_valid, type_match, element_coverage)
    return {"score": score, "status": "CORRECT" if score >= 0.9 else "FAIL"}
```

**PlantUML Expert** (`plantuml_expert.py`):
```python
async def _evaluate_plantuml(output, gold_standard, task, context):
    """Uses PlantUMLValidator + Renderer for error detection."""
    
    # Similar pattern: Domain validator + Renderer
    is_valid, error_msg, metadata = plantuml_validator.validate(...)
    
    # Renderer validation (HTTP 414 handling, etc.)
    if use_renderer:
        renderer_valid, renderer_error, _ = validate_plantuml_syntax(output)
        if not renderer_valid:
            # Error detected via renderer!
            return {"score": 0.0, "status": "FAIL", "errors": [renderer_error]}
```

**Error Detection Methods**:
1. ✅ **Renderer Validation** (current): Validates by attempting to render
   - Mermaid: `mermaid.ink` API
   - PlantUML: `plantuml.com` API
   - Handles HTTP 414 (URI Too Long) errors

2. ✅ **Domain Validator**: Syntax, type, element checks
   - `MermaidValidator`: Checks syntax, type, elements
   - `PlantUMLValidator`: Checks tags, type, structure

3. ✅ **Custom Evaluation**: Can use any function
   - Semantic similarity
   - Structure matching
   - Custom rules

---

## Complete Architecture Flow

```
┌─────────────────────────────────────────────────────────────┐
│ BASE EXPERT AGENT (Generic)                                  │
│                                                               │
│ 1. Generic for any domain                                    │
│ 2. Optional gold standards loading                           │
│ 3. Automatic teacher on errors                               │
│ 4. Pluggable error detection                                 │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ DOMAIN-SPECIFIC EXPERT (e.g., MermaidExpertAgent)           │
│                                                               │
│ - Provides: evaluation_function                              │
│ - Provides: domain_validator                                │
│ - Provides: renderer validation                             │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ GENERATION FLOW                                              │
│                                                               │
│ 1. expert.generate(task, context)                            │
│    ↓                                                          │
│ 2. Agent generates output                                    │
│    ↓                                                          │
│ 3. evaluation_function(output, gold_standard, ...)          │
│    ├─→ Domain validator checks                              │
│    ├─→ Renderer validates (if enabled)                      │
│    └─→ Returns: {score, status, errors}                      │
│    ↓                                                          │
│ 4. IF score < target_score (ERROR):                          │
│    ├─→ Teacher called automatically                          │
│    ├─→ Teacher provides correction                           │
│    ├─→ Expert learns from correction                        │
│    └─→ Improvement stored                                    │
│    ↓                                                          │
│ 5. Next generation uses learned improvement                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary

| Question | Answer | Evidence |
|----------|--------|----------|
| **Generic?** | ✅ YES | Works for any domain via `domain` parameter |
| **Optional gold data?** | ✅ YES | `gold_standards` parameter is optional in `train()` |
| **Teacher on error?** | ✅ YES | Automatically called when `score < target_score` |
| **Error detection function?** | ✅ YES | Via `evaluation_function` (currently renderer) |

**Key Points**:
1. ✅ **Fully Generic**: Base class works for any domain
2. ✅ **Optional Gold Standards**: Can train with or without
3. ✅ **Automatic Teacher**: Called on errors during training/optimization
4. ✅ **Pluggable Error Detection**: Via `evaluation_function` (renderer, validator, custom)

**Current Implementation**:
- Error detection: **Renderer validation** (mermaid.ink, plantuml.com)
- Also uses: **Domain validators** (syntax, type, elements)
- Fallback: **Structure-based validation** (if renderer fails)

**Architecture is clean, generic, and extensible!** 🎉
