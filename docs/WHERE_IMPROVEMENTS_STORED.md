# Where Improvements Are Stored

## Storage Locations

Improvements are stored in **multiple locations** depending on configuration:

### 1. JSON Files (Primary Storage)

#### Location
```
{output_path}/improvements.json
```

**For Expert Agents:**
```
{expert_data_dir}/improvements.json
```

**Example:**
```
test_outputs/mermaid_real_llm/improvements.json
expert_data/mermaid/improvements.json
```

#### Format
```json
[
  {
    "iteration": 1,
    "timestamp": "2026-01-13T22:23:59.463842",
    "task": "Generate simple flowchart",
    "student_output": "...",
    "teacher_output": "...",
    "student_score": 0.0,
    "teacher_score": 1.0,
    "improvement_type": "teacher_correction",
    "difference": "Output differs from gold standard",
    "learned_pattern": "When task is 'Generate simple flowchart', use '...' instead of '...'"
  }
]
```

### 2. Improvements Summary File

#### Location
```
{output_path}/improvements_summary.json
```

**Example:**
```
test_outputs/mermaid_real_llm/improvements_summary.json
```

#### Format
```json
{
  "optimization_complete": true,
  "total_iterations": 3,
  "total_improvements": 18,
  "improvements": [...],
  "final_result": {...},
  "summary": {
    "improvements_count": 18,
    "score_improvement": 1.0,
    "learned_patterns": [...]
  }
}
```

### 3. DSPy Instructions (Optional)

#### When Enabled
If `config.update_dspy_instructions = True`:

**Location**: In-memory DSPy module instructions

**How**: Updates the DSPy module's `instructions` attribute

**Code:**
```python
if hasattr(agent, 'instructions'):
    agent.instructions.append(learned_pattern)
```

### 4. Jotty Learned Instructions (Optional)

#### When Enabled
If `config.update_jotty_instructions = True` and `conductor` is provided:

**Location**: `conductor.learned_instructions`

**How**: Updates Jotty's internal learned instructions dictionary

## Code Flow

### 1. Recording Improvement

```python
# In OptimizationPipeline._record_improvement()
improvement = {
    "iteration": iteration,
    "timestamp": datetime.now().isoformat(),
    "task": task,
    "student_output": str(student_output),
    "teacher_output": str(teacher_output),
    "student_score": evaluation_result.get("score", 0.0),
    "teacher_score": teacher_eval.get("score", 1.0),
    "improvement_type": "teacher_correction",
    "difference": evaluation_result.get("difference"),
    "learned_pattern": self._extract_learned_pattern(...)
}

self.improvements.append(improvement)
```

### 2. Saving to File

```python
# Save to JSON file
if self.improvements_file:
    # Load existing
    existing = []
    if self.improvements_file.exists():
        with open(self.improvements_file, 'r') as f:
            existing = json.load(f)
    
    # Append new
    existing.append(improvement)
    
    # Save back
    with open(self.improvements_file, 'w') as f:
        json.dump(existing, f, indent=2)
```

### 3. File Path Determination

```python
# In OptimizationPipeline.__init__()
if self.config.save_improvements:
    if self.config.improvements_file:
        self.improvements_file = Path(self.config.improvements_file)
    elif self.output_path:
        self.improvements_file = self.output_path / "improvements.json"
```

## For Expert Agents

### Expert Agent Storage

**Base Directory:**
```python
self.data_dir = Path(config.expert_data_dir)
# Default: ./expert_data/{domain}
```

**Files:**
- `improvements.json` - All improvements
- `improvements_summary.json` - Summary
- `training_results.json` - Training results
- `validation_results.json` - Validation results
- `thinking.log` - Thinking log

### Loading Improvements

```python
# In ExpertAgent.__init__()
self.improvements_file = self.data_dir / "improvements.json"

def _load_improvements(self) -> List[Dict[str, Any]]:
    if self.improvements_file.exists():
        with open(self.improvements_file, 'r') as f:
            return json.load(f)
    return []
```

## Configuration

### Enable/Disable Storage

```python
# In OptimizationConfig
save_improvements: bool = True  # Enable/disable saving
improvements_file: Optional[Path] = None  # Custom path
update_dspy_instructions: bool = False  # Update DSPy modules
update_jotty_instructions: bool = False  # Update Jotty conductor
```

### Example Configuration

```python
pipeline = create_optimization_pipeline(
    agents=agents,
    save_improvements=True,  # Save to JSON
    output_path="./outputs/my_pipeline",
    update_dspy_instructions=False,  # Don't update DSPy
    update_jotty_instructions=False  # Don't update Jotty
)
```

## Accessing Improvements

### From File

```python
import json
from pathlib import Path

improvements_file = Path("./test_outputs/mermaid_real_llm/improvements.json")
with open(improvements_file, 'r') as f:
    improvements = json.load(f)

print(f"Total improvements: {len(improvements)}")
for imp in improvements:
    print(f"Task: {imp['task']}")
    print(f"Pattern: {imp['learned_pattern']}")
```

### From Expert Agent

```python
expert = MermaidExpertAgent()
status = expert.get_status()
print(f"Improvements: {status['improvements_count']}")

# Load directly
improvements = expert._load_improvements()
```

## Summary

| Storage Location | Type | Default | When Used |
|-----------------|------|---------|-----------|
| **JSON File** | File | `{output_path}/improvements.json` | Always (if `save_improvements=True`) |
| **Summary File** | File | `{output_path}/improvements_summary.json` | Always (if `save_improvements=True`) |
| **DSPy Instructions** | In-Memory | Module `instructions` attribute | If `update_dspy_instructions=True` |
| **Jotty Instructions** | In-Memory | `conductor.learned_instructions` | If `update_jotty_instructions=True` |

## Default Behavior

✅ **JSON files are saved by default**  
✅ **DSPy/Jotty updates are disabled by default**  
✅ **Improvements persist across runs**  
✅ **Can be loaded and reused**
