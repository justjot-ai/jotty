# PlantUML GitHub Training Implementation

## Summary

Implemented loading PlantUML training examples from GitHub and saving as JSON in expert directory.

---

## What is `gold_standard`?

**`gold_standard`** = **The correct answer** that expert should learn to produce.

### How `gold_standard` is Used:

1. **Training** (Primary)
   - Expert generates output from `task`
   - Output evaluated against `gold_standard`
   - If fails → teacher provides `gold_standard` as correction
   - Expert learns: "For this task, output should be `gold_standard`"

2. **Evaluation/Scoring**
   - Compare generated output with `gold_standard`
   - Calculate similarity score (0.0 to 1.0)

3. **Teacher Model Input**
   - Teacher receives `gold_standard` as input
   - Teacher returns `gold_standard` exactly

4. **Validation**
   - Verify expert performance after training

---

## Implementation

### 1. Training Data Loader ✅

**File**: `core/experts/training_data_loader.py`

**Features**:
- Loads examples from GitHub repositories
- Recursively searches directories (up to 2 levels deep)
- Supports multiple file extensions (.puml, .plantuml, .pu)
- Validates examples using domain validator
- Converts to `gold_standards` format

**Key Methods**:
- `load_from_github_repo()`: Loads files from GitHub
- `validate_examples()`: Validates using domain validator
- `convert_to_gold_standards()`: Converts to training format

---

### 2. PlantUML Expert Integration ✅

**File**: `core/experts/plantuml_expert.py`

**Method**: `load_training_examples_from_github()`

**Features**:
- Loads from GitHub repository
- Validates examples
- Converts to `gold_standards` format
- **Saves as JSON** to expert directory
- Returns `gold_standards` ready for training

**Usage**:
```python
# Load examples from GitHub
gold_standards = await PlantUMLExpertAgent.load_training_examples_from_github(
    repo_url="https://github.com/joelparkerhenderson/plantuml-examples",
    max_examples=50,
    save_to_file=True,  # Save as JSON
    expert_data_dir="./expert_data/plantuml_expert"
)

# Use for training
await expert.train(gold_standards=gold_standards)
```

---

### 3. JSON Storage ✅

**Location**: `./expert_data/plantuml_expert/github_training_examples.json`

**Format**:
```json
{
  "source": "https://github.com/...",
  "total_examples": 50,
  "loaded_at": "2026-01-14T...",
  "gold_standards": [
    {
      "task": "Generate sequence diagram: ...",
      "context": {
        "description": "...",
        "diagram_type": "sequence",
        "source": "github:owner/repo"
      },
      "gold_standard": "@startuml\n..."
    }
  ]
}
```

**Benefits**:
- ✅ Persistent storage (survives restarts)
- ✅ Can be reused without GitHub API calls
- ✅ Version controlled (can commit to repo)
- ✅ Easy to inspect/debug

---

## Testing

### Test File: `tests/test_plantuml_github_training.py`

**Tests**:
1. Load examples from GitHub
2. Validate format
3. Save as JSON
4. Verify gold_standards format

**Note**: GitHub API rate limits may affect testing. Consider:
- Using GitHub token for higher rate limits
- Caching examples locally
- Using mock data for testing

---

## Current Status

### ✅ Implemented:
- GitHub loading with recursive directory search
- File pattern matching (.puml, .plantuml, .pu)
- Domain validation
- Conversion to gold_standards format
- JSON saving to expert directory

### ⚠️ Known Issues:
- GitHub API rate limits (60 requests/hour without token)
- Files are in subdirectories (doc/activity-diagram/, etc.)
- Need to handle rate limits gracefully

### 🔧 Solutions:
1. **Use GitHub Token**: Set `GITHUB_TOKEN` env var for higher limits
2. **Cache Locally**: Save examples once, reuse JSON file
3. **Recursive Search**: Already implemented (searches up to 2 levels deep)

---

## Usage Example

```python
from core.experts import PlantUMLExpertAgent

# Load examples from GitHub
gold_standards = await PlantUMLExpertAgent.load_training_examples_from_github(
    repo_url="https://github.com/joelparkerhenderson/plantuml-examples",
    max_examples=50,
    save_to_file=True
)

# Examples saved to: ./expert_data/plantuml_expert/github_training_examples.json

# Use for training
expert = PlantUMLExpertAgent()
await expert.train(
    gold_standards=gold_standards,
    enable_pre_training=True,  # Extract patterns first
    training_mode="both"  # Both pre-training and iterative learning
)
```

---

## Why Use `gold_standards`?

**Answer**: `gold_standards` is the **correct answer** that expert learns to produce.

**Not creating new variables** - We consolidated:
- ✅ `gold_standards` used for **both** pre-training and iterative learning
- ✅ No separate `pre_training_examples` parameter needed
- ✅ Controlled by `enable_pre_training` and `training_mode` flags

**Use Cases**:
1. ✅ Pre-training (pattern extraction)
2. ✅ Iterative learning (fine-tuning)
3. ✅ Validation
4. ✅ Teacher input
5. ✅ Few-shot learning (potential)
6. ✅ Template learning (potential)
7. ✅ Domain adaptation (potential)

---

## Files Created/Modified

### New Files:
- `core/experts/training_data_loader.py` - GitHub loading and conversion

### Modified Files:
- `core/experts/plantuml_expert.py` - Added GitHub loading method
- `core/experts/expert_agent.py` - Consolidated to use `gold_standards` for all purposes

---

## Next Steps

1. ✅ **Implemented**: GitHub loading, JSON saving, gold_standards conversion
2. ⏳ **Testing**: Test with GitHub token or cached examples
3. ⏳ **Enhancement**: Add GitHub token support for rate limits
4. ⏳ **Usage**: Use loaded examples for training

---

## Summary

**`gold_standard`** = The correct answer expert learns to produce.

**Implementation**:
- ✅ Loads PlantUML examples from GitHub
- ✅ Saves as JSON in expert directory
- ✅ Converts to `gold_standards` format
- ✅ Ready for training

**Consolidation**:
- ✅ Using `gold_standards` for all purposes (no separate variables)
- ✅ Flexible training modes (pre-training, iterative, both)
