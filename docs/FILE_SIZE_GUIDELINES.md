# File Size Guidelines - Data-Driven Principles

## The Data: What Jotty Actually Looks Like

**Current Distribution (1,119 files):**
- **Median: 247 lines** ✓ (industry sweet spot)
- **Mean: 359 lines** ✓ (good)
- **75th percentile: 477 lines** ✓ (still good)
- **90th percentile: 801 lines** ⚠️ (getting large)
- **95th percentile: 1,108 lines** ❌ (too large)

**Size Breakdown:**
```
Tiny (<100 lines):         22.6% ███████████     - May be over-fragmented
Small (100-300 lines):     35.4% █████████████████ - ✓ SWEET SPOT
Medium (300-500 lines):    18.8% █████████        - ✓ IDEAL RANGE
Large (500-1,000 lines):   16.6% ████████         - Acceptable if cohesive
Very Large (1,000-2,000):   5.8% ██              - Should review
God Files (2,000+):         0.8%                 - MUST SPLIT
```

**Finding:** 54.2% of files are in the sweet spot (100-500 lines). Only 6.6% are problematic (>1,000 lines).

---

## The Principle: Cohesion > Size

**Good file size is about COHESION, not arbitrary line counts.**

### The "Single Responsibility" Test

Ask: **"Can I describe this file in one sentence without using 'and'?"**

✅ **Good (cohesive):**
- `calculator.py` - "Handles mathematical calculations and unit conversions"
- `memory_facade.py` - "Provides unified interface to memory system"
- `swarm_template.py` - "Base template for swarm implementations"

❌ **Bad (not cohesive):**
- `stock_ml.py` - "Handles data loading AND feature engineering AND model training AND visualization AND reporting AND..." (2,610 lines)
- `swarm_manager.py` - "Manages swarms AND handles learning AND does orchestration AND..." (2,964 lines)

---

## The Numbers: File Size Ranges

### Industry Standards (Python)

| Source | Recommendation |
|--------|----------------|
| **Google Style Guide** | "Modules should be focused and understandable" |
| **Clean Code (Martin)** | ~500 lines max for readability |
| **Linux Kernel** | 300-500 lines average |
| **Django** | 200-400 lines average |
| **scikit-learn** | 300-600 lines average |

### Recommended Ranges

| Size | Lines | When to Use | Examples |
|------|-------|-------------|----------|
| **Tiny** | <100 | Simple utilities, constants, single classes | `types.py`, `constants.py` |
| **Small** | 100-300 | ✓ **Sweet spot** - Single responsibility, complete | Most skills, facades |
| **Medium** | 300-500 | ✓ **Ideal** - Complex logic, multiple related classes | Orchestrators, managers |
| **Large** | 500-1,000 | ⚠️ Acceptable IF cohesive - Review periodically | Complex swarms, registries |
| **Very Large** | 1,000-2,000 | ❌ **Too big** - Should split unless exceptional | StockMLCommand (2,610) |
| **God File** | 2,000+ | ❌ **Must split** - Unmaintainable | swarm_manager.py (2,964) |

### The 80/20 Rule

**Target:** 80% of files should be 100-500 lines

**Jotty current:** 54.2% in sweet spot → Room for improvement

---

## When to Split a File

### Red Flags (Any ONE means consider splitting)

1. **"The Scroll Test"** - Can't see the whole file without scrolling > 3 screens
2. **"The Search Test"** - Takes >10 seconds to find a specific function
3. **"The Explain Test"** - Can't explain the file's purpose in one sentence
4. **"The Navigation Test"** - Need to check the file structure/outline to navigate
5. **"The Merge Conflict Test"** - Frequent merge conflicts from multiple developers
6. **"The Class Count Test"** - More than 3-5 classes in one file
7. **"The 1,000 Line Test"** - Over 1,000 lines (hard limit)

### How to Split

**Pattern 1: Extract by Responsibility**
```python
# Before: stock_ml.py (2,610 lines)
class StockMLCommand:
    # Data loading (300 lines)
    # Feature engineering (500 lines)
    # Model training (800 lines)
    # Visualization (600 lines)
    # Reporting (400 lines)

# After: Split into 5 files
stock_ml/
  ├── data_loader.py        (300 lines) - StockDataLoader
  ├── feature_engineer.py   (500 lines) - FeatureEngine
  ├── model_trainer.py      (800 lines) - ModelTrainer
  ├── visualizer.py         (600 lines) - StockVisualizer
  ├── reporter.py           (400 lines) - StockReporter
  └── command.py            (100 lines) - StockMLCommand (orchestrates)
```

**Pattern 2: Extract Mixins to Modules**
```python
# Before: swarm_manager.py (2,964 lines)
class SwarmManager(LearningMixin, OrchestrationMixin, StateMixin):
    # Learning methods (800 lines)
    # Orchestration methods (900 lines)
    # State methods (700 lines)
    # Core methods (500 lines)

# After: Extract mixins
swarm/
  ├── manager.py            (500 lines) - Core SwarmManager
  ├── learning.py           (800 lines) - SwarmLearningMixin
  ├── orchestration.py      (900 lines) - SwarmOrchestrationMixin
  └── state.py              (700 lines) - SwarmStateMixin
```

**Pattern 3: Extract Related Classes**
```python
# Before: Many small classes in one file (>1,000 lines)

# After: Group by domain
analysis/
  ├── drift_detector.py     (200 lines)
  ├── fairness_checker.py   (250 lines)
  ├── deployment_validator.py (300 lines)
  └── __init__.py           (50 lines) - Re-exports
```

---

## When NOT to Split

### Exceptions to the Rule

**DON'T split if:**

1. **Configuration Files** - Can be 1,000+ lines of related constants
   ```python
   # config.py with 1,500 lines of related settings - KEEP TOGETHER
   HYPERPARAMETERS = {...}  # 500 lines
   MODEL_CONFIGS = {...}     # 500 lines
   FEATURE_DEFINITIONS = {...}  # 500 lines
   ```

2. **Generated Code** - Auto-generated files can be any size
   ```python
   # gen_batch3.py (2,676 lines) - If auto-generated, document and ignore
   ```

3. **Schema/Types** - Type definitions benefit from being together
   ```python
   # types.py with 800 lines of dataclasses - OK if all related
   ```

4. **Highly Cohesive Complex Logic** - Some things just can't be split
   ```python
   # complex_algorithm.py (1,200 lines) - If it's ONE algorithm, keep together
   # BUT: Add detailed comments and section markers
   ```

---

## The Fragmentation Problem

**You asked: "Earlier you penalized for too fragmented"**

You're right! Here's the balance:

### Too Fragmented (Bad)
```
calculator/
  ├── add.py           (20 lines) - Just addition
  ├── subtract.py      (20 lines) - Just subtraction
  ├── multiply.py      (20 lines) - Just multiplication
  └── divide.py        (20 lines) - Just division
```
**Problem:** Jump between 4 files for basic arithmetic. Cognitive overhead.

### Just Right (Good)
```
calculator/
  └── tools.py         (230 lines) - All arithmetic + unit conversion
```
**Sweet spot:** Related operations together, single responsibility.

### Too Large (Bad)
```
calculator/
  └── tools.py         (2,500 lines) - Arithmetic + stats + linear algebra +
                                       calculus + graphing + ...
```
**Problem:** Too many responsibilities, hard to navigate.

---

## Practical Guidelines for Jotty

### Target Distribution (Ideal)

```
<100 lines:      15% (utilities, constants)
100-300 lines:   40% (skills, simple modules) ← INCREASE FROM 35%
300-500 lines:   25% (complex modules)        ← INCREASE FROM 19%
500-1,000 lines: 15% (registries, complex)    ← DECREASE FROM 17%
1,000-2,000:      4% (exceptional cases)      ← DECREASE FROM 6%
2,000+:           1% (config/generated only)  ← DECREASE FROM 1%
```

### Action Items by File Size

**For files >2,000 lines (9 files):**
1. `swarm_manager.py` (2,964) - ❌ **Must split** into 4-5 modules
2. `gen_batch3.py` (2,676) - 🗑️ **Delete if generated**
3. `stock_ml.py` (2,653) - ❌ **Must split** into 5 modules
4. `ml_report_generator.py` (2,571) - ❌ **Split into report components**
5. `_visualization_mixin.py` (2,407) - ❌ **Extract to visualization/ module**
6. `_analysis_sections_mixin.py` (2,344) - ❌ **Extract to analysis/ module**
7. `skills_registry.py` (2,154) - ⚠️ **Review** (registry, might be OK)
8. `gen_batch2.py` (2,068) - 🗑️ **Delete if generated**
9. `client.py` (2,007) - ⚠️ **Review** (might be cohesive)

**For files 1,000-2,000 lines (65 files):**
- Review each for cohesion
- Split if they fail the "one sentence" test
- Target: Reduce to 30-40 files in this range

---

## The Final Answer

### File Size Guiding Principles

1. **Cohesion First** - "Can I describe this in one sentence?"
2. **Sweet Spot: 100-500 lines** - Most files should be here
3. **Hard Limit: 1,000 lines** - Review anything above this
4. **Absolute Limit: 2,000 lines** - Must split (except config/generated)
5. **Don't Over-Fragment** - Related code should be together
6. **Use the Scroll Test** - If you can't see it all, it's too big

### For Jotty Specifically

**Current State:** Pretty good! (54% in sweet spot)

**Improvement:** Focus on the top 6.6% (>1,000 lines)
- Split 9 god files (2,000+ lines)
- Review 65 very large files (1,000-2,000 lines)
- Don't touch the other 93.4% - they're fine!

**Target:** 80% in sweet spot (100-500 lines)

---

## TL;DR

| Question | Answer |
|----------|--------|
| **What's ideal?** | 100-500 lines per file |
| **What's acceptable?** | 500-1,000 lines if cohesive |
| **What's too big?** | >1,000 lines (review), >2,000 lines (must split) |
| **What's too small?** | <50 lines if it could be merged with related code |
| **How to decide?** | Ask: "Can I describe this file in one sentence?" |
| **Jotty's status?** | 54% in sweet spot, 6.6% too big - Focus on splitting top 74 files |

**The Principle:** Make files as small as possible, but no smaller. Cohesion > arbitrary size limits.
