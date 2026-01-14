# Expert Agent Memory Contents Summary

## PlantUML Expert Memory

### File-Based Storage
**Location**: `test_outputs/plantuml_expert/improvements.json`  
**Total Improvements**: 8 entries

### Improvements Stored

All improvements are for task: **"Generate simple sequence diagram"**

**Key Patterns Learned:**

1. **Improvement 1** (Iteration 1)
   - **Issue**: Student generated Mermaid instead of PlantUML
   - **Pattern**: Use PlantUML syntax (`@startuml/@enduml`) not Mermaid

2. **Improvement 2** (Iteration 2)
   - **Issue**: Student used Mermaid syntax, added complexity
   - **Pattern**: Use PlantUML syntax, keep it simple (match gold standard)

3. **Improvement 3** (Iteration 1)
   - **Issue**: Student added too much detail (login, logout flow)
   - **Pattern**: Keep diagrams simple, match gold standard format

4. **Improvement 4** (Iteration 2)
   - **Issue**: Student output was more complex than required
   - **Pattern**: Simple sequence diagrams should be minimal

5. **Improvement 5** (Iteration 1)
   - **Issue**: Student used Mermaid syntax
   - **Pattern**: Use PlantUML syntax with `@startuml/@enduml` tags

6. **Improvement 6** (Iteration 2)
   - **Issue**: Different syntax format
   - **Pattern**: Match gold standard format exactly

7. **Improvement 7** (Iteration 1)
   - **Issue**: Exceeded requirements
   - **Pattern**: Keep it simple, match gold standard

8. **Improvement 8** (Iteration 2)
   - **Issue**: Different format and complexity
   - **Pattern**: Use PlantUML syntax, keep simple

### Memory Levels (After Sync)

- **PROCEDURAL**: 8 entries (all improvements stored)
- **SEMANTIC**: 0 entries (not yet consolidated)
- **META**: 0 entries

### Common Patterns Extracted

1. **Use PlantUML syntax** (`@startuml/@enduml`) not Mermaid
2. **Keep diagrams simple** - match gold standard format
3. **Use proper tags** - `@startuml` at start, `@enduml` at end
4. **Match gold standard** - don't add extra complexity

## Mermaid Expert Memory

### File-Based Storage
**Location**: `test_outputs/mermaid_complex_memory/improvements.json`  
**Total Improvements**: Varies by test run

### Common Patterns Learned

1. **Use `flowchart TD`** instead of `graph TD` for flowcharts
2. **Include proper node labels** - use brackets `[Label]`
3. **Use correct arrow syntax** - `-->` for connections
4. **Keep diagrams simple** - match requirements

## Memory Storage Format

Each improvement stored contains:

```json
{
  "iteration": 1,
  "timestamp": "2026-01-13T23:13:40.357485",
  "task": "Generate simple sequence diagram",
  "student_output": "...",
  "teacher_output": "...",
  "student_score": 0.0,
  "teacher_score": 1.0,
  "improvement_type": "teacher_correction",
  "difference": "Output differs from gold standard",
  "learned_pattern": "When task is 'X', use 'Y' instead of 'Z'"
}
```

## Memory Context

Each memory entry includes context:

```json
{
  "expert_name": "plantuml_expert_test",
  "domain": "plantuml",
  "task": "Generate simple sequence diagram",
  "iteration": 1,
  "improvement_type": "teacher_correction",
  "source": "optimization_pipeline"
}
```

## How Improvements Are Used

1. **Stored**: During training, improvements saved to file and memory
2. **Retrieved**: Expert agent loads improvements on initialization
3. **Applied**: Improvements injected into DSPy module via:
   - Signature docstring
   - Module instructions
   - Input field (`learned_improvements`)
4. **Used**: LLM uses improvements during generation

## Consolidation Status

- **PlantUML**: 8 improvements (ready for consolidation - need 3+)
- **Mermaid**: Varies (depends on test runs)

**Consolidation can be run** to create SEMANTIC-level patterns from PROCEDURAL improvements.

## Summary

✅ **Improvements are being stored correctly**  
✅ **Memory integration working**  
✅ **Patterns being learned** (PlantUML syntax, simplicity, tags)  
⚠️  **Teacher output sometimes returns evaluation text instead of diagram code** (needs fix)
