# Expert Agent Memory Storage - Detailed View

## PlantUML Expert

### Storage Location
- **File**: `test_outputs/plantuml_expert/improvements.json`
- **Memory**: HierarchicalMemory (PROCEDURAL level)
- **Total Improvements**: 8 entries

### What's Stored

All 8 improvements are for the same task: **"Generate simple sequence diagram"**

#### Pattern: PlantUML Syntax Required

**Common Issue**: Student keeps generating **Mermaid syntax** instead of **PlantUML syntax**

**What's Learned**:
- Use `@startuml/@enduml` tags (not Mermaid `sequenceDiagram`)
- Use PlantUML arrow syntax: `->` and `-->` (not Mermaid `->>`)
- Keep diagrams simple (match gold standard format)
- Don't add extra complexity

#### Example Improvement Entry

```json
{
  "iteration": 1,
  "task": "Generate simple sequence diagram",
  "student_output": "```mermaid\nsequenceDiagram\n    User->>System: Request\n```",
  "teacher_output": "[Gold standard PlantUML code]",
  "learned_pattern": "When task is 'Generate simple sequence diagram', use PlantUML syntax (@startuml/@enduml) instead of Mermaid"
}
```

### Memory Structure

**PROCEDURAL Level** (8 entries):
- Each entry contains full improvement JSON
- Context includes: expert_name, domain, task, iteration
- Content: Complete improvement dictionary as JSON string

**SEMANTIC Level**: 0 entries (not yet consolidated)

**META Level**: 0 entries

## Mermaid Expert

### Storage Locations
- **File 1**: `test_outputs/mermaid_complex_memory/improvements.json`
- **File 2**: `test_outputs/mermaid_quick_memory/improvements.json`
- **Memory**: HierarchicalMemory (PROCEDURAL level)

### What's Stored

#### Pattern: Flowchart Syntax

**Common Issue**: Student sometimes uses `graph TD` instead of `flowchart TD`

**What's Learned**:
- Use `flowchart TD` for flowcharts (not `graph TD`)
- Use proper node shapes: `([Start])` for rounded nodes
- Keep diagrams simple and match requirements

#### Example Improvement Entry

```json
{
  "iteration": 1,
  "task": "Generate a simple flowchart with Start and End nodes",
  "student_output": "[Student's attempt]",
  "teacher_output": "[Gold standard Mermaid code]",
  "learned_pattern": "When task is 'Generate simple flowchart', use 'flowchart TD' with proper node shapes"
}
```

## Key Observations

### ✅ What's Working

1. **Improvements are being stored**
   - File-based storage working
   - Memory storage working (after sync)

2. **Patterns are being learned**
   - PlantUML: Syntax format (PlantUML vs Mermaid)
   - Mermaid: Flowchart syntax preferences

3. **Memory structure is correct**
   - PROCEDURAL level for specific patterns
   - Context includes expert info
   - Content stored as JSON

### ⚠️ Issues Found

1. **Teacher Output Format**
   - Sometimes returns evaluation text instead of diagram code
   - Should return exact gold_standard code
   - This affects learning quality

2. **Pattern Extraction**
   - Learned patterns sometimes include full teacher output (evaluation text)
   - Should extract concise pattern: "Use PlantUML syntax" not full evaluation

3. **Memory Persistence**
   - Memory is in-memory only (not persisted to disk)
   - Need to sync from files on startup

## Memory Content Structure

### Each Memory Entry Contains:

```json
{
  "key": "hash_of_content",
  "content": "{full_improvement_json_as_string}",
  "level": "PROCEDURAL",
  "context": {
    "expert_name": "plantuml_expert_test",
    "domain": "plantuml",
    "task": "Generate simple sequence diagram",
    "iteration": 1,
    "improvement_type": "teacher_correction",
    "source": "optimization_pipeline"
  },
  "created_at": "2026-01-13T23:27:48",
  "access_count": 0,
  "goal_values": {
    "expert_plantuml_improvements": {
      "value": 1.0
    }
  }
}
```

### Improvement JSON Structure:

```json
{
  "iteration": 1,
  "timestamp": "2026-01-13T23:13:40",
  "task": "Generate simple sequence diagram",
  "student_output": "[what student generated]",
  "teacher_output": "[what teacher provided]",
  "student_score": 0.0,
  "teacher_score": 1.0,
  "improvement_type": "teacher_correction",
  "difference": "Output differs from gold standard",
  "learned_pattern": "When task is 'X', use 'Y' instead of 'Z'"
}
```

## How to View Stored Memory

### Option 1: View Files
```bash
cat test_outputs/plantuml_expert/improvements.json | python -m json.tool
cat test_outputs/mermaid_complex_memory/improvements.json | python -m json.tool
```

### Option 2: Use Inspection Script
```bash
python tests/inspect_expert_memory_detailed.py
```

### Option 3: Programmatic Access
```python
from core.experts.memory_integration import retrieve_improvements_from_memory
from core.memory.cortex import HierarchicalMemory

memory = HierarchicalMemory(...)
improvements = retrieve_improvements_from_memory(
    memory=memory,
    expert_name="plantuml_expert_test",
    domain="plantuml"
)
```

## Summary

✅ **PlantUML**: 8 improvements stored (learning PlantUML syntax)  
✅ **Mermaid**: Improvements stored (learning flowchart syntax)  
✅ **Memory Integration**: Working correctly  
⚠️  **Teacher Output**: Sometimes returns evaluation instead of code (needs fix)
