# Skills Optimization & Refactoring Analysis

## Current State

### Skills Inventory
- **Total Skills**: 34 skills
- **Total Tools**: 80 tools
- **Composite Skills**: 4 (using CompositeSkill framework)
- **Pipeline Skills**: 1 (using PipelineSkill framework)

### Skills by Category

#### Document Sources (1 skill)
- `arxiv-downloader` (2 tools)

#### Downloaders (1 skill)
- `youtube-downloader` (2 tools)

#### Converters (2 skills)
- `document-converter` (5 tools)
- `time-converter` (5 tools)

#### Text Processing (1 skill)
- `text-chunker` (1 tool)

#### Output/Upload (4 skills)
- `remarkable-upload` (3 tools)
- `remarkable-sender` (1 tool)
- `last30days-to-pdf-remarkable` (1 tool) ⚠️ Composite
- `v2v-to-pdf-telegram-remarkable` (1 tool)

#### Web/Search (6 skills)
- `web-search` (2 tools)
- `web-scraper` (1 tool)
- `search-summarize-pdf-telegram` (1 tool) ⚠️ Composite
- `search-summarize-pdf-telegram-v2` (1 tool) ✅ Pipeline
- `research-to-pdf` (1 tool)
- `v2v-trending-search` (1 tool)

#### Content Generation (1 skill)
- `image-generator` (3 tools)

#### System/File (3 skills)
- `file-operations` (7 tools)
- `shell-exec` (2 tools)
- `process-manager` (3 tools)

#### Text Utils (1 skill)
- `text-utils` (6 tools)

#### Utilities (2 skills)
- `calculator` (2 tools)
- `weather-checker` (2 tools)

#### Other (12 skills)
- `claude-cli-llm` (2 tools)
- `content-repurposer` (1 tool)
- `http-client` (3 tools)
- `last30days` (1 tool)
- `last30days-claude-cli` (3 tools)
- `last30days-to-epub-telegram` (1 tool) ⚠️ Composite
- `last30days-to-pdf-telegram` (1 tool) ⚠️ Composite
- `mcp-justjot` (10 tools)
- `mindmap-generator` (1 tool)
- `notebooklm-pdf` (1 tool)
- `oauth-automation` (1 tool)
- `telegram-sender` (2 tools)

---

## Optimization Opportunities

### 🔴 High Priority: Duplication Reduction

#### 1. **Consolidate `last30days-to-*` Skills** (3 → 1)

**Current State:**
- `last30days-to-pdf-telegram` (composite)
- `last30days-to-epub-telegram` (composite)
- `last30days-to-pdf-remarkable` (composite)

**Problem:**
- All 3 skills have nearly identical code (~300 lines each)
- Only difference: output format (PDF/EPUB) and destination (Telegram/reMarkable)
- Violates DRY principle

**Solution:**
```python
# Single parameterized pipeline skill
last30days_to_output_pipeline = [
    {
        "type": "source",
        "skill": "last30days-claude-cli",
        "tool": "research_topic_tool",
        "params": {"topic": "{{topic}}"}
    },
    {
        "type": "processor",
        "skill": "document-converter",
        "tool": "convert_to_{{format}}_tool",  # Dynamic format
        "params": {"content": "{{source.content}}"}
    },
    {
        "type": "sink",
        "skill": "{{destination}}",  # telegram-sender or remarkable-upload
        "tool": "send_{{destination}}_tool",
        "params": {"file_path": "{{processor.output_path}}"}
    }
]
```

**Benefits:**
- Reduce code from ~900 lines to ~100 lines
- Single skill handles all combinations
- Easy to add new formats/destinations

**Migration Path:**
1. Create `last30days-to-output` pipeline skill
2. Deprecate old skills (keep for backward compatibility)
3. Update callers to use new skill with parameters

---

#### 2. **Migrate Composite Skills to Pipeline Framework**

**Current State:**
- 4 composite skills using `CompositeSkill` framework
- 1 pipeline skill using `PipelineSkill` framework

**Problem:**
- Two different frameworks for same purpose
- Composite skills are more verbose (~300 lines each)
- Pipeline skills are declarative (~50 lines each)

**Solution:**
Migrate all composite skills to pipeline framework:

| Current Composite Skill | Pipeline Equivalent |
|------------------------|---------------------|
| `search-summarize-pdf-telegram` | ✅ Already migrated (`search-summarize-pdf-telegram-v2`) |
| `last30days-to-pdf-telegram` | → `last30days-to-output` (with params) |
| `last30days-to-epub-telegram` | → `last30days-to-output` (with params) |
| `last30days-to-pdf-remarkable` | → `last30days-to-output` (with params) |

**Benefits:**
- Consistent framework across all workflow skills
- 80% code reduction (300 lines → 50 lines per skill)
- Easier to maintain and extend
- Declarative configuration (easier to understand)

---

### 🟡 Medium Priority: Code Reuse

#### 3. **Extract Common Pipeline Templates**

**Current State:**
- Each composite/pipeline skill reimplements:
  - Error handling
  - Parameter validation
  - Result formatting
  - Logging

**Solution:**
Create reusable pipeline templates:

```python
# core/registry/pipeline_templates.py

RESEARCH_TO_PDF_TEMPLATE = [
    {"type": "source", "skill": "{{source_skill}}", "tool": "{{source_tool}}"},
    {"type": "processor", "skill": "document-converter", "tool": "convert_to_pdf_tool"},
    {"type": "sink", "skill": "{{sink_skill}}", "tool": "{{sink_tool}}"}
]

RESEARCH_TO_DOCUMENT_TEMPLATE = [
    {"type": "source", "skill": "{{source_skill}}", "tool": "{{source_tool}}"},
    {"type": "processor", "skill": "document-converter", "tool": "convert_to_{{format}}_tool"},
    {"type": "sink", "skill": "{{sink_skill}}", "tool": "{{sink_tool}}"}
]
```

**Benefits:**
- Reduce boilerplate code
- Standardize common patterns
- Faster creation of new workflows

---

#### 4. **Consolidate Similar Skills**

**Opportunities:**

**A. Merge `remarkable-upload` and `remarkable-sender`**
- Both handle reMarkable uploads
- Could be unified with different tools

**B. Consolidate `web-search` and `web-scraper`**
- Both fetch web content
- Could be unified with mode parameter

**C. Merge `research-to-pdf` and `search-summarize-pdf-telegram-v2`**
- Both follow same pattern (search → process → PDF)
- Could be parameterized pipeline

---

### 🟢 Low Priority: Performance & Architecture

#### 5. **Lazy Loading of Skills**

**Current State:**
- All skills loaded at registry initialization
- 34 skills × dependencies = slow startup

**Solution:**
```python
# Load skills on-demand
def get_skill(self, skill_name: str) -> SkillDefinition:
    if skill_name not in self.loaded_skills:
        self._load_skill(skill_name)
    return self.loaded_skills[skill_name]
```

**Benefits:**
- Faster startup time
- Lower memory usage
- Load only what's needed

---

#### 6. **Skill Dependency Graph**

**Current State:**
- No visibility into skill dependencies
- Hard to understand workflow chains

**Solution:**
```python
# Track skill dependencies
skill_dependencies = {
    'last30days-to-pdf-telegram': ['last30days-claude-cli', 'document-converter', 'telegram-sender'],
    'search-summarize-pdf-telegram-v2': ['web-search', 'claude-cli-llm', 'document-converter', 'telegram-sender']
}
```

**Benefits:**
- Understand skill relationships
- Detect circular dependencies
- Optimize loading order

---

## Refactoring Plan (Non-Breaking)

### Phase 1: Create Unified Pipeline Skill ✅ SAFE
1. Create `last30days-to-output` pipeline skill
2. Keep old skills (mark as deprecated)
3. Test new skill thoroughly
4. **No breaking changes** - old skills still work

### Phase 2: Migrate Composite Skills ✅ SAFE
1. Migrate each composite skill to pipeline framework
2. Keep old skills for backward compatibility
3. Add deprecation warnings
4. **No breaking changes** - gradual migration

### Phase 3: Extract Templates ✅ SAFE
1. Create pipeline templates
2. Refactor existing pipelines to use templates
3. **No breaking changes** - internal refactoring only

### Phase 4: Consolidate Skills ⚠️ BREAKING (Optional)
1. Merge similar skills
2. Update callers
3. **Breaking changes** - requires coordination

---

## Metrics & Impact

### Code Reduction
- **Before**: ~1,200 lines (4 composite skills × 300 lines)
- **After**: ~200 lines (1 pipeline skill × 50 lines + templates)
- **Reduction**: 83% code reduction

### Maintenance Benefits
- Single source of truth for workflows
- Easier to add new formats/destinations
- Consistent error handling
- Better testability

### Performance Benefits
- Faster startup (lazy loading)
- Lower memory usage
- Better caching opportunities

---

## Recommendations

### ✅ Do Now (Safe, High Impact)
1. **Create unified `last30days-to-output` pipeline skill**
   - Consolidates 3 skills into 1
   - No breaking changes
   - Immediate code reduction

2. **Migrate remaining composite skills to pipeline framework**
   - Consistent architecture
   - Easier maintenance
   - No breaking changes

### 🔄 Do Later (Requires Testing)
3. **Extract pipeline templates**
   - Reduce boilerplate
   - Standardize patterns
   - Requires careful design

### ⚠️ Consider (Breaking Changes)
4. **Consolidate similar skills**
   - Requires updating callers
   - Coordinate with users
   - High impact, needs planning

---

## Summary

**Current State:**
- 34 skills, 80 tools
- 4 composite skills, 1 pipeline skill
- Significant duplication in composite skills

**Optimization Potential:**
- **83% code reduction** possible in composite skills
- **Consistent framework** (pipeline-only)
- **Better maintainability** (declarative configs)
- **No breaking changes** (backward compatible migration)

**Next Steps:**
1. Create unified `last30days-to-output` pipeline skill
2. Migrate remaining composite skills
3. Extract common templates
4. Document migration guide
