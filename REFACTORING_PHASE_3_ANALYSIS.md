# Refactoring Phase 3+ Analysis - Additional Scope

**Current State**: conductor.py = 5,035 lines
**Extracted So Far**: 7 managers (~1,139 lines)
**Remaining Extractable**: ~3,200+ lines
**Target**: ~800 lines (pure orchestration)

---

## Major Subsystems Still in conductor.py

### D) Output Registry Management (~500 lines)

**Lines**: 4494-4694 (~200 lines of methods + usage throughout)

**Methods**:
- `_detect_output_type()` - Detect if output is dict, Prediction, dataclass, etc.
- `_extract_schema()` - Extract schema from output structure
- `_generate_preview()` - Generate preview string for outputs
- `_generate_tags()` - Generate semantic tags for outputs
- `_register_output_in_registry()` - Register outputs in IOManager
- `_register_output_in_registry_fallback()` - Fallback registration
- `get_actor_outputs()` - Get all actor outputs
- `get_output_from_actor()` - Get specific actor output

**Extraction Target**: `OutputRegistryManager`

**Responsibilities**:
- Output type detection and classification
- Schema extraction from various output formats
- Preview generation for debugging
- Tag generation for semantic search
- Output registration in IOManager
- Output retrieval API

**Benefits**:
- Centralized output handling logic
- Easier to support new output types
- Testable output detection/extraction
- Clear API for output access

---

### E) Actor Wrapping & Lifecycle (~450 lines)

**Lines**: 889-1337 (~450 lines)

**Methods**:
- `_should_wrap_actor()` - Determine if actor needs wrapping
- `_wrap_actor_with_jotty()` - Wrap actor with Jotty enhancements
- `_get_auto_discovered_dspy_tools()` - Auto-discover tools (already delegated to ToolDiscoveryManager)
- `_get_architect_tools()` - Filter tools for Planner (already delegated)
- `_get_auditor_tools()` - Filter tools for Reviewer (already delegated)
- `_should_inject_registry_tool()` - Decide registry injection
- `_load_annotations()` - Load actor annotations

**Extraction Target**: `ActorLifecycleManager`

**Responsibilities**:
- Actor wrapping decisions
- DSPy actor enhancement
- Tool injection
- Annotation loading
- Actor initialization

**Benefits**:
- Isolated actor lifecycle logic
- Easier to add new actor types
- Clear wrapping strategy
- Testable wrapping decisions

---

### F) Advanced Parameter Resolution (~800 lines)

**Lines**: 1025-3915 (~2,890 lines total parameter logic!)

**Methods** (in conductor.py still):
- `_resolve_param_from_iomanager()` - Resolve from IOManager
- `_resolve_param_by_type()` - Type-based resolution
- `resolve_input()` - Input specification resolution
- `_resolve_parameter()` - Main parameter resolution (DUPLICATE of ParameterResolutionManager!)
- `_extract_from_metadata_manager()` - Extract from metadata
- `_semantic_extract()` - LLM-based semantic extraction
- `_llm_match_field()` - LLM field matching
- `_extract_from_output()` - Extract from actor outputs
- `_build_param_mappings()` - Build parameter dependency map
- `_find_parameter_producer()` - Find which actor produces parameter
- `_introspect_actor_signature()` - Inspect actor signature

**Extraction Target**: **Deeper integration with existing ParameterResolutionManager**

**Current Issue**: We created ParameterResolutionManager as interface, but conductor.py still has ~800 lines of parameter logic!

**Strategy**:
1. Move ALL parameter resolution logic into ParameterResolutionManager
2. Remove duplicates from conductor.py
3. Update conductor.py to fully delegate to manager

**Benefits**:
- Single source of truth for parameter resolution
- No duplicated logic
- Easier to debug parameter issues
- Testable parameter resolution strategies

---

### G) State & Action Management (~350 lines)

**Lines**: 2474-2838 (~364 lines)

**Methods**:
- `_get_current_state()` - Get current episode state for RL
- `_get_available_actions()` - Get available actions for RL

**Extraction Target**: `StateActionManager`

**Responsibilities**:
- RL state representation
- Available action enumeration
- State-action space management
- Feature extraction for RL

**Benefits**:
- Centralized state/action logic
- Easier to modify state representation
- Testable state extraction
- Clear RL interface

---

### H) Pattern Learning & Prompt Evolution (~200 lines)

**Lines**: 261-399 (PolicyExplorer class) + usage

**Methods** (PolicyExplorer):
- `record_episode()` - Record episode for pattern learning
- `should_update_prompts()` - Check if prompts should evolve
- `update_prompt()` - Update prompt based on patterns
- `_extract_pattern()` - Extract patterns from trajectories

**Extraction Target**: `PromptEvolutionManager` (or enhance existing PolicyExplorer)

**Note**: PolicyExplorer already exists but is embedded in conductor.py. Should be extracted fully.

**Responsibilities**:
- Episode recording for pattern detection
- Trajectory pattern extraction
- Prompt evolution decisions
- Prompt update logic

**Benefits**:
- Isolated prompt evolution logic
- Testable pattern extraction
- Clear evolution strategy
- Easier to experiment with prompt learning

---

### I) Domain & Task Inference (~150 lines)

**Lines**: 4881-4914 + context

**Methods**:
- `_infer_domain_from_actor()` - Infer business domain from actor name
- `_infer_task_type_from_task()` - Infer task type from description

**Extraction Target**: `DomainInferenceManager`

**Responsibilities**:
- Domain classification from actor names
- Task type classification from descriptions
- Context enrichment
- Semantic understanding

**Benefits**:
- Centralized domain logic
- Easier to add new domains
- Testable inference
- Clear classification strategy

---

### J) Deeper Integration of Existing Managers (~400 lines)

**Current Issue**: We created managers but conductor.py still has duplicate methods!

**Duplicates to Remove**:

1. **Tool Execution** (lines 1101-1228):
   - `_call_tool_with_cache()` - DUPLICATE of ToolExecutionManager.call_tool_with_cache()
   - `_build_helpful_error_message()` - DUPLICATE of ToolExecutionManager.build_helpful_error_message()
   - `_build_enhanced_tool_description()` - DUPLICATE of ToolExecutionManager.build_enhanced_tool_description()

2. **Metadata Orchestration** (lines 4694-4880):
   - `_fetch_all_metadata_directly()` - DUPLICATE of MetadataOrchestrationManager.fetch_all_metadata_directly()
   - `_enrich_business_terms_with_filters()` - DUPLICATE of MetadataOrchestrationManager.enrich_business_terms_with_filters()
   - `_merge_filter_into_term()` - DUPLICATE of MetadataOrchestrationManager.merge_filter_into_term()

**Strategy**: Replace all calls with manager methods, delete duplicates

---

## Extraction Priority Ranking

### HIGH PRIORITY (Max Impact, Clear Boundaries)

**Phase 3.1: Remove Manager Duplicates** (~400 lines)
- Remove duplicate tool execution methods → use ToolExecutionManager
- Remove duplicate metadata methods → use MetadataOrchestrationManager
- Remove duplicate parameter methods → use ParameterResolutionManager
- **Impact**: Immediate 400-line reduction, zero logic changes

**Phase 3.2: OutputRegistryManager** (~500 lines)
- Extract all output handling logic
- Clear responsibility boundary
- High reusability
- **Impact**: 500-line reduction, improved output handling

**Phase 3.3: ActorLifecycleManager** (~450 lines)
- Extract actor wrapping/initialization
- Clear lifecycle management
- **Impact**: 450-line reduction, cleaner actor setup

### MEDIUM PRIORITY (Good Impact, Some Complexity)

**Phase 3.4: StateActionManager** (~350 lines)
- Extract RL state/action management
- Requires coordination with LearningManager
- **Impact**: 350-line reduction, better RL abstraction

**Phase 3.5: PromptEvolutionManager** (~200 lines)
- Extract or fully separate PolicyExplorer
- Pattern learning isolation
- **Impact**: 200-line reduction, testable prompt evolution

### LOWER PRIORITY (Smaller Impact)

**Phase 3.6: DomainInferenceManager** (~150 lines)
- Domain/task classification
- Smaller subsystem
- **Impact**: 150-line reduction

---

## Projected Outcome

### After Phase 3.1-3.6 Completion:

**Current**: 5,035 lines
**Remove duplicates (3.1)**: -400 lines = 4,635 lines
**OutputRegistryManager (3.2)**: -500 lines = 4,135 lines
**ActorLifecycleManager (3.3)**: -450 lines = 3,685 lines
**StateActionManager (3.4)**: -350 lines = 3,335 lines
**PromptEvolutionManager (3.5)**: -200 lines = 3,135 lines
**DomainInferenceManager (3.6)**: -150 lines = 2,985 lines

**Final conductor.py**: ~2,985 lines
**Total extracted**: ~2,050 lines (40% reduction from current)
**Total managers**: 13 (7 existing + 6 new)

### Remaining in conductor.py (~2,985 lines):
- Main orchestration loop (run_sync) - ~1,200 lines
- Episode initialization and coordination - ~800 lines
- Actor scheduling and execution flow - ~600 lines
- Helper utilities - ~385 lines

**Further reduction needed**: ~2,185 lines to reach 800-line target

---

## Next Steps Options

### Option 1: Quick Wins (Remove Duplicates)
**Phase 3.1 Only** - Remove manager duplicates
- Fastest impact (400 lines immediately)
- Zero risk (just removing duplicates)
- Improves manager usage
- Estimated time: 1-2 hours

### Option 2: Major Subsystems (Output + Actor)
**Phases 3.1 + 3.2 + 3.3** - Duplicates + Output + Actor
- High impact (1,350 lines total)
- Clear boundaries
- Significant improvement
- Estimated time: 4-6 hours

### Option 3: Complete Phase 3 (All 6 Phases)
**Phases 3.1 through 3.6** - All remaining subsystems
- Maximum impact (2,050 lines total)
- Conductor.py reduced to ~2,985 lines
- 13 total managers
- Estimated time: 8-12 hours

### Option 4: Strategic Focus
**Pick specific subsystems** based on current needs:
- Need better output handling? → Phase 3.2 (OutputRegistryManager)
- Need cleaner actor setup? → Phase 3.3 (ActorLifecycleManager)
- Need better RL state management? → Phase 3.4 (StateActionManager)

---

## Recommendation

**Start with Option 1 (Phase 3.1)** - Remove duplicates:

**Why**:
1. **Immediate impact** - 400 lines removed in <2 hours
2. **Zero risk** - Just using existing managers
3. **Validates managers** - Proves managers work in real usage
4. **Clean foundation** - Makes subsequent extractions cleaner

**Then assess**: After Phase 3.1, decide if you want to continue with 3.2+ based on results.

---

**Question**: Which option do you prefer?
- A) Phase 3.1 only (remove duplicates - quick win)
- B) Phases 3.1 + 3.2 + 3.3 (duplicates + output + actor - major subsystems)
- C) All Phase 3 (complete remaining extractions)
- D) Custom (pick specific subsystems)
