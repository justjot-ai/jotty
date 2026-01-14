# Conductor Refactoring Plan

## Current State

**File**: `core/orchestration/conductor.py`
**Lines**: 4,708
**Methods**: 58
**Classes**: 8 (7 helper classes + 1 main Conductor)

## Problem

The Conductor class violates the Single Responsibility Principle by handling:
1. **Parameter Resolution** - Resolving actor input parameters from multiple sources
2. **Tool Management** - Creating, caching, and injecting tools
3. **State Management** - Managing actor dependencies, outputs, and execution state
4. **Orchestration** - Running actors, managing episodes, learning

## Proposed Solution

Split into **4 focused components**:

### 1. ParameterResolver
**Responsibility**: Resolve actor input parameters from multiple sources

**Methods to Extract**:
- `_build_param_mappings()` - Build parameter mappings
- `_find_parameter_producer()` - Find which actor produces a parameter
- `resolve_input()` - Main entry point for parameter resolution
- `_resolve_parameter()` - Resolve a single parameter
- `_extract_from_metadata_manager()` - Extract from metadata
- `_semantic_extract()` - LLM-based semantic extraction
- `_llm_match_field()` - LLM field matching
- `_extract_from_output()` - Extract from actor outputs
- `_resolve_param_from_iomanager()` - Resolve from IOManager
- `_resolve_param_by_type()` - Resolve by type annotation

**Dependencies**:
- `AgenticParameterResolver` (already external)
- `IOManager` (already external)
- `metadata_provider`
- `param_mappings`
- `io_manager`

**Estimated Lines**: ~400-500 lines

---

### 2. ToolManager
**Responsibility**: Create, cache, and inject tools for actors

**Methods to Extract**:
- `_get_auto_discovered_dspy_tools()` - Auto-discover DSPy tools
- `_get_architect_tools()` - Get tools for Architect
- `_get_auditor_tools()` - Get tools for Auditor
- `_call_tool_with_cache()` - Call tool with caching
- `_build_helpful_error_message()` - Build error messages
- `_build_enhanced_tool_description()` - Build tool descriptions
- `_should_inject_registry_tool()` - Check if registry tool needed

**Dependencies**:
- `metadata_tool_registry`
- `data_registry_tool`
- `metadata_fetcher`
- `config`

**Estimated Lines**: ~300-400 lines

---

### 3. StateManager
**Responsibility**: Manage actor state, dependencies, and outputs

**Methods to Extract**:
- `_introspect_actor_signature()` - Introspect actor signature
- `_detect_output_type()` - Detect output type
- `_extract_schema()` - Extract schema
- `_generate_preview()` - Generate preview
- `_generate_tags()` - Generate tags
- `_register_output_in_registry()` - Register output
- `_register_output_in_registry_fallback()` - Fallback registration
- `get_actor_outputs()` - Get all actor outputs
- `get_output_from_actor()` - Get specific actor output
- `_get_current_state()` - Get current state
- `_get_available_actions()` - Get available actions

**Dependencies**:
- `io_manager`
- `data_registry`
- `registration_orchestrator`
- `actor_signatures`
- `dependency_graph`

**Estimated Lines**: ~500-600 lines

---

### 4. Orchestrator (Refactored Conductor)
**Responsibility**: Core orchestration logic only

**Keeps**:
- `__init__()` - Initialize and compose the 3 components above
- `run_sync()` - Main orchestration loop
- `_build_actor_context()` - Build execution context
- `_should_wrap_actor()` - Check if wrapping needed
- `_wrap_actor_with_jotty()` - Wrap actor
- `_fetch_all_metadata_directly()` - Fetch metadata
- `_enrich_business_terms_with_filters()` - Enrich terms
- `_merge_filter_into_term()` - Merge filters

**Dependencies**:
- **ParameterResolver** (new component)
- **ToolManager** (new component)
- **StateManager** (new component)
- All existing external components

**Estimated Lines**: ~1,200-1,500 lines (reduced from 4,708)

---

## Implementation Strategy

### Phase 1: Create Components (SAFE - No Integration)
1. Create `ParameterResolver` class in new file
2. Create `ToolManager` class in new file
3. Create `StateManager` class in new file
4. **Test each component in isolation**

### Phase 2: Integration (RISKY - Incremental)
1. Update Conductor `__init__` to create component instances
2. Replace method calls with component calls ONE AT A TIME
3. **Test after EACH replacement**
4. Keep old methods as deprecated fallbacks initially

### Phase 3: Cleanup (SAFE)
1. Remove deprecated methods once all tests pass
2. Update documentation
3. Final integration tests

---

## Risk Mitigation

### LOW RISK (Phase 1)
- Creating components doesn't affect existing code
- Can test components independently
- Easy to rollback if issues found

### MEDIUM RISK (Phase 2.1 - First Integration)
- Integrating ParameterResolver first (least dependencies)
- Keep old methods as fallbacks
- Test thoroughly before proceeding

### HIGH RISK (Phase 2.2-2.3 - Full Integration)
- ToolManager and StateManager integration
- More dependencies between components
- Requires extensive testing

---

## Testing Strategy

### Component Tests
- Test each component in isolation
- Mock dependencies
- Verify all methods work independently

### Integration Tests
- Test component interactions
- Verify Conductor still works after each integration
- Run full hello world test

### Regression Tests
- Run all existing baseline tests
- Verify backward compatibility
- Check no performance degradation

---

## Success Criteria

✅ All components tested in isolation
✅ All baseline tests pass
✅ Hello world integration test passes
✅ Code reduced from 4,708 lines to ~2,500-3,000 total
✅ Each component has single, clear responsibility
✅ No functionality broken
✅ Easier to maintain and test

---

## Timeline Estimate

- **Phase 1** (Component Creation): ~2-3 hours
- **Phase 2** (Integration): ~3-4 hours
- **Phase 3** (Cleanup): ~1 hour
- **Total**: ~6-8 hours of careful, methodical work

---

## Next Steps

1. ✅ Create this plan (DONE)
2. Get user approval
3. Create ParameterResolver component
4. Test ParameterResolver
5. Create ToolManager component
6. Test ToolManager
7. Create StateManager component
8. Test StateManager
9. Integrate into Conductor (incrementally)
10. Final testing and cleanup
