# DRY Analysis - What Already Exists in Jotty

## Existing Components (DO NOT DUPLICATE!)

### Orchestrators
- ✅ `MultiAgentsOrchestrator` - Base class
- ✅ `Conductor` - Main multi-agent orchestrator with tools, learning, validation
- ✅ `SingleAgentOrchestrator` - Single agent execution
- ✅ `TaskOrchestrator` - Task-based orchestration
- ✅ `LangGraphOrchestrator` - LangGraph integration

### Infrastructure
- ✅ `SharedContext` - Thread-safe key-value store
- ✅ `SharedScratchpad` - Message passing between agents
- ✅ `AgentMessage` - Inter-agent communication
- ✅ `ScratchpadPersistence` - Save/load sessions to disk

### Tools
- ✅ `MetadataToolRegistry` - Auto-discover tools
- ✅ `ToolManager` - Create DSPy tools with smart parameter resolution
- ✅ `ToolShed` - Agentic tool selection (LLM-based)
- ✅ `AgenticToolSelector` - Select tools based on task
- ✅ `CapabilityIndex` - Map outputs to producers

### Learning
- ✅ `TDLambdaLearner` - TD(λ) learning
- ✅ `LLMQPredictor` - Q-learning with LLM
- ✅ `LLMTrajectoryPredictor` - MARL trajectory prediction
- ✅ `CooperativeCreditAssigner` - Credit assignment
- ✅ `SwarmLearner` - Prompt updates as weight updates

### Validation
- ✅ `InspectorAgent` - Validation manager
- ✅ `Architect` → `Planner` - Pre-execution validation
- ✅ `Auditor` → `Reviewer` - Post-execution validation

### Memory
- ✅ `Cortex` → `HierarchicalMemoryManager` - 5-level memory hierarchy
- ✅ `ConsolidationEngine` → `ConsolidationManager` - Brain-inspired consolidation
- ✅ `BrainInspiredMemoryManager` - Neuroscience-based memory

### State Management
- ✅ `StateManager` - Episode state management
- ✅ `MarkovianTODO` - Long-horizon task management

### Workflow Templates
- ✅ `sequential_team_template.py` - Waterfall (A → B → C)
- ✅ `collaborative_team_template.py` - P2P (A ↔ B ↔ C)
- ✅ `hybrid_team_template.py` - P2P Discovery + Sequential Delivery

### Workflow Functions (Already Implemented!)
- ✅ `p2p_discovery_phase()` - Parallel discovery
- ✅ `sequential_delivery_phase()` - Ordered delivery

---

## What Conductor Already Does

From `core/orchestration/conductor.py`:

```python
class Conductor(MultiAgentsOrchestrator):
    def __init__(self, actors, config):
        # Initialize ALL infrastructure
        self.metadata_tool_registry = MetadataToolRegistry(...)
        self.tool_manager = ToolManager(...)
        self.state_manager = StateManager(...)
        self.parameter_resolver = ParameterResolver(...)
        self.shared_context = SharedContext()
        self.shared_scratchpad = SharedScratchpad()

        # Learning components
        self.td_lambda_learner = TDLambdaLearner(...)
        self.q_learner = LLMQPredictor(...)
        self.marl_predictor = LLMTrajectoryPredictor(...)
        self.credit_assigner = CooperativeCreditAssigner(...)

        # Memory
        self.memory = BrainInspiredMemoryManager(...)

        # Validation
        self.inspector = InspectorAgent(...)

    async def run(self, goal, max_iterations=100, **kwargs):
        # Full episode-based execution with:
        # - Actor scheduling
        # - Tool injection
        # - Validation (Architect/Auditor)
        # - Learning updates
        # - Memory consolidation
        # - State management
```

---

## What's MISSING (Need to Add WITHOUT Duplication)

### NEW Workflow Modes (Not Implemented)
- ❌ **Hierarchical** - Lead + sub-agents
- ❌ **Debate/Consensus** - Competing solutions → vote
- ❌ **Round-Robin** - Iterative refinement
- ❌ **Pipeline** - Data flow A → B → C
- ❌ **Swarm** - Self-organizing agents

### NEW Phases (Not Implemented)
- ❌ **Phase 0: Goal Analysis** - Auto-select workflow mode
- ❌ **Phase 1.5: Plan Validation** - Validate before delivery
- ❌ **Phase 2.5: Execution Validation** - Test during/after delivery
- ❌ **Phase 3: Learning & Retrospective** - Store patterns

### Context Flexibility (Partially Implemented)
- ✅ Already has `**kwargs` for flexible context
- ❌ Need structured context handling (data_folder, codebase, urls, etc.)

---

## DRY Implementation Strategy

### Approach: Thin Wrapper Pattern

Create `UniversalWorkflow` as a THIN WRAPPER that:
1. **USES Conductor** for all heavy lifting
2. **ADDS** only new workflow modes (hierarchical, debate, etc.)
3. **DELEGATES** to existing infrastructure
4. **NO DUPLICATION** of tool management, learning, validation

```python
class UniversalWorkflow:
    """
    Thin wrapper adding workflow modes to Conductor.

    DELEGATES everything to existing Conductor:
    - Tool management → Conductor.tool_manager
    - Learning → Conductor.td_lambda_learner, q_learner, etc.
    - Validation → Conductor.inspector
    - Memory → Conductor.memory
    - State → Conductor.state_manager

    ADDS only:
    - New workflow modes (hierarchical, debate, round-robin, pipeline, swarm)
    - Goal analysis (auto-select mode)
    - Context handling (flexible context types)
    """

    def __init__(self, actors, config):
        # Create Conductor (gets ALL infrastructure)
        self.conductor = Conductor(actors, config)

        # Use Conductor's components (NO DUPLICATION!)
        self.tool_registry = self.conductor.metadata_tool_registry
        self.shared_context = self.conductor.shared_context
        self.scratchpad = self.conductor.shared_scratchpad

        # Only NEW components
        self.goal_analyzer = GoalAnalyzer()  # NEW

    async def run(self, goal, context=None, mode=None, **kwargs):
        # Phase 0: Analyze goal (NEW)
        if mode is None:
            mode = await self.goal_analyzer.recommend_mode(goal, context)

        # Delegate to appropriate workflow
        if mode in ['sequential', 'parallel', 'p2p']:
            # Use existing Conductor.run()
            return await self.conductor.run(goal, **kwargs)

        elif mode == 'hierarchical':
            # NEW mode (implemented here)
            return await self._run_hierarchical(goal, context, **kwargs)

        elif mode == 'debate':
            # NEW mode (implemented here)
            return await self._run_debate(goal, context, **kwargs)

        # ... other new modes
```

---

## Files to Create (NO Duplication)

### 1. Core Workflow (Thin Wrapper)
```
core/orchestration/universal_workflow.py  (300 lines)
├─ UniversalWorkflow class
├─ GoalAnalyzer (auto-select mode)
├─ _run_hierarchical() - NEW
├─ _run_debate() - NEW
├─ _run_round_robin() - NEW
├─ _run_pipeline() - NEW
└─ _run_swarm() - NEW
```

### 2. Workflow Mode Implementations (Reuse Patterns)
```
core/orchestration/modes/
├─ __init__.py
├─ hierarchical.py     (use p2p_discovery_phase + Conductor.run)
├─ debate.py           (use p2p_discovery_phase + voting)
├─ round_robin.py      (use sequential_delivery_phase in loop)
├─ pipeline.py         (use sequential_delivery_phase with data flow)
└─ swarm.py            (use p2p_discovery_phase + dynamic task claiming)
```

### 3. Context Handling (Structured)
```
core/orchestration/context_handler.py  (150 lines)
├─ ContextHandler class
├─ parse_context(goal, context) → structured dict
├─ validate_context() → bool
└─ enrich_context() → add defaults
```

---

## Reuse Plan

### From Conductor (DO NOT REIMPLEMENT!)
- ✅ Tool injection → `conductor._get_auto_discovered_dspy_tools()`
- ✅ Actor execution → `conductor._execute_actor()`
- ✅ Validation → `conductor._run_multi_round_validation()`
- ✅ Learning updates → `conductor._update_learning_after_episode()`
- ✅ Memory consolidation → `conductor.memory.consolidate()`

### From Templates (DO NOT REIMPLEMENT!)
- ✅ P2P discovery → `hybrid_team_template.p2p_discovery_phase()`
- ✅ Sequential delivery → `hybrid_team_template.sequential_delivery_phase()`

### From Existing Infrastructure (DO NOT REIMPLEMENT!)
- ✅ Message passing → `SharedScratchpad`
- ✅ Data storage → `SharedContext`
- ✅ Persistence → `ScratchpadPersistence`
- ✅ Tool selection → `AgenticToolSelector`

---

## Total New Code Estimate

Without duplication:
- `universal_workflow.py`: ~300 lines (thin wrapper)
- `modes/*.py`: ~500 lines total (5 new modes × 100 lines each)
- `context_handler.py`: ~150 lines
- **Total**: ~950 lines of NEW code

Compare to if we duplicated everything: ~5,000+ lines!

**DRY Savings: 81% reduction** ✅

---

## Next Step

Implement `UniversalWorkflow` as thin wrapper that:
1. Creates Conductor internally
2. Delegates to Conductor for existing modes
3. Adds only NEW modes (hierarchical, debate, round-robin, pipeline, swarm)
4. Zero duplication with existing code

Ready to implement?
