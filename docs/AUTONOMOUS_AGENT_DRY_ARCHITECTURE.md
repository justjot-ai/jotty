# Autonomous Agent: DRY Architecture Analysis

## Executive Summary

**Goal**: Build truly autonomous agent product WITHOUT duplicating existing Jotty infrastructure.

**Key Insight**: Jotty already has 80% of what we need! We just need to:
1. Add intent parsing layer (natural language → structured)
2. Enhance AutoAgent with autonomous planning
3. Integrate with existing Conductor/SkillsRegistry/ParameterResolver

**DRY Principle**: Reuse, don't rebuild.

---

## Existing Jotty Components (What We Have)

### ✅ AutoAgent (`core/agents/auto_agent.py`)
**What it does**:
- Task type inference (RESEARCH, CREATION, COMPARISON, etc.)
- Skill discovery via `skill-discovery` skill
- Execution planning per task type
- Tool execution with parameter resolution
- Result handling

**What's missing**:
- Intent parsing (natural language → structured task graph)
- Autonomous planning (research → execution plan)
- Multi-step workflow orchestration
- Configuration management
- Glue code generation

### ✅ Conductor (`core/orchestration/conductor.py`)
**What it does**:
- Multi-agent orchestration
- Parameter resolution
- Tool discovery
- Memory integration
- Learning systems

**What we can reuse**:
- Orchestration engine
- Parameter resolution
- Tool management
- Memory systems

### ✅ SkillsRegistry (`core/registry/skills_registry.py`)
**What it does**:
- Loads skills from disk
- Auto-installs dependencies
- Tool registration
- Skill discovery

**What we can reuse**:
- Skill discovery
- Dependency installation
- Tool access

### ✅ ParameterResolver (`core/orchestration/parameter_resolver.py`)
**What it does**:
- Resolves parameters from multiple sources
- IOManager integration
- Type-based matching
- LLM-based field matching

**What we can reuse**:
- Parameter resolution
- Dependency handling

### ✅ MarkovianTODO (`core/orchestration/roadmap.py`)
**What it does**:
- Task decomposition
- Dependency tracking
- State management
- Q-value estimation

**What we can reuse**:
- Task planning
- Dependency resolution
- State tracking

### ✅ TaskOrchestrator (`core/orchestration/task_orchestrator.py`)
**What it does**:
- Task lifecycle management
- Agent spawning
- Monitoring
- Deployment hooks

**What we can reuse**:
- Lifecycle management
- Monitoring

---

## Architecture: Stitching Everything Together

### Layer 1: Intent Parser (NEW - Thin Layer)
**Purpose**: Convert natural language → structured task graph

**Reuses**: Nothing (this is the new abstraction layer)

**Output**: `TaskGraph` (structured representation)

```python
class IntentParser:
    """Thin layer: Natural language → TaskGraph"""
    def parse(self, user_request: str) -> TaskGraph:
        # Uses pattern matching or LLM
        # Returns structured TaskGraph
```

### Layer 2: Autonomous Planner (ENHANCED AutoAgent)
**Purpose**: TaskGraph → ExecutionPlan

**Reuses**:
- ✅ AutoAgent's `_infer_task_type()` → TaskGraph.task_type
- ✅ AutoAgent's `_discover_skills()` → SkillsRegistry
- ✅ AutoAgent's `_plan_execution()` → Enhanced with research
- ✅ MarkovianTODO → Task decomposition

**Enhancements**:
- Add research phase (web-search skill)
- Add tool discovery from web
- Add dependency planning
- Add configuration planning

```python
class AutonomousPlanner:
    """Enhanced AutoAgent planning"""
    def __init__(self):
        # Reuse AutoAgent components
        self.auto_agent = AutoAgent()
        self.skills_registry = get_skills_registry()
        self.todo = MarkovianTODO()
    
    async def plan(self, task_graph: TaskGraph) -> ExecutionPlan:
        # 1. Research (if needed) - use web-search skill
        # 2. Discover skills - reuse AutoAgent._discover_skills()
        # 3. Plan execution - reuse AutoAgent._plan_execution()
        # 4. Add dependencies - use MarkovianTODO
        # 5. Add configuration steps
        return ExecutionPlan(...)
```

### Layer 3: Autonomous Executor (ENHANCED AutoAgent + Conductor)
**Purpose**: Execute ExecutionPlan autonomously

**Reuses**:
- ✅ AutoAgent's `execute()` → Tool execution
- ✅ AutoAgent's `_execute_tool()` → Skill execution
- ✅ AutoAgent's `_resolve_params()` → Parameter resolution
- ✅ Conductor's orchestration → Multi-agent coordination
- ✅ ParameterResolver → Dependency resolution
- ✅ SkillsRegistry → Tool access
- ✅ SkillDependencyManager → Auto-installation

**Enhancements**:
- Add configuration management
- Add glue code generation
- Add error recovery
- Add scheduling setup

```python
class AutonomousExecutor:
    """Enhanced execution using existing components"""
    def __init__(self):
        # Reuse existing components
        self.auto_agent = AutoAgent()
        self.conductor = Conductor(...)  # For complex workflows
        self.parameter_resolver = ParameterResolver(...)
        self.skills_registry = get_skills_registry()
        self.dependency_manager = get_dependency_manager()
    
    async def execute(self, plan: ExecutionPlan) -> ExecutionResult:
        # For each step:
        # 1. Install dependencies - use SkillDependencyManager
        # 2. Configure - prompt user if needed
        # 3. Execute - use AutoAgent.execute() or Conductor.run()
        # 4. Handle errors - retry logic
        return ExecutionResult(...)
```

### Layer 4: Workflow Memory (REUSE Existing Memory)
**Purpose**: Learn and reuse patterns

**Reuses**:
- ✅ HierarchicalMemory → Pattern storage
- ✅ Memory consolidation → Pattern extraction
- ✅ Learning systems → Pattern matching

**Enhancements**:
- Add workflow pattern extraction
- Add pattern matching
- Add adaptation logic

```python
class WorkflowMemory:
    """Reuses existing memory systems"""
    def __init__(self):
        self.memory = HierarchicalMemory(...)
    
    def remember(self, task_graph, execution_plan, result):
        # Store pattern in memory
        pattern = self._extract_pattern(task_graph, execution_plan)
        self.memory.store(pattern, result)
    
    def recall(self, task_graph):
        # Find similar pattern
        similar = self.memory.find_similar(task_graph)
        return self._adapt_plan(similar, task_graph)
```

---

## Refactored Architecture Diagram

```
User Request (Natural Language)
    ↓
IntentParser (NEW - thin layer)
    ↓
TaskGraph (structured)
    ↓
AutonomousPlanner (ENHANCED AutoAgent)
    ├── Reuses: AutoAgent._infer_task_type()
    ├── Reuses: AutoAgent._discover_skills()
    ├── Reuses: AutoAgent._plan_execution()
    ├── Reuses: MarkovianTODO for dependencies
    └── Adds: Research phase, configuration planning
    ↓
ExecutionPlan (structured)
    ↓
AutonomousExecutor (ENHANCED AutoAgent + Conductor)
    ├── Reuses: AutoAgent.execute()
    ├── Reuses: AutoAgent._execute_tool()
    ├── Reuses: ParameterResolver
    ├── Reuses: SkillsRegistry
    ├── Reuses: SkillDependencyManager
    └── Adds: Configuration, glue code, scheduling
    ↓
ExecutionResult
    ↓
WorkflowMemory (REUSE HierarchicalMemory)
    ├── Reuses: HierarchicalMemory
    ├── Reuses: Memory consolidation
    └── Adds: Pattern extraction, matching
```

---

## Implementation Strategy (DRY)

### Phase 1: Intent Parser (NEW - 1 week)
**Create**: `core/autonomous/intent_parser.py`
- ✅ Already created (basic version)
- ⚠️ Needs LLM integration

**Reuses**: Nothing (new abstraction)

### Phase 2: Enhanced Planner (ENHANCE - 1 week)
**Enhance**: `core/agents/auto_agent.py`
- ✅ Keep existing `AutoAgent` class
- ✅ Add `AutonomousPlanner` that wraps `AutoAgent`
- ✅ Reuse `AutoAgent._discover_skills()`
- ✅ Reuse `AutoAgent._plan_execution()`
- ✅ Add research phase (use web-search skill)
- ✅ Add dependency planning (use MarkovianTODO)

**New File**: `core/autonomous/enhanced_planner.py`
```python
class AutonomousPlanner:
    """Wraps AutoAgent with autonomous planning"""
    def __init__(self):
        self.auto_agent = AutoAgent()  # Reuse existing
        self.todo = MarkovianTODO()    # Reuse existing
    
    async def plan(self, task_graph: TaskGraph) -> ExecutionPlan:
        # 1. Research (use web-search skill)
        # 2. Discover skills (reuse AutoAgent._discover_skills())
        # 3. Plan execution (reuse AutoAgent._plan_execution())
        # 4. Add dependencies (use MarkovianTODO)
        pass
```

### Phase 3: Enhanced Executor (ENHANCE - 2 weeks)
**Enhance**: `core/agents/auto_agent.py`
- ✅ Keep existing `AutoAgent.execute()`
- ✅ Add `AutonomousExecutor` that wraps `AutoAgent`
- ✅ Reuse `AutoAgent._execute_tool()`
- ✅ Reuse `AutoAgent._resolve_params()`
- ✅ Reuse `SkillDependencyManager` for installation
- ✅ Reuse `ParameterResolver` for dependencies
- ✅ Add configuration management
- ✅ Add glue code generation

**New File**: `core/autonomous/enhanced_executor.py`
```python
class AutonomousExecutor:
    """Wraps AutoAgent with autonomous execution"""
    def __init__(self):
        self.auto_agent = AutoAgent()           # Reuse existing
        self.dependency_manager = get_dependency_manager()  # Reuse existing
        self.parameter_resolver = ParameterResolver(...)     # Reuse existing
    
    async def execute(self, plan: ExecutionPlan) -> ExecutionResult:
        # 1. Install dependencies (use SkillDependencyManager)
        # 2. Configure (prompt if needed)
        # 3. Execute (use AutoAgent.execute())
        # 4. Handle errors
        pass
```

### Phase 4: Workflow Memory (REUSE - 1 week)
**Reuse**: `core/memory/cortex.py`
- ✅ Use existing `HierarchicalMemory`
- ✅ Add pattern extraction
- ✅ Add pattern matching
- ✅ Add adaptation logic

**New File**: `core/autonomous/workflow_memory.py`
```python
class WorkflowMemory:
    """Wraps HierarchicalMemory for workflow patterns"""
    def __init__(self):
        self.memory = HierarchicalMemory(...)  # Reuse existing
    
    def remember(self, task_graph, plan, result):
        # Extract pattern and store in memory
        pass
    
    def recall(self, task_graph):
        # Find similar pattern and adapt
        pass
```

---

## Code Reuse Matrix

| Component | What We Need | What Exists | Reuse Strategy |
|-----------|-------------|-------------|----------------|
| **Intent Parsing** | Natural language → TaskGraph | ❌ None | ✅ Create new (thin layer) |
| **Task Type Inference** | Infer task type | ✅ AutoAgent._infer_task_type() | ✅ Reuse directly |
| **Skill Discovery** | Find relevant skills | ✅ AutoAgent._discover_skills() | ✅ Reuse directly |
| **Execution Planning** | Plan execution steps | ✅ AutoAgent._plan_execution() | ✅ Reuse + enhance |
| **Tool Execution** | Execute tools | ✅ AutoAgent._execute_tool() | ✅ Reuse directly |
| **Parameter Resolution** | Resolve dependencies | ✅ ParameterResolver | ✅ Reuse directly |
| **Dependency Installation** | Install packages | ✅ SkillDependencyManager | ✅ Reuse directly |
| **Orchestration** | Multi-agent coordination | ✅ Conductor | ✅ Reuse for complex workflows |
| **Memory** | Store patterns | ✅ HierarchicalMemory | ✅ Reuse + enhance |
| **Task Management** | Track tasks | ✅ MarkovianTODO | ✅ Reuse directly |

---

## Refactored File Structure

```
core/
├── agents/
│   └── auto_agent.py              # ✅ KEEP (existing, working)
│
├── autonomous/                     # NEW: Thin orchestration layer
│   ├── __init__.py
│   ├── intent_parser.py           # ✅ NEW (already created)
│   ├── enhanced_planner.py        # ⚠️ NEW (wraps AutoAgent)
│   ├── enhanced_executor.py       # ⚠️ NEW (wraps AutoAgent)
│   └── workflow_memory.py         # ⚠️ NEW (wraps HierarchicalMemory)
│
├── orchestration/
│   ├── conductor.py               # ✅ KEEP (reuse for complex workflows)
│   ├── parameter_resolver.py      # ✅ KEEP (reuse for dependencies)
│   ├── roadmap.py                 # ✅ KEEP (reuse for task planning)
│   └── task_orchestrator.py       # ✅ KEEP (reuse for lifecycle)
│
├── registry/
│   ├── skills_registry.py         # ✅ KEEP (reuse for tool discovery)
│   └── skill_dependency_manager.py # ✅ KEEP (reuse for installation)
│
└── memory/
    └── cortex.py                  # ✅ KEEP (reuse for pattern storage)
```

---

## Integration Points

### 1. Intent Parser → Enhanced Planner
```python
# IntentParser creates TaskGraph
task_graph = intent_parser.parse("Set up daily Reddit scraping to Notion")

# Enhanced Planner uses AutoAgent internally
planner = AutonomousPlanner()
execution_plan = await planner.plan(task_graph)
```

### 2. Enhanced Planner → Enhanced Executor
```python
# Enhanced Planner creates ExecutionPlan
execution_plan = await planner.plan(task_graph)

# Enhanced Executor uses AutoAgent internally
executor = AutonomousExecutor()
result = await executor.execute(execution_plan)
```

### 3. Enhanced Executor → Existing Components
```python
# Enhanced Executor reuses:
# - AutoAgent.execute() for tool execution
# - SkillDependencyManager for installation
# - ParameterResolver for dependencies
# - SkillsRegistry for tool access
```

### 4. Workflow Memory → Existing Memory
```python
# Workflow Memory wraps HierarchicalMemory
memory = WorkflowMemory()
memory.remember(task_graph, execution_plan, result)
similar_plan = memory.recall(new_task_graph)
```

---

## Example: Complete Flow (DRY)

```python
# User request
user_request = "Set up daily Reddit scraping to Notion"

# Step 1: Intent Parsing (NEW - thin layer)
intent_parser = IntentParser()
task_graph = intent_parser.parse(user_request)
# → TaskGraph(task_type=DATA_PIPELINE, source="reddit", ...)

# Step 2: Autonomous Planning (ENHANCED - reuses AutoAgent)
planner = AutonomousPlanner()  # Wraps AutoAgent internally
execution_plan = await planner.plan(task_graph)
# → ExecutionPlan with steps (reuses AutoAgent._discover_skills(), etc.)

# Step 3: Autonomous Execution (ENHANCED - reuses AutoAgent)
executor = AutonomousExecutor()  # Wraps AutoAgent internally
result = await executor.execute(execution_plan)
# → ExecutionResult (reuses AutoAgent.execute(), SkillDependencyManager, etc.)

# Step 4: Workflow Memory (REUSE - wraps HierarchicalMemory)
memory = WorkflowMemory()  # Wraps HierarchicalMemory internally
memory.remember(task_graph, execution_plan, result)
# → Pattern stored in existing memory system
```

---

## Key Principles

### 1. **Reuse, Don't Rebuild**
- ✅ Use AutoAgent for execution
- ✅ Use Conductor for orchestration
- ✅ Use SkillsRegistry for tool discovery
- ✅ Use ParameterResolver for dependencies
- ✅ Use HierarchicalMemory for patterns

### 2. **Thin Wrapper Pattern**
- New components are **thin wrappers** around existing ones
- Add new capabilities without duplicating code
- Compose existing components, don't replace them

### 3. **Single Responsibility**
- IntentParser: Only parsing
- Enhanced Planner: Only planning (wraps AutoAgent)
- Enhanced Executor: Only execution (wraps AutoAgent)
- Workflow Memory: Only pattern storage (wraps HierarchicalMemory)

### 4. **Backward Compatibility**
- Keep existing AutoAgent unchanged
- New components are optional enhancements
- Existing code continues to work

---

## Next Steps

1. **Review AutoAgent** ✅ (done)
2. **Create Enhanced Planner** (wraps AutoAgent)
3. **Create Enhanced Executor** (wraps AutoAgent)
4. **Create Workflow Memory** (wraps HierarchicalMemory)
5. **Integrate with Conductor** (for complex workflows)
6. **Test end-to-end** (reuse existing tests)

---

## Conclusion

**We don't need to rebuild Jotty!** We just need to:
1. Add intent parsing (thin layer)
2. Enhance AutoAgent with planning (wrapper)
3. Enhance AutoAgent with execution (wrapper)
4. Reuse existing memory (wrapper)

**Result**: True autonomous agent product built on existing Jotty foundation, following DRY principles.
