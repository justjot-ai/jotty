# Jotty Architecture Clarification

## 🎯 Your Understanding vs Actual Architecture

### Your Flow (Confused):
```
Conductor → SAS/MAS (Auto Agent) → Agentic Planner → AutoAgents → AgentsTodo (Markovian TODO)
```

### ✅ Actual Architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Conductor (MultiAgentsOrchestrator)           │
│                    Orchestrates multiple agents                  │
└───────────────────────────┬─────────────────────────────────────┘
                             │
                             │ Uses
                             ▼
        ┌────────────────────────────────────────────┐
        │   SingleAgentOrchestrator (SAS)              │
        │   Wraps ONE agent with Architect/Auditor    │
        │   - Architect: Plans execution              │
        │   - Agent: Executes task                    │
        │   - Auditor: Validates output              │
        └────────────────────────────────────────────┘
                             │
                             │ Can wrap
                             ▼
        ┌────────────────────────────────────────────┐
        │   AutoAgent                                  │
        │   Autonomous task execution                  │
        │   - Discovers skills                        │
        │   - Uses AgenticPlanner for planning        │
        │   - Executes tools                          │
        └───────────────┬──────────────────────────────┘
                        │
                        │ Uses
                        ▼
        ┌────────────────────────────────────────────┐
        │   AgenticPlanner                            │
        │   Plans execution steps                    │
        │   - Infers task type                       │
        │   - Selects skills                          │
        │   - Creates ExecutionPlan                   │
        └────────────────────────────────────────────┘
                             │
                             │ Creates
                             ▼
        ┌────────────────────────────────────────────┐
        │   ExecutionPlan                             │
        │   List of ExecutionStep                    │
        │   - skill_name, tool_name, params          │
        └────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                    Conductor (MultiAgentsOrchestrator)           │
│                    ALSO uses:                                   │
└───────────────────────────┬─────────────────────────────────────┘
                             │
                             │ Uses for state tracking
                             ▼
        ┌────────────────────────────────────────────┐
        │   MarkovianTODO (from roadmap.py)          │
        │   Tracks task state/progress                │
        │   - Task dependencies                      │
        │   - Progress tracking                       │
        │   - RL state for Q-learning                 │
        │   - Checkpoint/resume                       │
        └────────────────────────────────────────────┘
```

---

## 📊 Key Components Explained

### 1. **Conductor** (`MultiAgentsOrchestrator`)
- **Purpose**: Orchestrates multiple agents
- **Uses**: 
  - `SingleAgentOrchestrator` (SAS) for each agent
  - `MarkovianTODO` for task state tracking
- **Flow**: Goal → Decompose → Assign to agents → Coordinate

### 2. **SingleAgentOrchestrator (SAS)**
- **Purpose**: Wraps ONE agent with validation
- **Components**:
  - Architect: Plans execution (pre-validation)
  - Agent: Executes task
  - Auditor: Validates output (post-validation)
- **Can wrap**: Any DSPy agent, including `AutoAgent`

### 3. **AutoAgent**
- **Purpose**: Autonomous task execution
- **Uses**: `AgenticPlanner` for planning
- **Flow**: Task → Plan → Execute → Result

### 4. **AgenticPlanner**
- **Purpose**: Plans execution steps
- **Input**: Task description
- **Output**: `ExecutionPlan` (list of steps)
- **NOT state tracking** - just planning!

### 5. **MarkovianTODO** (from `roadmap.py`)
- **Purpose**: State tracking (NOT planning!)
- **Used by**: Conductor for multi-agent coordination
- **Tracks**: Task state, dependencies, progress
- **Note**: This is technical/internal - users don't interact with it directly

---

## 🔄 Actual Flow Examples

### Example 1: Conductor with AutoAgent

```python
# Step 1: Create AutoAgent
auto_agent = AutoAgent()  # Uses AgenticPlanner internally

# Step 2: Wrap in SingleAgentOrchestrator (SAS)
sas = SingleAgentOrchestrator(
    agent=auto_agent,  # ← AutoAgent wrapped here
    architect_prompts=["plan.md"],
    auditor_prompts=["validate.md"]
)

# Step 3: Use in Conductor (MAS)
conductor = MultiAgentsOrchestrator(
    actors=[ActorConfig("auto", sas, ...)],
    ...
)

# Step 4: Conductor uses MarkovianTODO internally
result = await conductor.run(goal="Research topic")
# Conductor tracks state with MarkovianTODO
# But AutoAgent uses AgenticPlanner for planning
```

### Example 2: Direct AutoAgent (No Conductor)

```python
# Just AutoAgent + AgenticPlanner
auto_agent = AutoAgent()  # Has AgenticPlanner inside
result = await auto_agent.execute("Research topic")
# No Conductor, no MarkovianTODO
```

---

## ✅ Corrected Flow

### For Single Agent (AutoAgent):
```
User Task
    ↓
AutoAgent.execute()
    ↓
AgenticPlanner.plan_execution()  ← Plans steps
    ↓
ExecutionPlan (steps)
    ↓
AutoAgent executes steps
    ↓
Result
```

### For Multi-Agent (Conductor):
```
User Goal
    ↓
Conductor.run()
    ↓
MarkovianTODO (tracks state)  ← State tracking
    ↓
SingleAgentOrchestrator (SAS) for each agent
    ↓
AutoAgent (if wrapped)
    ↓
AgenticPlanner (plans execution)
    ↓
ExecutionPlan
    ↓
Result
```

---

## 🎯 What You Were Missing

1. **SAS/MAS distinction**:
   - **SAS** = SingleAgentOrchestrator (wraps ONE agent)
   - **MAS** = MultiAgentsOrchestrator (Conductor, orchestrates MULTIPLE agents)

2. **AgenticPlanner vs MarkovianTODO**:
   - **AgenticPlanner** = Plans execution steps (what to do)
   - **MarkovianTODO** = Tracks state/progress (what happened)

3. **Conductor doesn't directly use AutoAgent**:
   - Conductor uses `SingleAgentOrchestrator` (SAS)
   - SAS can wrap `AutoAgent` (or any agent)
   - AutoAgent uses `AgenticPlanner` internally

4. **MarkovianTODO is technical**:
   - Used internally by Conductor
   - Users don't interact with it directly
   - It's for state tracking, not planning

---

## 📝 Summary

**Correct Flow**:
```
Conductor (MAS)
    ↓
SingleAgentOrchestrator (SAS) - wraps agents
    ↓
AutoAgent (optional - can be any agent)
    ↓
AgenticPlanner - plans execution
    ↓
ExecutionPlan - list of steps

Conductor ALSO uses:
    ↓
MarkovianTODO - tracks state/progress
```

**Key Insight**: 
- **Planning** = AgenticPlanner (what to do)
- **State Tracking** = MarkovianTODO (what happened)
- **Orchestration** = Conductor (coordinates agents)
- **Validation** = SingleAgentOrchestrator (Architect/Auditor)

---

*Clarification completed: 2026-01-28*
