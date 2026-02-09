# RL Agent Selection + Planner/Architect - How They Work Together

**Date**: 2026-01-17
**Question**: "we also have planner which looks at available agents and then decides. how does that also fit together"

---

## 🎯 TL;DR - Two SEPARATE Concerns

| Component | Purpose | Answers | When |
|-----------|---------|---------|------|
| **Q-Learning (RL)** | **Which agent to run next** | "Should we run Fetcher or Processor next?" | BEFORE selecting task |
| **Planner/Architect** | **Should this agent execute now** | "Should Fetcher proceed given current state?" | AFTER selecting task, BEFORE execution |

**They are COMPLEMENTARY, not competing!**

---

## 📊 Execution Flow

### **Without RL (Original Behavior)**

```
1. Get next task → Uses fixed sequential order
2. Planner validates → "Should this agent proceed?"
   ├─ Yes → Execute agent
   └─ No → Block execution, mark as failed
3. Agent executes
4. Reviewer validates → "Was output valid?"
```

### **With RL (New Behavior)**

```
1. Get next task → Uses Q-value-based ε-greedy selection
   ├─ Get Q-values for all available agents
   ├─ Select best Q-value (70% of time)
   └─ Select random (30% of time - exploration)

2. Planner validates → "Should THIS agent proceed NOW?"
   ├─ Checks preconditions (does it have needed inputs?)
   ├─ Checks context (is this a good time?)
   └─ Decision: proceed=True/False

3. IF Planner says proceed:
   └─ Execute agent

4. Reviewer validates → "Was output valid?"
   ├─ Check output quality
   └─ Decision: valid=True/False

5. RL learns from outcome:
   ├─ If succeeded → increase Q-value for (state, agent) pair
   └─ If failed → decrease Q-value
```

---

## 🔍 Detailed Example: Data Pipeline

### **Scenario**: Process sales data (Fetch → Process → Visualize)

### **Episode 1** (Wrong Order - Visualizer First)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Q-LEARNING SELECTION (Iteration 1)                       │
├─────────────────────────────────────────────────────────────┤
│ Available tasks: [Visualizer, Fetcher, Processor]          │
│ Q-values: Visualizer=0.50, Fetcher=0.50, Processor=0.50    │
│ Selection: Visualizer (random among equals)                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. PLANNER VALIDATION                                       │
├─────────────────────────────────────────────────────────────┤
│ Agent: Visualizer                                           │
│ Context: No data available (no Fetcher output yet)         │
│ Planner checks:                                             │
│   - "Does Visualizer have the data it needs?" → NO          │
│   - "Is this the right time to visualize?" → NO             │
│ Decision: should_proceed = FALSE (BLOCKED!)                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. RL LEARNS FROM FAILURE                                   │
├─────────────────────────────────────────────────────────────┤
│ State: "No data fetched yet"                                │
│ Action: "Run Visualizer"                                    │
│ Reward: NEGATIVE (blocked by Planner)                      │
│ Q-value update: Visualizer Q-value ↓ (0.50 → 0.48)        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. Q-LEARNING SELECTION (Iteration 2)                       │
├─────────────────────────────────────────────────────────────┤
│ Available tasks: [Fetcher, Processor]                      │
│ Q-values: Fetcher=0.50, Processor=0.50                     │
│ Selection: Fetcher (random among equals)                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. PLANNER VALIDATION                                       │
├─────────────────────────────────────────────────────────────┤
│ Agent: Fetcher                                              │
│ Context: Start of pipeline (no dependencies)               │
│ Planner checks:                                             │
│   - "Does Fetcher have what it needs?" → YES (query)        │
│   - "Is this the right time to fetch?" → YES                │
│ Decision: should_proceed = TRUE (PROCEED!)                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. AGENT EXECUTES                                           │
├─────────────────────────────────────────────────────────────┤
│ Fetcher runs → Fetches sales data → Returns JSON           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. REVIEWER VALIDATION                                      │
├─────────────────────────────────────────────────────────────┤
│ Check output: Valid JSON with sales data                   │
│ Decision: is_valid = TRUE                                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. RL LEARNS FROM SUCCESS                                   │
├─────────────────────────────────────────────────────────────┤
│ State: "No data fetched yet"                                │
│ Action: "Run Fetcher"                                       │
│ Reward: POSITIVE (succeeded, produced valid data)          │
│ Q-value update: Fetcher Q-value ↑ (0.50 → 0.62)           │
└─────────────────────────────────────────────────────────────┘
```

---

## 📈 Learning Over Episodes

### **After 10 Episodes**:

```
Q-values learned:

State: "No data yet"
├─ Run Visualizer → Q = 0.35 (often blocked by Planner)
├─ Run Fetcher → Q = 0.75 (always succeeds, provides data)
└─ Run Processor → Q = 0.42 (blocked - needs Fetcher data first)

State: "Fetcher completed"
├─ Run Visualizer → Q = 0.38 (blocked - needs processed data)
├─ Run Fetcher → Q = 0.45 (redundant, already have data)
└─ Run Processor → Q = 0.80 (succeeds, uses Fetcher data)

State: "Processor completed"
├─ Run Visualizer → Q = 0.85 (succeeds, has processed data)
├─ Run Fetcher → Q = 0.40 (redundant)
└─ Run Processor → Q = 0.40 (redundant)
```

### **Result**: RL Learns Optimal Order

```
Episode 1-5:   Mixed (exploring)
Episode 6-15:  Fetcher first (60% of time)
Episode 16-30: Fetcher → Processor → Visualizer (80% of time)
Episode 31+:   Correct order 90%+ of time
```

---

## 🤝 How They Complement Each Other

### **Q-Learning Provides**:
- ✅ **Strategic ordering** - Learns which agent sequences work best
- ✅ **Exploration** - Tries different orders to discover optimal patterns
- ✅ **Adaptation** - Adjusts to changing conditions over time

### **Planner Provides**:
- ✅ **Tactical validation** - Checks if NOW is the right time to run this agent
- ✅ **Safety** - Prevents agents from running without needed inputs
- ✅ **Context awareness** - Understands current state and dependencies

### **Together They Create**:
- 🎯 **Smart ordering** (Q-learning) + **Smart execution** (Planner)
- 🎯 **Learn what works** (Q-learning) + **Validate before running** (Planner)
- 🎯 **Strategic** (which agent) + **Tactical** (should it run now)

---

## 🔄 Potential Conflict Resolution

### **Scenario**: Q-Learning vs Planner Disagreement

```
Q-Learning says: "Run Processor next" (high Q-value)
Planner says: "Block Processor" (no Fetcher data available yet)

Resolution:
1. Planner wins (safety first!)
2. Agent is blocked
3. Q-learning observes negative reward
4. Q-value for "Run Processor without Fetcher data" decreases
5. Next time: Q-learning learns NOT to select Processor in that state
```

**This is LEARNING IN ACTION!** Q-learning discovers through Planner feedback what works and what doesn't.

---

## 📊 Configuration Options

### **Disable Planner (Trust Q-Learning Completely)**
```python
config = JottyConfig(
    enable_rl=True,
    enable_architect=False  # No Planner validation
)
# Result: Q-learning has full control, no safety checks
```

### **Enable Both (Recommended)**
```python
config = JottyConfig(
    enable_rl=True,
    enable_architect=True  # Planner validates
)
# Result: Q-learning learns optimal order, Planner ensures safety
```

### **Planner Only (No RL)**
```python
config = JottyConfig(
    enable_rl=False,
    enable_architect=True  # Planner validates fixed order
)
# Result: Fixed sequential order, Planner blocks unsafe executions
```

---

## 💡 Real-World Analogy

### **Q-Learning = Strategic Planning**
*"Based on past experience, we should do Fetcher first, then Processor, then Visualizer"*

### **Planner = Tactical Validation**
*"Wait, we don't have the database credentials yet. Let's not run Fetcher right now."*

### **Together**:
- ✅ Q-Learning learns the ideal sequence over many episodes
- ✅ Planner ensures each step is safe given current context
- ✅ RL learns from Planner's blocks (low reward) and approvals (high reward)

---

## 🎯 Summary

| Question | Answer |
|----------|--------|
| **Do they conflict?** | No - they operate at different levels (strategic vs tactical) |
| **Which runs first?** | Q-learning selects agent, then Planner validates |
| **Can Planner override Q-learning?** | Yes - Planner can block execution for safety |
| **Does RL learn from Planner blocks?** | Yes! Blocked → negative reward → Q-value decreases |
| **Should I use both?** | Yes (recommended) - Q-learning for ordering, Planner for safety |

---

## 📝 Code Evidence

### **Q-Learning Selection** (`roadmap.py:584-678`)
```python
def get_next_task(self, q_predictor=None, current_state=None, goal=None, epsilon=0.1):
    """Select next task based on Q-values (ε-greedy)"""

    # Get Q-value for each available task
    for task in available_tasks:
        q_value, _, _ = q_predictor.predict_q_value(current_state, action, goal)

    # Select best Q-value (exploitation) or random (exploration)
    return best_task
```

### **Planner Validation** (`inspector.py`)
```python
class InspectorAgent:
    """Planner (Architect) and Reviewer (Auditor) validation"""

    def validate(self, actor, inputs):
        """Validate if actor should proceed"""

        # Planner checks preconditions
        result = self.agent(inputs)

        return ValidationResult(
            should_proceed=result.should_proceed,  # True/False
            reasoning=result.reasoning
        )
```

### **Integration** (`conductor.py`)
```python
async def run(self, goal):
    """Main execution loop"""

    # 1. Q-LEARNING: Select next task
    task = self.todo.get_next_task(
        q_predictor=self.q_learner,  # RL-based selection
        current_state=state,
        goal=goal
    )

    # 2. PLANNER: Validate execution
    if self.config.enable_architect:
        plan_result = await self._run_architect_for_actor(task.actor)
        if not plan_result.should_proceed:
            # Blocked! RL will learn from this
            return EpisodeResult(success=False, ...)

    # 3. EXECUTE: Run agent
    result = await self._execute_actor(task.actor)

    # 4. RL LEARNS: Update Q-values based on outcome
    reward = self._compute_reward(result)
    self.q_learner.update(state, action, reward)
```

---

**Generated**: 2026-01-17
**Purpose**: Clarify Q-Learning + Planner interaction
**Conclusion**: They work TOGETHER, not in conflict! 🤝
