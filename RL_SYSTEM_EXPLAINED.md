# 🧠 Jotty RL System Explained

**What the Reinforcement Learning System Does**

---

## 🎯 Overview

Jotty's RL system is **fundamentally different** from traditional RL. Instead of updating neural network weights, it uses **"Context as Gradient"** - learning happens by updating the context/prompts that agents see, not by updating model weights.

### Key Innovation: **LLM-Based RL**

```
Traditional RL:  weights -= learning_rate * gradient  (needs millions of simulations)
Jotty RL:        context += lessons_learned           (context IS learning)
```

---

## 🔄 How It Works

### 1. **TD(λ) Learning** (`learning.py`)

**What it does:**
- Learns the **value** of memories (which memories are useful for which goals)
- Uses Temporal Difference learning with eligibility traces
- Updates values based on actual outcomes vs. predicted values

**Key Components:**

#### **TDLambdaLearner**
```python
# Records memory access during episode
learner.record_access(memory, step_reward=0.1)

# At episode end, updates values based on final reward
updates = learner.end_episode(
    final_reward=1.0,  # Success!
    memories=all_memories
)
# Returns: [(memory_key, old_value, new_value), ...]
```

**How it learns:**
1. **During episode**: Tracks which memories are accessed (eligibility traces)
2. **At episode end**: Calculates TD error: `δ = R + γV(s') - V(s)`
3. **Updates values**: `V(s) ← V(s) + αδe(s)` where `e(s)` is eligibility trace
4. **Result**: Memories that led to success get higher values

**Example:**
```
Episode 1: Memory "Use partition column for date filters" accessed → Task succeeds
→ Value updated: V("Use partition column...") = 0.3 → 0.7

Episode 2: Similar query → Memory retrieved (high value) → Task succeeds faster
→ Context injection: "High-value pattern: Use partition column for date filters"
```

---

### 2. **LLM-Based Q-Learning** (`q_learning.py`)

**What it does:**
- Predicts **Q-values** (expected reward) for state-action pairs
- Uses **natural language Q-table** instead of numeric tables
- LLM reasons about expected outcomes

**Key Innovation: Natural Language Q-Table**

Instead of:
```python
Q[state_hash, action_hash] = 0.75  # Just a number
```

Jotty uses:
```python
Q["QUERY: Count P2P transactions | DOMAIN: UPI | TABLES: fact_upi_transactions",
  "ACTION: Use partition column dl_last_updated"] = {
    'value': 0.75,
    'context': [past experiences],
    'learned_lessons': ["✅ Use partition columns for date filters"],
    'visit_count': 5
}
```

**Why this works:**
- LLMs understand **semantic similarity** in natural language
- Can generalize: "Count P2P transactions" ≈ "Get P2P transaction count"
- No need for exact state matching

**Components:**

#### **LLMQPredictor**
```python
# Predict Q-value for a state-action pair
q_value, confidence, suggestion = predictor.predict_q_value(
    state={"query": "Count P2P transactions", "tables": ["fact_upi"]},
    action={"actor": "SQLGenerator", "task": "generate_query"}
)
# Returns: (0.75, 0.9, None)  # High Q-value, high confidence
```

**How it learns:**
1. **Records experience**: `add_experience(state, action, reward, next_state)`
2. **Updates Q-value**: `Q(s,a) ← Q(s,a) + α[r + γmax Q(s',a') - Q(s,a)]`
3. **Extracts lessons**: Converts numeric updates to natural language lessons
4. **Stores in Q-table**: Natural language state-action → Q-value + lessons

**Example:**
```
Experience: State="Count P2P", Action="Use transaction_category", Reward=-0.2 (failed)
→ Q-value updated: Q("Count P2P", "Use transaction_category") = 0.5 → 0.3
→ Lesson extracted: "❌ AVOID: Using transaction_category for P2P queries → FAILED"

Next similar query:
→ Q-value prediction: Low Q-value (0.3) → Avoid this action
→ Context injection: "Low-value pattern: AVOID using transaction_category for P2P"
```

---

### 3. **Predictive Multi-Agent RL** (`predictive_marl.py`)

**What it does:**
- Each agent **predicts what other agents will do**
- Compares predictions with actual outcomes
- Learns from **divergence** (prediction errors)

**Key Innovation: Theory of Mind**

Agents build **models of other agents**:
```python
agent_model = {
    'agent_name': 'SQLGenerator',
    'action_patterns': ['Uses partition columns', 'Prefers COUNT over SUM'],
    'cooperation_score': 0.8,  # High cooperation
    'predictability_score': 0.9  # Very predictable
}
```

**How it works:**

1. **Predict Trajectory**: Agent predicts what will happen next
   ```python
   predicted = predictor.predict_trajectory(
       current_state=state,
       acting_agent="SQLGenerator",
       proposed_action="Generate SQL",
       other_agents=[...],
       horizon=5  # Predict 5 steps ahead
   )
   # Returns: PredictedTrajectory with predicted_reward=0.8
   ```

2. **Execute**: Agents actually execute
   ```python
   actual = execute_agents()
   # Returns: ActualTrajectory with actual_reward=0.6
   ```

3. **Compare**: Calculate divergence
   ```python
   divergence = compare(predicted, actual)
   # Returns: Divergence with action_divergence=0.2, reward_divergence=0.2
   ```

4. **Learn**: Extract lessons from divergence
   ```python
   learning = extract_learning(divergence)
   # Returns: "Outcome worse than expected - SQLGenerator was slower than predicted"
   ```

**Why this matters:**
- Agents learn to **coordinate** better
- Predictions improve over time
- Emergent cooperation through predictive modeling

---

### 4. **Credit Assignment** (`algorithmic_credit.py`)

**What it does:**
- Assigns credit to agents in multi-agent scenarios
- Uses **Shapley Value** (fair distribution) and **Difference Rewards** (counterfactual impact)

**Shapley Value:**
- Fair credit based on **marginal contribution**
- "What would happen if this agent wasn't here?"

**Difference Rewards:**
- Counterfactual: `G - G_{-i}` where `G_{-i}` is reward without agent i
- Measures **actual impact** of each agent

**Example:**
```python
# Three agents: SQLGenerator, Validator, Executor
# Global reward: 0.8 (success)

credits = credit_assigner.assign_credit(
    agents=['SQLGenerator', 'Validator', 'Executor'],
    global_reward=0.8
)

# Returns:
# {
#   'SQLGenerator': 0.4,  # High marginal contribution
#   'Validator': 0.2,      # Medium contribution
#   'Executor': 0.2       # Medium contribution
# }
```

---

### 5. **Reward Shaping** (`shaped_rewards.py`)

**What it does:**
- Provides **intermediate rewards** for long-horizon tasks
- Solves sparse reward problem (only final reward is too sparse)

**Reward Conditions:**
- ✅ Architect approves → +0.1 reward
- ✅ Tool call succeeds → +0.05 reward
- ✅ Partial task completion → +0.1 reward
- ✅ Good reasoning step → +0.05 reward

**Example:**
```
Episode:
- Step 1: Architect approves → +0.1
- Step 2: Tool call succeeds → +0.05
- Step 3: Partial completion → +0.1
- Step 4: Final success → +0.5
Total: 0.75 (vs. 0.5 without shaping)
```

---

### 6. **Context as Gradient** (The Big Idea)

**Traditional RL:**
```
weights = weights - learning_rate * gradient
```

**Jotty RL:**
```
context = context + learned_lessons
```

**How it manifests:**

1. **Memory Values**: Memories with high values get injected into prompts
   ```python
   # High-value memory
   memory = "Use partition columns for date filters"  # V=0.8
   
   # Gets injected into agent prompt:
   prompt = f"""
   {base_prompt}
   
   # TD(λ) Learned Values:
   ## High-Value Patterns:
   - Use partition columns for date filters (V=0.800)
   """
   ```

2. **Q-Value Lessons**: Q-table lessons become context
   ```python
   # Q-table lesson
   lesson = "✅ LEARNED: Using partition columns → SUCCESS (reward=0.8)"
   
   # Gets injected:
   prompt = f"""
   {base_prompt}
   
   # Learned Lessons:
   {lesson}
   """
   ```

3. **Divergence Learning**: Prediction errors become context
   ```python
   # Divergence learning
   learning = "Outcome worse than expected - use simpler approach"
   
   # Gets injected:
   prompt = f"""
   {base_prompt}
   
   # Recent Learning:
   {learning}
   """
   ```

**Result:**
- Agents see **learned patterns** in their prompts
- No weight updates needed
- Learning happens **in-context**

---

## 📊 Learning Flow Example

### Episode 1: First Attempt

```
Goal: "Count P2P transactions yesterday"

1. Agent tries: Uses column "transaction_category"
2. Result: FAILED (column doesn't exist)
3. Reward: -0.2

Learning:
- TD(λ): Memory "transaction_category" → V = 0.3 → 0.1 (decreased)
- Q-Learning: Q("Count P2P", "Use transaction_category") = 0.5 → 0.3
- Lesson: "❌ AVOID: Using transaction_category for P2P queries"
```

### Episode 2: Second Attempt (With Learning)

```
Goal: "Count P2P transactions yesterday"

1. Context injection:
   "Low-value pattern: AVOID using transaction_category for P2P"
   
2. Agent tries: Uses partition column "dl_last_updated"
3. Result: SUCCESS!
4. Reward: +0.8

Learning:
- TD(λ): Memory "Use partition columns" → V = 0.3 → 0.7 (increased)
- Q-Learning: Q("Count P2P", "Use partition column") = 0.5 → 0.8
- Lesson: "✅ LEARNED: Using partition columns → SUCCESS"
```

### Episode 3: Similar Query (Fast Success)

```
Goal: "Get P2P count for yesterday"

1. Context injection:
   "High-value pattern: Use partition columns for date filters (V=0.700)"
   "✅ LEARNED: Using partition columns → SUCCESS"
   
2. Agent immediately uses partition column
3. Result: SUCCESS! (faster than Episode 2)
4. Reward: +0.9

Learning:
- TD(λ): Memory value increases further: V = 0.7 → 0.85
- Q-Learning: Q-value increases: Q = 0.8 → 0.9
```

---

## 🎯 Key Features

### 1. **No Weight Updates**
- Learning happens through **context updates**
- No neural network training needed
- Works with any LLM

### 2. **Semantic Generalization**
- Natural language Q-table enables generalization
- "Count P2P" ≈ "Get P2P count" (understood by LLM)
- No exact matching needed

### 3. **Multi-Agent Coordination**
- Predictive MARL enables coordination
- Agents learn to predict each other
- Emergent cooperation

### 4. **Fair Credit Assignment**
- Shapley values ensure fairness
- Difference rewards measure impact
- No free-riding

### 5. **Dense Learning Signal**
- Reward shaping provides intermediate rewards
- Faster learning on long-horizon tasks
- Better sample efficiency

---

## 🔬 Technical Details

### TD(λ) Algorithm

```python
# Eligibility trace update
e(s) = γλ * e(s) + 1_{s_t=s}  # Accumulating trace

# TD error
δ = R + γV(s') - V(s)

# Value update
V(s) ← V(s) + α * δ * e(s)
```

### Q-Learning Update

```python
# Q-value update
Q(s,a) ← Q(s,a) + α[r + γ * max_a' Q(s',a') - Q(s,a)]

# With natural language generalization
Q(semantic_state, semantic_action) ← updated_value
```

### Predictive MARL

```python
# Predict trajectory
predicted = LLM.predict(state, action, other_agents, horizon=5)

# Execute
actual = execute_agents()

# Learn from divergence
divergence = compare(predicted, actual)
learning = extract_lessons(divergence)
```

---

## 📈 Performance Characteristics

### Advantages:
- ✅ **Fast learning**: Context updates are immediate
- ✅ **No training data**: Learns from experience
- ✅ **Generalizes**: Semantic similarity enables transfer
- ✅ **Interpretable**: Lessons are human-readable
- ✅ **Multi-agent**: Handles coordination naturally

### Limitations:
- ⚠️ **Context limits**: Can't store infinite lessons
- ⚠️ **LLM dependency**: Requires good LLM reasoning
- ⚠️ **No guarantees**: No convergence proofs (yet)

---

## 🎓 Summary

Jotty's RL system is **revolutionary** because it:

1. **Learns without weight updates** - Context is the gradient
2. **Uses natural language** - Semantic generalization
3. **Enables multi-agent coordination** - Predictive MARL
4. **Provides fair credit** - Shapley + Difference rewards
5. **Solves sparse rewards** - Reward shaping

**The Big Idea:** In LLM-based agents, **prompts are weights**. Updating prompts with learned lessons IS learning. No neural network training needed!

---

*For more details, see:*
- `core/learning/learning.py` - TD(λ) implementation
- `core/learning/q_learning.py` - LLM-based Q-learning
- `core/learning/predictive_marl.py` - Predictive MARL
- `core/learning/algorithmic_credit.py` - Credit assignment
- `core/learning/shaped_rewards.py` - Reward shaping
