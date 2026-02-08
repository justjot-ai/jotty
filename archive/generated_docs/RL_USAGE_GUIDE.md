# 🧠 Reinforcement Learning (RL) Usage Guide

## When to Enable vs Disable RL

### ✅ Enable RL (`enable_rl=True`) - **RECOMMENDED for Production**

Use RL when you have **repeated, domain-specific workflows** where agents learn which contributions work best:

#### Example 1: SQL Query Generation System
```python
# Runs 100s of times per day translating natural language to SQL
config = SwarmConfig(
    enable_rl=True,  # ← Learn which agents are best at SQL generation
    rl_verbosity="normal"
)

# After 50 runs, RL learns:
# - SchemaAnalyzer: 40% contribution to success
# - SQLGenerator: 35% contribution
# - Validator: 25% contribution
# → Future queries route to best agents first
```

**Why RL helps**: Same problem class (NL→SQL) with consistent agent roles. RL optimizes agent contribution weights.

#### Example 2: Customer Support Chatbot
```python
# Handles 1000s of support tickets with similar patterns
config = SwarmConfig(
    enable_rl=True,  # ← Learn best response strategies
    rl_verbosity="normal"
)

# RL learns:
# - IntentClassifier: High value for triage
# - KnowledgeRetriever: Critical for technical issues
# - ResponseGenerator: Important for tone
```

**Why RL helps**: Recurring problem patterns allow RL to learn optimal agent coordination.

#### Example 3: Code Review System
```python
# Reviews 100s of PRs per week with similar quality checks
config = SwarmConfig(
    enable_rl=True,  # ← Learn which checks catch most issues
    rl_verbosity="quiet"  # Less logging in production
)

# RL learns:
# - SecurityScanner: Critical for API changes
# - PerformanceAnalyzer: Important for DB queries
# - StyleChecker: Lower priority
```

**Why RL helps**: Consistent review workflow where RL identifies high-value agents.

---

### ❌ Disable RL (`enable_rl=False`) - For One-Off Tasks

Use when tasks are **unrelated** or **run only once**:

#### Example 1: Quick Demo/Test
```python
# Single run to demonstrate functionality
config = SwarmConfig(
    enable_rl=False,  # ← No learning needed for demo
)
```

**Why disable**: No pattern to learn from single execution.

#### Example 2: Completely Different Tasks
```python
# Task 1: Analyze Python code
# Task 2: Write SQL query
# Task 3: Generate documentation
# Task 4: Parse JSON data

config = SwarmConfig(
    enable_rl=False,  # ← Tasks are unrelated
)
```

**Why disable**: No transferable knowledge between unrelated tasks.

#### Example 3: Research/Exploration
```python
# Trying different agent combinations to see what works
config = SwarmConfig(
    enable_rl=False,  # ← Still exploring, not optimizing
)
```

**Why disable**: Premature optimization - first find the right agents, then enable RL.

---

## 🎯 Key Principle: Problem Domain Consistency

**RL is valuable when:**
```
Same Agent Set + Similar Problem Class = RL learns optimal coordination
```

**Examples:**
- ✅ 100 SQL queries with same schema → RL learns best agent weights
- ✅ 1000 code reviews with similar patterns → RL learns valuable checks
- ✅ 500 customer questions with recurring themes → RL learns best responses
- ❌ 10 completely different one-off tasks → No pattern to learn

---

## ⚙️ Configuration Options

### Default Production Setup (Recommended)
```python
config = SwarmConfig(
    enable_rl=True,          # Enable RL
    rl_verbosity="quiet",    # Minimal logging
    gamma=0.99,              # Discount factor
    alpha=0.01,              # Learning rate
    lambda_trace=0.95,       # Eligibility traces
)
```

### Development/Debugging
```python
config = SwarmConfig(
    enable_rl=True,
    rl_verbosity="verbose",  # Show all RL decisions
)
```

### Quick Tests/Demos
```python
config = SwarmConfig(
    enable_rl=False,         # No RL overhead
)
```

---

## 📊 What RL Learns

### Agent Contribution Weights
RL tracks how much each agent contributes to successful outcomes:

```
Episode 1: SchemaAnalyzer=0.5, SQLGenerator=0.5, Validator=0.0
Episode 10: SchemaAnalyzer=0.6, SQLGenerator=0.4, Validator=0.0
Episode 50: SchemaAnalyzer=0.4, SQLGenerator=0.35, Validator=0.25
           ↑ Learned that validation prevents errors!
```

### Task Routing
With Q-learning, RL learns which agents to try first for different query types:

```
Query type: "Get all users" → SQLGenerator (high Q-value)
Query type: "Complex join" → SchemaAnalyzer first (higher Q-value)
```

### Cooperative Rewards
RL learns which agent combinations work well together:

```
Architect + Validator = 0.8 reward
Architect alone = 0.5 reward
→ RL learns validation is critical
```

---

## 🔄 RL Verbosity Levels

### `rl_verbosity="quiet"` (Default for Production)
- Only logs warnings/errors
- Minimal performance impact
- Best for production deployments

```python
config = SwarmConfig(enable_rl=True, rl_verbosity="quiet")
```

**Output**: Silent unless there's an issue

### `rl_verbosity="normal"` (Development)
- Logs RL initialization
- Shows learning milestones
- Helpful for monitoring

```python
config = SwarmConfig(enable_rl=True, rl_verbosity="normal")
```

**Output**:
```
📊 TD(λ) Learning initialized
📊 Episode 50: Avg reward = 0.75
```

### `rl_verbosity="verbose"` (Debugging)
- Logs every RL decision
- Shows Q-value updates
- Detailed credit assignment

```python
config = SwarmConfig(enable_rl=True, rl_verbosity="verbose")
```

**Output**:
```
📊 TD(λ) Learning initialized (eligibility traces)
🎯 Q-value for SchemaAnalyzer.analyze: 0.65
🎯 Eligibility trace updated: 0.85
💰 Reward assigned: 0.8 (cooperative bonus: 0.2)
```

---

## 💡 Best Practices

### 1. Start with RL Enabled (Default)
```python
# Default is enable_rl=True - keep it for production
config = SwarmConfig()  # RL enabled
```

### 2. Use Quiet Mode in Production
```python
config = SwarmConfig(
    enable_rl=True,
    rl_verbosity="quiet",  # Clean logs
    log_level="INFO"        # Standard logging
)
```

### 3. Only Disable for Demos/One-Offs
```python
# Quick demo
config = SwarmConfig(enable_rl=False)
```

### 4. Monitor Learning Progress
```python
# Development
config = SwarmConfig(
    enable_rl=True,
    rl_verbosity="normal",  # See learning progress
)
```

### 5. Persist Learned Q-Tables
```python
config = SwarmConfig(
    enable_rl=True,
    persist_q_tables=True,      # Save learned weights
    auto_load_on_start=True,    # Resume learning
    output_base_dir="./models"  # Where to save
)
```

---

## 🎓 Learning from Experience

### Cold Start (First 10 Runs)
```
Episode 1-10: Random exploration
- Trying different agent combinations
- Building initial Q-value estimates
```

### Warm Up (Runs 10-50)
```
Episode 10-50: Learning patterns
- Agent contribution weights stabilizing
- Q-values becoming more accurate
```

### Optimized (After 50+ Runs)
```
Episode 50+: Exploitation
- Using learned weights for routing
- Occasional exploration to avoid local optima
```

---

## 🚀 Migration Path

### If You're Currently Using `enable_rl=False` Everywhere

**For Production Systems:**
1. Identify systems with **repeated, similar tasks**
2. Enable RL: `enable_rl=True, rl_verbosity="quiet"`
3. Let it learn for 50+ episodes
4. Monitor improvement in success rate

**For Test/Demo Code:**
- Keep `enable_rl=False` - it's fine for one-offs

---

## 📈 Expected Benefits

### SQL Generation System (100 queries)
```
Without RL: 75% success rate
With RL (after 50 episodes): 85% success rate
Benefit: +10% from learned agent coordination
```

### Code Review System (500 PRs)
```
Without RL: Checks all agents every time
With RL: Focuses on high-value agents first
Benefit: 30% faster reviews, same quality
```

### Customer Support (1000 tickets)
```
Without RL: Equal weight to all agents
With RL: Learned response strategies
Benefit: 20% higher satisfaction scores
```

---

## 🎯 TL;DR

**Default (Production):**
```python
config = SwarmConfig()  # enable_rl=True, rl_verbosity="quiet"
```

**Demos/Tests:**
```python
config = SwarmConfig(enable_rl=False)
```

**When in doubt**: Leave RL enabled. The overhead is minimal, and you'll benefit as soon as you run similar tasks multiple times.
