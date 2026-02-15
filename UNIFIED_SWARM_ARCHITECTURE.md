# Unified Swarm Architecture
## World's Best Self-Learning Swarm Manager

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        UNIFIED BASE SWARM                                │
│                   (Single class, all learning integrated)               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
              ┌─────▼─────┐   ┌────▼────┐   ┌─────▼─────┐
              │   AGENTS  │   │ PATTERN │   │ LEARNING  │
              └───────────┘   └─────────┘   └───────────┘


═══════════════════════════════════════════════════════════════════════════
PATTERN SELECTION (AUTO Mode)
═══════════════════════════════════════════════════════════════════════════

    ┌──────────────────────────────────────────────────────────────┐
    │  Task: "Research 3 AI companies and compare them"            │
    └──────────────────┬───────────────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │   AUTO PATTERN SELECTOR   │
         └─────────────┬─────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
    ┌───▼───┐      ┌──▼──┐       ┌──▼──┐
    │Memory │      │ TD  │       │ SI  │
    │Search │      │Learn│       │Meta │
    └───┬───┘      └──┬──┘       └──┬──┘
        │             │             │
        └─────────────┼─────────────┘
                      ▼
            ┌─────────────────┐
            │   PARALLEL      │  ← Selected!
            │ (3 independent  │
            │   researchers)  │
            └─────────────────┘


═══════════════════════════════════════════════════════════════════════════
EXECUTION PATTERNS
═══════════════════════════════════════════════════════════════════════════

┌─────────────┐
│ SEQUENTIAL  │  A → B → C
└─────────────┘
  Research → Analyze → Write

┌─────────────┐
│  PARALLEL   │  A | B | C
└─────────────┘
  Researcher1 | Researcher2 | Researcher3

┌─────────────┐
│  CONSENSUS  │  Vote: best(A, B, C)
└─────────────┘
  Reviewer1, Reviewer2, Reviewer3 → Best review

┌─────────────┐
│   DEBATE    │  Multi-round + Synthesis
└─────────────┘
  Round 1: Propose
  Round 2: Critique
  Round 3: Refine
  Final: Synthesize → New insight

┌─────────────┐
│  ITERATIVE  │  Loop until threshold
└─────────────┘
  Generate → Evaluate → Improve → Repeat

┌─────────────┐
│HIERARCHICAL │  Manager → Workers
└─────────────┘
  Manager: Plan
  Workers: Execute (parallel)
  Manager: Aggregate

┌─────────────┐
│ BLACKBOARD  │  Shared workspace
└─────────────┘
  Agent1: Add finding
  Agent2: Add analysis
  Agent3: Add conclusion
  → Final state

┌─────────────┐
│   CUSTOM    │  User-defined STAGES
└─────────────┘
  STAGES = [
    Stage1 → Stage2 → Stage3
            ↓
          Stage4 (optional)
  ]


═══════════════════════════════════════════════════════════════════════════
LEARNING INTEGRATION (8 Layers)
═══════════════════════════════════════════════════════════════════════════

┌────────────────────────────────────────────────────────────────────────┐
│                         EXECUTION FLOW                                  │
└────────────────────────────────────────────────────────────────────────┘

1. PRE-EXECUTION
   ├─ Retrieve learned context (Memory Layer 1)
   ├─ Select pattern (Pattern Learner Layer 6)
   └─ Prepare with past experience (Transfer Layer 7)

2. EXECUTION
   ├─ Execute with selected pattern
   ├─ Monitor performance
   └─ Collect metrics

3. POST-EXECUTION
   ├─ Evaluate result (Gold Standards Layer 4)
   ├─ Calculate reward
   └─ Trigger improvement if needed (Layer 5)

4. LEARNING UPDATE (All 8 Layers)

   Layer 1: MEMORY (5 levels)
   ├─ EPISODIC ← Store raw experience
   ├─ SEMANTIC ← Extract facts
   ├─ PROCEDURAL ← Learn patterns
   ├─ META ← Learn about learning
   └─ CAUSAL ← Understand cause-effect

   Layer 2: TD-LAMBDA
   └─ Update Q-values (state, action, reward)

   Layer 3: SWARM INTELLIGENCE
   └─ Update meta-learning models

   Layer 4: GOLD STANDARDS
   ├─ Auto-curate excellent results
   └─ Update evaluation history

   Layer 5: IMPROVEMENT AGENTS
   ├─ Expert: Best practices
   ├─ Reviewer: Quality analysis
   ├─ Planner: Strategy
   └─ Actor: Improvements

   Layer 6: PATTERN LEARNER
   └─ Record which patterns work best

   Layer 7: TRANSFER LEARNING
   └─ Store task embeddings + outcomes

   Layer 8: ADAPTIVE COMPONENTS
   ├─ Adjust exploration rate
   ├─ Adjust learning rate
   └─ Adjust discount factor

5. CONSOLIDATION
   ├─ Episodic → Semantic (nightly)
   ├─ Semantic → Procedural (weekly)
   └─ Procedural → Meta (monthly)


═══════════════════════════════════════════════════════════════════════════
MERGE vs SYNTHESIZE
═══════════════════════════════════════════════════════════════════════════

┌────────────────────┐
│   MERGE/COMBINE    │  (Mechanical - No intelligence)
└────────────────────┘

CONCAT:    "A" + "B" + "C" = "ABC"
LIST:      [result_A, result_B, result_C]
VOTE:      max(A, A, B) = A
FIRST:     first_success(A, fail, C) = A
BEST:      max(A:0.8, B:0.95, C:0.7) = B

┌────────────────────┐
│    SYNTHESIZE      │  (Intelligent - Creates new insight)
└────────────────────┘

Input:     ["React is popular", "Vue is simple", "Svelte is fast"]

Synthesize: "For our team, Vue is optimal because:
            - Simpler learning curve (Vue) helps onboarding
            - Performance adequate for current needs
            - Ecosystem sufficient (React insight)
            - Migration path to Svelte later (Svelte insight)

            This hybrid approach addresses all concerns."

→ Creates NEW solution that's better than any individual input!


═══════════════════════════════════════════════════════════════════════════
CODE EXAMPLE: Complete Integration
═══════════════════════════════════════════════════════════════════════════

from Jotty.core.intelligence.swarms import BaseSwarm
from Jotty.core.intelligence.swarms.base.agent_team import AgentTeam, CoordinationPattern

class MySwarm(BaseSwarm):
    """
    Complete example showing all features.
    """

    # Define agent team
    AGENT_TEAM = AgentTeam.define(
        (ResearcherAgent, "Researcher"),
        (AnalystAgent, "Analyst"),
        (WriterAgent, "Writer"),
    )

    # Let swarm auto-select pattern!
    COORDINATION = CoordinationPattern.AUTO

    # Optional: Custom multi-stage (for CUSTOM pattern)
    STAGES = [
        StageConfig(
            name="research",
            agents=["_researcher"],
            description="Gather information",
        ),
        StageConfig(
            name="analyze",
            agents=["_analyst"],
            needs=["research"],
            description="Analyze findings",
        ),
        StageConfig(
            name="write",
            agents=["_writer"],
            needs=["analyze"],
            description="Create report",
        ),
    ]

# Usage
swarm = MySwarm()

# First execution - Swarm learns
result1 = await swarm.execute(task="Research AI trends")

# Pattern was auto-selected based on task analysis
print(f"Selected: {result1.metadata['pattern']}")  # → "sequential"

# Second execution - Uses learning
result2 = await swarm.execute(task="Research 3 companies")

# Different task → Different pattern
print(f"Selected: {result2.metadata['pattern']}")  # → "parallel"

# Learning improves over time
for i in range(10):
    result = await swarm.execute(task=f"Research task {i}")
    # Swarm gets better at selecting patterns
    # Execution gets faster
    # Quality improves


═══════════════════════════════════════════════════════════════════════════
KEY INNOVATIONS
═══════════════════════════════════════════════════════════════════════════

✅ Single unified architecture (not 3 separate concepts)
✅ AUTO pattern selection (learns which coordination works best)
✅ 8 learning layers (comprehensive, nothing lost)
✅ SYNTHESIZE vs COMBINE (intelligent vs mechanical)
✅ CUSTOM with STAGES (declarative multi-stage)
✅ Backward compatible (old code works)
✅ Examples & tests (validated)

Result: WORLD'S BEST SELF-LEARNING SWARM MANAGER! 🚀
