# Jotty Modular Configuration System

**Module-based composition** (not subjective tiers) - compose exactly what you need.

Inspired by [AIME's](https://github.com/argmax-ai/aime) Hydra approach:
```bash
# AIME example
python train.py env=walker world_model=rssm

# Jotty example
python run_jotty.py mas=full memory=cortex learning=td_lambda
```

---

## Philosophy

**Problem with tiers** (minimal, basic, standard, premium, advanced):
- ❌ Subjective names
- ❌ Locked into predefined combinations
- ❌ Can't mix features flexibly

**Solution: Module-based**:
- ✅ Choose by functionality (MAS, Memory, Learning, etc.)
- ✅ Compose your own setup
- ✅ Objective names (q_learning vs td_lambda vs marl)
- ✅ Presets for convenience

---

## Quick Start

### 1. Use a Preset (Easy)

```bash
# Minimal (1.5K lines - like MegaAgent)
python run_jotty.py --config-name presets/minimal goal="Write hello world"

# Development (7K lines - fast iteration)
python run_jotty.py --config-name presets/development goal="Test feature"

# Production (15K lines - RECOMMENDED)
python run_jotty.py --config-name presets/production goal="Generate guide"

# Research (30K lines - with RL)
python run_jotty.py --config-name presets/research goal="Learn optimal approach"

# Experimental (50K+ lines - everything)
python run_jotty.py --config-name presets/experimental goal="MARL experiment"
```

### 2. Compose Modules (Flexible)

```bash
# Choose exactly what you need
python run_jotty.py \
  mas=full \
  orchestrator=conductor \
  memory=hierarchical \
  learning=none \
  validation=planner_reviewer \
  goal="Custom setup"

# Add RL to production preset
python run_jotty.py \
  --config-name presets/production \
  learning=td_lambda \
  goal="Production with learning"

# Minimal + parallel execution
python run_jotty.py \
  mas=minimal \
  memory=chroma \
  execution.parallel=true \
  goal="Minimal but parallel"
```

---

## Module Categories

### 1. MAS (Multi-Agent System)

Core coordination capabilities.

| Config | Lines | Description |
|--------|-------|-------------|
| `minimal` | ~1.5K | Like MegaAgent, sequential |
| `full` | ~5K | Parallel, LLM spawning, hierarchical |

```bash
python run_jotty.py mas=full  # or mas=minimal
```

---

### 2. Orchestrator

How agents are coordinated.

| Config | Implementation | Best For |
|--------|----------------|----------|
| `simple` | `jotty_minimal.Orchestrator` | Learning, prototypes |
| `conductor` | `MultiAgentsOrchestrator` | Production workflows |
| `universal` | `UniversalWorkflow` | Adaptive tasks |

```bash
python run_jotty.py orchestrator=conductor  # or simple, universal
```

---

### 3. Memory

Storage and retrieval strategy.

| Config | Backend | Features |
|--------|---------|----------|
| `simple` | In-memory | Tags, keyword search |
| `chroma` | ChromaDB | Vector similarity |
| `hierarchical` | 3-level | Working → Episodic → Long-term |
| `cortex` | 5-level brain | Sharp Wave Ripple consolidation |

```bash
python run_jotty.py memory=cortex  # or simple, chroma, hierarchical
```

---

### 4. Learning (Reinforcement Learning)

RL algorithms for agent improvement.

| Config | Algorithm | Features |
|--------|-----------|----------|
| `none` | - | No learning |
| `q_learning` | Q-learning | Basic value estimation |
| `td_lambda` | TD(λ) | Eligibility traces, credit assignment |
| `marl` | Multi-Agent RL | Trajectory prediction, cooperation |

```bash
python run_jotty.py learning=td_lambda  # or none, q_learning, marl
```

---

### 5. Validation

Quality control (pre/post execution).

| Config | Features |
|--------|----------|
| `none` | No validation |
| `planner_reviewer` | Planner (pre) + Reviewer (post) |
| `multi_round` | Iterative improvement (5 rounds) |

```bash
python run_jotty.py validation=planner_reviewer  # or none, multi_round
```

---

### 6. Tools

Tool management and discovery.

| Config | Features |
|--------|----------|
| `simple` | Hardcoded tools |
| `registry` | Tool registry, capability matching |
| `auto_discovery` | LLM-driven tool generation |

```bash
python run_jotty.py tools=auto_discovery  # or simple, registry
```

---

### 7. Experts

Domain-specific expert agents.

| Config | Experts |
|--------|---------|
| `none` | No experts |
| `research` | Researcher, WebSearcher, Summarizer |
| `full` | All experts (research, analysis, code, data) |

```bash
python run_jotty.py experts=research  # or none, full
```

---

### 8. Communication

Agent-to-agent messaging.

| Config | Routing | Complexity |
|--------|---------|------------|
| `simple` | Direct | O(n) |
| `hierarchical` | Via supervisors | O(log n) |
| `slack` | Channel-based | O(n) with channels |

```bash
python run_jotty.py communication=hierarchical  # or simple, slack
```

---

## Presets

Convenience presets that compose modules.

### Preset: `minimal`

```yaml
# ~1,500 lines - MegaAgent equivalent
mas: minimal
orchestrator: simple
memory: simple
learning: none
validation: none
tools: simple
experts: none
communication: simple
```

**Use when**: Learning, prototypes, want simplicity

---

### Preset: `development`

```yaml
# ~7,000 lines - fast iteration
mas: full
orchestrator: conductor
memory: chroma
learning: none
validation: planner_reviewer
tools: registry
experts: none
communication: hierarchical
```

**Use when**: Development, testing, multi-step workflows

---

### Preset: `production` ⭐ **RECOMMENDED**

```yaml
# ~15,000 lines - battle-tested
mas: full
orchestrator: conductor
memory: hierarchical
learning: none
validation: planner_reviewer
tools: registry
experts: research
communication: hierarchical
```

**Use when**: Production, complex tasks, quality-critical

---

### Preset: `research`

```yaml
# ~30,000 lines - with RL
mas: full
orchestrator: conductor
memory: cortex
learning: td_lambda
validation: multi_round
tools: auto_discovery
experts: research
communication: slack
```

**Use when**: Research, optimization, learning tasks

---

### Preset: `experimental`

```yaml
# ~50,000+ lines - everything
mas: full
orchestrator: universal
memory: cortex
learning: marl
validation: multi_round
tools: auto_discovery
experts: full
communication: slack
```

**Use when**: MARL research, experiments, publications

---

## Configuration Structure

```
configs/
├── config.yaml                 # Default (uses production defaults)
├── mas/                        # Multi-Agent System
│   ├── minimal.yaml
│   └── full.yaml
├── orchestrator/               # Coordination strategy
│   ├── simple.yaml
│   ├── conductor.yaml
│   └── universal.yaml
├── memory/                     # Storage backend
│   ├── simple.yaml
│   ├── chroma.yaml
│   ├── hierarchical.yaml
│   └── cortex.yaml
├── learning/                   # RL algorithms
│   ├── none.yaml
│   ├── q_learning.yaml
│   ├── td_lambda.yaml
│   └── marl.yaml
├── validation/                 # Quality control
│   ├── none.yaml
│   ├── planner_reviewer.yaml
│   └── multi_round.yaml
├── tools/                      # Tool management
│   ├── simple.yaml
│   ├── registry.yaml
│   └── auto_discovery.yaml
├── experts/                    # Domain experts
│   ├── none.yaml
│   ├── research.yaml
│   └── full.yaml
├── communication/              # Message passing
│   ├── simple.yaml
│   ├── hierarchical.yaml
│   └── slack.yaml
├── presets/                    # Convenience presets
│   ├── minimal.yaml
│   ├── development.yaml
│   ├── production.yaml
│   ├── research.yaml
│   └── experimental.yaml
└── README.md                   # This file
```

---

## Common Recipes

### Recipe 1: Minimal + Parallel

```bash
python run_jotty.py \
  mas=minimal \
  memory=simple \
  execution.parallel=true \
  goal="Fast prototype"
```

### Recipe 2: Production + Learning

```bash
python run_jotty.py \
  --config-name presets/production \
  learning=q_learning \
  goal="Production with basic RL"
```

### Recipe 3: Research without Experts

```bash
python run_jotty.py \
  --config-name presets/research \
  experts=none \
  goal="RL research, no experts"
```

### Recipe 4: Custom Composition

```bash
python run_jotty.py \
  mas=full \
  orchestrator=conductor \
  memory=cortex \
  learning=td_lambda \
  validation=multi_round \
  tools=auto_discovery \
  experts=research \
  communication=slack \
  goal="Fully custom setup"
```

---

## Advantages Over Tier-Based

### Before (Tier-Based - Subjective)

```bash
# What does "standard" mean? 🤷
python run_jotty.py mode=standard

# Can't mix features easily
python run_jotty.py mode=standard  # Locked into predefined combo
```

### After (Module-Based - Objective)

```bash
# Clear what you're using
python run_jotty.py memory=cortex learning=td_lambda

# Mix freely
python run_jotty.py \
  mas=minimal \  # Minimal coordination
  memory=cortex \  # Brain-inspired memory
  learning=marl  # Advanced MARL
  # Whatever makes sense for YOUR use case!
```

---

## Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...  # or ANTHROPIC_API_KEY

# Optional
JOTTY_LOG_LEVEL=INFO
```

---

## Migration from Old Code

**Old (hardcoded)**:
```python
from core.orchestration.conductor import MultiAgentsOrchestrator

orchestrator = MultiAgentsOrchestrator(
    actors=[...],
    enable_learning=True,
    enable_validation=True,
    memory_backend="cortex",
    # ... 50+ parameters
)
```

**New (config-driven)**:
```python
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg: DictConfig):
    from jotty import create_orchestrator

    orchestrator = create_orchestrator(cfg)
    result = orchestrator.run(goal=cfg.goal)

if __name__ == "__main__":
    run()
```

---

## FAQ

### Q: Which preset should I use?

**A**: Start with **production**. It works for 90% of use cases.

### Q: Can I create my own modules?

**A**: Yes! Copy an existing config and customize:
```yaml
# configs/memory/my_memory.yaml
backend: custom
implementation: myproject.custom_memory.MyMemory
# ... your settings
```

Then use: `python run_jotty.py memory=my_memory`

### Q: How do I know what I'm using?

**A**: Hydra logs the full configuration:
```bash
python run_jotty.py --config-name presets/production --cfg job
# Shows full composed config
```

### Q: Can I override individual settings?

**A**: Yes, any setting:
```bash
python run_jotty.py \
  --config-name presets/production \
  learning.alpha=0.2 \
  execution.max_steps=100 \
  lm.model=gpt-4o
```

---

## Learn More

- **Hydra Docs**: https://hydra.cc/docs/intro/
- **AIME Example**: https://github.com/argmax-ai/aime
- **Jotty Architecture**: `MODULAR_JOTTY_ARCHITECTURE.md`
- **Implementation**: `NEXT_STEPS_PLAN.md`

---

## Summary

**Old approach**: Subjective tiers (minimal, basic, standard, premium)
**New approach**: Objective modules (MAS, Orchestrator, Memory, Learning, etc.)

**Benefits**:
- ✅ Clear, objective naming
- ✅ Flexible composition
- ✅ Presets for convenience
- ✅ Like AIME's approach
- ✅ Users choose complexity

**Usage**:
```bash
# Preset (easy)
python run_jotty.py --config-name presets/production goal="..."

# Compose (flexible)
python run_jotty.py mas=full memory=cortex learning=td_lambda goal="..."
```
