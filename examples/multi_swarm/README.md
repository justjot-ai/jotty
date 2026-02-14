# Multi-Swarm Examples

Examples demonstrating overnight enhancements for production deployments.

## Examples

### 1. Basic Multi-Swarm Coordination
**File:** `01_basic_multi_swarm.py`

Shows how to use Multi-Swarm Coordinator with different merge strategies:
- **Voting** - Majority consensus (2/3 votes)
- **Concatenation** - All perspectives combined
- **Best-of-N** - Highest confidence wins
- **Ensemble** - Weighted averaging
- **First Success** - First successful result

**Usage:**
```bash
python examples/multi_swarm/01_basic_multi_swarm.py
```

**Key Features:**
- Zero wrapper code needed
- `SwarmAdapter.quick_swarms()` creates swarms in 1 line
- Parallel execution (2-3x speedup)
- 5 merge strategies available

---

### 2. Cost-Aware Learning
**File:** `02_cost_aware_learning.py`

Demonstrates cost-aware reinforcement learning:
- Agents learn to prefer cheaper strategies
- Multi-objective optimization (quality + cost)
- Automatic cost tracking
- Savings calculation

**Usage:**
```bash
python examples/multi_swarm/02_cost_aware_learning.py
```

**Key Concepts:**
- Cost sensitivity parameter (0.1 to 10.0)
- Adjusted reward formula: `reward - (cost / sensitivity)`
- Cheaper strategies get higher rewards
- System learns automatically

---

### 3. Distributed Tracing
**File:** `03_distributed_tracing.py`

Shows distributed tracing for microservices:
- W3C Trace Context propagation
- Cross-service correlation
- Full execution context
- Header injection for downstream services

**Usage:**
```bash
python examples/multi_swarm/03_distributed_tracing.py
```

**Key Features:**
- W3C traceparent headers
- Nested trace support
- Integration with monitoring tools
- Full context extraction

---

## Quick Start

All examples require an Anthropic API key:

```bash
# Set API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Or use .env file
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# Run examples
python examples/multi_swarm/01_basic_multi_swarm.py
python examples/multi_swarm/02_cost_aware_learning.py
python examples/multi_swarm/03_distributed_tracing.py
```

## Key Concepts

### SwarmAdapter
Zero-code adapter for multi-swarm:

```python
from Jotty.core.orchestration import SwarmAdapter

# Quick swarms (simplest)
swarms = SwarmAdapter.quick_swarms([
    ("Name1", "System prompt 1"),
    ("Name2", "System prompt 2"),
])

# From existing swarms
swarms = SwarmAdapter.from_swarms([existing_swarm1, existing_swarm2])

# From agents
swarms = SwarmAdapter.from_agents([agent1, agent2])
```

### Merge Strategies

| Strategy | Use Case | Example |
|----------|----------|---------|
| **VOTING** | Consensus | "Is this positive?" (2/3 agree = positive) |
| **CONCATENATE** | All perspectives | Research from multiple angles |
| **BEST_OF_N** | Confidence-based | Choose highest confidence answer |
| **ENSEMBLE** | Numeric predictions | Average stock price predictions |
| **FIRST_SUCCESS** | Redundancy | First successful API call |

### Performance

**Sequential (before):**
- 3 swarms × 10s each = 30s total

**Parallel (after):**
- 3 swarms × 10s (concurrent) = 10s total
- **3x speedup!**

## Additional Resources

- **Full Guide:** `docs/OVERNIGHT_ENHANCEMENTS.md`
- **API Reference:** `docs/API_REFERENCE.md`
- **Tests:** `tests/test_swarm_adapter.py`

## Support

For issues or questions:
- GitHub Issues: [Jotty Issues](https://github.com/yourusername/jotty/issues)
- Documentation: `docs/`
