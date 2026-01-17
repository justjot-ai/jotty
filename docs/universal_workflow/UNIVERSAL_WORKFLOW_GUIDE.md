# Universal Workflow - Complete Guide

## 🎯 Overview

**UniversalWorkflow** is Jotty's adaptive multi-agent orchestrator supporting 8+ workflow patterns with:
- ✅ **Auto-mode selection** based on goal analysis
- ✅ **Flexible context** (data folders, codebases, URLs, databases, etc.)
- ✅ **ALL Jotty tools** (file ops, execution, git, data, metadata)
- ✅ **ZERO duplication** (thin wrapper around Conductor)
- ✅ **8 workflow modes** (sequential, parallel, p2p, hierarchical, debate, round-robin, pipeline, swarm)

---

## 📦 Installation

```python
from core.orchestration.universal_workflow import UniversalWorkflow
from core.foundation.agent_config import AgentConfig
from core.foundation.jotty_config import JottyConfig

# Create workflow
config = JottyConfig()
actors = []  # Will be created dynamically based on mode
workflow = UniversalWorkflow(actors, config)
```

---

## 🚀 Usage Patterns

### 1. Auto-Mode (Recommended)

Let Jotty analyze the goal and pick the best workflow:

```python
result = await workflow.run(
    goal="Build a stock market screening system to find undervalued growth stocks",
    context={'data_folder': '/path/to/FUNDAMENTALS'},
    mode='auto'  # Jotty picks mode based on goal analysis
)

# Jotty will analyze:
# - Complexity: simple/medium/complex
# - Uncertainty: clear/ambiguous/exploratory
# - Recommends: sequential/p2p/hierarchical/etc.
```

### 2. Explicit Mode Selection

Specify the workflow mode directly:

```python
result = await workflow.run(
    goal="Build stock screener",
    context={'data_folder': '/path/to/data'},
    mode='hierarchical'  # Force specific mode
)
```

### 3. Flexible Context Types

Context is FLEXIBLE - use what's relevant to your task:

```python
# Example 1: Data analysis task
result = await workflow.run(
    goal="Analyze sales data and find trends",
    context={
        'data_folder': '/path/to/sales_data',
        'database': 'postgres://user:pass@host/sales',
        'time_limit': '1 hour',
        'quality_threshold': 0.9
    },
    mode='auto'
)

# Example 2: Code refactoring task
result = await workflow.run(
    goal="Refactor authentication module",
    context={
        'codebase': '/path/to/repo',
        'requirements_doc': 'docs/AUTH_REQUIREMENTS.md',
        'coding_style': 'Google Python Style Guide',
        'frameworks': ['FastAPI', 'SQLAlchemy']
    },
    mode='auto'
)

# Example 3: API integration task
result = await workflow.run(
    goal="Integrate Stripe payment API",
    context={
        'codebase': '/path/to/repo',
        'api_docs': 'https://stripe.com/docs/api',
        'api_key': 'sk_test_...',
        'github_repo': 'https://github.com/user/repo'
    },
    mode='auto'
)

# Example 4: Resume previous session
result = await workflow.run(
    goal="Continue previous analysis",
    context={
        'session_id': 'sess_123',
        'previous_output': 'output.json'
    },
    mode='auto'
)
```

---

## 🎬 Workflow Modes

### 1. Sequential (Waterfall)

**Pattern**: A → B → C → D

**Use When**: Strict dependencies, linear workflow

**Example**:
```python
result = await workflow.run(
    goal="Build user authentication system",
    mode='sequential'
)

# Workflow: PM → UX → Designer → Frontend → Backend → QA
```

---

### 2. Parallel

**Pattern**: A, B, C (all at once)

**Use When**: Independent tasks, no dependencies

**Example**:
```python
result = await workflow.run(
    goal="Analyze multiple data sources independently",
    mode='parallel'
)

# Workflow: Sales Analyzer || Marketing Analyzer || Finance Analyzer
```

---

### 3. P2P / Hybrid

**Pattern**:
```
Phase 1 (P2P):     Agent A ↘
                   Agent B → SharedScratchpad → Insights
                   Agent C ↗

Phase 2 (Sequential): Insights → Agent D → Agent E → Agent F
```

**Use When**: Problem needs exploration before solution

**Example**:
```python
result = await workflow.run(
    goal="Build ML model for churn prediction",
    context={'data_folder': '/path/to/customer_data'},
    mode='p2p'  # or 'hybrid'
)

# Phase 1 (P2P Discovery):
#   - Data Explorer (analyze data quality)
#   - Feature Engineer (identify features)
#   - Algorithm Researcher (recommend models)
#
# Phase 2 (Sequential Delivery):
#   - Data Prep → Model Training → Evaluation → Deployment
```

---

### 4. Hierarchical

**Pattern**:
```
Lead Agent (coordinates)
├─ Sub-Agent 1 → reports results
├─ Sub-Agent 2 → reports results
└─ Sub-Agent 3 → reports results
```

**Use When**: Complex task with clear decomposition, centralized coordination

**Example**:
```python
result = await workflow.run(
    goal="Build complete e-commerce platform",
    mode='hierarchical',
    num_sub_agents=5
)

# Lead Agent decomposes:
#   - Sub-Agent 1: User authentication & authorization
#   - Sub-Agent 2: Product catalog & search
#   - Sub-Agent 3: Shopping cart & checkout
#   - Sub-Agent 4: Payment integration
#   - Sub-Agent 5: Order management & tracking
#
# Lead Agent aggregates and integrates all components
```

---

### 5. Debate / Consensus

**Pattern**:
```
Phase 1: Propose (agents propose different solutions)
Phase 2: Critique (agents review each other's proposals)
Phase 3: Vote/Combine (select best or synthesize)
```

**Use When**: Multiple valid approaches, need to explore solution space

**Example**:
```python
result = await workflow.run(
    goal="Choose best architecture for real-time chat system",
    mode='debate',
    num_debaters=3
)

# Debater 1 proposes: WebSockets + Redis
# Debater 2 proposes: Server-Sent Events + PostgreSQL
# Debater 3 proposes: WebRTC + Firestore
#
# Critics review each proposal (performance, scalability, cost)
# Judge selects best or synthesizes hybrid approach
```

---

### 6. Round-Robin

**Pattern**:
```
Round 1: A adds → B refines → C extends
Round 2: A reviews → B improves → C polishes
Round 3: Final pass
```

**Use When**: Iterative improvement needed, quality improves with multiple passes

**Example**:
```python
result = await workflow.run(
    goal="Write comprehensive technical documentation",
    mode='round-robin',
    num_rounds=3,
    num_agents=3
)

# Round 1:
#   Agent 1 (Writer): Draft initial content
#   Agent 2 (Editor): Edit for clarity
#   Agent 3 (Technical Reviewer): Add technical accuracy
#
# Round 2:
#   Agent 1: Improve based on feedback
#   Agent 2: Polish grammar and flow
#   Agent 3: Add examples and diagrams
#
# Round 3:
#   All agents: Final review and polish
```

---

### 7. Pipeline

**Pattern**: Data → Stage 1 → Intermediate → Stage 2 → Intermediate → Stage 3 → Output

**Use When**: Clear data transformation pipeline, each stage has distinct responsibility

**Example**:
```python
result = await workflow.run(
    goal="Process customer feedback data",
    context={'data_folder': '/path/to/feedback'},
    mode='pipeline',
    stages=[
        'Load and validate CSV files',
        'Clean and normalize text data',
        'Perform sentiment analysis',
        'Extract key topics and themes',
        'Generate summary report and visualizations'
    ]
)

# Data flows through stages sequentially
# Each stage expects specific input type, produces specific output type
```

---

### 8. Swarm

**Pattern**: Self-organizing agents claim tasks dynamically

**Use When**: Many independent subtasks, agents have varying capabilities

**Example**:
```python
result = await workflow.run(
    goal="Code review for entire repository",
    context={'codebase': '/path/to/repo'},
    mode='swarm',
    num_agents=5
)

# Task announced: "Review all Python files in repo"
# Agents self-select files based on:
#   - Expertise (frontend/backend/ML/data/etc.)
#   - Current workload
#   - File complexity
#
# Agents coordinate via scratchpad:
#   "I'm reviewing auth.py"
#   "I found security issue in payments.py"
#   "I'll help with tests for API endpoints"
```

---

## 🔧 Advanced Configuration

### Custom Agent Count

```python
result = await workflow.run(
    goal="...",
    mode='p2p',
    num_discovery_agents=5,  # Phase 1: 5 discovery agents
    num_delivery_agents=7     # Phase 2: 7 delivery agents
)
```

### Quality Thresholds

```python
result = await workflow.run(
    goal="...",
    context={'quality_threshold': 0.95},  # 95% quality required
    mode='auto'
)
```

### Time Limits

```python
result = await workflow.run(
    goal="...",
    context={'time_limit': '30 minutes'},
    mode='auto'
)
```

---

## 📊 Result Structure

```python
result = {
    'status': 'success',
    'results': {
        # Mode-specific results
        'discoveries': {...},  # P2P/hybrid modes
        'deliverables': {...},  # Sequential/delivery modes
        'proposals': {...},     # Debate mode
        'rounds': [...]         # Round-robin mode
    },
    'session_id': 'sess_abc123',
    'mode_used': 'hierarchical',
    'analysis': {
        'complexity': 'complex',
        'uncertainty': 'ambiguous',
        'recommended_mode': 'hierarchical',
        'reasoning': 'Task has clear decomposition but requires centralized coordination',
        'num_agents': 5
    }
}
```

---

## 🎯 Real-World Examples

### Example 1: Stock Market Screener (Original Request)

```python
from core.orchestration.universal_workflow import UniversalWorkflow
from core.foundation.jotty_config import JottyConfig

workflow = UniversalWorkflow([], JottyConfig())

result = await workflow.run(
    goal="Build a stock market screening system to find undervalued growth stocks",
    context={
        'data_folder': '/var/www/sites/personal/stock_market/common/Data/FUNDAMENTALS'
    },
    mode='auto'  # Jotty will likely pick 'p2p' or 'hierarchical'
)

# Expected workflow:
# Phase 1 (P2P Discovery):
#   - Financial Data Analyst: Analyze balance sheets, P&L, cash flow
#   - Ratio Expert: Define valuation metrics (P/E, P/B, etc.)
#   - Technical Analyst: Identify momentum indicators
#   - System Architect: Design pipeline architecture
#
# Phase 2 (Sequential Delivery):
#   - Data Engineer: Build data loader with cleaning
#   - Screening Engineer: Implement multi-factor scoring
#   - Backend Developer: Create CLI/API interface
#   - Test Engineer: Write tests and backtesting
#   - Documentation Writer: Create README and methodology
#
# All agents have access to:
#   - File operations (read/write data files)
#   - Data operations (pandas, CSV, Excel)
#   - Code execution (run Python, tests)
#   - Git operations (commit code)
```

### Example 2: Security Audit

```python
result = await workflow.run(
    goal="Perform comprehensive security audit of authentication system",
    context={
        'codebase': '/path/to/repo',
        'requirements_doc': 'docs/SECURITY_REQUIREMENTS.md'
    },
    mode='debate'  # Multiple perspectives on vulnerabilities
)

# Debate workflow:
# Proposals:
#   - Auth Security Expert: Focus on session management
#   - API Security Expert: Focus on endpoint protection
#   - Data Security Expert: Focus on encryption and PII
#   - Infrastructure Expert: Focus on network security
#
# Critiques:
#   - Each reviews others' findings
#   - Identifies overlaps and gaps
#   - Scores severity of issues
#
# Final Decision:
#   - Judge prioritizes all findings
#   - Creates remediation plan
#   - Assigns severity levels
```

### Example 3: Content Creation Pipeline

```python
result = await workflow.run(
    goal="Create blog post about AI trends in 2026",
    mode='pipeline',
    stages=[
        'Research: Gather sources and data',
        'Outline: Create structure and key points',
        'Draft: Write initial content',
        'Edit: Polish grammar and flow',
        'SEO: Optimize for search engines',
        'Publish: Format and export'
    ]
)

# Pipeline workflow:
# Each stage passes output to next stage
# Agents have file write access to save intermediate results
```

---

## 🔍 How It Works (Architecture)

### Thin Wrapper Pattern (ZERO Duplication!)

```
UniversalWorkflow
├─ Wraps Conductor internally
├─ Delegates to Conductor for:
│  ├─ Tool management (file, execution, git, data)
│  ├─ Learning (TD-lambda, Q-learning, MARL)
│  ├─ Validation (Planner/Reviewer)
│  ├─ Memory (hierarchical, consolidation)
│  └─ State management
│
├─ Reuses existing workflow functions:
│  ├─ p2p_discovery_phase (from hybrid_team_template)
│  └─ sequential_delivery_phase (from hybrid_team_template)
│
└─ Adds ONLY:
   ├─ GoalAnalyzer (auto-select mode)
   ├─ ContextHandler (flexible context)
   └─ 5 NEW modes (hierarchical, debate, round-robin, pipeline, swarm)
```

### Code Stats

- **Total new code**: ~950 lines
- **Code reused**: ~5,000+ lines from existing infrastructure
- **DRY savings**: 81% reduction
- **Duplication**: 0%

---

## 🎓 Best Practices

1. **Use auto mode** for most tasks → let Jotty decide
2. **Provide relevant context** → more context = better results
3. **Start simple** → try auto mode before forcing specific modes
4. **Iterate** → use round-robin or debate for quality-critical tasks
5. **Monitor sessions** → check session files for agent collaboration

---

## 🚦 Quick Start

```bash
cd /var/www/sites/personal/stock_market/Jotty

# Run demo
python3 -c "
import asyncio
from core.orchestration.universal_workflow import UniversalWorkflow
from core.foundation.jotty_config import JottyConfig

async def main():
    workflow = UniversalWorkflow([], JottyConfig())
    result = await workflow.run(
        goal='Build a simple calculator',
        mode='auto'
    )
    print(result)

asyncio.run(main())
"
```

---

## 📚 Further Reading

- `DRY_ANALYSIS.md` - Architecture analysis and DRY compliance
- `HYBRID_WORKFLOW_DEMO.md` - P2P + Sequential demo results
- `templates/hybrid_team_template.py` - Reusable workflow functions
- `core/orchestration/conductor.py` - Base orchestrator
- `core/orchestration/modes/` - New mode implementations

---

**This is the most flexible multi-agent framework with ZERO duplication!** 🚀
