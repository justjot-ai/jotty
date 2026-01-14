# JOTTY Framework - Architecture & Execution Flow

**Last Updated:** 2026-01-10

## 📋 Table of Contents

1. [Overview](#overview)
2. [Execution Flow](#execution-flow)
3. [File Hierarchy](#file-hierarchy)
4. [Module Dependencies](#module-dependencies)
5. [Folder Structure](#folder-structure)
6. [Execution Order](#execution-order)

---

## 🎯 Overview

JOTTY is a multi-agent orchestration framework with brain-inspired learning and game-theoretic cooperation. The architecture is organized into logical layers:

```
┌─────────────────────────────────────────────┐
│         USER INTERFACE LAYER                │
│  (Jotty, Conductor, AgentConfig)           │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│      ORCHESTRATION LAYER                    │
│  (Conductor, JottyCore, DependencyGraph)   │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│       AGENT EXECUTION LAYER                 │
│  (Agents, Validation, Communication)       │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│         LEARNING LAYER                      │
│  (Memory, Q-Learning, Credit Assignment)   │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│        INFRASTRUCTURE LAYER                 │
│  (Context, Persistence, Utils)             │
└─────────────────────────────────────────────┘
```

---

## 🔄 Execution Flow

### Phase 1: Initialization (User Code)

```
1. User creates AgentConfig instances
2. User creates JottyConfig
3. User creates Conductor with agents + config
4. Conductor.__init__() initializes all subsystems
```

### Phase 2: Conductor Initialization

```
Conductor.__init__()
├── Load Configuration (data_structures.py)
├── Initialize Metadata Provider (metadata_protocol.py)
├── Initialize IOManager (io_manager.py)
├── Initialize DataRegistry (data_registry.py)
├── Initialize SmartDataTransformer (smart_data_transformer.py)
├── Initialize AgentSlack (axon.py)
├── Initialize Parameter Resolver (agentic_parameter_resolver.py)
├── Initialize Memory System (cortex.py)
├── Initialize Q-Learning (q_learning.py)
├── Initialize Roadmap (roadmap.py)
├── Build Dependency Graph (dynamic_dependency_graph.py)
└── Initialize Brain Modes (brain_modes.py)
```

### Phase 3: Execution (conductor.run())

```
conductor.run(goal="...", **kwargs)
├── 1. Initialize Episode
│   ├── Create Markovian TODO from goal
│   ├── Reset episode state
│   └── Retrieve relevant memories
│
├── 2. Build Execution Plan
│   ├── Resolve dependencies (dynamic_dependency_graph.py)
│   ├── Determine execution order
│   └── Initialize task queue (roadmap.py)
│
├── 3. Execute Agents (Loop)
│   ├── a. Get Next Agent (roadmap.py)
│   ├── b. Resolve Parameters (agentic_parameter_resolver.py)
│   ├── c. Architect Validation (jotty_core.py)
│   │   ├── Load architect prompts
│   │   ├── Run pre-execution validation
│   │   └── Get input suggestions
│   ├── d. Execute Agent (jotty_core.py)
│   │   ├── Call agent.forward(**params)
│   │   ├── Monitor execution time
│   │   └── Capture output
│   ├── e. Auditor Validation (jotty_core.py)
│   │   ├── Load auditor prompts
│   │   ├── Run post-execution validation
│   │   └── Provide feedback if failed
│   ├── f. Store Results (io_manager.py)
│   │   ├── Register in DataRegistry
│   │   ├── Update agent outputs
│   │   └── Store in memory
│   ├── g. Update Learning (learning.py, q_learning.py)
│   │   ├── Calculate reward
│   │   ├── Update Q-values
│   │   ├── Credit assignment
│   │   └── Store experience
│   ├── h. Agent Communication (axon.py)
│   │   ├── Broadcast results
│   │   ├── Share relevant data
│   │   └── Track cooperation
│   └── i. Update Roadmap (roadmap.py)
│       ├── Mark task complete/failed
│       ├── Update dependencies
│       └── Predict next task
│
├── 4. Consolidate Memory (cortex.py, brain_modes.py)
│   ├── Hippocampal extraction
│   ├── Sharp-wave ripple consolidation
│   └── Synaptic pruning
│
├── 5. Persist State (persistence.py)
│   ├── Save Q-tables
│   ├── Save memories
│   ├── Save roadmap state
│   └── Save episode history
│
└── 6. Return Results (io_manager.py)
    ├── Package SwarmResult
    ├── Include trajectory
    └── Return to user
```

---

## 📂 File Hierarchy

### Level 1: Entry Points
User directly interacts with these files.

```
├── interface.py           # Clean API wrapper (Jotty class)
├── __init__.py           # Main exports
└── core/
    ├── conductor.py      # Main orchestrator (PRIMARY ENTRY)
    ├── agent_config.py   # Agent configuration
    └── data_structures.py # Core data types (JottyConfig, etc.)
```

### Level 2: Core Orchestration
Core execution engine called by Conductor.

```
core/
├── jotty_core.py              # Wraps agents with Architect/Auditor
├── dynamic_dependency_graph.py # Builds agent execution order
├── roadmap.py                  # Markovian TODO management
└── policy_explorer.py          # Exploration when stuck
```

### Level 3: Agent Execution
Components involved in agent execution.

```
core/
├── axon.py                    # Agent-to-agent communication
├── feedback_channel.py        # Agent coordination messages
├── inspector.py               # Agent inspection & debugging
└── modern_agents.py           # Retry handlers, critics
```

### Level 4: Memory & Learning
Learning and memory systems.

```
core/
├── cortex.py                  # 5-level hierarchical memory
├── brain_memory_manager.py    # Brain-inspired memory
├── brain_modes.py             # Hippocampal, sharp-wave ripple
├── simple_brain.py            # Simplified memory API
├── learning.py                # TD(λ) learning
├── q_learning.py              # Q-table & LLM Q-predictor
├── rl_components.py           # RL building blocks
├── offline_learning.py        # Offline training
├── shaped_rewards.py          # Reward shaping
├── predictive_marl.py         # Multi-agent RL prediction
├── predictive_cooperation.py  # Cooperation prediction
└── algorithmic_credit.py      # Credit assignment
```

### Level 5: Data & Parameter Management
Data processing and parameter resolution.

```
core/
├── data_registry.py              # Agentic data discovery
├── agentic_parameter_resolver.py # LLM-based param matching
├── agentic_feedback_router.py    # Route feedback to agents
├── io_manager.py                 # Input/output management
├── smart_data_transformer.py     # Data format transformation
├── smart_data_extractor.py       # Extract structured data
└── information_storage.py        # Information persistence
```

### Level 6: Context Management
Managing token limits and context windows.

```
core/
├── smart_context_manager.py   # Auto-chunking, compression
├── global_context_guard.py    # Global context protection
├── context_guard.py           # Context overflow prevention
├── context_gradient.py        # Context-as-gradient learning
├── content_gate.py            # Content filtering
├── agentic_chunker.py         # LLM-based chunking
└── agentic_compressor.py      # LLM-based compression
```

### Level 7: Metadata & Tools
Tool discovery and metadata management.

```
core/
├── protocols.py               # Core protocols (MetadataProvider)
├── metadata_protocol.py       # Metadata protocol definitions
├── metadata_fetcher.py        # Fetch metadata
├── metadata_tool_registry.py  # Register & discover tools
├── base_metadata_provider.py  # Base metadata implementation
├── tool_shed.py               # Tool management & caching
└── tool_interceptor.py        # Tool call interception
```

### Level 8: Persistence & State
State management and persistence.

```
core/
├── persistence.py        # Vault - save/load state
├── session_manager.py    # Session management
└── shared_context.py     # Shared context across agents
```

### Level 9: Utilities
Low-level utilities and helpers.

```
core/
├── token_counter.py           # Count tokens
├── token_utils.py             # Token utilities
├── robust_parsing.py          # Robust parsing utilities
├── timeouts.py                # Timeout management
├── model_limits_catalog.py    # Model context limits
├── trajectory_parser.py       # Parse trajectories
├── enhanced_logging_and_context.py # Logging utilities
└── algorithmic_foundations.py # Core algorithms
```

### Level 10: Integration & Wrappers
Integration helpers and wrappers.

```
core/
├── universal_wrapper.py   # Wrap any module with Jotty
├── integration.py         # Integration helpers
├── compression_agent.py   # Compression agent wrapper
├── jotty_fixes.py        # Backward compatibility fixes
├── llm_rag.py            # RAG integration
└── __init__.py           # Core exports
```

### Special Directories

```
core/
├── agentic_discovery/     # Agentic data discovery
│   └── __init__.py       # Discovery orchestrator
├── swarm_prompts/        # Swarm coordination prompts
│   ├── architect_orchestration.md
│   ├── auditor_coordination.md
│   └── auditor_goal_alignment.md
└── validation_prompts/   # Validation prompts
    └── generic_auditor.md
```

---

## 🔗 Module Dependencies

### Core Dependencies (Must load first)

```
1. data_structures.py       (No dependencies - defines types)
2. protocols.py             (No dependencies - defines protocols)
3. agent_config.py          (Depends on: data_structures)
4. robust_parsing.py        (Utility - minimal dependencies)
5. token_utils.py           (Utility - minimal dependencies)
```

### Subsystem Dependencies

```
Memory System:
  cortex.py → data_structures.py
  brain_modes.py → cortex.py
  brain_memory_manager.py → cortex.py, brain_modes.py

Learning System:
  rl_components.py → data_structures.py
  learning.py → rl_components.py, data_structures.py
  q_learning.py → learning.py, data_structures.py
  shaped_rewards.py → learning.py

Context Management:
  context_guard.py → token_counter.py
  global_context_guard.py → context_guard.py
  smart_context_manager.py → context_guard.py, agentic_chunker.py

Agent Execution:
  jotty_core.py → ALL subsystems
  conductor.py → jotty_core.py, ALL subsystems
```

---

## 📁 Folder Structure

### Current Structure (Flat)
```
Jotty/
├── core/           # 62 files (all mixed together)
├── tests/
├── interface.py
└── __init__.py
```

### Proposed Structure (Organized)
```
Jotty/
├── core/
│   ├── 01_foundation/         # Core types, protocols, config
│   │   ├── data_structures.py
│   │   ├── protocols.py
│   │   ├── agent_config.py
│   │   └── __init__.py
│   │
│   ├── 02_orchestration/      # Conductor, execution engine
│   │   ├── conductor.py
│   │   ├── jotty_core.py
│   │   ├── roadmap.py
│   │   ├── dynamic_dependency_graph.py
│   │   ├── policy_explorer.py
│   │   └── __init__.py
│   │
│   ├── 03_agents/             # Agent execution & communication
│   │   ├── axon.py
│   │   ├── feedback_channel.py
│   │   ├── inspector.py
│   │   ├── modern_agents.py
│   │   └── __init__.py
│   │
│   ├── 04_memory/             # Memory systems
│   │   ├── cortex.py
│   │   ├── brain_modes.py
│   │   ├── brain_memory_manager.py
│   │   ├── simple_brain.py
│   │   └── __init__.py
│   │
│   ├── 05_learning/           # RL & learning
│   │   ├── learning.py
│   │   ├── q_learning.py
│   │   ├── rl_components.py
│   │   ├── offline_learning.py
│   │   ├── shaped_rewards.py
│   │   ├── predictive_marl.py
│   │   ├── predictive_cooperation.py
│   │   ├── algorithmic_credit.py
│   │   └── __init__.py
│   │
│   ├── 06_data/               # Data management
│   │   ├── data_registry.py
│   │   ├── io_manager.py
│   │   ├── agentic_parameter_resolver.py
│   │   ├── agentic_feedback_router.py
│   │   ├── smart_data_transformer.py
│   │   ├── smart_data_extractor.py
│   │   ├── information_storage.py
│   │   ├── agentic_discovery/
│   │   └── __init__.py
│   │
│   ├── 07_context/            # Context management
│   │   ├── smart_context_manager.py
│   │   ├── global_context_guard.py
│   │   ├── context_guard.py
│   │   ├── context_gradient.py
│   │   ├── content_gate.py
│   │   ├── agentic_chunker.py
│   │   ├── agentic_compressor.py
│   │   └── __init__.py
│   │
│   ├── 08_metadata/           # Metadata & tools
│   │   ├── metadata_protocol.py
│   │   ├── metadata_fetcher.py
│   │   ├── metadata_tool_registry.py
│   │   ├── base_metadata_provider.py
│   │   ├── tool_shed.py
│   │   ├── tool_interceptor.py
│   │   └── __init__.py
│   │
│   ├── 09_persistence/        # State persistence
│   │   ├── persistence.py
│   │   ├── session_manager.py
│   │   ├── shared_context.py
│   │   └── __init__.py
│   │
│   ├── 10_utils/              # Utilities
│   │   ├── token_counter.py
│   │   ├── token_utils.py
│   │   ├── robust_parsing.py
│   │   ├── timeouts.py
│   │   ├── model_limits_catalog.py
│   │   ├── trajectory_parser.py
│   │   ├── enhanced_logging_and_context.py
│   │   ├── algorithmic_foundations.py
│   │   └── __init__.py
│   │
│   ├── 11_integration/        # Wrappers & integration
│   │   ├── universal_wrapper.py
│   │   ├── integration.py
│   │   ├── compression_agent.py
│   │   ├── jotty_fixes.py
│   │   ├── llm_rag.py
│   │   └── __init__.py
│   │
│   ├── prompts/               # Prompt templates
│   │   ├── swarm/
│   │   │   ├── architect_orchestration.md
│   │   │   ├── auditor_coordination.md
│   │   │   └── auditor_goal_alignment.md
│   │   └── validation/
│   │       └── generic_auditor.md
│   │
│   └── __init__.py            # Core exports
│
├── tests/                     # Test suite
├── interface.py               # Clean API
├── __init__.py               # Main exports
├── default_config.yml        # Default config
├── ARCHITECTURE.md           # This file
└── README.md                 # User guide
```

---

## ⚡ Execution Order

### Startup Sequence (Order Matters!)

```
1. IMPORT PHASE
   ├── data_structures.py      (Core types)
   ├── protocols.py             (Core protocols)
   ├── agent_config.py          (Agent config)
   ├── Utilities                (token_utils, robust_parsing, etc.)
   ├── Memory components        (cortex, brain_modes)
   ├── Learning components      (learning, q_learning)
   ├── Context components       (context_guard, smart_context_manager)
   ├── Metadata components      (metadata_protocol, tool_shed)
   ├── Data components          (data_registry, io_manager)
   ├── Agent components         (axon, feedback_channel)
   ├── Orchestration            (roadmap, dynamic_dependency_graph)
   ├── Core execution           (jotty_core)
   └── Conductor                (conductor.py - imports everything)

2. INITIALIZATION PHASE (Conductor.__init__)
   ├── 1. Load configuration
   ├── 2. Initialize IOManager
   ├── 3. Initialize DataRegistry
   ├── 4. Initialize MetadataToolRegistry
   ├── 5. Initialize SmartDataTransformer
   ├── 6. Initialize AgentSlack (Axon)
   ├── 7. Initialize ParameterResolver
   ├── 8. Initialize FeedbackChannel
   ├── 9. Initialize Memory (Cortex)
   ├── 10. Initialize Q-Learning
   ├── 11. Initialize Roadmap
   ├── 12. Build Dependency Graph
   ├── 13. Initialize Brain Modes
   └── 14. Initialize each Agent's JottyCore wrapper

3. EXECUTION PHASE (conductor.run)
   For each episode:
     ├── Create Markovian TODO
     ├── Resolve execution order
     └── For each agent:
         ├── 1. Resolve parameters
         ├── 2. Run Architect (if enabled)
         ├── 3. Execute agent
         ├── 4. Run Auditor (if enabled)
         ├── 5. Store results
         ├── 6. Update Q-values
         ├── 7. Broadcast to other agents
         └── 8. Update roadmap

4. CLEANUP PHASE
   ├── Memory consolidation
   ├── Save state to disk
   └── Return results
```

### Critical Path (Hot Path)

These files are called on EVERY agent execution:

```
conductor.run()
  → roadmap.get_next_task()
  → agentic_parameter_resolver.resolve()
  → jotty_core.execute()
      → Agent Architect validation
      → agent.forward()
      → Agent Auditor validation
  → io_manager.store_result()
  → data_registry.register()
  → q_learning.update()
  → cortex.store_memory()
  → axon.broadcast()
  → roadmap.update()
```

---

## 🎯 Key Takeaways

1. **Foundation First**: data_structures.py and protocols.py have NO dependencies
2. **Layered Architecture**: Each layer builds on previous layers
3. **Conductor is Central**: All subsystems converge at conductor.py
4. **Hot Path Optimization**: Files in critical path should be fast
5. **Lazy Loading**: Non-critical components can be loaded on-demand

---

## 📚 Related Documentation

- [README.md](README.md) - User guide & quick start
- [TESTING_PLAN.md](TESTING_PLAN.md) - Testing strategy
- [default_config.yml](default_config.yml) - Configuration reference

---

**Generated by:** JOTTY Framework Analysis
**Date:** 2026-01-10
