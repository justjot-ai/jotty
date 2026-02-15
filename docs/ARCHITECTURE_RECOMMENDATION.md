# Jotty Architecture Recommendation - Clean Layering

**Date:** 2026-02-15
**Status:** 🚨 ARCHITECTURAL ISSUES IDENTIFIED
**Priority:** HIGH - Core design principle violation

---

## 🚨 Current Problems

### **Problem 1: CLI in Wrong Layer**
```
❌ CURRENT (WRONG):
Jotty/
├── core/              # Framework internals
│   └── interface/
│       └── cli/       # ❌ CLI application in core layer!
├── sdk/               # External SDK
└── apps/              # Applications
    ├── frontend/
    └── telegram_bot/
```

**Issue:** CLI is an APPLICATION, not part of the core framework.

### **Problem 2: CLI Bypasses SDK Layer**
```python
# In Jotty/core/interface/cli/app.py (WRONG!)
from Jotty.core.intelligence.orchestration import Orchestrator  # ❌ Direct core import
from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig  # ❌ Direct core import
```

**Issue:** CLI imports directly from `core/`, bypassing the SDK layer.

### **Problem 3: No Dogfooding**
- SDK is not being used by internal applications
- SDK gets no real-world testing
- SDK API not validated by internal use

---

## ✅ Recommended Architecture

### **Clean Layer Hierarchy**

```
┌─────────────────────────────────────────────────────┐
│  LAYER 5: APPLICATIONS (apps/)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │   CLI    │  │ Frontend │  │  Telegram Bot    │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
│       ↓              ↓                  ↓           │
└───────────────────────────────────────────────────┬─┘
                                                    │
┌───────────────────────────────────────────────────┴─┐
│  LAYER 4: SDK (sdk/)                                │
│  ┌─────────────────────────────────────────────┐   │
│  │  Jotty SDK - Clean, Stable API              │   │
│  │  • Jotty() client                           │   │
│  │  • Event emitters                           │   │
│  │  • Handles (skill, agent, session)          │   │
│  └─────────────────────────────────────────────┘   │
│       ↓                                             │
└─────────────────────────────────────────────────┬───┘
                                                  │
┌─────────────────────────────────────────────────┴───┐
│  LAYER 3: CORE API (core/interface/)                │
│  Internal interfaces for SDK consumption            │
│  • JottyAPI (unified.py)                            │
│  • ChatAPI, WorkflowAPI                             │
│  • Use cases (chat, workflow)                       │
└─────────────────────────────────────────────────┬───┘
                                                  │
┌─────────────────────────────────────────────────┴───┐
│  LAYER 2: CORE FRAMEWORK (core/)                    │
│  • modes/ (agent, workflow, execution)              │
│  • capabilities/ (skills, registry)                 │
│  • intelligence/ (memory, learning, orchestration)  │
│  • infrastructure/ (foundation, utils, context)     │
└─────────────────────────────────────────────────────┘
```

### **Directory Structure**

```
Jotty/
├── core/                          # LAYER 2-3: Framework internals
│   ├── interface/                 # LAYER 3: Internal API (for SDK)
│   │   ├── api/                   # JottyAPI, ChatAPI, etc.
│   │   ├── use_cases/             # Chat, workflow use cases
│   │   └── interfaces/            # Messages, hosts, adapters
│   ├── modes/                     # Agent, workflow, execution
│   ├── capabilities/              # Skills, registry, tools
│   ├── intelligence/              # Memory, learning, orchestration
│   └── infrastructure/            # Foundation, utils, context
│
├── sdk/                           # LAYER 4: External SDK
│   ├── client.py                  # Jotty() SDK client
│   ├── __init__.py                # Public exports
│   └── generated/                 # Multi-language SDKs
│
├── apps/                          # LAYER 5: Applications
│   ├── cli/                       # ✅ CLI app (MOVED HERE!)
│   │   ├── main.py                # Entry point
│   │   ├── repl/                  # REPL engine
│   │   ├── commands/              # Slash commands
│   │   ├── ui/                    # Rich rendering
│   │   └── config/                # CLI config
│   ├── frontend/                  # Web UI
│   ├── telegram_bot/              # Telegram bot
│   └── discord_bot/               # (future)
│
├── examples/                      # Usage examples
├── tests/                         # Test suite
└── docs/                          # Documentation
```

---

## 🎯 Key Principles

### **1. Dependency Flow (CRITICAL)**

```
Applications ──> SDK ──> Core API ──> Core Framework
    ↑             ↑          ↑            ↑
  LAYER 5      LAYER 4    LAYER 3     LAYER 2
```

**Rules:**
- ✅ Applications ONLY import from SDK
- ✅ SDK ONLY imports from core/interface/api/
- ✅ Core API can use core framework
- ❌ Applications NEVER import from core directly
- ❌ SDK NEVER imports from applications

### **2. SDK as Contract**

The SDK is the **stable public API**. Core can change internally without breaking apps.

```python
# ✅ GOOD: CLI uses SDK
from jotty import Jotty

client = Jotty()
result = await client.chat("Hello")

# ❌ BAD: CLI bypasses SDK
from Jotty.core.intelligence.orchestration import Orchestrator
swarm = Orchestrator(...)
```

### **3. Dogfooding**

Internal apps use the SDK = SDK gets real-world testing.

```
CLI      ──┐
Frontend ──┼──> SDK ──> Core
Telegram ──┘
           ↑
      Same API that
      external devs use
```

---

## 📋 Migration Plan

### **Phase 1: Move CLI to apps/ (HIGH PRIORITY)**

#### Step 1: Create apps/cli structure
```bash
mkdir -p Jotty/apps/cli
mv Jotty/core/interface/cli/* Jotty/apps/cli/
```

#### Step 2: Update CLI imports
**Before:**
```python
# apps/cli/app.py (WRONG)
from Jotty.core.intelligence.orchestration import Orchestrator
from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
```

**After:**
```python
# apps/cli/app.py (CORRECT)
from jotty import Jotty
from jotty.sdk import EventEmitter, SDKEventType

client = Jotty()
```

#### Step 3: Update entry points
```python
# apps/cli/main.py (new entry point)
if __name__ == "__main__":
    from Jotty.apps.cli.app import main
    main()
```

#### Step 4: Update imports across codebase
```bash
# Find all imports
grep -r "from Jotty.core.interface.cli" Jotty/

# Replace with
from Jotty.apps.cli
```

### **Phase 2: Ensure SDK Completeness (MEDIUM PRIORITY)**

Verify SDK exposes everything CLI needs:

```python
# SDK should provide:
✅ client.chat()           # Chat mode
✅ client.workflow()       # Workflow mode
✅ client.stream()         # Streaming
✅ client.skill()          # Direct skill access
✅ client.agent()          # Direct agent access
✅ client.on()             # Event callbacks
✅ client.session()        # Session management

# If CLI needs more, ADD to SDK (don't bypass it!)
```

### **Phase 3: Frontend/Telegram Bot (LOW PRIORITY)**

Verify other apps use SDK:

```python
# apps/web/ should use SDK
from jotty import Jotty

# apps/telegram/ should use SDK
from jotty import Jotty
```

---

## 🏗️ Updated CLAUDE.md Architecture Section

```markdown
## 🏗️ Clean 5-Layer Architecture

Jotty follows strict layering principles:

```
Layer 5: APPLICATIONS → CLI, frontend, bots (use SDK)
Layer 4: SDK          → Stable public API (jotty.Jotty)
Layer 3: CORE API     → Internal interfaces (JottyAPI, ChatAPI)
Layer 2: CORE         → Framework internals (modes, intelligence, etc.)
Layer 1: FOUNDATION   → Utils, context, monitoring
```

**CRITICAL RULES:**
- ✅ Apps import ONLY from `jotty` (SDK)
- ✅ SDK imports ONLY from `Jotty.core.interface.api`
- ❌ Apps NEVER import from `Jotty.core` directly
- ❌ SDK NEVER imports from apps

**Example:**
```python
# ✅ CORRECT: CLI uses SDK
from jotty import Jotty
client = Jotty()
result = await client.chat("Hello")

# ❌ WRONG: CLI bypasses SDK
from Jotty.core.intelligence.orchestration import Orchestrator  # NO!
```
```

---

## 📊 Benefits of Proper Layering

| Benefit | Description |
|---------|-------------|
| **Stability** | Core can change without breaking apps (SDK is stable contract) |
| **Dogfooding** | Internal apps test SDK = better SDK quality |
| **Consistency** | Internal and external developers use same API |
| **Maintainability** | Clear boundaries, easier to refactor |
| **Documentation** | Internal apps serve as reference examples |
| **Versioning** | SDK versioning protects apps from core changes |

---

## 🚦 Decision Matrix: Should X be in SDK or Apps?

| Component | Layer | Reason |
|-----------|-------|--------|
| **CLI** | apps/ | Application consuming SDK |
| **Web Frontend** | apps/ | Application consuming SDK |
| **Telegram Bot** | apps/ | Application consuming SDK |
| **SDK Client** | sdk/ | Public API for all apps |
| **JottyAPI** | core/interface/api/ | Internal API for SDK |
| **ChatAPI** | core/interface/api/ | Internal API for SDK |
| **Orchestrator** | core/intelligence/ | Framework internals |
| **Memory** | core/intelligence/ | Framework internals |
| **Skills** | core/capabilities/ | Framework internals |

---

## 🎯 Action Items

### **Immediate (This Week)**

- [ ] **Move CLI to apps/cli/**
  - Create `Jotty/apps/cli/` directory
  - Move all CLI code from `core/interface/cli/`
  - Update imports to use SDK instead of core
  - Update entry points
  - Test CLI still works

- [ ] **Verify SDK Completeness**
  - Ensure SDK exposes all CLI needs
  - Add missing methods if needed
  - Document SDK API

- [ ] **Update Documentation**
  - Update CLAUDE.md with clean architecture
  - Update README with correct import examples
  - Add architecture diagram

### **Short Term (This Month)**

- [ ] **Verify Other Apps Use SDK**
  - Check `apps/web/` imports
  - Check `apps/telegram/` imports
  - Fix any direct core imports

- [ ] **Add Architecture Tests**
  - Test that apps don't import from core
  - Test that SDK only imports from core/interface
  - Fail build if layering violated

### **Long Term**

- [ ] **Enforce with Linting**
  - Add import-linter to pre-commit
  - Block `from Jotty.core` in apps/
  - Block `from Jotty.apps` in sdk/

---

## 🔍 Current vs Recommended

### **Current (WRONG)**
```python
# apps implicitly in core/interface/cli/
from Jotty.core.intelligence.orchestration import Orchestrator  # ❌
from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig  # ❌

swarm = Orchestrator(agents="...")
result = await swarm.run(goal="...")
```

### **Recommended (CORRECT)**
```python
# apps/ explicitly separate
from jotty import Jotty  # ✅ SDK layer

client = Jotty()
result = await client.workflow(
    goal="...",
    agents="..."
)
```

---

## 📚 References

**Clean Architecture Principles:**
- [Uncle Bob's Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Hexagonal Architecture](https://alistair.cockburn.us/hexagonal-architecture/)
- [Domain-Driven Design](https://martinfowler.com/bliki/DomainDrivenDesign.html)

**Python Package Layering:**
- [Python Application Layouts](https://realpython.com/python-application-layouts/)
- [Structuring Your Project](https://docs.python-guide.org/writing/structure/)

---

## ✅ Success Criteria

Migration is complete when:

- ✅ CLI is in `apps/cli/` (not `core/interface/cli/`)
- ✅ CLI imports only from `jotty` (SDK), never `Jotty.core`
- ✅ All apps use SDK consistently
- ✅ Tests pass
- ✅ Documentation updated
- ✅ Architecture diagram added
- ✅ Import linting enforced

---

**Author:** Claude Code Analysis
**Date:** 2026-02-15
**Status:** Recommendation Ready for Review
