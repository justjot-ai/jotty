# Critical Evaluation: JustJot.ai, Jotty & cmd.dev

## Current State Analysis

### 1. Naming Confusion

| Term | What It Actually Is | Problem |
|------|---------------------|---------|
| **JustJot.ai** | Ideas/notes platform (separate app) | Sounds like "Jotty" - easily confused |
| **Jotty** | Multi-agent orchestration framework | Lives under `stock_market/` directory |
| **cmd.dev** | Deployment host/platform | Generic name, unclear purpose |
| **jotty.justjot.ai** | Jotty's web gateway subdomain | Nests Jotty under JustJot brand |

### 2. Directory Path Issue

```
/var/www/sites/personal/stock_market/Jotty
```

This reveals the legacy: originally built for **PlanMInvesting** (stock analysis), but now Jotty is a **generic multi-agent framework**. The path creates confusion:
- LLMs see "stock_market" and may assume financial context
- Doesn't reflect Jotty's domain-agnostic design

### 3. Deployment Ambiguity

- `cmd.dev` hosts both JustJot.ai and Jotty
- Blue-green containers reference `justjot-ai-blue/green`
- Multiple entry points: `web.py`, `gateway`, `cli`
- No clear separation between staging/production

---

## Recommended Architecture

### Option A: Clear Separation (Recommended)

```
github.com/justjot-ai/
├── jotty/                 # Framework (rename to "swarm" or "axon"?)
├── justjot-app/           # Notes/ideas platform
├── paper2slides/          # Extract to own repo
└── paper2code/            # Extract to own repo

Domains:
├── swarm.dev or axon.ai   # Jotty framework (new identity)
├── justjot.ai             # Ideas platform
└── cmd.dev                # Rename to "workbench.dev" (your dev environment)
```

### Option B: Unified Monorepo with Clear Boundaries

```
github.com/justjot-ai/platform/
├── packages/
│   ├── jotty-core/        # Framework engine
│   ├── jotty-cli/         # CLI interface
│   ├── jotty-web/         # Web gateway
│   ├── justjot-app/       # Notes platform
│   ├── paper2slides/      # Document conversion
│   └── paper2code/        # Paper to code
├── deployments/
│   ├── docker-compose.yml # Local dev
│   ├── kubernetes/        # Production
│   └── cmd-dev/           # SSH deployment scripts
└── CLAUDE.md              # LLM context file (critical!)
```

---

## LLM-Friendly Organization

**Problem:** LLMs get confused by:
1. Multiple codebases in same workspace
2. Unclear which service handles what
3. Legacy paths that don't match current purpose

**Solution: Create clear `CONTEXT.md` files**

```markdown
# /CONTEXT.md (root of each repo)

## Identity
- **Name:** Jotty
- **Purpose:** Multi-agent orchestration framework (DSPy-based)
- **NOT:** A stock trading app, a notes app, or JustJot.ai

## Boundaries
- This repo: Agent orchestration, memory, learning
- JustJot.ai: Separate app for ideas (https://justjot.ai)
- Paper2Slides: Document conversion (consider extracting)

## Deployment
- Local: `docker-compose up` or `python web.py`
- Production: cmd.dev SSH → `screen -dmS jotty python web.py`
- Domain: jotty.justjot.ai (gateway) or standalone

## Key Entry Points
- CLI: `python -m Jotty.cli`
- Web: `python web.py --port 8766`
- Gateway: `python -m Jotty.cli.gateway`
```

---

## Specific Recommendations

| Issue | Recommendation |
|-------|----------------|
| **Path: `/stock_market/Jotty`** | Move to `/var/www/sites/jotty` or `/home/coder/projects/jotty` |
| **cmd.dev naming** | Rename to `workbench.dev` or `dev.justjot.ai` (clearer purpose) |
| **JustJot vs Jotty confusion** | Consider renaming Jotty to **Axon**, **Swarm**, or **Conductor** |
| **Paper2Slides/Paper2Code** | Extract to separate repos - different concerns |
| **Deployment scripts** | Create `/deploy/` folder with env-specific configs |
| **Docker vs SSH** | Standardize: Docker for local, SSH for cmd.dev, K8s for scale |

---

## Proposed New Structure

```
Your Development Machine (IDE/Docker)
└── ~/projects/
    ├── axon/                    # Renamed from Jotty (the framework)
    │   ├── core/
    │   ├── cli/
    │   ├── web/
    │   ├── CONTEXT.md           # LLM context
    │   └── deploy/
    │       ├── local/docker-compose.yml
    │       └── workbench/deploy.sh
    │
    ├── justjot-app/             # Ideas platform
    │   ├── api/
    │   ├── frontend/
    │   └── CONTEXT.md
    │
    └── paper2slides/            # Separate tool
        └── CONTEXT.md

Workbench.dev (renamed from cmd.dev)
├── /home/coder/
│   ├── axon/                    # Framework deployment
│   └── justjot/                 # Ideas app deployment
└── Nginx config pointing to correct ports
```

---

## Summary

| Current | Proposed | Rationale |
|---------|----------|-----------|
| `stock_market/Jotty` | `~/projects/axon` | Clean path, no legacy baggage |
| `cmd.dev` | `workbench.dev` | Clearer purpose |
| JustJot + Jotty merged | Separate repos | Different concerns |
| No CONTEXT.md | Add per-repo | LLM-friendly context |
| Paper2Slides inside | Extract to own repo | Cleaner separation |

---

## Next Steps

1. Create a migration plan with specific steps
2. Draft the CONTEXT.md files for each component
3. Design the deploy scripts for workbench.dev
