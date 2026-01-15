# Jotty Management UI

**Standalone deployable UI for managing all Jotty entities**

---

## Overview

This folder contains all UI components and pages for managing Jotty-specific entities. It is designed to be self-contained and deployable independently.

---

## Structure

```
src/jotty/
├── components/
│   └── management/
│       ├── ExtendedEntityManager.tsx  # Main CRUD manager for Jotty entities
│       └── EntityForm.tsx             # Create/edit forms
├── pages/
│   └── ManagementPage.tsx             # Main management dashboard
├── types.ts                           # Jotty entity type definitions
├── index.ts                           # Public exports
└── README.md                          # This file
```

---

## Jotty Entities Managed

### Core Entities
- **Agents** - Custom AI agents with prompts, tools, temperature
- **Swarms** - Multi-agent swarms and configurations
- **Workflows** - Agent workflows and execution presets
- **Tools** - Agent tools and capabilities
- **MCP Tools** - MCP server tool registrations

### Advanced Entities
- **Presets** - Agent and swarm presets
- **Memory Entities** - Memory graph entities (from Jotty memory system)
- **RL Configs** - Reinforcement learning configurations

---

## What's NOT Included

This UI does NOT manage JustJot.ai-specific entities:
- ❌ Sections (content sections)
- ❌ Collections (content collections)
- ❌ Templates (content templates)
- ❌ Ideas (content ideas)

These are managed separately in JustJot.ai.

---

## Usage

### Import Components
```typescript
import { JottyManagementPage } from '@/jotty';
import { ExtendedEntityManager } from '@/jotty/components/management/ExtendedEntityManager';
import type { JottyEntityType } from '@/jotty/types';
```

### Access Management UI
- `/dashboard/jotty` - Main management page
- `/dashboard/diy` - Alias (re-exports JottyManagementPage)

---

## API Integration

All API calls go to `/api/jotty/entities`:
- `GET /api/jotty/entities?type={type}` - List entities
- `POST /api/jotty/entities` - Create entity
- `GET /api/jotty/entities/{id}?type={type}` - Get entity
- `PUT /api/jotty/entities/{id}?type={type}` - Update entity
- `DELETE /api/jotty/entities/{id}?type={type}` - Delete entity

---

## Mobile-First Design

- ✅ 44px+ touch targets (WCAG 2.5.5 Level AA)
- ✅ Material Design 3 components
- ✅ Responsive layouts
- ✅ Dark mode support
- ✅ Smooth animations

---

## Standalone Deployment

This folder can be extracted and deployed independently:
1. Copy `src/jotty/` folder
2. Copy API routes from `src/app/api/jotty/`
3. Copy dashboard routes from `src/app/dashboard/jotty/`
4. Ensure UI components are available (`@/components/ui/*`)

All dependencies are external, making it easy to extract.
