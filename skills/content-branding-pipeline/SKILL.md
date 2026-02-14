---
name: running-content-branding-pipeline
description: "This composite skill combines: 1. **Domain Brainstorming** (Source): domain-name-brainstormer 2. **Artifact Creation** (Processor): artifacts-builder 3. **Brand Guidelines** (Processor): brand-guidelines 4. **Theme Application** (Sink): theme-factory. Use when the user wants to brand, branding, brand guidelines."
---

# Content & Branding Pipeline Composite Skill

Complete content creation workflow: domain brainstorming → artifact creation → brand → theme.

## Description

This composite skill combines:
1. **Domain Brainstorming** (Source): domain-name-brainstormer
2. **Artifact Creation** (Processor): artifacts-builder
3. **Brand Guidelines** (Processor): brand-guidelines
4. **Theme Application** (Sink): theme-factory


## Type
composite

## Base Skills
- claude-cli-llm
- brand-guidelines

## Execution
sequential


## Capabilities
- document
- media

## Usage

```python
from skills.content_branding_pipeline.tools import content_branding_pipeline_tool

result = await content_branding_pipeline_tool({
    'project_description': 'AI-powered code review tool',
    'artifact_type': 'html',
    'include_domain_brainstorm': True,
    'apply_brand': True,
    'apply_theme': True,
    'theme_name': 'ocean_depths'
})
```

## Parameters

- `project_description` (str, required): Project description
- `project_name` (str, optional): Project name (will brainstorm if not provided)
- `artifact_type` (str, optional): 'html', 'presentation', 'document' (default: 'html')
- `include_domain_brainstorm` (bool, optional): Brainstorm domains (default: True)
- `create_artifact` (bool, optional): Create artifact (default: True)
- `apply_brand` (bool, optional): Apply brand guidelines (default: True)
- `apply_theme` (bool, optional): Apply theme (default: True)
- `brand_style` (str, optional): Brand style (default: 'anthropic')
- `theme_name` (str, optional): Theme name (default: 'ocean_depths')
- `max_domain_suggestions` (int, optional): Max domain suggestions (default: 10)

## Architecture

Source → Processor → Processor → Sink pattern:
- **Source**: Domain name brainstormer
- **Processor**: Artifacts builder → Brand guidelines
- **Sink**: Theme factory

No code duplication - reuses existing skills.

## Workflow

```
Task Progress:
- [ ] Step 1: Brainstorm domain names
- [ ] Step 2: Create artifact
- [ ] Step 3: Apply brand guidelines
- [ ] Step 4: Apply theme
```

**Step 1: Brainstorm domain names**
Generate creative domain name suggestions for the project.

**Step 2: Create artifact**
Build the HTML, presentation, or document artifact.

**Step 3: Apply brand guidelines**
Apply Anthropic brand colors, typography, and styling.

**Step 4: Apply theme**
Finalize the artifact with the selected visual theme.

## Triggers
- "content branding pipeline"
- "brand"
- "branding"
- "brand guidelines"

## Category
media-creation
