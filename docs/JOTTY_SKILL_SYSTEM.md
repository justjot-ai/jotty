# Jotty Skill System - Complete Guide

## Overview

The Jotty skill system is a framework-level infrastructure for dynamically loading and managing tools that extend agent capabilities. Skills are modular, reusable components that can be manually written or AI-generated.

## Core Concepts

### What is a Skill?

A **skill** is a collection of related tools organized in a directory. Each skill provides specific functionality (e.g., web search, document conversion, image generation).

### Skill Structure

Every skill follows a standard structure:

```
skills/
  skill-name/
    SKILL.md          # Skill metadata and documentation
    tools.py          # Tool implementations
    requirements.txt  # Optional: Python dependencies
    scripts/          # Optional: For Claude Code skills
```

## Components

### 1. SkillsRegistry (`core/registry/skills_registry.py`)

The central registry that loads and manages all skills.

**Key Features:**
- Auto-discovers skills from `~/jotty/skills` and `~/.claude/skills`
- Dynamically loads Python tools from `tools.py`
- Supports both standard Python skills and Claude Code skills (with `scripts/` directory)
- Auto-installs dependencies via `SkillDependencyManager`
- Hot-reloads skills on changes

**Usage:**
```python
from core.registry.skills_registry import get_skills_registry

registry = get_skills_registry()
registry.init()  # Load all skills

# List all skills
skills = registry.list_skills()

# Get a specific skill
skill = registry.get_skill('calculator')

# Get all registered tools
tools = registry.get_registered_tools()
```

### 2. SkillDefinition

Represents a loaded skill with:
- `name`: Skill name
- `description`: Skill description (from SKILL.md)
- `tools`: Dict mapping tool names to execute functions
- `metadata`: Additional metadata (path, type, etc.)

### 3. Tool Structure

Tools are Python functions that:
- Accept `params: Dict[str, Any]` as input
- Return `Dict[str, Any]` with `success` boolean and results/error
- Can be async or sync

**Example Tool:**
```python
def calculate_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform mathematical calculations."""
    try:
        expression = params.get('expression')
        result = eval(expression, safe_dict)
        return {
            'success': True,
            'result': float(result),
            'expression': expression
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
```

## Skill Types

### 1. Standard Skills

Standard Python skills with `tools.py` containing tool functions.

**Example:** `calculator`, `web-search`, `file-operations`

### 2. Claude Code Skills

Skills that use Python scripts in a `scripts/` directory instead of `tools.py`. The registry creates wrapper tools that execute these scripts.

**Example:** `last30days-claude-cli`

### 3. Composite Skills (`core/registry/composite_skill.py`)

Skills that combine multiple existing skills into workflows.

**Execution Modes:**
- **Sequential**: Steps execute one after another
- **Parallel**: Steps execute simultaneously
- **Mixed**: Steps execute based on dependencies

**Example:**
```python
from core.registry.composite_skill import create_composite_skill, ExecutionMode

skill = create_composite_skill(
    name='research-to-pdf',
    description='Research topic and generate PDF',
    steps=[
        {
            'skill_name': 'last30days-claude-cli',
            'tool_name': 'last30days_claude_cli_tool',
            'params': lambda p, r: {'topic': p.get('topic')}
        },
        {
            'skill_name': 'document-converter',
            'tool_name': 'convert_to_pdf_tool',
            'params': lambda p, r: {
                'input_file': r['step_0']['md_path']
            }
        }
    ],
    execution_mode=ExecutionMode.SEQUENTIAL
)
```

### 4. Pipeline Skills (`core/registry/pipeline_skill.py`)

Skills following the **Source → Processor → Sink** pattern.

**Step Types:**
- `source`: Data retrieval/generation
- `processor`: Data transformation
- `sink`: Data output/delivery

**Template Variables:**
- `{{variable}}` - from current parameters
- `{{step.field}}` - from previous step results
- `{{step.field.nested}}` - nested access

**Example:**
```python
from core.registry.pipeline_skill import create_pipeline_skill

skill = create_pipeline_skill(
    name='search-summarize-pdf-telegram',
    description='Search → Summarize → PDF → Telegram',
    pipeline=[
        {
            "type": "source",
            "skill": "web-search",
            "tool": "search_web_tool",
            "params": {"query": "{{topic}}", "max_results": 10}
        },
        {
            "type": "processor",
            "skill": "claude-cli-llm",
            "tool": "summarize_text_tool",
            "params": {"content": "{{source.results}}"}
        },
        {
            "type": "sink",
            "skill": "telegram-sender",
            "tool": "send_telegram_file_tool",
            "params": {"file_path": "{{processor.pdf_path}}"}
        }
    ]
)
```

## Skill Generation

### SkillGenerator (`core/registry/skill_generator.py`)

AI-powered skill creation using Jotty's unified LLM interface.

**Features:**
- Generate skills from natural language descriptions
- Auto-create `SKILL.md` and `tools.py`
- Validate generated skills
- Improve existing skills based on feedback

**Usage:**
```python
from core.registry.skill_generator import get_skill_generator

generator = get_skill_generator(skills_registry=registry)

# Generate a new skill
result = generator.generate_skill(
    skill_name="todoist-automation",
    description="Automate Todoist tasks - create, complete, and list tasks",
    requirements="Use Todoist API, handle authentication",
    examples=["Create task 'Buy milk'", "List all tasks"]
)

# Improve an existing skill
generator.improve_skill(
    skill_name="calculator",
    feedback="Add support for matrix operations",
    changes="Add matrix multiplication and inversion"
)
```

## Dependency Management

### SkillDependencyManager (`core/registry/skill_dependency_manager.py`)

Automatically manages skill dependencies.

**Features:**
- Reads `requirements.txt` files
- Extracts dependencies from `tools.py` imports
- Auto-installs missing packages via `SkillVenvManager`
- Maps common imports to package names (e.g., `PIL` → `pillow`)

**Auto-installation:**
Dependencies are automatically installed when skills are loaded by the registry.

## Virtual Environment Management

### SkillVenvManager (`core/registry/skill_venv_manager.py`)

Manages isolated Python virtual environments for skills.

**Features:**
- Creates shared venv for all skills
- Installs packages in isolated environment
- Prevents dependency conflicts

## Available Skills

### Core Skills
- `calculator`: Mathematical calculations and unit conversions
- `file-operations`: File system operations
- `web-search`: Web search using DuckDuckGo
- `web-scraper`: Web page scraping
- `http-client`: HTTP requests
- `shell-exec`: Shell command execution
- `text-utils`: Text processing utilities
- `time-converter`: Time zone and format conversions
- `weather-checker`: Weather information

### Content Skills
- `document-converter`: Convert between document formats (PDF, Markdown, etc.)
- `text-chunker`: Split text into chunks
- `image-generator`: Generate images using AI
- `mindmap-generator`: Generate mindmaps from content
- `content-repurposer`: Repurpose content for different platforms

### Research Skills
- `arxiv-downloader`: Download papers from arXiv
- `last30days-claude-cli`: Research topics from last 30 days
- `search-to-justjot-idea`: Convert search results to ideas
- `trending-topics-to-ideas`: Convert trending topics to ideas
- `v2v-trending-search`: Video-to-video trending search

### Integration Skills
- `telegram-sender`: Send messages/files via Telegram
- `remarkable-sender`: Send files to reMarkable tablet
- `remarkable-upload`: Upload files to reMarkable cloud
- `oauth-automation`: OAuth authentication automation
- `notebooklm-pdf`: NotebookLM PDF integration

### Composite/Pipeline Skills
- `research-to-pdf`: Research → PDF generation
- `last30days-to-pdf-telegram`: Research → PDF → Telegram
- `last30days-to-pdf-remarkable`: Research → PDF → reMarkable
- `search-summarize-pdf-telegram`: Search → Summarize → PDF → Telegram
- `v2v-to-pdf-telegram-remarkable`: V2V search → PDF → Telegram + reMarkable

### MCP Skills
- `mcp-justjot`: JustJot MCP server integration
- `mcp-justjot-mcp-client`: MCP client for JustJot

## Creating a New Skill

### Step 1: Create Skill Directory

```bash
mkdir -p ~/jotty/skills/my-skill
cd ~/jotty/skills/my-skill
```

### Step 2: Create SKILL.md

```markdown
# My Skill

## Description
Brief description of what the skill does.

## Tools

### my_tool
Description of the tool.

**Parameters:**
- `param1` (str, required): Description
- `param2` (int, optional): Description

**Returns:**
- `success` (bool): Whether operation succeeded
- `result` (any): Result data
- `error` (str, optional): Error message if failed
```

### Step 3: Create tools.py

```python
from typing import Dict, Any

def my_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool description.
    
    Args:
        params: Dictionary containing tool parameters
    
    Returns:
        Dictionary with success status and results
    """
    try:
        param1 = params.get('param1')
        if not param1:
            return {
                'success': False,
                'error': 'param1 is required'
            }
        
        # Tool logic here
        result = do_something(param1)
        
        return {
            'success': True,
            'result': result
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
```

### Step 4: Add Dependencies (Optional)

Create `requirements.txt`:
```
requests>=2.28.0
beautifulsoup4>=4.11.0
```

### Step 5: Test the Skill

```python
from core.registry.skills_registry import get_skills_registry

registry = get_skills_registry()
registry.init()

skill = registry.get_skill('my-skill')
tool = skill.tools['my_tool']

result = tool({'param1': 'value'})
print(result)
```

## Integration with Jotty Framework

Skills integrate seamlessly with Jotty's tool system:

1. **Tool Registry**: Skills are loaded into the unified tool registry
2. **Agent Access**: Agents can discover and use skill tools automatically
3. **Metadata**: Tool metadata is available for agent decision-making
4. **Hot Reload**: Skills can be reloaded without restarting the system

## Best Practices

1. **Tool Naming**: Use descriptive names ending with `_tool` (optional but recommended)
2. **Error Handling**: Always return `success` boolean and handle errors gracefully
3. **Documentation**: Provide clear documentation in `SKILL.md`
4. **Type Hints**: Use type hints for better IDE support
5. **Dependencies**: List all dependencies in `requirements.txt`
6. **Testing**: Test tools with various inputs before deployment
7. **DRY**: Use composite/pipeline skills to reuse existing skills

## Architecture Principles

1. **DRY (Don't Repeat Yourself)**: Composite skills reuse existing skills
2. **Modularity**: Each skill is self-contained
3. **Discoverability**: Skills are auto-discovered from directories
4. **Extensibility**: Easy to add new skills without modifying core code
5. **Isolation**: Dependencies are managed in isolated environments

## Directory Structure

```
Jotty/
├── core/
│   └── registry/
│       ├── skills_registry.py          # Main registry
│       ├── skill_generator.py          # AI skill generation
│       ├── composite_skill.py          # Composite skills
│       ├── pipeline_skill.py           # Pipeline skills
│       ├── skill_dependency_manager.py # Dependency management
│       └── skill_venv_manager.py       # Venv management
└── skills/
    ├── calculator/
    │   ├── SKILL.md
    │   └── tools.py
    ├── web-search/
    │   ├── SKILL.md
    │   └── tools.py
    └── ...
```

## Environment Variables

- `JOTTY_SKILLS_DIR`: Override default skills directory (default: `~/jotty/skills`)

## Future Enhancements

- Skill versioning
- Skill marketplace
- Skill testing framework
- Skill performance metrics
- Skill dependency graph visualization
