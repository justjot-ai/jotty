# Composite Skills Framework

## Overview

Composite skills combine multiple existing skills into reusable workflows. They follow DRY principles - no code duplication, just composition of existing skills.

## Architecture

### Execution Modes

1. **Sequential** (default): Steps execute one after another
   - Step 2 receives output from Step 1
   - Failures stop the workflow

2. **Parallel**: Steps execute simultaneously
   - All steps receive initial parameters
   - Results collected after all complete

3. **Mixed**: Steps execute based on dependencies
   - Steps with satisfied dependencies run in parallel
   - Supports complex workflows

### Step Definition

Each step in a composite skill defines:

```python
{
    'skill_name': 'last30days-claude-cli',  # Skill to use
    'tool_name': 'last30days_claude_cli_tool',  # Tool in that skill
    'params': {  # Parameters (dict or function)
        'topic': 'multi agent systems'
    },
    'output_key': 'research',  # Key to store result (optional)
    'required': True,  # Stop workflow if this fails (default: True)
    'depends_on': []  # Step indices this depends on (for mixed mode)
}
```

### Parameter Functions

Parameters can be functions that receive current params and results:

```python
'params': lambda params, results: {
    'input_file': results['step_0']['md_path'],
    'output_file': params.get('output_file')
}
```

## Examples

### Example 1: Sequential Workflow

```python
from core.registry.composite_skill import create_composite_skill, ExecutionMode

skill = create_composite_skill(
    name='research-to-pdf',
    description='Research topic and generate PDF',
    steps=[
        {
            'skill_name': 'last30days-claude-cli',
            'tool_name': 'last30days_claude_cli_tool',
            'params': lambda p, r: {'topic': p.get('topic'), 'emit': 'md'}
        },
        {
            'skill_name': 'document-converter',
            'tool_name': 'convert_to_pdf_tool',
            'params': lambda p, r: {
                'input_file': r['step_0']['md_path'],
                'output_file': p.get('output_file')
            }
        }
    ],
    execution_mode=ExecutionMode.SEQUENTIAL
)
```

### Example 2: Parallel Workflow

```python
skill = create_composite_skill(
    name='multi-source-research',
    description='Research from multiple sources in parallel',
    steps=[
        {
            'skill_name': 'last30days-claude-cli',
            'tool_name': 'last30days_claude_cli_tool',
            'params': {'topic': 'AI', 'emit': 'json'},
            'output_key': 'last30days'
        },
        {
            'skill_name': 'web-search',
            'tool_name': 'search_web_tool',
            'params': {'query': 'AI trends', 'max_results': 10},
            'output_key': 'web_search'
        }
    ],
    execution_mode=ExecutionMode.PARALLEL
)
```

### Example 3: Mixed Workflow

```python
skill = create_composite_skill(
    name='research-pdf-telegram',
    description='Research → PDF → Telegram',
    steps=[
        {
            'skill_name': 'last30days-claude-cli',
            'tool_name': 'last30days_claude_cli_tool',
            'params': lambda p, r: {'topic': p.get('topic'), 'emit': 'md'},
            'output_key': 'research'
        },
        {
            'skill_name': 'document-converter',
            'tool_name': 'convert_to_pdf_tool',
            'params': lambda p, r: {
                'input_file': r['research']['md_path']
            },
            'depends_on': [0],  # Depends on step 0
            'output_key': 'pdf'
        },
        {
            'skill_name': 'telegram-sender',
            'tool_name': 'send_telegram_file_tool',
            'params': lambda p, r: {
                'file_path': r['pdf']['output_path']
            },
            'depends_on': [1],  # Depends on step 1
            'output_key': 'telegram'
        }
    ],
    execution_mode=ExecutionMode.MIXED
)
```

## Integration with Registry

Composite skills can be registered like regular skills:

```python
from core.registry.skills_registry import get_skills_registry
from core.registry.composite_skill import create_composite_skill

registry = get_skills_registry()

# Create composite skill
composite = create_composite_skill(...)

# Register it
registry.register_composite_skill(composite)
```

## Benefits

1. **DRY**: No code duplication - reuses existing skills
2. **Composable**: Mix and match skills easily
3. **Flexible**: Sequential, parallel, or mixed execution
4. **Maintainable**: Changes to individual skills automatically benefit composites
5. **Testable**: Each step can be tested independently

## Current Composite Skills

1. **last30days-to-pdf-telegram**: Research → PDF → Telegram
2. **v2v-to-pdf-telegram-remarkable**: V2V search → PDF → Telegram + reMarkable
3. **research-to-pdf**: Research → PDF (existing)

## Creating New Composite Skills

1. Define steps using existing skills
2. Choose execution mode
3. Use `create_composite_skill()` factory
4. Register with registry (or create as regular skill file)
