# Composite Skill Template Guide

## Overview

Templates are reusable composite skills built using `skill-composer`. They solve common workflow patterns so you don't have to reinvent the wheel.

## Available Templates

### 1. simple-pdf-generator
**Use case:** Convert text/markdown to PDF  
**Pattern:** Sequential (Write → Convert)  
**Best for:** Document generation, report creation

### 2. research-to-pdf
**Use case:** Research topic and generate PDF report  
**Pattern:** Parallel → Sequential (Search → Summarize → PDF)  
**Best for:** Research reports, information gathering

### 3. multi-source-aggregator
**Use case:** Collect from multiple sources and combine  
**Pattern:** Parallel → Combine → Output  
**Best for:** Data aggregation, multi-source reports

## Using Templates

### Direct Use

```python
from core.registry.skills_registry import get_skills_registry

registry = get_skills_registry()
skill = registry.get_skill('simple-pdf-generator')
result = await skill.tools['generate_pdf_tool']({
    'content': '# My Document',
    'output_file': 'output.pdf'
})
```

### Customization

1. **Copy template:**
   ```bash
   cp skills/composite-templates/simple-pdf-generator/tools.py skills/my-custom-pdf/tools.py
   ```

2. **Modify workflow:**
   - Change skill composition
   - Add/remove steps
   - Customize parameters

3. **Register skill:**
   - Create `SKILL.md` with description
   - Skill auto-discovers on next load

## Creating New Templates

### Step 1: Identify Pattern

Common patterns:
- **Sequential:** A → B → C
- **Parallel:** A || B || C
- **Conditional:** if X then A else B
- **Loop:** repeat A N times
- **Complex:** Mix of above

### Step 2: Build Workflow

Use `skill-composer` to build workflow:

```python
workflow = {
    'workflow': [
        {
            'type': 'parallel',
            'skills': [...]
        },
        {
            'type': 'single',
            'skill': '...',
            'params': {...}
        }
    ]
}
```

### Step 3: Wrap in Tool Function

```python
async def my_template_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    # Get composer
    composer = registry.get_skill('skill-composer')
    compose_tool = composer.tools.get('compose_skills_tool')
    
    # Build workflow from params
    workflow = build_workflow(params)
    
    # Execute
    result = await compose_tool(workflow)
    
    # Return formatted result
    return format_result(result)
```

### Step 4: Document

Create:
- `SKILL.md` - Tool documentation
- `example.py` - Usage examples
- `customization.md` - How to customize

## Template Best Practices

1. **Parameterize everything** - Make workflow configurable
2. **Handle errors gracefully** - Cleanup temp files, return clear errors
3. **Document well** - Include examples and customization guide
4. **Keep it simple** - One template = one clear use case
5. **Make it reusable** - Don't hardcode specific values

## Popular Use Cases to Template

- ✅ PDF generation (simple-pdf-generator)
- ✅ Research to PDF (research-to-pdf)
- ✅ Multi-source aggregation (multi-source-aggregator)
- ⏳ Web scrape → Process → Store
- ⏳ Data collection → Analysis → Report
- ⏳ Content generation → Format → Publish
- ⏳ API → Transform → Database
- ⏳ File processing pipeline

## Next Steps

1. Use existing templates for common tasks
2. Customize templates for specific needs
3. Create new templates for recurring patterns
4. Share templates with community
