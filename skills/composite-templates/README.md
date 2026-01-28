# Composite Skill Templates

Reusable templates for common workflows using skill composition.

## Available Templates

1. **simple-pdf-generator** - Text/Markdown → PDF
2. **research-to-pdf** - Web Search → Summarize → PDF
3. **multi-source-aggregator** - Parallel Sources → Combine → Output
4. **data-pipeline** - Extract → Transform → Load
5. **content-pipeline** - Generate → Format → Publish

## How to Use

Each template is a composite skill that uses `skill-composer` internally.

### Option 1: Use Template Directly

```python
from core.registry.skills_registry import get_skills_registry

registry = get_skills_registry()
skill = registry.get_skill('simple-pdf-generator')
result = await skill.tools['generate_pdf_tool']({
    'content': '# My Document',
    'output_file': 'output.pdf'
})
```

### Option 2: Customize Template

Copy template and modify for your specific needs:

```bash
cp skills/composite-templates/simple-pdf-generator/tools.py skills/my-custom-pdf/tools.py
# Edit to customize
```

## Template Structure

Each template includes:
- `tools.py` - Implementation using skill-composer
- `SKILL.md` - Documentation
- `example.py` - Usage examples
- `customization.md` - How to customize

## Creating New Templates

1. Identify common workflow pattern
2. Create template using skill-composer
3. Document use cases
4. Add examples
5. Submit to templates directory
