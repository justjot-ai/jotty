# Canvas Design Skill

Creates visual art and designs in PNG and PDF formats using design philosophy principles.

## Description

This skill creates visual art and design pieces by first establishing a design philosophy, then expressing it visually through form, space, color, and composition. Outputs original visual designs as PNG or PDF files.


## Type
derived

## Base Skills
- image-generator


## Capabilities
- media
- visualize

## Tools

### `create_design_artwork_tool`

Create visual artwork based on design philosophy.

**Parameters:**
- `design_brief` (str, required): Brief description or concept for the artwork
- `output_format` (str, optional): Format - 'png', 'pdf' (default: 'png')
- `dimensions` (tuple, optional): Dimensions (width, height) in pixels (default: (1920, 1080))
- `design_philosophy` (str, optional): Pre-defined design philosophy (auto-generated if not provided)
- `style` (str, optional): Style hint - 'minimalist', 'bold', 'organic', 'geometric' (default: 'minimalist')

**Returns:**
- `success` (bool): Whether creation succeeded
- `artwork_path` (str): Path to created artwork
- `design_philosophy` (str): Design philosophy used
- `format` (str): Output format
- `error` (str, optional): Error message if failed

## Usage Examples

### Create PNG Artwork

```python
result = await create_design_artwork_tool({
    'design_brief': 'A poster about sustainable energy',
    'output_format': 'png',
    'dimensions': (1920, 1080)
})
```

### Create PDF Design

```python
result = await create_design_artwork_tool({
    'design_brief': 'A minimalist brand identity',
    'output_format': 'pdf',
    'style': 'minimalist'
})
```

## Dependencies

- `PIL` (Pillow): For image creation
- `claude-cli-llm`: For design philosophy generation
- `reportlab`: For PDF creation (optional)
