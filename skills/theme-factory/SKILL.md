---
name: theme-factory
description: "This skill provides a curated collection of professional themes with carefully selected color palettes and font pairings. Each theme includes cohesive colors and complementary fonts suitable for different contexts and audiences."
---

# Theme Factory Skill

Provides professional font and color themes for styling artifacts like slides, documents, and HTML pages.

## Description

This skill provides a curated collection of professional themes with carefully selected color palettes and font pairings. Each theme includes cohesive colors and complementary fonts suitable for different contexts and audiences.


## Type
derived

## Base Skills
- image-generator


## Capabilities
- media

## Tools

### `apply_theme_tool`

Apply a theme to an artifact.

**Parameters:**
- `theme_name` (str, required): Theme name - 'ocean_depths', 'sunset_boulevard', 'forest_canopy', 'modern_minimalist', 'golden_hour', 'arctic_frost', 'desert_rose', 'tech_innovation', 'botanical_garden', 'midnight_galaxy', or 'custom'
- `artifact_path` (str, required): Path to artifact file
- `artifact_type` (str, optional): Type - 'pptx', 'html', 'css', 'auto' (default: 'auto')
- `custom_colors` (dict, optional): Custom colors if theme_name is 'custom'
- `custom_fonts` (dict, optional): Custom fonts if theme_name is 'custom'

**Returns:**
- `success` (bool): Whether theme application succeeded
- `theme_applied` (str): Name of theme applied
- `colors` (dict): Color palette used
- `fonts` (dict): Font pairings used
- `output_path` (str): Path to styled artifact
- `error` (str, optional): Error message if failed

## Available Themes

1. **Ocean Depths** - Professional and calming maritime theme
2. **Sunset Boulevard** - Warm and vibrant sunset colors
3. **Forest Canopy** - Natural and grounded earth tones
4. **Modern Minimalist** - Clean and contemporary grayscale
5. **Golden Hour** - Rich and warm autumnal palette
6. **Arctic Frost** - Cool and crisp winter-inspired theme
7. **Desert Rose** - Soft and sophisticated dusty tones
8. **Tech Innovation** - Bold and modern tech aesthetic
9. **Botanical Garden** - Fresh and organic garden colors
10. **Midnight Galaxy** - Dramatic and cosmic deep tones

## Usage Examples

### Apply Theme to Presentation

```python
result = await apply_theme_tool({
    'theme_name': 'ocean_depths',
    'artifact_path': 'presentation.pptx',
    'artifact_type': 'pptx'
})
```

### Custom Theme

```python
result = await apply_theme_tool({
    'theme_name': 'custom',
    'artifact_path': 'document.html',
    'custom_colors': {
        'primary': '#1a1a2e',
        'secondary': '#16213e',
        'accent': '#0f3460'
    },
    'custom_fonts': {
        'heading': 'Roboto',
        'body': 'Open Sans'
    }
})
```

## Dependencies

- `python-pptx`: For PowerPoint styling
- `claude-cli-llm`: For custom theme generation

## Triggers
- "theme factory"

## Category
workflow-automation
