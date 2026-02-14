---
name: media-production-pipeline
description: "This composite skill combines: 1. **Image Enhancement** (Source): image-enhancer 2. **Design Creation** (Processor): canvas-design 3. **GIF Creation** (Sink): slack-gif-creator."
---

# Media Production Pipeline Composite Skill

Complete media production workflow: image enhancement → design creation → GIF generation.

## Description

This composite skill combines:
1. **Image Enhancement** (Source): image-enhancer
2. **Design Creation** (Processor): canvas-design
3. **GIF Creation** (Sink): slack-gif-creator


## Type
composite

## Base Skills
- image-generator
- voice
- video-downloader

## Execution
sequential


## Capabilities
- media

## Usage

```python
from skills.media_production_pipeline.tools import media_production_pipeline_tool

# Full workflow
result = await media_production_pipeline_tool({
    'workflow_type': 'full',
    'image_path': 'screenshot.png',
    'design_brief': 'A modern UI design',
    'gif_description': 'A loading animation'
})

# Just design
result = await media_production_pipeline_tool({
    'workflow_type': 'design',
    'design_brief': 'A minimalist logo',
    'design_dimensions': (1200, 800)
})
```

## Parameters

- `workflow_type` (str, required): 'enhance', 'design', 'gif', or 'full'
- `image_path` (str, optional): Path to image for enhancement
- `target_resolution` (str, optional): Target resolution (default: '2x')
- `design_brief` (str, optional): Design brief
- `design_dimensions` (tuple, optional): Dimensions (default: (800, 600))
- `design_output_format` (str, optional): 'png' or 'pdf' (default: 'png')
- `gif_description` (str, optional): GIF description
- `gif_type` (str, optional): 'emoji' or 'animation' (default: 'emoji')
- `animation_type` (str, optional): Animation type (default: 'bounce')
- `enhance_image` (bool, optional): Enhance image
- `create_design` (bool, optional): Create design
- `create_gif` (bool, optional): Create GIF

## Architecture

Source → Processor → Sink pattern:
- **Source**: Image enhancer
- **Processor**: Canvas design
- **Sink**: Slack GIF creator

No code duplication - reuses existing skills.

## Triggers
- "media production pipeline"

## Category
content-creation
