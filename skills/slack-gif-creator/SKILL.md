# Slack GIF Creator Skill

Creates animated GIFs optimized for Slack with size constraints and animation primitives.

## Description

This skill creates animated GIFs specifically optimized for Slack's requirements. Supports both message GIFs and emoji GIFs with different size constraints and optimization strategies.


## Type
composite

## Base Skills
- gif-creator
- slack

## Execution
sequential


## Capabilities
- media
- communicate

## Tools

### `create_slack_gif_tool`

Create a Slack-optimized animated GIF.

**Parameters:**
- `description` (str, required): Description of the GIF animation
- `gif_type` (str, optional): Type - 'message' or 'emoji' (default: 'message')
- `output_path` (str, optional): Output file path
- `width` (int, optional): Width in pixels (auto-selected based on type)
- `height` (int, optional): Height in pixels (auto-selected based on type)
- `fps` (int, optional): Frames per second (auto-selected based on type)
- `duration` (float, optional): Duration in seconds (default: 3.0)
- `animation_type` (str, optional): Type - 'shake', 'bounce', 'pulse', 'spin', 'custom' (default: 'custom')

**Returns:**
- `success` (bool): Whether creation succeeded
- `gif_path` (str): Path to created GIF
- `file_size_kb` (float): File size in KB
- `meets_requirements` (bool): Whether meets Slack requirements
- `error` (str, optional): Error message if failed

## Slack Requirements

**Message GIFs:**
- Max size: ~2MB
- Optimal dimensions: 480x480
- Typical FPS: 15-20
- Duration: 2-5s

**Emoji GIFs:**
- Max size: 64KB (strict)
- Optimal dimensions: 128x128
- Typical FPS: 10-12
- Duration: 1-2s

## Usage Examples

### Message GIF

```python
result = await create_slack_gif_tool({
    'description': 'A bouncing ball',
    'gif_type': 'message',
    'animation_type': 'bounce'
})
```

### Emoji GIF

```python
result = await create_slack_gif_tool({
    'description': 'A shaking emoji',
    'gif_type': 'emoji',
    'animation_type': 'shake'
})
```

## Dependencies

- `PIL` (Pillow): For image processing
- `imageio`: For GIF creation

## Triggers
- "slack gif creator"
- "send to slack"
- "slack message"
- "post to slack"
- "create gif"
- "make gif"
- "animated"

## Category
communication
