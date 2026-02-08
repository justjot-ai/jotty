# Meeting Intelligence Pipeline Composite Skill

Complete meeting workflow: analyze insights → prepare materials → generate communications.

## Description

This composite skill combines:
1. **Meeting Analysis** (Source): meeting-insights-analyzer
2. **Meeting Preparation** (Processor): notion-meeting-intelligence
3. **Internal Communications** (Sink): internal-comms


## Type
composite

## Base Skills
- voice
- claude-cli-llm
- notion

## Execution
sequential


## Capabilities
- analyze
- document

## Usage

```python
from skills.meeting_intelligence_pipeline.tools import meeting_intelligence_pipeline_tool

result = await meeting_intelligence_pipeline_tool({
    'transcript_files': ['meeting1.txt', 'meeting2.txt'],
    'user_name': 'John Doe',
    'meeting_topic': 'Q4 Planning',
    'meeting_type': 'planning',
    'create_pre_read': True,
    'create_agenda': True,
    'send_comm': True
})
```

## Parameters

- `transcript_files` (list, required): List of transcript file paths
- `user_name` (str, required): User's name for analysis
- `meeting_topic` (str, required): Meeting topic
- `meeting_type` (str, optional): 'status_update', 'planning', 'retrospective', 'decision_making' (default: 'status_update')
- `analysis_types` (list, optional): Analysis types (default: ['speaking_ratios', 'action_items', 'decisions'])
- `create_pre_read` (bool, optional): Create pre-read (default: True)
- `create_agenda` (bool, optional): Create agenda (default: True)
- `comm_type` (str, optional): Communication type (default: '3p_update')
- `send_comm` (bool, optional): Generate internal comm (default: True)

## Architecture

Source → Processor → Sink pattern:
- **Source**: Meeting insights analyzer
- **Processor**: Notion meeting intelligence
- **Sink**: Internal communications

No code duplication - reuses existing skills.
