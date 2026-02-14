---
name: formatting-transcripts
description: "Clean raw transcripts with speaker labels, timestamps, and paragraph breaks. Use when the user wants to clean transcript, format transcript, add speaker labels."
---

# Transcript Formatter Skill

Clean raw transcripts with speaker labels, timestamps, and paragraph breaks. Use when the user wants to clean transcript, format transcript, add speaker labels.

## Type
base

## Capabilities
- generate
- analyze

## Reference
For detailed tool documentation, see [REFERENCE.md](REFERENCE.md).

## Workflow

```
Task Progress:
- [ ] Step 1: Parse input parameters
- [ ] Step 2: Execute operation
- [ ] Step 3: Return results
```

## Triggers
- "transcript"
- "format transcript"
- "clean transcript"
- "speaker labels"
- "subtitles"

## Category
content-creation

## Tools

### format_transcript_tool
Format a raw transcript with speaker labels and timestamps.

**Parameters:**
- `text` (str, required): Raw transcript text
- `merge_speakers` (bool, optional): Merge consecutive lines from same speaker (default: true)
- `include_timestamps` (bool, optional): Keep timestamps in output (default: true)

**Returns:**
- `success` (bool)
- `formatted` (str): Cleaned transcript
- `speakers` (list): Unique speakers found
- `duration` (str): Estimated duration

## Dependencies
None
