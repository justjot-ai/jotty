---
name: analyzing-meeting-insights
description: "This skill transforms your meeting transcripts into actionable insights about your communication patterns, helping you become a more effective communicator and leader. Identifies when you avoid conflict, use filler words, dominate conversations, or miss opportunities to listen. Use when the user wants to meeting notes, meeting summary, meeting insights."
---

# Meeting Insights Analyzer Skill

Analyzes meeting transcripts and recordings to uncover behavioral patterns, communication insights, and actionable feedback.

## Description

This skill transforms your meeting transcripts into actionable insights about your communication patterns, helping you become a more effective communicator and leader. Identifies when you avoid conflict, use filler words, dominate conversations, or miss opportunities to listen.


## Type
derived

## Base Skills
- claude-cli-llm


## Capabilities
- analyze

## Tools

### `analyze_meeting_insights_tool`

Analyze meeting transcripts for communication patterns and insights.

**Parameters:**
- `transcript_files` (list, required): List of transcript file paths
- `user_name` (str, optional): Your name/identifier in transcripts
- `analysis_types` (list, optional): Types of analysis to perform
  - `conflict_avoidance`: Detect indirect communication and conflict avoidance
  - `speaking_ratios`: Calculate speaking time and turn-taking
  - `filler_words`: Count filler words and hedging language
  - `active_listening`: Identify listening indicators
  - `leadership`: Analyze facilitation and leadership style
- `output_file` (str, optional): Path to save analysis report

**Returns:**
- `success` (bool): Whether analysis succeeded
- `insights` (dict): Dictionary of insights by type
- `statistics` (dict): Speaking statistics and metrics
- `recommendations` (list): Actionable improvement recommendations
- `output_file` (str, optional): Path to saved report
- `error` (str, optional): Error message if failed

## Usage Examples

### Basic Usage

```python
result = await analyze_meeting_insights_tool({
    'transcript_files': ['meeting1.txt', 'meeting2.txt'],
    'user_name': 'John',
    'analysis_types': ['conflict_avoidance', 'speaking_ratios']
})
```

### Comprehensive Analysis

```python
result = await analyze_meeting_insights_tool({
    'transcript_files': ['meetings/*.txt'],
    'user_name': 'John',
    'analysis_types': ['all'],
    'output_file': 'meeting_insights_report.md'
})
```

## Dependencies

- `file-operations`: For reading transcript files
- `claude-cli-llm`: For AI-powered pattern analysis

## Triggers
- "meeting insights analyzer"
- "meeting notes"
- "meeting summary"
- "meeting insights"

## Category
workflow-automation
