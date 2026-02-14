---
name: multi-source-aggregator
description: "Template for aggregating data from multiple sources."
---

# Multi-Source Aggregator

Template for aggregating data from multiple sources.


## Type
derived

## Base Skills
- web-search


## Capabilities
- research
- data-fetch

## Use Cases

- Collect data from multiple APIs
- Aggregate search results
- Combine multiple file sources
- Multi-source reporting

## Tools

### aggregate_sources_tool

Aggregate data from multiple sources.

**Parameters:**
- `sources` (list, required): List of source configurations
- `combine_strategy` (str, optional): 'merge', 'append', 'aggregate' (default: 'merge')
- `output_format` (str, optional): 'json', 'csv', 'pdf', 'markdown' (default: 'json')
- `output_file` (str, optional): Output file path

**Returns:**
- `success` (bool): Whether aggregation succeeded
- `output_file` (str): Path to output file
- `file_size` (int): Size of output file
- `sources_count` (int): Number of sources aggregated

## Example

```python
from core.registry.skills_registry import get_skills_registry

registry = get_skills_registry()
skill = registry.get_skill('multi-source-aggregator')
result = await skill.tools['aggregate_sources_tool']({
    'sources': [
        {'type': 'web-search', 'params': {'query': 'Python'}},
        {'type': 'web-search', 'params': {'query': 'JavaScript'}}
    ],
    'output_format': 'pdf'
})
```

## Triggers
- "multi source aggregator"

## Category
workflow-automation
