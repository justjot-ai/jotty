# Claude CLI LLM Skill

Text generation, summarization, and **Prompt Ensembling** using DSPy LM.

## Description

This skill provides LLM capabilities including cutting-edge prompt ensembling techniques:
- **Multi-Perspective Synthesis**: Virtual expert panel with different lenses
- **Self-Consistency**: Multiple samples with synthesis
- **Generative Self-Aggregation (GSA)**: Diverse drafts + context-enriched synthesis
- **Debate**: Multi-round argumentation with final judgment

Based on latest research (2025-2026) on prompt engineering and LLM ensembling.

## Usage

### Basic LLM Call
```python
from core.registry.skills_registry import get_skills_registry

registry = get_skills_registry()
registry.init()

skill = registry.get_skill('claude-cli-llm')
tool = skill.tools['claude_cli_llm_tool']

result = tool({
    'prompt': 'Explain quantum computing'
})
```

### Prompt Ensembling (Multi-Perspective)
```python
result = tool({
    'prompt': 'Should we invest in AI stocks?',
    'ensemble': True,
    'ensemble_strategy': 'multi_perspective',
    'synthesis_style': 'structured'  # consensus, tensions, blind spots
})
```

### Self-Consistency Ensembling
```python
tool = skill.tools['ensemble_prompt_tool']

result = tool({
    'prompt': 'What is 23 * 47?',
    'strategy': 'self_consistency',
    'num_samples': 5
})
```

### Debate Strategy
```python
result = tool({
    'prompt': 'Is remote work better than office work?',
    'strategy': 'debate',
    'debate_rounds': 2
})
```

## Ensemble Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `self_consistency` | Same prompt, N samples, synthesis | Math, factual questions |
| `multi_perspective` | Different expert personas | Complex decisions, analysis |
| `gsa` | Diverse drafts + synthesis | Creative writing, problem-solving |
| `debate` | Advocate/Critic debate rounds | Controversial topics, trade-offs |

## Parameters

### claude_cli_llm_tool
- `prompt` (str, required): The prompt/question
- `ensemble` (bool): Enable ensembling (default: False)
- `ensemble_strategy` (str): Strategy for ensembling
- `max_tokens` (int): Maximum tokens

### ensemble_prompt_tool
- `prompt` (str, required): Question to analyze
- `strategy` (str): 'self_consistency', 'multi_perspective', 'gsa', 'debate'
- `num_samples` (int): Samples for self_consistency (default: 5)
- `perspectives` (list): Custom perspectives for multi_perspective
- `debate_rounds` (int): Rounds for debate (default: 2)
- `synthesis_style` (str): 'detailed', 'concise', 'structured'

### summarize_text_tool
- `content` (str, required): Text to summarize
- `prompt` (str): Custom summarization prompt

### generate_text_tool
- `prompt` (str, required): Generation prompt
- `max_tokens` (int): Maximum tokens

## Returns

Ensemble responses include:
- `response`: Synthesized final answer
- `individual_responses`: Each perspective's raw response
- `perspectives_used`: List of perspectives/samples
- `confidence`: Agreement score (0-1)
- `strategy`: Strategy used

## Research References

- Multi-Perspective Simulation identifies overlooked considerations ~70% of time
- Self-Consistency accounts for most gains in multi-agent approaches
- GSA enables synthesis better than any individual draft
- Debate reduces hallucinations and improves factual validity

## Version

2.0.0

## Author

Jotty Framework
