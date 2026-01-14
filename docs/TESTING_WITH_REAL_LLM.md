# Testing Expert Agents with Real LLM

## Overview

To test that expert agents actually learn and work with real LLMs (Claude/Cursor), you need to configure DSPy with an LLM API key.

## Setup

### 1. Get API Key

**For Claude:**
- Get API key from: https://console.anthropic.com/
- Set environment variable: `export ANTHROPIC_API_KEY='your-key-here'`

**For OpenAI:**
- Get API key from: https://platform.openai.com/
- Set environment variable: `export OPENAI_API_KEY='your-key-here'`

### 2. Run Test

```bash
# Set API key
export ANTHROPIC_API_KEY='your-key-here'

# Run test
python tests/test_mermaid_expert_real_llm.py
```

## What the Test Does

1. **Configures LLM**: Sets up DSPy with Claude/OpenAI
2. **Trains Expert**: Trains MermaidExpertAgent on examples
3. **Tests Learning**: Tests with complex descriptions it hasn't seen
4. **Verifies Learning**: Checks if it generates appropriate diagrams

## Expected Results

### Training Phase

```
Training Results:
  Overall Success: True
  Passed Cases: 2/2
  
  Case 1: Generate simple flowchart
    Success: True
    Final Score: 1.00
    Iterations: 2
  
  Case 2: Generate decision flowchart
    Success: True
    Final Score: 1.00
    Iterations: 2
```

### Testing Phase

```
Test 1: Complex Multi-Branch Decision Tree
  Generated Diagram:
  ```mermaid
  graph TD
      A[User Authentication]
      B{Has Admin?}
      C{Data Valid?}
      D[Process Request]
      E[Show Error]
      A --> B
      B -->|Yes| C
      B -->|No| E
      C -->|Yes| D
      C -->|No| E
  ```
  
  Validation:
    ✅ Has Graph: True
    ✅ Has Nodes: True
    ✅ Has Arrows: True
    ✅ Has Decision Nodes: True
    ✅ Has Labels: True
  Score: 1.00 / 1.0
```

## Troubleshooting

### "Missing Anthropic API Key"

**Solution**: Set the environment variable:
```bash
export ANTHROPIC_API_KEY='your-key-here'
```

### "No LM is loaded"

**Solution**: The test will try to configure automatically, but you need to set API keys first.

### Training Fails

**Possible causes**:
1. API key invalid or expired
2. Rate limits exceeded
3. Network issues

**Solution**: Check API key and try again.

## Mock Testing

For testing syntax without real LLM calls, use mock mode:

```python
# The test will detect no LLM and show structure
python tests/test_mermaid_expert_learning.py
```

This shows:
- Test structure
- Expected behavior
- What would happen with real LLM

## Next Steps

Once you have real LLM working:

1. **Train on more examples**: Add more training cases
2. **Test complex scenarios**: Try edge cases
3. **Verify learning**: Check if improvements are stored
4. **Use in production**: Deploy trained expert agents
