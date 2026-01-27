# Deep Stock Research Skill

## Description

Performs **deep research** using multi-stage context intelligence methodology. Unlike shallow research (single-pass), this skill implements:

1. **Initial Broad Research** - Parallel searches across all aspects
2. **Gap Analysis** - AI identifies what's missing
3. **Targeted Follow-up** - Fills knowledge gaps
4. **Cross-Referencing** - Verifies information
5. **Progressive Synthesis** - Builds context layer by layer
6. **Quality Validation** - Ensures comprehensive coverage
7. **Iterative Refinement** - Refines weak sections

## How Deep Research Works

### Context Intelligence

Deep research builds understanding progressively:

```
Stage 1: Search → Get Results
Stage 2: Analyze → Identify Gaps
Stage 3: Research More → Fill Gaps
Stage 4: Synthesize → Build Context
Stage 5: Validate → Check Quality
Stage 6: Refine → Improve Weak Areas
```

### Key Differences from Shallow Research

| Aspect | Shallow Research | Deep Research |
|--------|------------------|---------------|
| **Passes** | Single | Multiple (3-5) |
| **Gap Analysis** | None | AI-powered |
| **Follow-up** | None | Targeted queries |
| **Context Building** | All at once | Progressive |
| **Quality Check** | None | Explicit validation |
| **Refinement** | None | Iterative |

## Tools

### deep_stock_research_tool

Performs deep stock research with context intelligence.

**Parameters:**
- `ticker` (str, required): Stock ticker symbol
- `company_name` (str, optional): Full company name
- `country` (str, optional): Country/Exchange
- `exchange` (str, optional): Exchange name
- `target_pages` (int, optional): Target report length (default: 10)
- `max_iterations` (int, optional): Max refinement iterations (default: 2)
- `send_telegram` (bool, optional): Send to Telegram (default: True)

**Returns:**
- `success` (bool): Whether operation succeeded
- `md_path` (str): Path to markdown file
- `pdf_path` (str): Path to PDF file
- `word_count` (int): Word count of generated report
- `research_stages` (dict): Research statistics
- `telegram_sent` (bool): Whether Telegram send succeeded

## Usage

```python
from core.registry.skills_registry import get_skills_registry

registry = get_skills_registry()
registry.init()

skill = registry.get_skill('stock-research-deep')
tool = skill.tools['deep_stock_research_tool']

result = await tool({
    'ticker': 'COLPAL',
    'company_name': 'Colgate Palmolive India',
    'country': 'India',
    'exchange': 'NSE',
    'target_pages': 10,
    'send_telegram': True
})
```

## Research Methodology

### Stage 1: Initial Research
- Parallel searches across 12 aspects
- 15 results per aspect
- ~180 initial results

### Stage 2: Gap Analysis
- AI analyzes research data
- Identifies missing information
- Generates targeted follow-up queries

### Stage 3: Targeted Follow-up
- Executes gap-filling queries
- Merges results with initial research
- Builds comprehensive dataset

### Stage 4: Progressive Synthesis
- Generates report section by section
- Uses accumulated context
- Builds comprehensive understanding

### Stage 5: Quality Validation
- Checks word count
- Validates coverage
- Identifies weak sections

### Stage 6: Refinement (if needed)
- Deep dive into weak areas
- Additional research
- Report refinement

## Benefits

1. **Comprehensive Coverage** - Identifies and fills gaps
2. **Higher Quality** - Multi-stage validation
3. **Context Intelligence** - Progressive understanding
4. **Adaptive** - Adjusts based on findings
5. **Thorough** - Ensures all aspects covered

## Requirements

- `web-search` skill
- `claude-cli-llm` skill
- `file-operations` skill
- `document-converter` skill
- `telegram-sender` skill
- Claude CLI installed and authenticated
