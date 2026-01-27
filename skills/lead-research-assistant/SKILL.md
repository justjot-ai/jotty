# Lead Research Assistant Skill

Identifies high-quality leads for your product or service by analyzing your business, searching for target companies, and providing actionable contact strategies.

## Description

This skill helps you identify and qualify potential leads for your business by analyzing your product/service, understanding your ideal customer profile, and providing actionable outreach strategies. Perfect for sales, business development, and marketing professionals.

## Tools

### `research_leads_tool`

Research and identify potential leads for your product/service.

**Parameters:**
- `product_description` (str, required): Description of your product/service
- `industry` (str, optional): Target industry
- `location` (str, optional): Geographic location preference
- `company_size` (str, optional): Company size range (e.g., "50-500 employees")
- `pain_points` (list, optional): Pain points your product solves
- `technologies` (list, optional): Technologies they might use
- `max_leads` (int, optional): Maximum number of leads to find (default: 10)
- `output_format` (str, optional): Output format - 'markdown', 'csv', 'json' (default: 'markdown')

**Returns:**
- `success` (bool): Whether research succeeded
- `leads` (list): List of lead dictionaries with company info, fit score, contact strategy
- `summary` (dict): Summary statistics
- `output_file` (str, optional): Path to saved output file
- `error` (str, optional): Error message if failed

## Usage Examples

### Basic Usage

```python
result = await research_leads_tool({
    'product_description': 'AI-powered code review tool for development teams',
    'industry': 'Technology',
    'company_size': '50-500 employees',
    'max_leads': 10
})
```

### Advanced Usage

```python
result = await research_leads_tool({
    'product_description': 'Remote team productivity platform',
    'industry': 'Software',
    'location': 'Bay Area',
    'pain_points': ['remote collaboration', 'team communication'],
    'technologies': ['Slack', 'Jira'],
    'max_leads': 20,
    'output_format': 'csv'
})
```

## Dependencies

- `web-search`: For finding companies
- `claude-cli-llm`: For analyzing and qualifying leads
