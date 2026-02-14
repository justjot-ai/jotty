# Financial Visualization Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`extract_financial_data_tool`](#extract_financial_data_tool) | Extract structured financial data from research content using AI and pattern ... |
| [`generate_intelligent_charts_tool`](#generate_intelligent_charts_tool) | Intelligently generate financial charts with AI-powered selection, analysis, ... |
| [`generate_financial_charts_tool`](#generate_financial_charts_tool) | Generate financial charts from extracted data. |
| [`generate_financial_tables_tool`](#generate_financial_tables_tool) | Generate formatted financial tables from extracted data. |

### Helper Functions

| Function | Description |
|----------|-------------|
| [`safe_num`](#safe_num) | Safely convert value to number, handling None and invalid types. |
| [`safe_get_num`](#safe_get_num) | Safely get numeric value from dict, handling None values. |

---

## `extract_financial_data_tool`

Extract structured financial data from research content using AI and pattern matching.

**Parameters:**

- **research_content** (`str`): Research text/content to extract from
- **data_types** (`list, optional`): Types of data to extract
- **use_llm** (`bool, optional`): Use LLM for extraction (default: True)

**Returns:** Dictionary with extracted_data, confidence_scores, success status

---

## `generate_intelligent_charts_tool`

Intelligently generate financial charts with AI-powered selection, analysis, and insights.  This is the "best of AI" version - it: 1. Analyzes data completeness 2. Selects optimal chart types 3. Detects anomalies 4. Generates forecasts 5. Creates contextual narratives 6. Provides section placements

**Parameters:**

- **ticker** (`str`): Stock ticker
- **company_name** (`str`): Company name
- **research_data** (`dict`): Research results
- **chart_types** (`list, optional`): Specific chart types (auto-selected if not provided)
- **enable_intelligence** (`bool, optional`): Enable intelligent features (default: True)
- **output_dir** (`str, optional`): Output directory
- **format** (`str`): Chart format ('png', 'svg', 'pdf')

**Returns:** Dictionary with charts, insights, narratives, anomalies, forecasts, section placements

---

## `generate_financial_charts_tool`

Generate financial charts from extracted data.

**Parameters:**

- **ticker** (`str`): Stock ticker
- **company_name** (`str`): Company name
- **research_data** (`dict`): Research results
- **chart_types** (`list`): Types of charts to generate
- **output_dir** (`str, optional`): Output directory
- **format** (`str`): Chart format ('png', 'svg', 'pdf')

**Returns:** Dictionary with charts list, descriptions, success status

---

## `generate_financial_tables_tool`

Generate formatted financial tables from extracted data.

**Parameters:**

- **ticker** (`str`): Stock ticker
- **company_name** (`str`): Company name
- **research_data** (`dict`): Research results
- **table_types** (`list`): Types of tables to generate
- **format** (`str`): Table format ('markdown', 'html', 'latex')

**Returns:** Dictionary with tables dict, descriptions, success status

---

## `safe_num`

Safely convert value to number, handling None and invalid types.

**Parameters:**

- **value** (`Any`)
- **default** (`float`)

**Returns:** `float`

---

## `safe_get_num`

Safely get numeric value from dict, handling None values.

**Parameters:**

- **d** (`Dict`)
- **key** (`str`)
- **default** (`float`)

**Returns:** `float`
