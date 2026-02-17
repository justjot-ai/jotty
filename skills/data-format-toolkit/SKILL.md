---
name: data-format-toolkit
description: "Unified data format conversion and validation toolkit. Convert CSV/JSON/YAML/TOML, validate schemas, diff JSON objects, generate markdown/HTML tables, and convert markdown to HTML."
---

# Data Format Toolkit

Unified data format conversion, validation, and table generation skill. Consolidates
csv-to-json, json-transformer, toml-parser, yaml-validator, csv-analyzer,
json-diff, markdown-to-html, markdown-table-generator, html-table-generator.

## Type
base

## Capabilities
- data-conversion
- format-validation
- table-generation
- diff

## Triggers
- "csv to json"
- "json to csv"
- "yaml validate"
- "toml parse"
- "json diff"
- "compare json"
- "markdown table"
- "html table"
- "markdown to html"
- "convert format"

## Category
data-processing

## Tools

### csv_to_json_tool
Convert CSV text to JSON array of objects.

**Parameters:**
- `csv_text` (str, required): CSV content
- `delimiter` (str, optional): Column delimiter (default: ,)
- `has_header` (bool, optional): First row is header (default: true)

### json_to_csv_tool
Convert JSON array of objects to CSV text.

**Parameters:**
- `data` (array, required): JSON array of objects

### yaml_validate_tool
Parse and validate YAML, returning structured data or error details.

**Parameters:**
- `yaml_text` (str, required): YAML content to validate

### toml_parse_tool
Parse TOML text and return structured data.

**Parameters:**
- `toml_text` (str, required): TOML content to parse

### json_diff_tool
Compare two JSON objects and return differences.

**Parameters:**
- `a` (object, required): First JSON object
- `b` (object, required): Second JSON object

### markdown_table_tool
Generate a markdown table from data.

**Parameters:**
- `headers` (array, required): Column headers
- `rows` (array, required): Array of row arrays
- `alignment` (array, optional): Column alignments: left, center, right

### html_table_tool
Generate an HTML table from data.

**Parameters:**
- `headers` (array, required): Column headers
- `rows` (array, required): Array of row arrays
- `css_class` (str, optional): CSS class for table element

### markdown_to_html_tool
Convert markdown text to HTML.

**Parameters:**
- `markdown` (str, required): Markdown text
- `extensions` (array, optional): Markdown extensions: tables, fenced_code, toc

## Dependencies
- pyyaml
