---
name: text-toolkit
description: "Unified text processing toolkit. Case conversion, slug generation, text statistics, regex testing/replacing, word frequency analysis, and color conversion."
---

# Text Toolkit

Unified text processing and manipulation skill. Consolidates string-case-converter,
slug-generator, text-statistics, word-frequency-analyzer, regex-tester, regex-builder,
color-converter, date-calculator, and lorem-ipsum-generator.

## Type
base

## Capabilities
- text-processing
- string-manipulation
- regex
- analysis

## Triggers
- "convert case"
- "camel case"
- "snake case"
- "slugify"
- "slug"
- "text statistics"
- "word count"
- "reading time"
- "regex"
- "regular expression"
- "word frequency"
- "color convert"

## Category
text-processing

## Tools

### convert_case_tool
Convert string between naming conventions (camelCase, snake_case, PascalCase, kebab-case, etc.).

**Parameters:**
- `text` (str, required): Text to convert
- `to_case` (str, required): Target case: camelCase, PascalCase, snake_case, kebab-case, UPPER_CASE, Title Case, dot.case

### slugify_tool
Generate a URL-friendly slug from text with unicode transliteration.

**Parameters:**
- `text` (str, required): Text to slugify
- `separator` (str, optional): Word separator (default: -)
- `max_length` (int, optional): Max slug length (default: 200)
- `lowercase` (bool, optional): Force lowercase (default: true)

### analyze_text_tool
Text statistics: word count, char count, sentence count, reading time, Flesch-Kincaid grade.

**Parameters:**
- `text` (str, required): Text to analyze

### word_frequency_tool
Analyze word frequency in text with top-N results.

**Parameters:**
- `text` (str, required): Text to analyze
- `top_n` (int, optional): Number of top words (default: 20)
- `min_length` (int, optional): Minimum word length (default: 1)

### regex_match_tool
Test a regex pattern against text and return all matches with positions and groups.

**Parameters:**
- `pattern` (str, required): Regular expression pattern
- `text` (str, required): Text to search
- `flags` (str, optional): Regex flags: i (ignore case), m (multiline), s (dotall), x (verbose)

### regex_replace_tool
Replace regex matches in text.

**Parameters:**
- `pattern` (str, required): Regular expression pattern
- `text` (str, required): Text to search
- `replacement` (str, optional): Replacement string (default: empty)
- `count` (int, optional): Max replacements, 0=all (default: 0)
- `flags` (str, optional): Regex flags

### regex_split_tool
Split text by regex pattern.

**Parameters:**
- `pattern` (str, required): Regular expression pattern
- `text` (str, required): Text to split
- `flags` (str, optional): Regex flags

### color_convert_tool
Convert colors between hex, RGB, and HSL formats.

**Parameters:**
- `color` (str, required): Color value (hex like #FF5733, rgb like 255,87,51, or named color)
- `to_format` (str, optional): Target format: hex, rgb, hsl, all (default: all)

## Dependencies
None
