# Data Validator Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`validate_schema_tool`](#validate_schema_tool) | Validate data against an expected schema. |
| [`validate_quality_tool`](#validate_quality_tool) | Check data quality metrics. |
| [`validate_drift_tool`](#validate_drift_tool) | Detect distribution drift between reference and current data. |
| [`validate_constraints_tool`](#validate_constraints_tool) | Validate business constraints on data. |
| [`validate_completeness_tool`](#validate_completeness_tool) | Check data completeness and coverage. |

---

## `validate_schema_tool`

Validate data against an expected schema.

**Parameters:**

- **data**: DataFrame to validate
- **schema**: Dict with column specifications e.g., {'col1': {'type': 'int', 'nullable': False}, ...}
- **strict**: If True, fail on extra columns (default False)

**Returns:** Dict with validation results

---

## `validate_quality_tool`

Check data quality metrics.

**Parameters:**

- **data**: DataFrame to validate
- **thresholds**: Dict with quality thresholds e.g., {'missing_pct': 5, 'duplicate_pct': 1}

**Returns:** Dict with quality scores and issues

---

## `validate_drift_tool`

Detect distribution drift between reference and current data.

**Parameters:**

- **reference**: Reference DataFrame
- **current**: Current DataFrame to check
- **columns**: Optional list of columns to check
- **threshold**: P-value threshold for drift (default 0.05)

**Returns:** Dict with drift detection results

---

## `validate_constraints_tool`

Validate business constraints on data.

**Parameters:**

- **data**: DataFrame to validate
- **constraints**: List of constraint dicts e.g., [ {'type': 'positive', 'columns': ['price', 'quantity']}, {'type': 'range', 'column': 'age', 'min': 0, 'max': 120}, {'type': 'unique', 'columns': ['id']}, {'type': 'relationship', 'condition': 'end_date >= start_date'}, ]

**Returns:** Dict with constraint validation results

---

## `validate_completeness_tool`

Check data completeness and coverage.

**Parameters:**

- **data**: DataFrame to validate
- **required_columns**: List of required column names
- **min_rows**: Minimum required rows
- **coverage_threshold**: Min coverage percentage per column (default 95)

**Returns:** Dict with completeness metrics
