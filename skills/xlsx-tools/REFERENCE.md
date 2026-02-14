# Excel Tools Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`read_excel_tool`](#read_excel_tool) | Read Excel file to dictionary or dataframe representation. |
| [`write_excel_tool`](#write_excel_tool) | Write data to Excel file. |
| [`get_sheets_tool`](#get_sheets_tool) | List all sheets in an Excel workbook. |
| [`add_sheet_tool`](#add_sheet_tool) | Add a new sheet to an existing Excel workbook. |
| [`update_cells_tool`](#update_cells_tool) | Update specific cells in an Excel workbook. |
| [`add_formula_tool`](#add_formula_tool) | Add a formula to a cell in an Excel workbook. |
| [`create_chart_tool`](#create_chart_tool) | Create a chart in an Excel workbook. |

---

## `read_excel_tool`

Read Excel file to dictionary or dataframe representation.

**Parameters:**

- **file_path** (`str, required`): Path to the Excel file
- **sheet_name** (`str, optional`): Sheet name to read (default: first sheet)
- **header_row** (`int, optional`): Row number for column headers (0-indexed, default: 0)
- **as_dataframe** (`bool, optional`): Return as dataframe dict (default: False, returns list of dicts)
- **skip_rows** (`int, optional`): Number of rows to skip from top
- **max_rows** (`int, optional`): Maximum number of rows to read

**Returns:** Dictionary with: - success (bool): Whether operation succeeded - data (list/dict): Data from Excel file - columns (list): Column names - row_count (int): Number of rows - error (str, optional): Error message if failed

---

## `write_excel_tool`

Write data to Excel file.

**Parameters:**

- **data** (`list/dict, required`): Data to write (list of dicts or dataframe-style dict)
- **output_path** (`str, required`): Output file path
- **sheet_name** (`str, optional`): Sheet name (default: 'Sheet1')
- **index** (`bool, optional`): Include index column (default: False)
- **overwrite** (`bool, optional`): Overwrite existing file (default: True)

**Returns:** Dictionary with: - success (bool): Whether operation succeeded - file_path (str): Path to written file - row_count (int): Number of rows written - error (str, optional): Error message if failed

---

## `get_sheets_tool`

List all sheets in an Excel workbook.

**Parameters:**

- **file_path** (`str, required`): Path to the Excel file

**Returns:** Dictionary with: - success (bool): Whether operation succeeded - sheets (list): List of sheet names - count (int): Number of sheets - error (str, optional): Error message if failed

---

## `add_sheet_tool`

Add a new sheet to an existing Excel workbook.

**Parameters:**

- **file_path** (`str, required`): Path to the Excel file
- **sheet_name** (`str, required`): Name for the new sheet
- **data** (`list, optional`): Data to write (list of dicts)
- **position** (`int, optional`): Position for new sheet (0-indexed, default: end)

**Returns:** Dictionary with: - success (bool): Whether operation succeeded - sheet_name (str): Name of created sheet - file_path (str): Path to modified file - error (str, optional): Error message if failed

---

## `update_cells_tool`

Update specific cells in an Excel workbook.

**Parameters:**

- **file_path** (`str, required`): Path to the Excel file
- **sheet_name** (`str, optional`): Sheet name (default: first sheet)
- **updates** (`dict, required`): Dictionary of cell:value pairs (e.g., {'A1': 'Hello', 'B2': 42})

**Returns:** Dictionary with: - success (bool): Whether operation succeeded - updated_cells (int): Number of cells updated - file_path (str): Path to modified file - error (str, optional): Error message if failed

---

## `add_formula_tool`

Add a formula to a cell in an Excel workbook.

**Parameters:**

- **file_path** (`str, required`): Path to the Excel file
- **sheet_name** (`str, optional`): Sheet name (default: first sheet)
- **cell** (`str, required`): Cell reference (e.g., 'C1')
- **formula** (`str, required`): Excel formula (e.g., '=SUM(A1:B1)')

**Returns:** Dictionary with: - success (bool): Whether operation succeeded - cell (str): Cell where formula was added - formula (str): Formula that was added - file_path (str): Path to modified file - error (str, optional): Error message if failed

---

## `create_chart_tool`

Create a chart in an Excel workbook.

**Parameters:**

- **file_path** (`str, required`): Path to the Excel file
- **sheet_name** (`str, optional`): Sheet name (default: first sheet)
- **chart_type** (`str, required`): Type of chart ('bar', 'line', 'pie', 'area', 'scatter', 'column')
- **data_range** (`str, required`): Data range for chart (e.g., 'A1:B10')
- **position** (`str, optional`): Cell position for chart (default: 'E1')
- **title** (`str, optional`): Chart title
- **categories_range** (`str, optional`): Range for category labels (e.g., 'A2:A10')
- **width** (`int, optional`): Chart width in EMUs (default: 15)
- **height** (`int, optional`): Chart height in EMUs (default: 10)

**Returns:** Dictionary with: - success (bool): Whether operation succeeded - chart_type (str): Type of chart created - position (str): Cell position of chart - file_path (str): Path to modified file - error (str, optional): Error message if failed
