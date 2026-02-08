# Excel Tools Skill

## Description
Excel file manipulation toolkit using openpyxl and pandas. Provides tools for reading, writing, and manipulating Excel workbooks including cell updates, formulas, and charts.


## Type
base

## Tools

### read_excel_tool
Read Excel file to dictionary or dataframe representation.

**Parameters:**
- `file_path` (str, required): Path to the Excel file
- `sheet_name` (str, optional): Sheet name to read (default: first sheet)
- `header_row` (int, optional): Row number for column headers (0-indexed, default: 0)
- `as_dataframe` (bool, optional): Return as dataframe dict (default: False)
- `skip_rows` (int, optional): Number of rows to skip from top
- `max_rows` (int, optional): Maximum number of rows to read

### write_excel_tool
Write data to Excel file.

**Parameters:**
- `data` (list/dict, required): Data to write (list of dicts or dataframe-style dict)
- `output_path` (str, required): Output file path
- `sheet_name` (str, optional): Sheet name (default: 'Sheet1')
- `index` (bool, optional): Include index column (default: False)
- `overwrite` (bool, optional): Overwrite existing file (default: True)

### get_sheets_tool
List all sheets in an Excel workbook.

**Parameters:**
- `file_path` (str, required): Path to the Excel file

### add_sheet_tool
Add a new sheet to an existing Excel workbook.

**Parameters:**
- `file_path` (str, required): Path to the Excel file
- `sheet_name` (str, required): Name for the new sheet
- `data` (list, optional): Data to write (list of dicts)
- `position` (int, optional): Position for new sheet (0-indexed, default: end)

### update_cells_tool
Update specific cells in an Excel workbook.

**Parameters:**
- `file_path` (str, required): Path to the Excel file
- `sheet_name` (str, optional): Sheet name (default: first sheet)
- `updates` (dict, required): Dictionary of cell:value pairs (e.g., {'A1': 'Hello', 'B2': 42})

### add_formula_tool
Add a formula to a cell in an Excel workbook.

**Parameters:**
- `file_path` (str, required): Path to the Excel file
- `sheet_name` (str, optional): Sheet name (default: first sheet)
- `cell` (str, required): Cell reference (e.g., 'C1')
- `formula` (str, required): Excel formula (e.g., '=SUM(A1:B1)')

### create_chart_tool
Create a chart in an Excel workbook.

**Parameters:**
- `file_path` (str, required): Path to the Excel file
- `sheet_name` (str, optional): Sheet name (default: first sheet)
- `chart_type` (str, required): Type of chart ('bar', 'line', 'pie', 'area', 'scatter', 'column')
- `data_range` (str, required): Data range for chart (e.g., 'A1:B10')
- `position` (str, optional): Cell position for chart (default: 'E1')
- `title` (str, optional): Chart title
- `categories_range` (str, optional): Range for category labels
- `width` (int, optional): Chart width (default: 15)
- `height` (int, optional): Chart height (default: 10)

## Requirements
```bash
pip install openpyxl pandas
```

## Examples

### Read Excel file
```python
result = read_excel_tool({
    'file_path': '/path/to/file.xlsx',
    'sheet_name': 'Data',
    'header_row': 0
})
```

### Write data to Excel
```python
result = write_excel_tool({
    'data': [
        {'Name': 'John', 'Age': 30},
        {'Name': 'Jane', 'Age': 25}
    ],
    'output_path': '/path/to/output.xlsx',
    'sheet_name': 'People'
})
```

### Add formula
```python
result = add_formula_tool({
    'file_path': '/path/to/file.xlsx',
    'cell': 'C1',
    'formula': '=SUM(A1:B1)'
})
```

### Create chart
```python
result = create_chart_tool({
    'file_path': '/path/to/file.xlsx',
    'chart_type': 'bar',
    'data_range': 'A1:B10',
    'position': 'D1',
    'title': 'Sales Data'
})
```
