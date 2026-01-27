# Invoice Organizer Skill

Automatically organizes invoices and receipts for tax preparation by reading files, extracting information, and renaming consistently.

## Description

This skill transforms chaotic folders of invoices, receipts, and financial documents into a clean, tax-ready filing system without manual effort. Extracts key information from PDFs and images, renames files consistently, and organizes them into logical folders.

## Tools

### `organize_invoices_tool`

Organize invoices and receipts automatically.

**Parameters:**
- `invoice_directory` (str, required): Directory containing invoices
- `organization_strategy` (str, optional): Strategy - 'by_vendor', 'by_category', 'by_date', 'by_tax_category' (default: 'by_date')
- `output_directory` (str, optional): Where to organize invoices (default: organized_invoices/)
- `rename_format` (str, optional): Filename format (default: 'YYYY-MM-DD Vendor - Invoice - Description')
- `generate_csv` (bool, optional): Generate CSV summary (default: True)
- `extract_amounts` (bool, optional): Extract amounts from invoices (default: True)

**Returns:**
- `success` (bool): Whether organization succeeded
- `invoices_processed` (int): Number of invoices processed
- `invoices` (list): List of processed invoices with metadata
- `csv_path` (str, optional): Path to generated CSV
- `statistics` (dict): Organization statistics
- `error` (str, optional): Error message if failed

## Usage Examples

### Basic Usage

```python
result = await organize_invoices_tool({
    'invoice_directory': '~/Downloads/invoices',
    'organization_strategy': 'by_vendor'
})
```

### Tax Preparation

```python
result = await organize_invoices_tool({
    'invoice_directory': '~/Documents/receipts',
    'organization_strategy': 'by_tax_category',
    'generate_csv': True,
    'extract_amounts': True
})
```

## Dependencies

- `file-operations`: For file operations
- `claude-cli-llm`: For extracting information from invoices
- `document-converter`: For reading PDF invoices
