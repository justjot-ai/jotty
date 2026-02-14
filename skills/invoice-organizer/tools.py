"""
Invoice Organizer Skill - Automatically organize invoices and receipts.

Extracts information from invoices, renames them consistently, and
organizes them into logical folders for tax preparation.
"""
import asyncio
import logging
import inspect
import re
import csv
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import os
import shutil

from Jotty.core.utils.skill_status import SkillStatus
from Jotty.core.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper

# Status emitter for progress updates
status = SkillStatus("invoice-organizer")


logger = logging.getLogger(__name__)


@async_tool_wrapper()
async def organize_invoices_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Organize invoices and receipts automatically.
    
    Args:
        params:
            - invoice_directory (str): Directory with invoices
            - organization_strategy (str, optional): Organization strategy
            - output_directory (str, optional): Output directory
            - rename_format (str, optional): Filename format
            - generate_csv (bool, optional): Generate CSV summary
            - extract_amounts (bool, optional): Extract amounts
    
    Returns:
        Dictionary with processed invoices, CSV path, statistics
    """
    status.set_callback(params.pop('_status_callback', None))

    invoice_dir = params.get('invoice_directory', '')
    strategy = params.get('organization_strategy', 'by_date')
    output_dir = params.get('output_directory', 'organized_invoices')
    rename_format = params.get('rename_format', 'YYYY-MM-DD Vendor - Invoice - Description')
    generate_csv = params.get('generate_csv', True)
    extract_amounts = params.get('extract_amounts', True)
    
    if not invoice_dir:
        return {
            'success': False,
            'error': 'invoice_directory is required'
        }
    
    invoice_path = Path(os.path.expanduser(invoice_dir))
    if not invoice_path.exists():
        return {
            'success': False,
            'error': f'Directory does not exist: {invoice_dir}'
        }
    
    output_path = Path(os.path.expanduser(output_dir))
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find invoice files
    invoice_files = []
    for ext in ['.pdf', '.jpg', '.jpeg', '.png', '.docx']:
        invoice_files.extend(list(invoice_path.glob(f'*{ext}')))
        invoice_files.extend(list(invoice_path.glob(f'*{ext.upper()}')))
    
    # Extract information from invoices
    processed_invoices = []
    for invoice_file in invoice_files:
        invoice_data = await _extract_invoice_info(invoice_file, extract_amounts)
        if invoice_data:
            processed_invoices.append(invoice_data)
    
    # Organize invoices
    organized = await _organize_invoices(processed_invoices, strategy, output_path, rename_format)
    
    # Generate CSV if requested
    csv_path = None
    if generate_csv:
        csv_path = await _generate_invoice_csv(organized, output_path)
    
    statistics = {
        'total_invoices': len(invoice_files),
        'processed': len(organized),
        'vendors': len(set(i.get('vendor', '') for i in organized)),
        'total_amount': sum(float(i.get('amount', 0) or 0) for i in organized)
    }
    
    return {
        'success': True,
        'invoices_processed': len(organized),
        'invoices': organized,
        'csv_path': str(csv_path) if csv_path else None,
        'statistics': statistics
    }


async def _extract_invoice_info(invoice_file: Path, extract_amounts: bool) -> Optional[Dict]:
    """Extract information from invoice file."""
    
    invoice_data = {
        'original_file': str(invoice_file),
        'vendor': '',
        'date': '',
        'invoice_number': '',
        'amount': '',
        'description': '',
        'file_type': invoice_file.suffix.lower()
    }
    
    # Try to extract from filename first
    filename = invoice_file.stem
    date_match = re.search(r'(\d{4}[-/]\d{2}[-/]\d{2})', filename)
    if date_match:
        invoice_data['date'] = date_match.group(1).replace('/', '-')
    
    # Use AI to extract from content
    try:
        try:
            from Jotty.core.registry.skills_registry import get_skills_registry
        except ImportError:
            from Jotty.core.registry.skills_registry import get_skills_registry
        
        registry = get_skills_registry()
        registry.init()
        claude_skill = registry.get_skill('claude-cli-llm')
        
        if claude_skill:
            generate_tool = claude_skill.tools.get('generate_text_tool')
            
            if generate_tool:
                # Read file content (limit size for PDFs/images)
                content = ""
                if invoice_file.suffix.lower() == '.pdf':
                    try:
                        # Try to read PDF text
                        import PyPDF2
                        with open(invoice_file, 'rb') as f:
                            pdf_reader = PyPDF2.PdfReader(f)
                            content = "\n".join([page.extract_text() for page in pdf_reader.pages[:3]])[:2000]
                    except:
                        content = f"Invoice file: {invoice_file.name}"
                else:
                    content = f"Invoice file: {invoice_file.name}"
                
                prompt = f"""Extract key information from this invoice:

**File:** {invoice_file.name}
**Content:** {content[:1500]}

Extract:
1. Vendor/Company name
2. Invoice date (format as YYYY-MM-DD)
3. Invoice number
4. Amount/Total (if available)
5. Product/service description

Return JSON:
{{
  "vendor": "Company Name",
  "date": "YYYY-MM-DD",
  "invoice_number": "INV-123",
  "amount": "123.45",
  "description": "Product or service"
}}"""

                if inspect.iscoroutinefunction(generate_tool):
                    result = await generate_tool({
                        'prompt': prompt,
                        'model': 'sonnet',
                        'timeout': 60
                    })
                else:
                    result = generate_tool({
                        'prompt': prompt,
                        'model': 'sonnet',
                        'timeout': 60
                    })
                
                if result.get('success'):
                    import json
                    text = result.get('text', '')
                    json_match = re.search(r'\{.*\}', text, re.DOTALL)
                    if json_match:
                        try:
                            extracted = json.loads(json_match.group())
                            invoice_data.update(extracted)
                        except json.JSONDecodeError:
                            pass
    except Exception as e:
        logger.debug(f"Invoice extraction failed: {e}")
    
    # Fallback: use file metadata
    if not invoice_data.get('date'):
        try:
            mtime = datetime.fromtimestamp(invoice_file.stat().st_mtime)
            invoice_data['date'] = mtime.strftime('%Y-%m-%d')
        except:
            invoice_data['date'] = datetime.now().strftime('%Y-%m-%d')
    
    if not invoice_data.get('vendor'):
        invoice_data['vendor'] = invoice_file.stem.split('_')[0].split('-')[0]
    
    return invoice_data


async def _organize_invoices(
    invoices: List[Dict],
    strategy: str,
    output_path: Path,
    rename_format: str
) -> List[Dict]:
    """Organize invoices according to strategy."""
    
    organized = []
    
    for invoice in invoices:
        original_file = Path(invoice['original_file'])
        
        # Determine new location based on strategy
        if strategy == 'by_vendor':
            vendor = invoice.get('vendor', 'Unknown').replace('/', '-')
            new_dir = output_path / vendor
        elif strategy == 'by_date':
            date = invoice.get('date', datetime.now().strftime('%Y-%m-%d'))
            year = date.split('-')[0] if '-' in date else str(datetime.now().year)
            new_dir = output_path / year
        elif strategy == 'by_category':
            # Would need AI to categorize
            new_dir = output_path / 'Uncategorized'
        else:
            new_dir = output_path
        
        new_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate new filename
        vendor = invoice.get('vendor', 'Unknown').replace('/', '-')
        date = invoice.get('date', datetime.now().strftime('%Y-%m-%d'))
        desc = invoice.get('description', 'Invoice')[:50].replace('/', '-')
        invoice_num = invoice.get('invoice_number', '')
        
        new_filename = f"{date} {vendor} - Invoice"
        if invoice_num:
            new_filename += f" - {invoice_num}"
        if desc and desc != 'Invoice':
            new_filename += f" - {desc}"
        new_filename += original_file.suffix
        
        new_path = new_dir / new_filename
        
        # Copy file
        try:
            shutil.copy2(original_file, new_path)
            invoice['new_path'] = str(new_path)
            invoice['new_filename'] = new_filename
            organized.append(invoice)
        except Exception as e:
            logger.error(f"Failed to copy {original_file}: {e}")
    
    return organized


async def _generate_invoice_csv(invoices: List[Dict], output_path: Path) -> Path:
    """Generate CSV summary of invoices."""
    
    csv_path = output_path / f"invoice_summary_{datetime.now().strftime('%Y%m%d')}.csv"
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Date', 'Vendor', 'Invoice Number', 'Description', 
            'Amount', 'File Path', 'Original File'
        ])
        
        for invoice in invoices:
            writer.writerow([
                invoice.get('date', ''),
                invoice.get('vendor', ''),
                invoice.get('invoice_number', ''),
                invoice.get('description', ''),
                invoice.get('amount', ''),
                invoice.get('new_path', ''),
                invoice.get('original_file', '')
            ])
    
    return csv_path
