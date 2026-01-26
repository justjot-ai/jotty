"""
NotebookLM PDF Generator Skill

Uploads markdown/text content to Google NotebookLM and generates PDFs.
Supports multiple methods:
1. NotebookLM API (when available)
2. Browser automation with Playwright
3. Fallback to Pandoc
"""
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import json
import tempfile
import logging
import os

logger = logging.getLogger(__name__)


async def notebooklm_pdf_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate PDF using Google NotebookLM by uploading markdown/text content.
    
    Args:
        params: Dictionary containing:
            - content (str, optional): Markdown or text content to upload
            - content_file (str, optional): Path to markdown/text file (alternative to content)
            - title (str, optional): Document title
            - output_file (str, optional): Output PDF path
            - output_dir (str, optional): Output directory (defaults to stock_market/outputs)
            - use_browser (bool, optional): Use browser automation (default: True if API unavailable)
    
    Returns:
        Dictionary with:
            - success (bool): Whether PDF generation succeeded
            - pdf_path (str): Path to generated PDF
            - notebook_id (str, optional): NotebookLM notebook ID
            - error (str, optional): Error message if failed
    """
    try:
        # Get content from file or direct input
        content = params.get('content')
        content_file = params.get('content_file')
        
        if not content and not content_file:
            return {
                'success': False,
                'error': 'Either content or content_file parameter is required'
            }
        
        # Read content if file provided
        if content_file:
            content_path = Path(content_file)
            if not content_path.exists():
                return {
                    'success': False,
                    'error': f'Content file not found: {content_file}'
                }
            content = content_path.read_text(encoding='utf-8')
        
        if not content:
            return {
                'success': False,
                'error': 'No content provided'
            }
        
        # Determine output path
        output_dir = params.get('output_dir')
        if not output_dir:
            # Default to stock_market/outputs
            current_file = Path(__file__).resolve()
            jotty_dir = current_file.parent.parent.parent
            stock_market_root = jotty_dir.parent
            output_dir = stock_market_root / 'outputs'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        title = params.get('title', 'NotebookLM Document')
        output_file = params.get('output_file')
        if not output_file:
            # Generate filename from title
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_title = safe_title.replace(' ', '_').lower()
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = output_dir / f"{safe_title}_{timestamp}.pdf"
        else:
            output_file = Path(output_file)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating PDF via NotebookLM: {output_file}")
        
        # Try NotebookLM API first, then fallback to browser automation
        use_browser = params.get('use_browser', True)
        
        # Method 1: Try NotebookLM API (if available)
        api_result = await _try_notebooklm_api(content, title, output_file)
        if api_result.get('success'):
            return api_result
        
        # Method 2: Use browser automation with Playwright
        if use_browser:
            browser_result = await _try_browser_automation(content, title, output_file)
            if browser_result.get('success'):
                return browser_result
        
        # Method 3: Fallback to Pandoc (if NotebookLM unavailable)
        logger.warning("NotebookLM unavailable, falling back to Pandoc conversion")
        return await _fallback_pandoc(content, title, output_file, params)
        
    except Exception as e:
        logger.error(f"Error generating PDF via NotebookLM: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Failed to generate PDF: {str(e)}'
        }


async def _try_notebooklm_api(content: str, title: str, output_file: Path) -> Dict[str, Any]:
    """
    Try to use NotebookLM API if available.
    
    This is a placeholder for when NotebookLM API becomes available.
    """
    # Check for API credentials
    import os
    notebooklm_api_key = os.getenv('NOTEBOOKLM_API_KEY')
    notebooklm_project_id = os.getenv('NOTEBOOKLM_PROJECT_ID')
    
    if not notebooklm_api_key:
        return {'success': False, 'error': 'NotebookLM API not configured'}
    
    # TODO: Implement NotebookLM API calls when available
    # This would involve:
    # 1. Creating a notebook via API
    # 2. Uploading content
    # 3. Requesting PDF export
    # 4. Downloading the PDF
    
    return {'success': False, 'error': 'NotebookLM API not yet implemented'}


async def _try_browser_automation(content: str, title: str, output_file: Path) -> Dict[str, Any]:
    """
    Use browser automation (Playwright) to interact with NotebookLM web interface.
    
    Note: This requires manual authentication setup. The user needs to:
    1. Sign in to NotebookLM manually in the browser
    2. Or provide authentication cookies/tokens
    """
    try:
        # Check if Playwright is available
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.warning("Playwright not installed. Install with: pip install playwright && playwright install")
            return {'success': False, 'error': 'Playwright not available. Install with: pip install playwright && playwright install'}
        
        # Check for user data directory (for persistent login)
        user_data_dir = os.getenv('NOTEBOOKLM_USER_DATA_DIR')
        if not user_data_dir:
            # Use default location
            user_data_dir = str(Path.home() / '.notebooklm_browser')
        
        # Create temporary markdown file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            async with async_playwright() as p:
                # Launch browser with persistent context (to maintain login)
                browser = await p.chromium.launch_persistent_context(
                    user_data_dir=user_data_dir,
                    headless=False,  # Set to False to see what's happening
                    viewport={'width': 1920, 'height': 1080}
                )
                
                # Get or create page
                pages = browser.pages
                if pages:
                    page = pages[0]
                else:
                    page = await browser.new_page()
                
                # Navigate to NotebookLM
                logger.info("Navigating to NotebookLM...")
                await page.goto('https://notebooklm.google.com', wait_until='networkidle', timeout=30000)
                
                # Check if we need to sign in
                await page.wait_for_timeout(3000)
                sign_in_elements = await page.query_selector_all('text=Sign in')
                if sign_in_elements:
                    logger.warning("NotebookLM requires sign-in.")
                    logger.info("Please sign in manually in the browser window, then the script will continue.")
                    logger.info("Waiting 60 seconds for manual sign-in...")
                    await page.wait_for_timeout(60000)  # Wait for manual sign-in
                
                # Try to create new notebook or upload
                logger.info("Looking for upload/create notebook options...")
                
                # Method 1: Look for "New notebook" or "+" button
                new_notebook_selectors = [
                    'button:has-text("New notebook")',
                    'button:has-text("New")',
                    '[aria-label*="New notebook"]',
                    '[aria-label*="Create"]',
                    'button[aria-label*="new"]'
                ]
                
                for selector in new_notebook_selectors:
                    try:
                        button = await page.query_selector(selector)
                        if button:
                            await button.click()
                            await page.wait_for_timeout(2000)
                            logger.info(f"Clicked: {selector}")
                            break
                    except:
                        continue
                
                # Method 2: Look for file upload
                upload_selectors = [
                    'input[type="file"]',
                    '[aria-label*="upload"]',
                    '[aria-label*="Upload"]',
                    'button:has-text("Upload")'
                ]
                
                uploaded = False
                for selector in upload_selectors:
                    try:
                        upload_element = await page.query_selector(selector)
                        if upload_element:
                            if selector.startswith('input'):
                                await upload_element.set_input_files(tmp_path)
                            else:
                                # Click button and then find file input
                                await upload_element.click()
                                await page.wait_for_timeout(1000)
                                file_input = await page.query_selector('input[type="file"]')
                                if file_input:
                                    await file_input.set_input_files(tmp_path)
                            
                            logger.info(f"Uploaded file via: {selector}")
                            uploaded = True
                            await page.wait_for_timeout(5000)  # Wait for upload
                            break
                    except Exception as e:
                        logger.debug(f"Selector {selector} failed: {e}")
                        continue
                
                if not uploaded:
                    logger.warning("Could not find upload option. Trying drag-and-drop...")
                    # Try drag and drop
                    try:
                        await page.evaluate(f"""
                            const input = document.createElement('input');
                            input.type = 'file';
                            input.accept = '.md,.txt,.markdown';
                            input.onchange = (e) => {{
                                const file = e.target.files[0];
                                const reader = new FileReader();
                                reader.onload = (e) => {{
                                    // Try to paste content
                                    document.body.dispatchEvent(new ClipboardEvent('paste', {{
                                        clipboardData: new DataTransfer()
                                    }}));
                                }};
                                reader.readAsText(file);
                            }};
                            input.click();
                        """)
                        await page.wait_for_timeout(3000)
                    except:
                        pass
                
                # Wait for processing
                logger.info("Waiting for NotebookLM to process content...")
                await page.wait_for_timeout(10000)
                
                # Try to export as PDF
                logger.info("Looking for export/download options...")
                export_selectors = [
                    'button:has-text("Export")',
                    'button:has-text("Download")',
                    '[aria-label*="Export"]',
                    '[aria-label*="Download"]',
                    'button:has-text("PDF")',
                    'button:has-text("Save")'
                ]
                
                for selector in export_selectors:
                    try:
                        export_button = await page.query_selector(selector)
                        if export_button:
                            async with page.expect_download(timeout=10000) as download_info:
                                await export_button.click()
                            
                            download = await download_info.value
                            await download.save_as(output_file)
                            await browser.close()
                            
                            if output_file.exists():
                                logger.info(f"PDF downloaded successfully: {output_file}")
                                return {
                                    'success': True,
                                    'pdf_path': str(output_file),
                                    'method': 'browser_automation'
                                }
                    except Exception as e:
                        logger.debug(f"Export selector {selector} failed: {e}")
                        continue
                
                # If we get here, export didn't work
                logger.warning("Could not find export option. Browser will remain open for manual export.")
                logger.info(f"Please manually export the notebook as PDF and save to: {output_file}")
                await browser.close()
                
                return {
                    'success': False,
                    'error': 'Could not find export option. Please export manually from NotebookLM.',
                    'note': f'Content uploaded. Please export PDF manually to: {output_file}'
                }
                
        finally:
            # Clean up temp file
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()
            
    except Exception as e:
        logger.error(f"Browser automation failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Browser automation failed: {str(e)}'
        }


async def _fallback_pandoc(content: str, title: str, output_file: Path, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fallback to Pandoc conversion if NotebookLM is unavailable.
    """
    try:
        # Create temporary markdown file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Use document-converter skill
            from skills.document_converter.tools import convert_to_pdf_tool
            
            result = await convert_to_pdf_tool({
                'input_file': tmp_path,
                'output_file': str(output_file),
                'title': title,
                'author': params.get('author', 'Jotty Framework'),
                'page_size': params.get('page_size', 'a4')
            })
            
            if result.get('success'):
                return {
                    'success': True,
                    'pdf_path': result.get('output_path', str(output_file)),
                    'method': 'pandoc_fallback',
                    'note': 'NotebookLM unavailable, used Pandoc instead'
                }
            else:
                return result
                
        finally:
            # Clean up temp file
            Path(tmp_path).unlink()
            
    except Exception as e:
        logger.error(f"Pandoc fallback failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Pandoc fallback failed: {str(e)}'
        }


__all__ = ['notebooklm_pdf_tool']
