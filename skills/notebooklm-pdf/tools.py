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
        
        # Method 1: Use browser automation with Playwright (primary method)
        if use_browser:
            try:
                browser_result = await _try_browser_automation(content, title, output_file, params)
                if browser_result.get('success'):
                    return browser_result
                # If browser automation fails but gives useful info, log it
                if browser_result.get('error'):
                    logger.warning(f"Browser automation failed: {browser_result.get('error')}")
            except Exception as e:
                logger.error(f"Browser automation error: {e}", exc_info=True)
        
        # Method 2: Fallback to Pandoc (if NotebookLM unavailable)
        logger.info("NotebookLM browser automation unavailable, falling back to Pandoc conversion")
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


async def _try_browser_automation(content: str, title: str, output_file: Path, params: Dict[str, Any]) -> Dict[str, Any]:
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
        user_data_dir = params.get('user_data_dir') or os.getenv('NOTEBOOKLM_USER_DATA_DIR')
        if not user_data_dir:
            # Use default location
            user_data_dir = str(Path.home() / '.notebooklm_browser')
        
        headless_mode = params.get('headless', True)  # Default to headless, but allow override
        
        # Create temporary markdown file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            async with async_playwright() as p:
                # Launch browser with persistent context (to maintain login)
                browser = await p.chromium.launch_persistent_context(
                    user_data_dir=user_data_dir,
                    headless=headless_mode,
                    viewport={'width': 1920, 'height': 1080},
                    args=['--disable-blink-features=AutomationControlled']  # Avoid detection
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
                
                # Wait for page to load
                await page.wait_for_timeout(3000)
                
                # Check if we need to sign in
                sign_in_selectors = [
                    'text=Sign in',
                    'a:has-text("Sign in")',
                    'button:has-text("Sign in")',
                    '[aria-label*="Sign in"]'
                ]
                
                needs_sign_in = False
                for selector in sign_in_selectors:
                    try:
                        sign_in_element = await page.query_selector(selector)
                        if sign_in_element and await sign_in_element.is_visible():
                            needs_sign_in = True
                            break
                    except:
                        continue
                
                if needs_sign_in:
                    if headless_mode:
                        logger.warning("NotebookLM requires sign-in, but browser is in headless mode.")
                        logger.info("Please run with headless=False or sign in manually first.")
                        logger.info("Setting NOTEBOOKLM_USER_DATA_DIR will persist your login.")
                        await browser.close()
                        return {
                            'success': False,
                            'error': 'NotebookLM requires sign-in. Run with headless=False or sign in manually first.',
                            'hint': 'Set headless=False in params or sign in manually to persist session'
                        }
                    else:
                        logger.warning("NotebookLM requires sign-in.")
                        logger.info("Please sign in manually in the browser window, then press Enter to continue...")
                        logger.info("Waiting 90 seconds for manual sign-in...")
                        await page.wait_for_timeout(90000)  # Wait for manual sign-in
                
                # Try to create new notebook or upload
                logger.info("Looking for upload/create notebook options...")
                
                # Wait for page to be fully loaded
                await page.wait_for_load_state('networkidle')
                await page.wait_for_timeout(2000)
                
                # Method 1: Look for "New notebook" or "+" button
                new_notebook_selectors = [
                    'button:has-text("New notebook")',
                    'button:has-text("New")',
                    'button:has-text("+")',
                    '[aria-label*="New notebook"]',
                    '[aria-label*="Create"]',
                    '[aria-label*="new notebook"]',
                    'button[aria-label*="New"]',
                    '.new-notebook-button',
                    '[data-testid*="new"]'
                ]
                
                notebook_created = False
                for selector in new_notebook_selectors:
                    try:
                        button = await page.query_selector(selector)
                        if button and await button.is_visible():
                            await button.click()
                            await page.wait_for_timeout(3000)  # Wait for notebook creation
                            logger.info(f"✅ Clicked: {selector}")
                            notebook_created = True
                            break
                    except Exception as e:
                        logger.debug(f"Selector {selector} failed: {e}")
                        continue
                
                if not notebook_created:
                    logger.info("No 'New notebook' button found, trying to find upload area...")
                
                # Method 2: Look for file upload or paste content
                # NotebookLM might support pasting text directly
                logger.info("Attempting to paste content directly...")
                
                # Try to find a text area or content editor
                editor_selectors = [
                    'textarea',
                    '[contenteditable="true"]',
                    '[role="textbox"]',
                    '.editor',
                    '[data-testid*="editor"]',
                    'div[contenteditable]'
                ]
                
                content_pasted = False
                for selector in editor_selectors:
                    try:
                        editor = await page.query_selector(selector)
                        if editor and await editor.is_visible():
                            await editor.click()
                            await page.wait_for_timeout(500)
                            # Paste content
                            await editor.fill(content)
                            await page.wait_for_timeout(2000)
                            logger.info(f"✅ Pasted content into: {selector}")
                            content_pasted = True
                            break
                    except Exception as e:
                        logger.debug(f"Editor selector {selector} failed: {e}")
                        continue
                
                # If no editor found, try file upload
                if not content_pasted:
                    logger.info("No editor found, trying file upload...")
                    upload_selectors = [
                        'input[type="file"]',
                        '[aria-label*="upload"]',
                        '[aria-label*="Upload"]',
                        'button:has-text("Upload")',
                        '[data-testid*="upload"]'
                    ]
                    
                    for selector in upload_selectors:
                        try:
                            upload_element = await page.query_selector(selector)
                            if upload_element:
                                if selector.startswith('input'):
                                    await upload_element.set_input_files(tmp_path)
                                    logger.info(f"✅ Uploaded file via: {selector}")
                                    content_pasted = True
                                    await page.wait_for_timeout(5000)  # Wait for upload
                                    break
                                else:
                                    # Click button and then find file input
                                    await upload_element.click()
                                    await page.wait_for_timeout(2000)
                                    file_input = await page.query_selector('input[type="file"]')
                                    if file_input:
                                        await file_input.set_input_files(tmp_path)
                                        logger.info(f"✅ Uploaded file via button: {selector}")
                                        content_pasted = True
                                        await page.wait_for_timeout(5000)
                                        break
                        except Exception as e:
                            logger.debug(f"Upload selector {selector} failed: {e}")
                            continue
                
                if not content_pasted:
                    logger.warning("Could not find upload option. Trying keyboard paste...")
                    # Try keyboard paste as last resort
                    try:
                        # Focus on page body and paste
                        await page.keyboard.press('Tab')  # Focus somewhere
                        await page.keyboard.press('Tab')
                        await page.keyboard.press('Tab')
                        await page.keyboard.type(content[:100])  # Type first 100 chars as test
                        await page.wait_for_timeout(2000)
                        logger.info("Attempted keyboard input")
                    except Exception as e:
                        logger.debug(f"Keyboard paste failed: {e}")
                
                # Wait for processing
                logger.info("Waiting for NotebookLM to process content...")
                await page.wait_for_timeout(8000)
                
                # Check if content was successfully added
                page_content = await page.content()
                if content[:50].lower() in page_content.lower():
                    logger.info("✅ Content appears to be in the notebook")
                else:
                    logger.warning("⚠️  Content may not have been added successfully")
                
                # Try to export as PDF
                logger.info("Looking for export/download options...")
                await page.wait_for_timeout(2000)
                
                export_selectors = [
                    'button:has-text("Export")',
                    'button:has-text("Download")',
                    'button:has-text("PDF")',
                    'button:has-text("Save")',
                    '[aria-label*="Export"]',
                    '[aria-label*="Download"]',
                    '[aria-label*="export"]',
                    '[data-testid*="export"]',
                    '[data-testid*="download"]',
                    'a:has-text("Export")',
                    'a:has-text("Download")'
                ]
                
                pdf_exported = False
                for selector in export_selectors:
                    try:
                        export_button = await page.query_selector(selector)
                        if export_button and await export_button.is_visible():
                            logger.info(f"Found export button: {selector}")
                            # Wait for download
                            try:
                                async with page.expect_download(timeout=15000) as download_info:
                                    await export_button.click()
                                
                                download = await download_info.value
                                await download.save_as(output_file)
                                
                                if output_file.exists():
                                    logger.info(f"✅ PDF downloaded successfully: {output_file}")
                                    pdf_exported = True
                                    break
                            except Exception as e:
                                logger.debug(f"Download wait failed for {selector}: {e}")
                                # Try clicking again without download wait
                                await export_button.click()
                                await page.wait_for_timeout(5000)
                                # Check if file was downloaded to default location
                                downloads_dir = Path.home() / 'Downloads'
                                if downloads_dir.exists():
                                    pdf_files = list(downloads_dir.glob('*.pdf'))
                                    if pdf_files:
                                        latest_pdf = max(pdf_files, key=lambda p: p.stat().st_mtime)
                                        import shutil
                                        shutil.copy2(latest_pdf, output_file)
                                        if output_file.exists():
                                            logger.info(f"✅ PDF copied from Downloads: {output_file}")
                                            pdf_exported = True
                                            break
                    except Exception as e:
                        logger.debug(f"Export selector {selector} failed: {e}")
                        continue
                
                await browser.close()
                
                if pdf_exported and output_file.exists():
                    return {
                        'success': True,
                        'pdf_path': str(output_file),
                        'method': 'browser_automation',
                        'file_size': output_file.stat().st_size
                    }
                else:
                    return {
                        'success': False,
                        'error': 'Could not find export option or download failed.',
                        'note': f'Content may have been uploaded. Please export PDF manually from NotebookLM to: {output_file}',
                        'hint': 'Try running with headless=False to see what\'s happening'
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Import document-converter skill using relative path
            import sys
            from pathlib import Path
            
            # Get the skills directory
            current_file = Path(__file__).resolve()
            skills_dir = current_file.parent.parent
            
            # Add to path if not already there
            if str(skills_dir) not in sys.path:
                sys.path.insert(0, str(skills_dir))
            
            # Import using relative import
            import importlib.util
            doc_converter_path = skills_dir / 'document-converter' / 'tools.py'
            if doc_converter_path.exists():
                spec = importlib.util.spec_from_file_location("document_converter_tools", doc_converter_path)
                if spec and spec.loader:
                    doc_converter_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(doc_converter_module)
                    convert_to_pdf_tool = doc_converter_module.convert_to_pdf_tool
                    
                    # Check if it's async or sync
                    import inspect
                    if inspect.iscoroutinefunction(convert_to_pdf_tool):
                        result = await convert_to_pdf_tool({
                            'input_file': tmp_path,
                            'output_file': str(output_file),
                            'title': title,
                            'author': params.get('author', 'Jotty Framework'),
                            'page_size': params.get('page_size', 'a4')
                        })
                    else:
                        result = convert_to_pdf_tool({
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
            else:
                # Direct Pandoc call if module not available
                return await _direct_pandoc_call(tmp_path, output_file, title, params)
                
        finally:
            # Clean up temp file
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()
            
    except Exception as e:
        logger.error(f"Pandoc fallback failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Pandoc fallback failed: {str(e)}'
        }


async def _direct_pandoc_call(input_path: Path, output_path: Path, title: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Direct Pandoc call without using document-converter skill.
    """
    try:
        import subprocess
        
        # Check if pandoc is available
        pandoc_check = subprocess.run(['which', 'pandoc'], capture_output=True)
        if pandoc_check.returncode != 0:
            return {
                'success': False,
                'error': 'Pandoc not found. Install with: sudo apt-get install pandoc or brew install pandoc'
            }
        
        # Build pandoc command
        cmd = [
            'pandoc',
            str(input_path),
            '-o', str(output_path),
            '--pdf-engine=xelatex',
            '--standalone',
            '-V', 'geometry:a4paper,margin=1in',
            '-V', 'fontsize=11pt',
            '-V', 'linestretch=1.15',
            '-V', 'urlcolor=blue',
            '-V', 'linkcolor=blue',
            '--toc',
            '--toc-depth=3',
        ]
        
        if title:
            cmd.extend(['-M', f'title={title}'])
        
        author = params.get('author', 'Jotty Framework')
        if author:
            cmd.extend(['-M', f'author={author}'])
        
        # Execute pandoc
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            return {
                'success': False,
                'error': f'Pandoc conversion failed: {result.stderr}'
            }
        
        if output_path.exists():
            return {
                'success': True,
                'pdf_path': str(output_path),
                'method': 'pandoc_direct',
                'note': 'NotebookLM unavailable, used Pandoc directly'
            }
        else:
            return {
                'success': False,
                'error': 'Pandoc completed but output file not found'
            }
            
    except Exception as e:
        logger.error(f"Direct Pandoc call failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Direct Pandoc call failed: {str(e)}'
        }


__all__ = ['notebooklm_pdf_tool']
