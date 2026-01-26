#!/usr/bin/env python3
"""
Standalone NotebookLM PDF Generator

Generate PDFs using Google NotebookLM without importing from Jotty skills.
This script can be used independently or in Docker environments.

Usage:
    # From markdown content
    python3 scripts/notebooklm_pdf.py --content "# My Doc\n\nContent" --title "My Doc"
    
    # From markdown file
    python3 scripts/notebooklm_pdf.py --file document.md --title "Document"
    
    # With custom output
    python3 scripts/notebooklm_pdf.py --file doc.md --output /path/to/output.pdf
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


def check_pandoc() -> bool:
    """Check if Pandoc is available for fallback."""
    try:
        subprocess.run(['which', 'pandoc'], capture_output=True, check=True)
        return True
    except:
        return False


def convert_with_pandoc(input_file: Path, output_file: Path, title: str, author: str = "NotebookLM PDF Generator") -> Dict[str, Any]:
    """Convert markdown to PDF using Pandoc."""
    try:
        cmd = [
            'pandoc',
            str(input_file),
            '-o', str(output_file),
            '--pdf-engine=xelatex',
            '--standalone',
            '-V', 'geometry:a4paper,margin=1in',
            '-V', 'fontsize=11pt',
            '-V', 'linestretch=1.15',
            '-V', 'urlcolor=blue',
            '-V', 'linkcolor=blue',
            '--toc',
            '--toc-depth=3',
            '-M', f'title={title}',
            '-M', f'author={author}'
        ]
        
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
        
        if output_file.exists():
            return {
                'success': True,
                'pdf_path': str(output_file),
                'method': 'pandoc',
                'file_size': output_file.stat().st_size
            }
        else:
            return {
                'success': False,
                'error': 'Pandoc completed but output file not found'
            }
    except Exception as e:
        return {
            'success': False,
            'error': f'Pandoc conversion error: {str(e)}'
        }


async def generate_pdf_with_notebooklm(
    content: str,
    title: str,
    output_file: Path,
    profile_dir: Optional[str] = None,
    headless: bool = True
) -> Dict[str, Any]:
    """
    Generate PDF using NotebookLM browser automation.
    
    Args:
        content: Markdown content
        title: Document title
        output_file: Output PDF path
        profile_dir: Browser profile directory (for authentication)
        headless: Run browser in headless mode
    """
    if not PLAYWRIGHT_AVAILABLE:
        return {
            'success': False,
            'error': 'Playwright not available. Install with: pip install playwright && playwright install chromium'
        }
    
    from playwright.async_api import async_playwright
    
    # Determine profile directory
    if not profile_dir:
        profile_dir = os.getenv('NOTEBOOKLM_USER_DATA_DIR', str(Path.home() / '.notebooklm_browser'))
    
    # Create temporary markdown file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp_file:
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        async with async_playwright() as p:
            # Launch browser with persistent context
            browser = await p.chromium.launch_persistent_context(
                user_data_dir=profile_dir,
                headless=headless,
                viewport={'width': 1920, 'height': 1080},
                args=['--disable-blink-features=AutomationControlled']
            )
            
            # Get or create page
            pages = browser.pages
            if pages:
                page = pages[0]
            else:
                page = await browser.new_page()
            
            # Navigate to NotebookLM
            print("🌐 Opening NotebookLM...")
            await page.goto('https://notebooklm.google.com', wait_until='networkidle', timeout=30000)
            await page.wait_for_timeout(3000)
            
            # Check if signed in
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
                if headless:
                    await browser.close()
                    return {
                        'success': False,
                        'error': 'NotebookLM requires sign-in. Run auth_notebooklm.py first or use headless=False',
                        'hint': 'Run: python3 scripts/auth_notebooklm.py'
                    }
                else:
                    print("⚠️  Please sign in to NotebookLM in the browser window...")
                    print("   Press Enter here when signed in...")
                    input()
            
            # Try to create new notebook
            print("📝 Creating notebook...")
            await page.wait_for_load_state('networkidle')
            await page.wait_for_timeout(2000)
            
            # Look for new notebook button
            new_notebook_selectors = [
                'button:has-text("New notebook")',
                'button:has-text("New")',
                'button:has-text("+")',
                '[aria-label*="New notebook"]'
            ]
            
            for selector in new_notebook_selectors:
                try:
                    button = await page.query_selector(selector)
                    if button and await button.is_visible():
                        await button.click()
                        await page.wait_for_timeout(3000)
                        print("✅ Notebook created")
                        break
                except:
                    continue
            
            # Paste content
            print("📋 Adding content...")
            editor_selectors = [
                'textarea',
                '[contenteditable="true"]',
                '[role="textbox"]',
                '.editor'
            ]
            
            content_added = False
            for selector in editor_selectors:
                try:
                    editor = await page.query_selector(selector)
                    if editor and await editor.is_visible():
                        await editor.click()
                        await page.wait_for_timeout(500)
                        await editor.fill(content)
                        await page.wait_for_timeout(2000)
                        print("✅ Content added")
                        content_added = True
                        break
                except:
                    continue
            
            if not content_added:
                # Try file upload
                upload_input = await page.query_selector('input[type="file"]')
                if upload_input:
                    await upload_input.set_input_files(tmp_path)
                    await page.wait_for_timeout(5000)
                    print("✅ File uploaded")
                    content_added = True
            
            if not content_added:
                await browser.close()
                return {
                    'success': False,
                    'error': 'Could not add content to notebook'
                }
            
            # Wait for processing
            print("⏳ Waiting for NotebookLM to process...")
            await page.wait_for_timeout(8000)
            
            # Export as PDF
            print("📄 Exporting PDF...")
            export_selectors = [
                'button:has-text("Export")',
                'button:has-text("Download")',
                'button:has-text("PDF")',
                '[aria-label*="Export"]'
            ]
            
            pdf_exported = False
            for selector in export_selectors:
                try:
                    export_button = await page.query_selector(selector)
                    if export_button and await export_button.is_visible():
                        async with page.expect_download(timeout=15000) as download_info:
                            await export_button.click()
                        
                        download = await download_info.value
                        await download.save_as(output_file)
                        print("✅ PDF downloaded")
                        pdf_exported = True
                        break
                except:
                    continue
            
            await browser.close()
            
            if pdf_exported and output_file.exists():
                return {
                    'success': True,
                    'pdf_path': str(output_file),
                    'method': 'notebooklm',
                    'file_size': output_file.stat().st_size
                }
            else:
                return {
                    'success': False,
                    'error': 'Could not export PDF from NotebookLM'
                }
    
    finally:
        # Clean up temp file
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()


async def main():
    parser = argparse.ArgumentParser(
        description='Generate PDF using NotebookLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From markdown content
  python3 scripts/notebooklm_pdf.py --content "# Title\\n\\nContent" --title "My Doc"
  
  # From markdown file
  python3 scripts/notebooklm_pdf.py --file document.md --title "Document"
  
  # With custom output and profile
  python3 scripts/notebooklm_pdf.py --file doc.md --output output.pdf --profile-dir /path/to/profile
  
  # Use Pandoc fallback only
  python3 scripts/notebooklm_pdf.py --file doc.md --pandoc-only
        """
    )
    
    parser.add_argument('--content', help='Markdown content (alternative to --file)')
    parser.add_argument('--file', help='Path to markdown file')
    parser.add_argument('--title', default='Document', help='Document title')
    parser.add_argument('--author', default='NotebookLM PDF Generator', help='Document author')
    parser.add_argument('--output', help='Output PDF path (default: auto-generated)')
    parser.add_argument('--output-dir', help='Output directory (default: current directory)')
    parser.add_argument('--profile-dir', help='Browser profile directory (default: ~/.notebooklm_browser)')
    parser.add_argument('--headless', action='store_true', default=True, help='Run browser in headless mode (default: True)')
    parser.add_argument('--no-headless', dest='headless', action='store_false', help='Show browser window')
    parser.add_argument('--pandoc-only', action='store_true', help='Use Pandoc only, skip NotebookLM')
    
    args = parser.parse_args()
    
    # Get content
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ File not found: {args.file}")
            sys.exit(1)
        content = file_path.read_text(encoding='utf-8')
    elif args.content:
        content = args.content
    else:
        print("❌ Either --content or --file must be provided")
        parser.print_help()
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_file = Path(args.output)
    else:
        output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_title = "".join(c for c in args.title if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_title = safe_title.replace(' ', '_').lower()
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"{safe_title}_{timestamp}.pdf"
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📄 Generating PDF: {output_file}")
    print(f"📝 Title: {args.title}")
    
    # Try NotebookLM first (unless pandoc-only)
    if not args.pandoc_only and PLAYWRIGHT_AVAILABLE:
        print("\n🌐 Attempting NotebookLM generation...")
        result = await generate_pdf_with_notebooklm(
            content=content,
            title=args.title,
            output_file=output_file,
            profile_dir=args.profile_dir,
            headless=args.headless
        )
        
        if result.get('success'):
            print(f"\n✅ PDF generated successfully!")
            print(f"   Method: {result.get('method')}")
            print(f"   File: {result.get('pdf_path')}")
            print(f"   Size: {result.get('file_size', 0) / 1024:.2f} KB")
            sys.exit(0)
        else:
            print(f"\n⚠️  NotebookLM failed: {result.get('error')}")
            if result.get('hint'):
                print(f"   💡 Hint: {result.get('hint')}")
            print("\n🔄 Falling back to Pandoc...")
    
    # Fallback to Pandoc
    if check_pandoc():
        print("\n📄 Using Pandoc conversion...")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            result = convert_with_pandoc(
                Path(tmp_path),
                output_file,
                args.title,
                args.author
            )
            
            if result.get('success'):
                print(f"\n✅ PDF generated successfully!")
                print(f"   Method: {result.get('method')}")
                print(f"   File: {result.get('pdf_path')}")
                print(f"   Size: {result.get('file_size', 0) / 1024:.2f} KB")
                sys.exit(0)
            else:
                print(f"\n❌ Pandoc conversion failed: {result.get('error')}")
                sys.exit(1)
        finally:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()
    else:
        print("\n❌ Pandoc not available. Install with: sudo apt-get install pandoc")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
