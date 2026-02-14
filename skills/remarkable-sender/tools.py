"""
reMarkable Sender Skill

Send PDFs to reMarkable tablet via cloud API using rmapi.
"""
import subprocess
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from Jotty.core.utils.skill_status import SkillStatus
from Jotty.core.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper

# Status emitter for progress updates
status = SkillStatus("remarkable-sender")


logger = logging.getLogger(__name__)


@async_tool_wrapper()
async def send_to_remarkable_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send PDF to reMarkable tablet via cloud API.
    
    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to PDF file
            - folder (str, optional): reMarkable folder path (default: '/')
            - document_name (str, optional): Document name (default: filename)
            - force (bool, optional): Overwrite if exists (default: True)
    
    Returns:
        Dictionary with:
            - success (bool): Whether upload succeeded
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        file_path = params.get('file_path')
        if not file_path:
            return {
                'success': False,
                'error': 'file_path parameter is required'
            }
        
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            return {
                'success': False,
                'error': f'File not found: {file_path}'
            }
        
        # Check if it's a PDF
        if file_path_obj.suffix.lower() != '.pdf':
            return {
                'success': False,
                'error': f'Only PDF files are supported. Got: {file_path_obj.suffix}'
            }
        
        # Find rmapi
        rmapi_path = shutil.which('rmapi')
        if not rmapi_path:
            # Try common locations
            for alt_path in ['/usr/local/bin/rmapi', '/tmp/rmapi', '~/go/bin/rmapi']:
                alt_path_expanded = Path(alt_path).expanduser()
                if alt_path_expanded.exists():
                    rmapi_path = str(alt_path_expanded)
                    break
        
        if not rmapi_path:
            return {
                'success': False,
                'error': 'rmapi not found. Install with: go install github.com/juruen/rmapi@latest',
                'hint': 'See: https://github.com/juruen/rmapi'
            }
        
        # Get parameters
        folder = params.get('folder', '/')
        document_name = params.get('document_name')
        force = params.get('force', True)
        
        # Prepare upload file (rename if needed)
        upload_file = file_path_obj
        temp_file = None
        
        if document_name and document_name != file_path_obj.stem:
            temp_file = file_path_obj.parent / f"{document_name}.pdf"
            shutil.copy2(file_path_obj, temp_file)
            upload_file = temp_file
        
        try:
            # Create folder if specified and not root
            if folder and folder != '/':
                logger.info(f"Creating folder: {folder}")
                mkdir_result = subprocess.run(
                    [rmapi_path, 'mkdir', folder],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                # Ignore errors (folder might already exist)
            
            # Upload file
            logger.info(f"Uploading to reMarkable: {upload_file.name}")
            cmd = [rmapi_path, 'put']
            if force:
                cmd.append('--force')
            cmd.append(str(upload_file))
            if folder and folder != '/':
                cmd.append(folder)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # Longer timeout for large files
            )
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'file_path': str(file_path_obj),
                    'document_name': document_name or file_path_obj.stem,
                    'folder': folder,
                    'message': f'Successfully uploaded to reMarkable: {folder}'
                }
            else:
                error_msg = result.stderr or result.stdout or 'Unknown error'
                return {
                    'success': False,
                    'error': f'rmapi upload failed: {error_msg[:500]}'
                }
        
        finally:
            # Clean up temp file
            if temp_file and temp_file.exists():
                temp_file.unlink()
                
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': 'Upload timeout. File may still be uploading.'
        }
    except Exception as e:
        logger.error(f"reMarkable upload error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Failed to upload to reMarkable: {str(e)}'
        }


__all__ = ['send_to_remarkable_tool']
