from pathlib import Path
from typing import Dict, Any, Optional
import json
import hashlib
import platform
import requests


def upload_to_remarkable_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Upload PDF to reMarkable cloud.
    
    Requires device registration first (use register_remarkable_tool).
    
    Args:
        params: Dictionary containing:
            - pdf_path (str, required): Path to PDF file
            - folder (str, optional): reMarkable folder path, default: '/'
            - force (bool, optional): Overwrite if exists, default: False
    
    Returns:
        Dictionary with:
            - success (bool): Whether upload succeeded
            - document_id (str): reMarkable document ID
            - path (str): Path on device
            - error (str, optional): Error message if failed
    """
    try:
        pdf_path = params.get('pdf_path')
        if not pdf_path:
            return {
                'success': False,
                'error': 'pdf_path parameter is required'
            }
        
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            return {
                'success': False,
                'error': f'PDF file not found: {pdf_path}'
            }
        
        folder = params.get('folder', '/')
        
        # Check if rmapy is available
        try:
            from rmapy.api import Client
            from rmapy.document import ZipDocument
            RMAPY_AVAILABLE = True
        except ImportError:
            return {
                'success': False,
                'error': 'rmapy library not installed. Install: pip install rmapy'
            }
        
        # Check registration
        config_file = Path.home() / '.md2pdf' / 'remarkable_config.json'
        if not config_file.exists():
            return {
                'success': False,
                'error': 'reMarkable device not registered. Use register_remarkable_tool first.'
            }
        
        # Load config and upload
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        client = Client()
        client.renew_token()
        
        # Create document
        doc = ZipDocument(doc=pdf_file)
        doc.metadata['VissibleName'] = pdf_file.stem
        
        # Upload
        client.upload(doc, folder)
        client.sync()
        
        return {
            'success': True,
            'document_id': doc.ID,
            'path': folder,
            'filename': pdf_file.name
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error uploading to reMarkable: {str(e)}'
        }


def register_remarkable_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Register device with reMarkable cloud.
    
    Args:
        params: Dictionary containing:
            - one_time_code (str, required): 8-character code from https://my.remarkable.com/device/browser/connect
    
    Returns:
        Dictionary with:
            - success (bool): Whether registration succeeded
            - message (str): Status message
            - error (str, optional): Error message if failed
    """
    try:
        one_time_code = params.get('one_time_code')
        if not one_time_code:
            return {
                'success': False,
                'error': 'one_time_code parameter is required. Get it from: https://my.remarkable.com/device/browser/connect'
            }
        
        if len(one_time_code) != 8:
            return {
                'success': False,
                'error': 'one_time_code must be 8 characters'
            }
        
        # Generate device ID
        device_id = hashlib.md5(platform.node().encode()).hexdigest()
        
        # Register with reMarkable API
        url = 'https://webapp-prod.cloud.remarkable.engineering/token/json/2/device/new'
        body = {
            'code': one_time_code,
            'deviceDesc': 'desktop-windows',
            'deviceID': device_id
        }
        
        response = requests.post(url, json=body, timeout=30)
        
        if response.status_code == 200:
            device_token = response.text
            
            # Save config
            config_file = Path.home() / '.md2pdf' / 'remarkable_config.json'
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump({
                    'device_token': device_token,
                    'device_id': device_id
                }, f, indent=2)
            
            return {
                'success': True,
                'message': 'Device registered successfully',
                'device_id': device_id[:16] + '...'
            }
        else:
            return {
                'success': False,
                'error': f'Registration failed: HTTP {response.status_code}'
            }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error registering device: {str(e)}'
        }


def check_remarkable_status_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check reMarkable connection status.
    
    Args:
        params: Dictionary (can be empty)
    
    Returns:
        Dictionary with:
            - success (bool): Whether check succeeded
            - registered (bool): Whether device is registered
            - connected (bool): Whether currently connected
            - error (str, optional): Error message if failed
    """
    try:
        config_file = Path.home() / '.md2pages' / 'remarkable_config.json'
        
        registered = config_file.exists()
        connected = False
        
        if registered:
            try:
                from rmapy.api import Client
                client = Client()
                client.renew_token()
                connected = True
            except:
                connected = False
        
        return {
            'success': True,
            'registered': registered,
            'connected': connected,
            'config_file': str(config_file) if registered else None
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error checking status: {str(e)}'
        }
