import subprocess
import sys
import io
from typing import Dict, Any
from pathlib import Path


def execute_command_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a shell command and return the output.
    
    WARNING: This tool executes shell commands. Use with caution.
    
    Args:
        params: Dictionary containing:
            - command (str, required): Shell command to execute
            - timeout (int, optional): Timeout in seconds (default: 30)
            - working_directory (str, optional): Working directory for command
            - shell (bool, optional): Use shell execution (default: True)
    
    Returns:
        Dictionary with:
            - success (bool): Whether execution succeeded
            - stdout (str): Standard output
            - stderr (str): Standard error
            - exit_code (int): Exit code
            - error (str, optional): Error message if failed
    """
    try:
        command = params.get('command')
        if not command:
            return {
                'success': False,
                'error': 'command parameter is required'
            }
        
        timeout = params.get('timeout', 30)
        working_directory = params.get('working_directory')
        use_shell = params.get('shell', True)
        
        # Prepare working directory
        cwd = None
        if working_directory:
            cwd = Path(working_directory)
            if not cwd.exists() or not cwd.is_dir():
                return {
                    'success': False,
                    'error': f'Invalid working directory: {working_directory}'
                }
            cwd = str(cwd)
        
        # Execute command
        try:
            result = subprocess.run(
                command,
                shell=use_shell,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd
            )
            
            return {
                'success': True,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'exit_code': result.returncode,
                'command': command
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f'Command timed out after {timeout} seconds',
                'command': command
            }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error executing command: {str(e)}'
        }


def execute_script_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a Python script and return the output.
    
    WARNING: This tool executes Python code. Use with extreme caution.
    
    Args:
        params: Dictionary containing:
            - script (str, required): Python script code to execute
            - timeout (int, optional): Timeout in seconds (default: 30)
    
    Returns:
        Dictionary with:
            - success (bool): Whether execution succeeded
            - output (str): Script output
            - error (str, optional): Error message if failed
    """
    try:
        script = params.get('script')
        if not script:
            return {
                'success': False,
                'error': 'script parameter is required'
            }
        
        timeout = params.get('timeout', 30)
        
        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Redirect stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            # Execute script with timeout using subprocess
            # This is safer than exec() as it provides timeout
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script)
                temp_script = f.name
            
            try:
                result = subprocess.run(
                    [sys.executable, temp_script],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                output = result.stdout + result.stderr
                
                return {
                    'success': result.returncode == 0,
                    'output': output,
                    'exit_code': result.returncode,
                    'error': result.stderr if result.returncode != 0 else None
                }
            except subprocess.TimeoutExpired:
                return {
                    'success': False,
                    'error': f'Script timed out after {timeout} seconds'
                }
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_script)
                except:
                    pass
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    except Exception as e:
        return {
            'success': False,
            'error': f'Error executing script: {str(e)}'
        }
