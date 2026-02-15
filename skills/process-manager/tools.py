import psutil
import os
from typing import Dict, Any, List

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper

# Status emitter for progress updates
status = SkillStatus("process-manager")



@tool_wrapper()
def list_processes_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List running processes on the system.
    
    Args:
        params: Dictionary containing:
            - filter (str, optional): Filter processes by name pattern
            - user (str, optional): Filter by username
            - limit (int, optional): Maximum number of processes to return (default: 50)
    
    Returns:
        Dictionary with:
            - success (bool): Whether listing succeeded
            - processes (list): List of process info dicts
            - count (int): Number of processes found
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        filter_pattern = params.get('filter', '').lower()
        user_filter = params.get('user')
        limit = params.get('limit', 50)
        
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent', 'status']):
            try:
                pinfo = proc.info
                
                # Filter by name pattern
                if filter_pattern and filter_pattern not in pinfo['name'].lower():
                    continue
                
                # Filter by user
                if user_filter and pinfo['username'] != user_filter:
                    continue
                
                processes.append({
                    'pid': pinfo['pid'],
                    'name': pinfo['name'],
                    'user': pinfo['username'] or 'N/A',
                    'cpu_percent': round(pinfo['cpu_percent'] or 0, 2),
                    'memory_percent': round(pinfo['memory_percent'] or 0, 2),
                    'status': pinfo['status'] or 'unknown'
                })
                
                if len(processes) >= limit:
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        # Sort by CPU usage (descending)
        processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
        
        return {
            'success': True,
            'processes': processes,
            'count': len(processes)
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error listing processes: {str(e)}'
        }


@tool_wrapper()
def get_process_info_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get detailed information about a specific process.
    
    Args:
        params: Dictionary containing:
            - pid (int, required): Process ID
    
    Returns:
        Dictionary with:
            - success (bool): Whether retrieval succeeded
            - pid (int): Process ID
            - name (str): Process name
            - status (str): Process status
            - cpu_percent (float): CPU usage percentage
            - memory_percent (float): Memory usage percentage
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        pid = params.get('pid')
        
        if pid is None:
            return {
                'success': False,
                'error': 'pid parameter is required'
            }
        
        try:
            pid = int(pid)
        except (ValueError, TypeError):
            return {
                'success': False,
                'error': 'pid must be an integer'
            }
        
        try:
            proc = psutil.Process(pid)
            
            with proc.oneshot():
                info = {
                    'pid': proc.pid,
                    'name': proc.name(),
                    'status': proc.status(),
                    'cpu_percent': round(proc.cpu_percent(interval=0.1), 2),
                    'memory_percent': round(proc.memory_percent(), 2),
                    'memory_info': {
                        'rss': proc.memory_info().rss,
                        'vms': proc.memory_info().vms
                    },
                    'create_time': proc.create_time(),
                    'username': proc.username(),
                    'cmdline': proc.cmdline()[:5]  # First 5 args
                }
            
            return {
                'success': True,
                **info
            }
        except psutil.NoSuchProcess:
            return {
                'success': False,
                'error': f'Process {pid} not found'
            }
        except psutil.AccessDenied:
            return {
                'success': False,
                'error': f'Access denied to process {pid}'
            }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error getting process info: {str(e)}'
        }


@tool_wrapper()
def kill_process_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Terminate a process.
    
    WARNING: This tool can terminate processes. Use with caution.
    
    Args:
        params: Dictionary containing:
            - pid (int, required): Process ID to terminate
            - force (bool, optional): Force kill (SIGKILL) vs graceful (SIGTERM) (default: False)
    
    Returns:
        Dictionary with:
            - success (bool): Whether termination succeeded
            - pid (int): Process ID
            - method (str): Termination method used
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        pid = params.get('pid')
        force = params.get('force', False)
        
        if pid is None:
            return {
                'success': False,
                'error': 'pid parameter is required'
            }
        
        try:
            pid = int(pid)
        except (ValueError, TypeError):
            return {
                'success': False,
                'error': 'pid must be an integer'
            }
        
        try:
            proc = psutil.Process(pid)
            
            if force:
                proc.kill()
                method = 'SIGKILL (force)'
            else:
                proc.terminate()
                method = 'SIGTERM (graceful)'
            
            # Wait a bit to see if process terminated
            try:
                proc.wait(timeout=3)
            except psutil.TimeoutExpired:
                # Process didn't terminate gracefully, might need force kill
                if not force:
                    return {
                        'success': False,
                        'error': 'Process did not terminate gracefully. Set force=True for force kill.',
                        'pid': pid
                    }
            
            return {
                'success': True,
                'pid': pid,
                'method': method
            }
        except psutil.NoSuchProcess:
            return {
                'success': False,
                'error': f'Process {pid} not found'
            }
        except psutil.AccessDenied:
            return {
                'success': False,
                'error': f'Access denied to process {pid}. May need elevated permissions.'
            }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error killing process: {str(e)}'
        }
