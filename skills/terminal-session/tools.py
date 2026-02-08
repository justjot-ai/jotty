"""
Terminal Session Skill
======================

Persistent terminal sessions using pexpect for interactive command execution.
Supports SSH connections, sudo operations, and pattern-based output matching.
"""

import os
import uuid
import logging
from typing import Dict, Any, Optional

from Jotty.core.utils.skill_status import SkillStatus

logger = logging.getLogger(__name__)

# Try to import pexpect

# Status emitter for progress updates
status = SkillStatus("terminal-session")

PEXPECT_AVAILABLE = False
try:
    import pexpect
    PEXPECT_AVAILABLE = True
except ImportError:
    logger.warning("pexpect not installed. Run: pip install pexpect")


class TerminalSessionManager:
    """Manages persistent terminal sessions."""

    _instance: Optional["TerminalSessionManager"] = None
    _sessions: Dict[str, Any] = {}

    @classmethod
    def get_instance(cls) -> "TerminalSessionManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = TerminalSessionManager()
        return cls._instance

    def create_session(self, shell: str = "/bin/bash") -> Dict[str, Any]:
        """Create a new terminal session."""
        if not PEXPECT_AVAILABLE:
            raise ImportError("pexpect not installed")

        session_id = str(uuid.uuid4())[:8]
        child = pexpect.spawn(shell, encoding='utf-8', timeout=30)
        child.expect(r'[\$#>]\s*')  # Wait for prompt

        self._sessions[session_id] = {
            'child': child,
            'shell': shell,
            'cwd': os.getcwd(),
            'created': True
        }

        return {
            'session_id': session_id,
            'shell': shell,
            'status': 'active'
        }

    def get_session(self, session_id: str) -> Optional[Any]:
        """Get session by ID."""
        return self._sessions.get(session_id)

    def close_session(self, session_id: str) -> bool:
        """Close a session."""
        session = self._sessions.get(session_id)
        if session:
            try:
                session['child'].close()
            except Exception:
                pass
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> list:
        """List all active sessions."""
        return [
            {
                'session_id': sid,
                'shell': info['shell'],
                'status': 'active' if info['child'].isalive() else 'dead'
            }
            for sid, info in self._sessions.items()
        ]

    def execute_command(
        self,
        session_id: str,
        command: str,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Execute command in session."""
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        child = session['child']
        child.timeout = timeout

        # Send command
        child.sendline(command)

        # Wait for prompt
        try:
            child.expect(r'[\$#>]\s*', timeout=timeout)
            output = child.before.strip()
            # Remove the command echo from output
            lines = output.split('\n')
            if lines and command in lines[0]:
                output = '\n'.join(lines[1:])
            return {
                'success': True,
                'output': output,
                'exit_code': 0
            }
        except pexpect.TIMEOUT:
            return {
                'success': False,
                'output': child.before if hasattr(child, 'before') else '',
                'error': 'Command timed out'
            }
        except pexpect.EOF:
            return {
                'success': False,
                'output': child.before if hasattr(child, 'before') else '',
                'error': 'Session terminated'
            }


def terminal_create_session_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new persistent terminal session.

    Args:
        params: Dictionary containing:
            - shell (str, optional): Shell to use (default: /bin/bash)

    Returns:
        Dictionary with:
            - success (bool): Whether session was created
            - session_id (str): Session identifier
            - shell (str): Shell being used
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    if not PEXPECT_AVAILABLE:
        return {
            'success': False,
            'error': 'pexpect not installed. Run: pip install pexpect'
        }

    try:
        shell = params.get('shell', '/bin/bash')
        manager = TerminalSessionManager.get_instance()
        result = manager.create_session(shell)
        return {
            'success': True,
            **result
        }
    except Exception as e:
        logger.error(f"Failed to create session: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


def terminal_execute_command_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a command in an existing terminal session.

    Args:
        params: Dictionary containing:
            - session_id (str, required): Session ID
            - command (str, required): Command to execute
            - timeout (int, optional): Timeout in seconds (default: 30)

    Returns:
        Dictionary with:
            - success (bool): Whether command succeeded
            - output (str): Command output
            - exit_code (int): Exit code (0 for success)
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    if not PEXPECT_AVAILABLE:
        return {'success': False, 'error': 'pexpect not installed'}

    try:
        session_id = params.get('session_id')
        command = params.get('command')
        timeout = params.get('timeout', 30)

        if not session_id:
            return {'success': False, 'error': 'session_id is required'}
        if not command:
            return {'success': False, 'error': 'command is required'}

        manager = TerminalSessionManager.get_instance()
        result = manager.execute_command(session_id, command, timeout)
        return result

    except Exception as e:
        logger.error(f"Command execution failed: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


def terminal_ssh_connect_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Connect to SSH server and create a session.

    Args:
        params: Dictionary containing:
            - host (str, required): SSH host
            - username (str, required): SSH username
            - password (str, optional): SSH password
            - port (int, optional): SSH port (default: 22)
            - key_file (str, optional): Path to private key file

    Returns:
        Dictionary with:
            - success (bool): Whether connection succeeded
            - session_id (str): Session ID for subsequent commands
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    if not PEXPECT_AVAILABLE:
        return {'success': False, 'error': 'pexpect not installed'}

    try:
        host = params.get('host')
        username = params.get('username')
        password = params.get('password')
        port = params.get('port', 22)
        key_file = params.get('key_file')

        if not host:
            return {'success': False, 'error': 'host is required'}
        if not username:
            return {'success': False, 'error': 'username is required'}

        # Build SSH command
        ssh_cmd = f"ssh -p {port}"
        if key_file:
            ssh_cmd += f" -i {key_file}"
        ssh_cmd += f" {username}@{host}"

        session_id = str(uuid.uuid4())[:8]
        child = pexpect.spawn(ssh_cmd, encoding='utf-8', timeout=30)

        # Handle authentication
        i = child.expect([
            r'password:',
            r'Are you sure you want to continue connecting',
            r'[\$#>]\s*',
            pexpect.TIMEOUT,
            pexpect.EOF
        ], timeout=30)

        if i == 0:  # Password prompt
            if not password:
                child.close()
                return {'success': False, 'error': 'Password required'}
            child.sendline(password)
            child.expect(r'[\$#>]\s*', timeout=30)
        elif i == 1:  # Host key verification
            child.sendline('yes')
            j = child.expect([r'password:', r'[\$#>]\s*'], timeout=30)
            if j == 0 and password:
                child.sendline(password)
                child.expect(r'[\$#>]\s*', timeout=30)
        elif i == 2:  # Already logged in (key auth)
            pass
        elif i == 3:  # Timeout
            child.close()
            return {'success': False, 'error': 'SSH connection timed out'}
        elif i == 4:  # EOF
            child.close()
            return {'success': False, 'error': 'SSH connection failed'}

        # Store session
        manager = TerminalSessionManager.get_instance()
        manager._sessions[session_id] = {
            'child': child,
            'shell': 'ssh',
            'host': host,
            'username': username
        }

        return {
            'success': True,
            'session_id': session_id,
            'host': host,
            'username': username
        }

    except Exception as e:
        logger.error(f"SSH connection failed: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


def terminal_expect_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wait for a pattern in session output.

    Args:
        params: Dictionary containing:
            - session_id (str, required): Session ID
            - pattern (str, required): Regex pattern to wait for
            - timeout (int, optional): Timeout in seconds (default: 30)

    Returns:
        Dictionary with:
            - success (bool): Whether pattern was found
            - matched (str): Text matched by pattern
            - before (str): Text before match
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    if not PEXPECT_AVAILABLE:
        return {'success': False, 'error': 'pexpect not installed'}

    try:
        session_id = params.get('session_id')
        pattern = params.get('pattern')
        timeout = params.get('timeout', 30)

        if not session_id:
            return {'success': False, 'error': 'session_id is required'}
        if not pattern:
            return {'success': False, 'error': 'pattern is required'}

        manager = TerminalSessionManager.get_instance()
        session = manager.get_session(session_id)
        if not session:
            return {'success': False, 'error': f'Session {session_id} not found'}

        child = session['child']
        child.expect(pattern, timeout=timeout)

        return {
            'success': True,
            'matched': child.after if hasattr(child, 'after') else '',
            'before': child.before if hasattr(child, 'before') else ''
        }

    except pexpect.TIMEOUT:
        return {'success': False, 'error': 'Pattern not found (timeout)'}
    except pexpect.EOF:
        return {'success': False, 'error': 'Session terminated'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def terminal_send_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send input to session without waiting for output.

    Args:
        params: Dictionary containing:
            - session_id (str, required): Session ID
            - text (str, required): Text to send
            - newline (bool, optional): Add newline (default: True)

    Returns:
        Dictionary with:
            - success (bool): Whether send succeeded
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    if not PEXPECT_AVAILABLE:
        return {'success': False, 'error': 'pexpect not installed'}

    try:
        session_id = params.get('session_id')
        text = params.get('text')
        newline = params.get('newline', True)

        if not session_id:
            return {'success': False, 'error': 'session_id is required'}
        if not text:
            return {'success': False, 'error': 'text is required'}

        manager = TerminalSessionManager.get_instance()
        session = manager.get_session(session_id)
        if not session:
            return {'success': False, 'error': f'Session {session_id} not found'}

        child = session['child']
        if newline:
            child.sendline(text)
        else:
            child.send(text)

        return {'success': True, 'sent': text}

    except Exception as e:
        return {'success': False, 'error': str(e)}


def terminal_close_session_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Close a terminal session.

    Args:
        params: Dictionary containing:
            - session_id (str, required): Session ID to close

    Returns:
        Dictionary with:
            - success (bool): Whether close succeeded
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        session_id = params.get('session_id')
        if not session_id:
            return {'success': False, 'error': 'session_id is required'}

        manager = TerminalSessionManager.get_instance()
        closed = manager.close_session(session_id)

        if closed:
            return {'success': True, 'message': f'Session {session_id} closed'}
        else:
            return {'success': False, 'error': f'Session {session_id} not found'}

    except Exception as e:
        return {'success': False, 'error': str(e)}


def terminal_list_sessions_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List all active terminal sessions.

    Args:
        params: Empty dictionary (no parameters needed)

    Returns:
        Dictionary with:
            - success (bool): Always True
            - sessions (list): List of session info dicts
            - count (int): Number of active sessions
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        manager = TerminalSessionManager.get_instance()
        sessions = manager.list_sessions()
        return {
            'success': True,
            'sessions': sessions,
            'count': len(sessions)
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


__all__ = [
    'terminal_create_session_tool',
    'terminal_execute_command_tool',
    'terminal_ssh_connect_tool',
    'terminal_expect_tool',
    'terminal_send_tool',
    'terminal_close_session_tool',
    'terminal_list_sessions_tool'
]
