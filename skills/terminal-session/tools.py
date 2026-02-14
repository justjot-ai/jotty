"""
Terminal Session Skill
======================

Persistent terminal sessions using pexpect for interactive command execution.
Supports SSH connections, sudo operations, and pattern-based output matching.
"""

import atexit
import os
import signal
import uuid
import logging
from typing import Dict, Any, Optional

from Jotty.core.utils.skill_status import SkillStatus
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper

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
            atexit.register(cls._instance.cleanup_all)
            try:
                signal.signal(signal.SIGTERM, lambda *_: cls._instance.cleanup_all())
            except (OSError, ValueError):
                pass  # signal handlers only work in main thread
        return cls._instance

    def cleanup_all(self) -> None:
        """Close all open sessions — called at exit or on SIGTERM."""
        for sid in list(self._sessions.keys()):
            try:
                self._sessions[sid]['child'].close()
            except Exception:
                pass
        self._sessions.clear()
        self._emit_event("tool_end", {"action": "cleanup", "sessions_closed": "all"})

    @staticmethod
    def _emit_event(event_type: str, data: dict) -> None:
        """Emit an event via AgentEventBroadcaster (lazy import)."""
        try:
            from Jotty.core.utils.async_utils import AgentEventBroadcaster, AgentEvent
            broadcaster = AgentEventBroadcaster.get_instance()
            data["skill"] = "terminal-session"
            broadcaster.emit_async(AgentEvent(type=event_type, data=data))
        except Exception:
            pass

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

        self._emit_event("tool_start", {"command": command, "session_id": session_id})

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
            self._emit_event("tool_end", {"command": command, "success": True})
            return {
                'success': True,
                'output': output,
                'exit_code': 0
            }
        except pexpect.TIMEOUT:
            self._emit_event("tool_end", {"command": command, "success": False, "error": "timeout"})
            return {
                'success': False,
                'output': child.before if hasattr(child, 'before') else '',
                'error': 'Command timed out'
            }
        except pexpect.EOF:
            self._emit_event("tool_end", {"command": command, "success": False, "error": "eof"})
            return {
                'success': False,
                'output': child.before if hasattr(child, 'before') else '',
                'error': 'Session terminated'
            }


@tool_wrapper()
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


@tool_wrapper()
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


@tool_wrapper()
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


@tool_wrapper()
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


@tool_wrapper()
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


@tool_wrapper()
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


@tool_wrapper()
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


class AutoTerminalSession:
    """
    Auto-initializing terminal session for simple command execution.

    Ported from Synapse surface/tools/terminal_tools.py.
    Provides send_command/get_state/get_incremental without manual session management.
    """

    _instance: Optional["AutoTerminalSession"] = None
    _session = None

    @classmethod
    def get_instance(cls) -> "AutoTerminalSession":
        if cls._instance is None:
            cls._instance = AutoTerminalSession()
        return cls._instance

    # Known corporate SSL proxy indicators
    _PROXY_INDICATORS = (
        '/Library/Application Support/Zscaler',
        '/opt/zscaler',
        '/usr/local/zscaler',
    )

    _PROXY_ENV_KEYWORDS = ('zscaler', 'bluecoat', 'forcepoint', 'mcafee')

    def _detect_corporate_proxy(self) -> bool:
        """Detect Zscaler/BlueCoat/corporate SSL proxy presence.

        Checks filesystem paths and environment variables for known
        corporate proxy indicators.
        """
        # Check known install paths
        for path in self._PROXY_INDICATORS:
            if os.path.exists(path):
                return True

        # Check proxy-related env vars
        for var in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
            val = os.environ.get(var, '').lower()
            if any(kw in val for kw in self._PROXY_ENV_KEYWORDS):
                return True

        return False

    def _get_env_overrides(self) -> Dict[str, str]:
        """Return env vars for SSL bypass when corporate proxy is detected.

        These overrides disable SSL certificate verification for tools
        running inside the terminal session, preventing Zscaler/BlueCoat
        MITM certificate errors.
        """
        if not self._detect_corporate_proxy():
            return {}

        logger.info("Corporate proxy detected — applying SSL overrides for terminal session")
        return {
            'CURL_CA_BUNDLE': '',
            'PYTHONHTTPSVERIFY': '0',
            'REQUESTS_CA_BUNDLE': '',
            'NODE_TLS_REJECT_UNAUTHORIZED': '0',
        }

    def _ensure_session(self) -> Dict[str, Any]:
        """Auto-initialize a pexpect session if not already running."""
        if self._session is not None and self._session.isalive():
            return {"status": "success"}

        if not PEXPECT_AVAILABLE:
            return {"status": "error", "error": "pexpect not installed. Run: pip install pexpect"}

        try:
            import time
            # Merge current env with any proxy/SSL overrides
            spawn_env = dict(os.environ)
            spawn_env.update(self._get_env_overrides())

            self._session = pexpect.spawn(
                "/bin/bash", cwd=os.getcwd(), echo=False, encoding='utf-8',
                timeout=30, env=spawn_env)
            time.sleep(0.5)
            if not self._session.isalive():
                return {"status": "error", "error": "Terminal session created but not alive"}
            return {"status": "success"}
        except Exception as e:
            return {"status": "error", "error": f"Failed to initialize: {e}"}

    def send_command(self, keystrokes: str, duration: float = 2.0) -> Dict[str, Any]:
        """Send command and return output."""
        import time
        init = self._ensure_session()
        if init["status"] != "success":
            return init

        try:
            self._session.sendline(keystrokes)
            time.sleep(duration)

            output = ""
            try:
                self._session.expect(pexpect.TIMEOUT, timeout=0.1)
                output = self._session.before or ""
                if hasattr(self._session, 'buffer'):
                    output += self._session.buffer
            except (EOFError, OSError, Exception):
                pass

            return {"status": "success", "command": keystrokes, "output": output}
        except pexpect.TIMEOUT:
            output = self._session.before if hasattr(self._session, 'before') else ""
            return {"status": "timeout", "command": keystrokes, "output": output,
                    "error": "Command timed out"}
        except Exception as e:
            return {"status": "error", "command": keystrokes, "output": "",
                    "error": f"Error: {e}"}

    def get_state(self) -> Dict[str, Any]:
        """Get current terminal state."""
        init = self._ensure_session()
        if init["status"] != "success":
            return init

        output = ""
        try:
            self._session.expect(pexpect.TIMEOUT, timeout=0.1)
            if hasattr(self._session, 'before'):
                output = self._session.before or ""
            if hasattr(self._session, 'buffer'):
                output += self._session.buffer
        except (EOFError, OSError, Exception):
            pass

        return {"status": "success", "output": output,
                "is_alive": self._session.isalive() if self._session else False}

    def get_incremental(self) -> Dict[str, Any]:
        """Get new output since last read."""
        init = self._ensure_session()
        if init["status"] != "success":
            return init

        output = ""
        try:
            self._session.expect(pexpect.TIMEOUT, timeout=0.1)
            if hasattr(self._session, 'before'):
                output = self._session.before or ""
        except (EOFError, OSError, Exception):
            pass

        return {"status": "success", "output": output}

    def close(self) -> Dict[str, Any]:
        """Close auto-session."""
        if self._session is not None:
            try:
                self._session.close(force=True)
            except Exception:
                pass
            self._session = None
        return {"status": "success", "message": "Auto terminal closed"}


@tool_wrapper()
def terminal_auto_command_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a command to an auto-managed terminal session.

    No session management needed - automatically creates/reuses a persistent session.

    Args:
        params: Dictionary containing:
            - command (str, required): Command to execute
            - duration (float, optional): Wait time after command (default: 2.0)

    Returns:
        Dictionary with status, command, output
    """
    status.set_callback(params.pop('_status_callback', None))

    command = params.get('command')
    if not command:
        return {'success': False, 'error': 'command is required'}

    duration = float(params.get('duration', 2.0))
    auto = AutoTerminalSession.get_instance()
    result = auto.send_command(command, duration)
    result['success'] = result['status'] == 'success'
    return result


@tool_wrapper()
def terminal_get_state_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get current state of the auto-managed terminal.

    Args:
        params: Empty dictionary (no parameters needed)

    Returns:
        Dictionary with status, output, is_alive
    """
    status.set_callback(params.pop('_status_callback', None))
    auto = AutoTerminalSession.get_instance()
    result = auto.get_state()
    result['success'] = result['status'] == 'success'
    return result


@tool_wrapper()
def terminal_get_incremental_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get incremental output from auto-managed terminal (new output since last read).

    Args:
        params: Empty dictionary (no parameters needed)

    Returns:
        Dictionary with status, output
    """
    status.set_callback(params.pop('_status_callback', None))
    auto = AutoTerminalSession.get_instance()
    result = auto.get_incremental()
    result['success'] = result['status'] == 'success'
    return result


__all__ = [
    'terminal_create_session_tool',
    'terminal_execute_command_tool',
    'terminal_ssh_connect_tool',
    'terminal_expect_tool',
    'terminal_send_tool',
    'terminal_close_session_tool',
    'terminal_list_sessions_tool',
    # Auto-managed terminal (no session management needed)
    'terminal_auto_command_tool',
    'terminal_get_state_tool',
    'terminal_get_incremental_tool',
]
