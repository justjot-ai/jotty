# Terminal Session - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`terminal_create_session_tool`](#terminal_create_session_tool) | Create a new persistent terminal session. |
| [`terminal_execute_command_tool`](#terminal_execute_command_tool) | Execute a command in an existing terminal session. |
| [`terminal_ssh_connect_tool`](#terminal_ssh_connect_tool) | Connect to SSH server and create a session. |
| [`terminal_expect_tool`](#terminal_expect_tool) | Wait for a pattern in session output. |
| [`terminal_send_tool`](#terminal_send_tool) | Send input to session without waiting for output. |
| [`terminal_close_session_tool`](#terminal_close_session_tool) | Close a terminal session. |
| [`terminal_list_sessions_tool`](#terminal_list_sessions_tool) | List all active terminal sessions. |
| [`terminal_auto_command_tool`](#terminal_auto_command_tool) | Send a command to an auto-managed terminal session. |
| [`terminal_get_state_tool`](#terminal_get_state_tool) | Get current state of the auto-managed terminal. |
| [`terminal_get_incremental_tool`](#terminal_get_incremental_tool) | Get incremental output from auto-managed terminal (new output since last read). |

### Helper Functions

| Function | Description |
|----------|-------------|
| [`get_instance`](#get_instance) | Get singleton instance. |
| [`cleanup_all`](#cleanup_all) | Close all open sessions — called at exit or on SIGTERM. |
| [`create_session`](#create_session) | Create a new terminal session. |
| [`get_session`](#get_session) | Get session by ID. |
| [`close_session`](#close_session) | Close a session. |
| [`list_sessions`](#list_sessions) | List all active sessions. |
| [`execute_command`](#execute_command) | Execute command in session. |
| [`send_command`](#send_command) | Send command and return output. |
| [`get_state`](#get_state) | Get current terminal state. |
| [`get_incremental`](#get_incremental) | Get new output since last read. |
| [`close`](#close) | Close auto-session. |

---

## `terminal_create_session_tool`

Create a new persistent terminal session.

**Parameters:**

- **shell** (`str, optional`): Shell to use (default: /bin/bash)

**Returns:** Dictionary with: - success (bool): Whether session was created - session_id (str): Session identifier - shell (str): Shell being used - error (str, optional): Error message if failed

---

## `terminal_execute_command_tool`

Execute a command in an existing terminal session.

**Parameters:**

- **session_id** (`str, required`): Session ID
- **command** (`str, required`): Command to execute
- **timeout** (`int, optional`): Timeout in seconds (default: 30)

**Returns:** Dictionary with: - success (bool): Whether command succeeded - output (str): Command output - exit_code (int): Exit code (0 for success) - error (str, optional): Error message if failed

---

## `terminal_ssh_connect_tool`

Connect to SSH server and create a session.

**Parameters:**

- **host** (`str, required`): SSH host
- **username** (`str, required`): SSH username
- **password** (`str, optional`): SSH password
- **port** (`int, optional`): SSH port (default: 22)
- **key_file** (`str, optional`): Path to private key file

**Returns:** Dictionary with: - success (bool): Whether connection succeeded - session_id (str): Session ID for subsequent commands - error (str, optional): Error message if failed

---

## `terminal_expect_tool`

Wait for a pattern in session output.

**Parameters:**

- **session_id** (`str, required`): Session ID
- **pattern** (`str, required`): Regex pattern to wait for
- **timeout** (`int, optional`): Timeout in seconds (default: 30)

**Returns:** Dictionary with: - success (bool): Whether pattern was found - matched (str): Text matched by pattern - before (str): Text before match - error (str, optional): Error message if failed

---

## `terminal_send_tool`

Send input to session without waiting for output.

**Parameters:**

- **session_id** (`str, required`): Session ID
- **text** (`str, required`): Text to send
- **newline** (`bool, optional`): Add newline (default: True)

**Returns:** Dictionary with: - success (bool): Whether send succeeded - error (str, optional): Error message if failed

---

## `terminal_close_session_tool`

Close a terminal session.

**Parameters:**

- **session_id** (`str, required`): Session ID to close

**Returns:** Dictionary with: - success (bool): Whether close succeeded - error (str, optional): Error message if failed

---

## `terminal_list_sessions_tool`

List all active terminal sessions.

**Parameters:**

- **params** (`Dict[str, Any]`)

**Returns:** Dictionary with: - success (bool): Always True - sessions (list): List of session info dicts - count (int): Number of active sessions

---

## `terminal_auto_command_tool`

Send a command to an auto-managed terminal session.  No session management needed - automatically creates/reuses a persistent session.

**Parameters:**

- **command** (`str, required`): Command to execute
- **duration** (`float, optional`): Wait time after command (default: 2.0)

**Returns:** Dictionary with status, command, output

---

## `terminal_get_state_tool`

Get current state of the auto-managed terminal.

**Parameters:**

- **params** (`Dict[str, Any]`)

**Returns:** Dictionary with status, output, is_alive

---

## `terminal_get_incremental_tool`

Get incremental output from auto-managed terminal (new output since last read).

**Parameters:**

- **params** (`Dict[str, Any]`)

**Returns:** Dictionary with status, output

---

## `get_instance`

Get singleton instance.

**Returns:** `'TerminalSessionManager'`

---

## `cleanup_all`

Close all open sessions — called at exit or on SIGTERM.

**Returns:** `None`

---

## `create_session`

Create a new terminal session.

**Parameters:**

- **shell** (`str`)

**Returns:** `Dict[str, Any]`

---

## `get_session`

Get session by ID.

**Parameters:**

- **session_id** (`str`)

**Returns:** `Optional[Any]`

---

## `close_session`

Close a session.

**Parameters:**

- **session_id** (`str`)

**Returns:** `bool`

---

## `list_sessions`

List all active sessions.

**Returns:** `list`

---

## `execute_command`

Execute command in session.

**Parameters:**

- **session_id** (`str`)
- **command** (`str`)
- **timeout** (`int`)

**Returns:** `Dict[str, Any]`

---

## `send_command`

Send command and return output.

**Parameters:**

- **keystrokes** (`str`)
- **duration** (`float`)

**Returns:** `Dict[str, Any]`

---

## `get_state`

Get current terminal state.

**Returns:** `Dict[str, Any]`

---

## `get_incremental`

Get new output since last read.

**Returns:** `Dict[str, Any]`

---

## `close`

Close auto-session.

**Returns:** `Dict[str, Any]`
