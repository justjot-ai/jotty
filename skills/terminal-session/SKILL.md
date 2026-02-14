# Terminal Session

Persistent terminal sessions with pexpect for interactive command execution.

## Description

Provides persistent terminal sessions that maintain state across commands, support for SSH connections, and interactive command execution with expect-based pattern matching.


## Type
base


## Capabilities
- code

## Features

- Persistent sessions (state maintained across commands)
- SSH connection support
- Interactive command execution
- Pattern-based output matching (expect)
- Session management (create, list, close)
- Timeout handling
- Sudo password support

## Tools

- `terminal_create_session`: Create a new terminal session
- `terminal_execute_command`: Execute command in session
- `terminal_ssh_connect`: Connect to SSH server
- `terminal_expect`: Wait for pattern in output
- `terminal_send`: Send input without waiting
- `terminal_close_session`: Close a session
- `terminal_list_sessions`: List active sessions

## Usage

```python
# Create a session
session = terminal_create_session({})

# Execute commands
result = terminal_execute_command({
    'session_id': session['session_id'],
    'command': 'cd /tmp && ls -la'
})

# SSH connection
result = terminal_ssh_connect({
    'host': 'server.example.com',
    'username': 'user',
    'password': 'pass'
})
```

## Dependencies

- pexpect

## Triggers
- "terminal session"

## Category
development
