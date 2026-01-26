# Shell Execution Skill

## Description
Provides safe shell command execution capabilities with timeout and output capture.

## Tools

### execute_command_tool
Executes a shell command and returns the output.

**Parameters:**
- `command` (str, required): Shell command to execute
- `timeout` (int, optional): Timeout in seconds (default: 30)
- `working_directory` (str, optional): Working directory for command (default: current directory)
- `shell` (bool, optional): Use shell execution (default: True)

**Returns:**
- `success` (bool): Whether execution succeeded
- `stdout` (str): Standard output
- `stderr` (str): Standard error
- `exit_code` (int): Exit code
- `error` (str, optional): Error message if failed

### execute_script_tool
Executes a Python script and returns the output.

**Parameters:**
- `script` (str, required): Python script code to execute
- `timeout` (int, optional): Timeout in seconds (default: 30)

**Returns:**
- `success` (bool): Whether execution succeeded
- `output` (str): Script output
- `error` (str, optional): Error message if failed
