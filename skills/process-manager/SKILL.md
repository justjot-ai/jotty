---
name: process-manager
description: "Provides process management capabilities: list running processes, get process details, and terminate processes. Use when the user wants to process, system process, kill process."
---

# Process Manager Skill

## Description
Provides process management capabilities: list running processes, get process details, and terminate processes.


## Type
base


## Capabilities
- code

## Tools

### list_processes_tool
Lists running processes on the system.

**Parameters:**
- `filter` (str, optional): Filter processes by name pattern
- `user` (str, optional): Filter by username
- `limit` (int, optional): Maximum number of processes to return (default: 50)

**Returns:**
- `success` (bool): Whether listing succeeded
- `processes` (list): List of process info dicts with pid, name, user, cpu, memory
- `count` (int): Number of processes found
- `error` (str, optional): Error message if failed

### get_process_info_tool
Gets detailed information about a specific process.

**Parameters:**
- `pid` (int, required): Process ID

**Returns:**
- `success` (bool): Whether retrieval succeeded
- `pid` (int): Process ID
- `name` (str): Process name
- `status` (str): Process status
- `cpu_percent` (float): CPU usage percentage
- `memory_percent` (float): Memory usage percentage
- `error` (str, optional): Error message if failed

### kill_process_tool
Terminates a process.

**Parameters:**
- `pid` (int, required): Process ID to terminate
- `force` (bool, optional): Force kill (SIGKILL) vs graceful (SIGTERM) (default: False)

**Returns:**
- `success` (bool): Whether termination succeeded
- `pid` (int): Process ID
- `method` (str): Termination method used
- `error` (str, optional): Error message if failed

## Triggers
- "process manager"
- "process"
- "system process"
- "kill process"

## Category
development
