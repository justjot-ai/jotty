---
name: skill-to-agent-converter
description: "Converts Jotty skills into autonomous agents. Takes a skill and creates an agent that can execute that skill independently."
---

# Skill to Agent Converter

## Description
Converts Jotty skills into autonomous agents. Takes an existing skill and wraps it in an agent interface, allowing the skill to be executed as a standalone agent with its own decision-making capabilities.

## Type
base

## Capabilities
- agent-creation
- skill-conversion

## Triggers
- "convert skill to agent"
- "make skill into agent"
- "create agent from skill"
- "turn skill into agent"

## Category
developer-tools

## Tools

### convert_skill_to_agent
Converts a skill into an autonomous agent.

**Parameters:**
- `skill_name` (str, required): Name of the skill to convert
- `agent_name` (str, optional): Name for the created agent (defaults to skill name)
- `model` (str, optional): LLM model to use (default: "sonnet")
- `system_prompt` (str, optional): Custom system prompt for the agent

**Returns:**
- `success` (bool): Whether conversion succeeded
- `agent_name` (str): Name of the created agent
- `agent_config` (dict): Configuration of the created agent
- `error` (str, optional): Error message if failed

### test_skill_agent
Tests a converted skill agent with a sample task.

**Parameters:**
- `agent_name` (str, required): Name of the agent to test
- `test_task` (str, required): Task to test the agent with

**Returns:**
- `success` (bool): Whether test succeeded
- `result` (str): Result from the agent
- `execution_time` (float): How long the test took
- `error` (str, optional): Error message if failed
