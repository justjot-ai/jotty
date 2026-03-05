---
description: "Creates new skills from natural language descriptions. Meta-skill that generates skill code and metadata files."
---

# create-skill

## Description
A meta-skill that generates new Jotty skills from natural language descriptions.
Given a description of what the skill should do, it generates the `tools.py`
implementation and `SKILL.md` metadata file, writes them to `skills/user/`,
and the registry auto-discovers them on the next run.

## Type
base

## Capabilities
- skill_generation
- code_generation
- meta_programming

## Tools
- create_skill: Generate a new skill from a natural language description

## Triggers
- "create a skill"
- "make a new skill"
- "generate skill"
- "new skill that"

## Category
meta-tools
