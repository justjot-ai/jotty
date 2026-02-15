---
name: skill-package-manager
description: "Manages Python packages and virtual environments for skills. Installs dependencies, creates isolated venvs, and ensures skills have the packages they need."
---

# Skill Package Manager

## Description
Manages Python packages and virtual environments for skills. Automatically installs skill dependencies, creates isolated virtual environments, and ensures each skill has access to the packages it needs without conflicts.

## Type
base

## Capabilities
- package-management
- environment-management

## Triggers
- "install skill dependencies"
- "manage skill packages"
- "create skill venv"
- "install requirements for skill"

## Category
developer-tools

## Tools

### install_skill_dependencies
Installs all required packages for a skill.

**Parameters:**
- `skill_name` (str, required): Name of the skill
- `requirements` (list[str], optional): Specific packages to install

**Returns:**
- `success` (bool): Whether installation succeeded
- `installed_packages` (list[str]): Packages that were installed
- `error` (str, optional): Error message if failed

### create_skill_venv
Creates an isolated virtual environment for a skill.

**Parameters:**
- `skill_name` (str, required): Name of the skill
- `shared` (bool, optional): Use shared venv vs isolated (default: true)

**Returns:**
- `success` (bool): Whether venv was created
- `venv_path` (str): Path to the virtual environment
- `error` (str, optional): Error message if failed

### check_skill_dependencies
Checks if all dependencies for a skill are installed.

**Parameters:**
- `skill_name` (str, required): Name of the skill to check

**Returns:**
- `success` (bool): Whether all dependencies are satisfied
- `missing_packages` (list[str]): Packages that need to be installed
- `installed_packages` (list[str]): Packages already installed
