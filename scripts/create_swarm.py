#!/usr/bin/env python3
"""
Create New Swarm - Template Generator
======================================

Quickly scaffold a new swarm with all required files.

Usage:
    python scripts/create_swarm.py my_swarm "My Swarm Description"
"""

import sys
from pathlib import Path


def create_swarm(swarm_name: str, description: str):
    """Create a new swarm from template."""

    # Convert to snake_case if needed
    swarm_snake = swarm_name.lower().replace(" ", "_").replace("-", "_")
    if not swarm_snake.endswith("_swarm"):
        swarm_snake += "_swarm"

    # Convert to PascalCase
    swarm_pascal = "".join(word.capitalize() for word in swarm_snake.replace("_swarm", "").split("_")) + "Swarm"

    swarm_dir = Path(f"Jotty/core/swarms/{swarm_snake}")

    if swarm_dir.exists():
        print(f"❌ Swarm already exists: {swarm_dir}")
        sys.exit(1)

    swarm_dir.mkdir(parents=True)
    print(f"✅ Created directory: {swarm_dir}")

    # __init__.py
    init_content = f'''"""{description}"""

from .swarm import {swarm_pascal}
from .types import {swarm_pascal}Config, {swarm_pascal}Result

__all__ = ["{swarm_pascal}", "{swarm_pascal}Config", "{swarm_pascal}Result"]
'''

    # types.py
    types_content = f'''"""{swarm_pascal} - Types and Configuration"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from ..swarm_types import SwarmBaseConfig, SwarmResult


@dataclass
class {swarm_pascal}Config(SwarmBaseConfig):
    """Configuration for {swarm_pascal}."""

    # Add your custom config parameters here
    custom_param: str = "default_value"

    def __post_init__(self) -> None:
        self.name = "{swarm_pascal}"
        self.domain = "{swarm_snake.replace('_swarm', '')}"


@dataclass
class {swarm_pascal}Result(SwarmResult):
    """Result from {swarm_pascal}."""

    # Add your custom result fields here
    custom_output: str = ""
'''

    # swarm.py
    swarm_content = f'''"""{swarm_pascal} - {description}"""

import logging
from typing import Dict, Any

from ..base.domain_swarm import DomainSwarm
from .types import {swarm_pascal}Config, {swarm_pascal}Result

logger = logging.getLogger(__name__)


class {swarm_pascal}(DomainSwarm):
    """{description}

    Features:
    - Feature 1
    - Feature 2
    - Feature 3
    """

    def __init__(self, config: {swarm_pascal}Config = None):
        super().__init__(config or {swarm_pascal}Config())

        # Define your agents here
        self._define_agents([
            # Example:
            # ("AgentName", AgentRole.EXPERT, AgentClass, "agent_field"),
        ])

    async def execute(self, task: str, **kwargs) -> {swarm_pascal}Result:
        """Execute {swarm_snake.replace('_', ' ').title()} task.

        Args:
            task: Description of task to perform
            **kwargs: Additional parameters

        Returns:
            {swarm_pascal}Result with execution results
        """
        logger.info(f"Executing {swarm_pascal}: {{task}}")

        try:
            # Your swarm logic here
            result_data = {{"result": "success"}}

            return {swarm_pascal}Result(
                success=True,
                swarm_name=self.config.name,
                domain=self.config.domain,
                output=result_data,
                execution_time=0.0,
                custom_output="your custom data"
            )

        except Exception as e:
            logger.error(f"{swarm_pascal} execution failed: {{e}}")
            return {swarm_pascal}Result(
                success=False,
                swarm_name=self.config.name,
                domain=self.config.domain,
                output={{"error": str(e)}},
                execution_time=0.0,
                error=str(e)
            )
'''

    # README.md
    readme_content = f'''# {swarm_pascal}

{description}

## 🎯 Purpose

DESCRIBE: Describe what this swarm does and when to use it.

## 🚀 Quick Start

```python
from Jotty.core.swarms.{swarm_snake} import {swarm_pascal}

swarm = {swarm_pascal}()
result = await swarm.execute("your task here")
```

## 📋 Configuration

```python
from Jotty.core.swarms.{swarm_snake}.types import {swarm_pascal}Config

config = {swarm_pascal}Config(
    custom_param="value",
)

swarm = {swarm_pascal}(config)
```

## 💡 Use Cases

DESCRIBE: Add common use cases and examples.

## 📄 License

Part of Jotty AI Framework
'''

    # Write files
    files = {
        "__init__.py": init_content,
        "types.py": types_content,
        "swarm.py": swarm_content,
        "README.md": readme_content,
    }

    for filename, content in files.items():
        filepath = swarm_dir / filename
        filepath.write_text(content)
        print(f"✅ Created: {filepath}")

    print(f"\n🎉 Swarm created successfully!")
    print(f"\nNext steps:")
    print(f"1. Edit {swarm_dir}/swarm.py to implement your logic")
    print(f"2. Add agents in _define_agents()")
    print(f"3. Update {swarm_dir}/README.md with documentation")
    print(f"4. Create tests in tests/test_{swarm_snake}.py")
    print(f"5. Add to CLAUDE.md task→swarm mapping")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/create_swarm.py <name> <description>")
        print('\nExample: python scripts/create_swarm.py blog_writer "Content generation for blogs"')
        sys.exit(1)

    name = sys.argv[1]
    description = " ".join(sys.argv[2:])

    create_swarm(name, description)
