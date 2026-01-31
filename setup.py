"""
Setup script for Jotty AI package.
This is a fallback for older tools that don't support pyproject.toml.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read version from __init__.py (which is in current directory)
version = "10.0.0"
try:
    init_file = Path(__file__).parent / "__init__.py"
    if init_file.exists():
        for line in init_file.read_text().splitlines():
            if line.startswith("__version__"):
                version = line.split("=")[1].strip().strip('"').strip("'")
                break
except Exception:
    pass

# Build package_dir mapping for all subpackages
# Since setup.py is INSIDE the Jotty directory, map current dir to Jotty package
core_packages = find_packages("core", exclude=["tests*", "examples*", "test_*"])
package_dir_map = {"Jotty": "."}
# Map Jotty.core to core directory
package_dir_map["Jotty.core"] = "core"
for pkg in core_packages:
    # Map Jotty.core.presets -> core/presets
    # For nested packages like data.agentic_discovery, replace dots with slashes
    pkg_path = pkg.replace(".", "/")
    package_dir_map[f"Jotty.core.{pkg}"] = f"core/{pkg_path}"

# Map CLI packages
package_dir_map["Jotty.cli"] = "cli"
package_dir_map["Jotty.cli.repl"] = "cli/repl"
package_dir_map["Jotty.cli.commands"] = "cli/commands"
package_dir_map["Jotty.cli.ui"] = "cli/ui"
package_dir_map["Jotty.cli.config"] = "cli/config"
package_dir_map["Jotty.cli.plugins"] = "cli/plugins"

# CLI packages
cli_packages = [
    "cli",
    "cli.repl",
    "cli.commands",
    "cli.ui",
    "cli.config",
    "cli.plugins",
    "cli.gateway",
]

# Map gateway package
package_dir_map["Jotty.cli.gateway"] = "cli/gateway"

# Additional packages (telegram, web, core interfaces)
package_dir_map["Jotty.telegram_bot"] = "telegram_bot"
package_dir_map["Jotty.web"] = "web"
package_dir_map["Jotty.core.interfaces"] = "core/interfaces"

extra_packages = [
    "telegram_bot",
    "web",
    "core.interfaces",
]

setup(
    name="jotty-ai",
    version=version,
    description="Production-Ready Multi-Agent RL Wrapper for DSPy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Soham Acharya, Anshul Chauhan",
    author_email="",  # Add if available
    url="https://github.com/yourusername/jotty",
    package_dir=package_dir_map,
    packages=["Jotty", "Jotty.core"] + [f"Jotty.core.{pkg}" for pkg in core_packages] + [f"Jotty.{pkg}" for pkg in cli_packages] + [f"Jotty.{pkg}" for pkg in extra_packages],
    package_data={
        "Jotty.core.swarm_prompts": ["*.md"],
        "Jotty.core.validation_prompts": ["*.md"],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "jotty=Jotty.cli.__main__:main",
            "jotty-gateway=Jotty.cli.gateway:main",
        ],
    },
    install_requires=[
        "dspy-ai>=2.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "all": [
            "pymongo>=4.0.0",
            "redis>=4.0.0",
            "sqlalchemy>=2.0.0",
            "rich>=13.0.0",
            "prompt-toolkit>=3.0.0",
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            "websockets>=12.0",
            "python-multipart>=0.0.6",
            "python-telegram-bot>=20.0",
            "python-dotenv>=1.0.0",
        ],
        "cli": [
            "rich>=13.0.0",
            "prompt-toolkit>=3.0.0",
        ],
        "web": [
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            "websockets>=12.0",
            "python-multipart>=0.0.6",
        ],
        "telegram": [
            "python-telegram-bot>=20.0",
            "python-dotenv>=1.0.0",
        ],
        "mongodb": ["pymongo>=4.0.0"],
        "redis": ["redis>=4.0.0"],
        "sql": ["sqlalchemy>=2.0.0"],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "black>=23.12.0",
            "isort>=5.13.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="dspy multi-agent reinforcement-learning llm ai orchestration",
)
