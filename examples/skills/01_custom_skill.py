"""
Example 1: Creating a Custom Skill

Demonstrates:
- Creating a custom skill from scratch
- Skill definition (skill.yaml)
- Tool implementation (tools.py)
- Using the custom skill in an agent
"""
import os
from pathlib import Path


def create_custom_skill():
    """Create a custom skill for domain-specific tasks."""

    print("=== Creating Custom Skill ===\n")

    # Create skill directory
    skill_dir = Path("skills/weather-api")
    skill_dir.mkdir(parents=True, exist_ok=True)

    # Create skill.yaml
    skill_yaml = """name: weather-api
description: "Get current weather for any city using WeatherAPI"
version: "1.0.0"
author: "Your Name"
tools:
  - get_current_weather
  - get_forecast
"""

    (skill_dir / "skill.yaml").write_text(skill_yaml)
    print("✅ Created skill.yaml")

    # Create tools.py
    tools_py = '''"""
Weather API tools
"""
import requests
from typing import Dict, Any


def get_current_weather(city: str, units: str = "metric") -> Dict[str, Any]:
    """
    Get current weather for a city.

    Args:
        city: City name (e.g., "London", "New York")
        units: Temperature units ("metric" or "imperial")

    Returns:
        dict: Weather data with temperature, conditions, humidity
    """
    # NOTE: Replace with actual API key
    api_key = os.environ.get("WEATHER_API_KEY", "demo-key")

    # Mock response for demo (replace with actual API call)
    return {
        "city": city,
        "temperature": 22 if units == "metric" else 72,
        "units": "°C" if units == "metric" else "°F",
        "conditions": "Partly cloudy",
        "humidity": 65,
        "wind_speed": 15
    }


def get_forecast(city: str, days: int = 3) -> Dict[str, Any]:
    """
    Get weather forecast for a city.

    Args:
        city: City name
        days: Number of days (1-7)

    Returns:
        dict: Forecast data
    """
    # Mock response for demo
    return {
        "city": city,
        "forecast": [
            {"day": i + 1, "high": 25, "low": 15, "conditions": "Sunny"}
            for i in range(days)
        ]
    }
'''

    (skill_dir / "tools.py").write_text(tools_py)
    print("✅ Created tools.py")

    # Create __init__.py
    (skill_dir / "__init__.py").write_text("# Weather API Skill\n")
    print("✅ Created __init__.py")

    print("\n=== Skill Structure ===\n")
    print("skills/weather-api/")
    print("├── skill.yaml       (metadata + tool list)")
    print("├── tools.py         (tool implementations)")
    print("└── __init__.py      (package marker)")

    print("\n=== Using Your Custom Skill ===\n")
    print("""
from Jotty.core.registry import get_unified_registry
from Jotty.core.agents import AutoAgent

# Get registry
registry = get_unified_registry()

# Discover your skill
tools = registry.get_claude_tools(['weather-api'])

# Use in an agent
agent = AutoAgent(
    name="WeatherBot",
    goal="Provide weather information",
    skills=['weather-api']
)

result = await agent.execute("What's the weather in London?")
    """)

    print("✅ Custom skill created!")
    print("\n💡 Tip: Skills are auto-discovered via plugin system")
    print("   Place in skills/ directory and restart Jotty")


if __name__ == "__main__":
    create_custom_skill()
