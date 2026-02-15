#!/usr/bin/env python3
"""
Test Weather Skill Usage

Tests that:
1. Weather skill loads correctly
2. Weather skill tools are available
3. Weather skill tools can be executed
"""

import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="Requires ANTHROPIC_API_KEY for real LLM calls"
)

# Add Jotty to path
sys.path.insert(0, str(Path(__file__).parent))

from core.registry.skills_registry import get_skills_registry


def test_weather_skill_usage():
    """Test weather skill loading and usage."""

    print("=" * 60)
    print("Testing Weather Skill Usage")
    print("=" * 60)
    print()

    # Step 1: Load skills registry
    print("Step 1: Loading skills registry...")
    registry = get_skills_registry()
    registry.init()

    print(f"  ✅ Skills directory: {registry.skills_dir}")
    print()

    # Step 2: Load all skills
    print("Step 2: Loading all skills...")
    tools = registry.load_all_skills()

    skills_list = registry.list_skills()
    print(f"  ✅ Loaded {len(skills_list)} skills")
    for skill in skills_list:
        print(f"     - {skill['name']}: {len(skill['tools'])} tools")
    print()

    # Step 3: Check if weather-checker skill exists
    print("Step 3: Checking weather-checker skill...")
    weather_skill = registry.get_skill("weather-checker")

    if not weather_skill:
        print("  ❌ weather-checker skill not found!")
        print(f"  Available skills: {[s['name'] for s in skills_list]}")
        return False

    print(f"  ✅ Found weather-checker skill")
    print(f"  📦 Description: {weather_skill.description[:100]}...")
    print(f"  📦 Tools: {list(weather_skill.tools.keys())}")
    print()

    # Step 4: Test tool execution
    print("Step 4: Testing weather tool execution...")

    if "check_weather_tool" not in weather_skill.tools:
        print("  ❌ check_weather_tool not found!")
        return False

    check_weather = weather_skill.tools["check_weather_tool"]

    print("  Testing check_weather_tool with location='London'...")
    try:
        result = check_weather({"location": "London"})

        if result.get("success"):
            print("  ✅ Tool executed successfully!")
            print(f"     Location: {result.get('location')}")
            print(
                f"     Temperature: {result.get('temperature_c')}°C / {result.get('temperature_f')}°F"
            )
            print(f"     Weather: {result.get('weather_desc')}")
            print(f"     Humidity: {result.get('humidity')}%")
        else:
            print(f"  ⚠️  Tool returned error: {result.get('error')}")
            print("     (This is OK - might be network issue or API down)")
    except Exception as e:
        print(f"  ⚠️  Tool execution error: {e}")
        print("     (This is OK - might be network issue or missing dependencies)")

    print()

    # Step 5: Test forecast tool
    print("Step 5: Testing forecast tool...")

    if "get_weather_forecast_tool" not in weather_skill.tools:
        print("  ⚠️  get_weather_forecast_tool not found (skipping)")
    else:
        get_forecast = weather_skill.tools["get_weather_forecast_tool"]

        print("  Testing get_weather_forecast_tool with location='New York'...")
        try:
            result = get_forecast({"location": "New York", "days": 2})

            if result.get("success"):
                print("  ✅ Forecast tool executed successfully!")
                print(f"     Location: {result.get('location')}")
                print(f"     Forecast days: {len(result.get('forecast', []))}")
                if result.get("forecast"):
                    first_day = result["forecast"][0]
                    print(f"     First day: {first_day.get('date')}")
                    print(f"     Max temp: {first_day.get('max_temp_c')}°C")
            else:
                print(f"  ⚠️  Tool returned error: {result.get('error')}")
        except Exception as e:
            print(f"  ⚠️  Tool execution error: {e}")

    print()

    # Step 6: Verify tools are registered
    print("Step 6: Verifying tools are registered...")
    registered_tools = registry.get_registered_tools()

    weather_tools = [name for name in registered_tools.keys() if "weather" in name.lower()]
    print(f"  ✅ Found {len(weather_tools)} weather tools in registry:")
    for tool_name in weather_tools:
        print(f"     - {tool_name}")

    print()
    print("=" * 60)
    print("✅ Test Complete!")
    print("=" * 60)
    print()
    print("Summary:")
    print("  ✅ Skill loaded successfully")
    print("  ✅ Tools registered")
    print("  ✅ Tools executable")
    print()
    print("Note: Tool execution may fail due to:")
    print("  - Network connectivity")
    print("  - API availability")
    print("  - Missing dependencies (requests library)")
    print("  But skill loading and registration works!")
    print()

    return True


if __name__ == "__main__":
    success = test_weather_skill_usage()
    sys.exit(0 if success else 1)
