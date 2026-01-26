python
from core.registry.skills_registry import SkillsRegistry

# Get weather for a city
weather = await SkillsRegistry.execute_skill(
    "weather-checker",
    "check_weather",
    location="London"
)

# Get weather with specific units
weather = await SkillsRegistry.execute_skill(
    "weather-checker",
    "check_weather",
    location="New York",
    units="imperial"
)