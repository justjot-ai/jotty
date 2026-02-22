---
description: "This skill provides weather information including temperature, humidity, wind speed, and multi-day forecasts for any city or location worldwide. Use when the user wants to weather, forecast, temperature."
---

# Weather Checker

Check current weather and forecast for any location using the wttr.in API.

## Description

This skill provides weather information including temperature, humidity, wind speed, and multi-day forecasts for any city or location worldwide.


## Type
base


## Capabilities
- data-fetch

## Tools

### check_weather_tool
Get current weather conditions for any city or location.

**Parameters:**
- `location` (str, required): City name or location (e.g., "London", "New York", "Tokyo")

### get_weather_forecast_tool
Get weather forecast for next 1-3 days.

**Parameters:**
- `location` (str, required): City name or location (e.g., "London", "Delhi")
- `days` (int, optional): Number of forecast days (default: 3)

## Examples

```python
# Get current weather for Delhi
result = check_weather_tool({"location": "Delhi"})

# Get 3-day forecast for New York
result = get_weather_forecast_tool({"location": "New York", "days": 3})
```

## Triggers
- "weather checker"
- "weather"
- "forecast"
- "temperature"

## Category
workflow-automation
