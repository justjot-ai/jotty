---
name: weatherforecast
description: "Fetch weather forecast for any location using OpenWeather API. Provides current conditions, temperature, humidity, and 5-day forecast. Use when user wants to check weather, get forecast."
---

# Weather Forecast Skill

## Description
Fetch weather forecast for any location using OpenWeather API. Provides current conditions, temperature, humidity, and 5-day forecast with detailed metrics.

## Type
base

## Capabilities
- data-fetch
- analyze

## Triggers
- "get weather"
- "check weather"
- "weather forecast"
- "what's the weather"
- "temperature in"

## Category
data-analysis

## Tools

### weather_forecast_tool
Fetch current weather and forecast for any location.

**Parameters:**
- `location` (str, required): City name or "City, Country" (e.g., "London", "New York, US")
- `units` (str, optional): Temperature units - "metric" (Celsius), "imperial" (Fahrenheit), "kelvin". Defaults to "metric"
- `forecast_days` (int, optional): Number of forecast days (1-5). Defaults to 3

**Returns:**
- `success` (bool): Whether operation succeeded
- `location` (str): Location name
- `current` (dict): Current weather (temp, feels_like, humidity, description)
- `forecast` (list): Daily forecasts
- `error` (str, optional): Error message if failed

## Usage Examples
```python
# Example 1: Current weather for New York
result = weather_forecast_tool({
    'location': 'New York, US'
})

# Example 2: 5-day forecast in Fahrenheit
result = weather_forecast_tool({
    'location': 'London, UK',
    'units': 'imperial',
    'forecast_days': 5
})
```

## Requirements
OpenWeather API key (get free at https://openweathermap.org/api)
Set environment variable: OPENWEATHER_API_KEY

## Error Handling
Common errors and solutions:
- **Invalid location**: Use format "City" or "City, Country". Example: "London, UK"
- **API key missing**: Set OPENWEATHER_API_KEY environment variable
- **Rate limit exceeded**: Free tier limited to 60 calls/minute
