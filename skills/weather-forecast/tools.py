"""
Weather Forecast Skill

Fetch weather forecasts using OpenWeather API.
Follows Anthropic best practices for tool design.
"""
from typing import Dict, Any
import os
import requests
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

# Status emitter for progress updates
status = SkillStatus("weather-forecast")


@tool_wrapper(required_params=['location'])
def weather_forecast_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch weather forecast for any location using OpenWeather API.

    Provides current conditions, temperature, humidity, and multi-day forecast.

    Args:
        params: Dictionary containing:
            - location (str, required): City name or "City, Country"
            - units (str, optional): "metric", "imperial", or "kelvin". Default: "metric"
            - forecast_days (int, optional): Number of days (1-5). Default: 3

    Returns:
        Dictionary with success, location, current, forecast, error
    """
    status.set_callback(params.pop('_status_callback', None))

    location = params.get('location', '').strip()
    units = params.get('units', 'metric')
    forecast_days = params.get('forecast_days', 3)

    # Validate units
    valid_units = ['metric', 'imperial', 'kelvin']
    if units not in valid_units:
        return tool_error(
            f'Invalid units: "{units}". Must be one of: metric, imperial, kelvin. '
            f'Example: {{"location": "London", "units": "metric"}}'
        )

    # Validate forecast_days
    if not isinstance(forecast_days, int) or not 1 <= forecast_days <= 5:
        return tool_error(
            f'Parameter "forecast_days" must be integer between 1-5, got: {forecast_days}. '
            f'Example: {{"forecast_days": 3}}'
        )

    # Get API key
    api_key = os.getenv('OPENWEATHER_API_KEY')
    if not api_key:
        return tool_error(
            'OpenWeather API key not found. '
            'Get free key at https://openweathermap.org/api and set OPENWEATHER_API_KEY environment variable. '
            'Example: export OPENWEATHER_API_KEY="your_key_here"'
        )

    try:
        # Fetch current weather
        status.emit("Fetching", f"🌤️ Fetching weather for {location}...")

        current_url = "https://api.openweathermap.org/data/2.5/weather"
        current_params = {
            'q': location,
            'appid': api_key,
            'units': units
        }

        current_response = requests.get(current_url, params=current_params, timeout=10)

        if current_response.status_code == 404:
            return tool_error(
                f'Location not found: "{location}". '
                f'Use format "City" or "City, Country". '
                f'Examples: "London", "New York, US", "Tokyo, JP"'
            )
        elif current_response.status_code == 401:
            return tool_error(
                'Invalid API key. Verify OPENWEATHER_API_KEY is correct. '
                'Get new key at https://openweathermap.org/api'
            )
        elif current_response.status_code == 429:
            return tool_error(
                'Rate limit exceeded. Free tier: 60 calls/minute. '
                'Wait 60 seconds or upgrade at https://openweathermap.org/price'
            )

        current_response.raise_for_status()
        current_data = current_response.json()

        # Parse current weather
        current_weather = {
            'temperature': current_data['main']['temp'],
            'feels_like': current_data['main']['feels_like'],
            'humidity': current_data['main']['humidity'],
            'pressure': current_data['main']['pressure'],
            'description': current_data['weather'][0]['description'],
            'wind_speed': current_data['wind']['speed'],
        }

        # Fetch forecast
        status.emit("Fetching", f"📅 Fetching {forecast_days}-day forecast...")

        forecast_url = "https://api.openweathermap.org/data/2.5/forecast"
        forecast_params = {
            'q': location,
            'appid': api_key,
            'units': units,
            'cnt': forecast_days * 8  # 8 forecasts per day (3-hour intervals)
        }

        forecast_response = requests.get(forecast_url, params=forecast_params, timeout=10)
        forecast_response.raise_for_status()
        forecast_data = forecast_response.json()

        # Parse forecast (group by day)
        daily_forecasts = []
        current_day = None
        day_data = []

        for item in forecast_data['list'][:forecast_days * 8]:
            date = item['dt_txt'].split(' ')[0]
            
            if date != current_day:
                if day_data:
                    # Average the day's forecasts
                    avg_temp = sum(d['main']['temp'] for d in day_data) / len(day_data)
                    daily_forecasts.append({
                        'date': current_day,
                        'temperature': round(avg_temp, 1),
                        'description': day_data[len(day_data)//2]['weather'][0]['description'],
                        'humidity': day_data[len(day_data)//2]['main']['humidity']
                    })
                current_day = date
                day_data = [item]
            else:
                day_data.append(item)

        # Add last day
        if day_data:
            avg_temp = sum(d['main']['temp'] for d in day_data) / len(day_data)
            daily_forecasts.append({
                'date': current_day,
                'temperature': round(avg_temp, 1),
                'description': day_data[len(day_data)//2]['weather'][0]['description'],
                'humidity': day_data[len(day_data)//2]['main']['humidity']
            })

        status.emit("Complete", "✅ Weather data retrieved!")

        # Unit symbol
        unit_symbol = {
            'metric': '°C',
            'imperial': '°F',
            'kelvin': 'K'
        }[units]

        return tool_response(
            location=current_data['name'] + ', ' + current_data['sys']['country'],
            current=current_weather,
            forecast=daily_forecasts[:forecast_days],
            units=units,
            unit_symbol=unit_symbol
        )

    except requests.Timeout:
        return tool_error(
            'Request timeout. Check internet connection. '
            'OpenWeather API may be slow, try again in a few seconds.'
        )
    except requests.RequestException as e:
        return tool_error(
            f'Weather API request failed: {str(e)}. '
            f'Check internet connection and API key validity.'
        )
    except KeyError as e:
        return tool_error(
            f'Unexpected API response format. Missing field: {str(e)}. '
            f'OpenWeather API may have changed. Contact support.'
        )
    except Exception as e:
        return tool_error(
            f'Weather fetch failed: {str(e)}. '
            f'Verify location format: "City" or "City, Country"'
        )


__all__ = ['weather_forecast_tool']
