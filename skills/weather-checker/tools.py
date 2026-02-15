import requests
from typing import Dict, Any

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper

# Status emitter for progress updates
status = SkillStatus("weather-checker")



@tool_wrapper()
def check_weather_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check current weather for a location using wttr.in API.

    Args:
        - location (str, required): The city or location name (e.g., "London", "Delhi", "New York")

    Returns:
        Dictionary with weather information including temperature, humidity, wind speed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        location = params.get('location')
        if not location:
            return {
                'success': False,
                'error': 'Location parameter is required'
            }

        status.emit("Fetching", f"🌤️ Getting weather for {location}...")
        url = f'https://wttr.in/{location}?format=j1'
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        current = data['current_condition'][0]
        
        return {
            'success': True,
            'location': location,
            'temperature_c': current['temp_C'],
            'temperature_f': current['temp_F'],
            'feels_like_c': current['FeelsLikeC'],
            'feels_like_f': current['FeelsLikeF'],
            'weather_desc': current['weatherDesc'][0]['value'],
            'humidity': current['humidity'],
            'wind_speed_kmph': current['windspeedKmph'],
            'wind_speed_mph': current['windspeedMiles'],
            'precipitation_mm': current['precipMM'],
            'visibility_km': current['visibility'],
            'uv_index': current['uvIndex']
        }
        
    except requests.exceptions.Timeout:
        return {
            'success': False,
            'error': 'Request timeout while fetching weather data'
        }
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'error': f'Failed to fetch weather data: {str(e)}'
        }
    except (KeyError, IndexError) as e:
        return {
            'success': False,
            'error': f'Invalid response format from weather API: {str(e)}'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Unexpected error: {str(e)}'
        }


@tool_wrapper()
def get_weather_forecast_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get weather forecast for a location using wttr.in API.

    Args:
        - location (str, required): The city or location name (e.g., "London", "Delhi")
        - days (int, optional): Number of forecast days (1-3, default: 3)

    Returns:
        Dictionary with forecast information including daily temperatures
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        location = params.get('location')
        if not location:
            return {
                'success': False,
                'error': 'Location parameter is required'
            }
        
        days = params.get('days', 3)
        if not isinstance(days, int) or days < 1 or days > 3:
            days = 3
        
        url = f'https://wttr.in/{location}?format=j1'
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        forecast_list = []
        
        for day_data in data['weather'][:days]:
            forecast_list.append({
                'date': day_data['date'],
                'max_temp_c': day_data['maxtempC'],
                'max_temp_f': day_data['maxtempF'],
                'min_temp_c': day_data['mintempC'],
                'min_temp_f': day_data['mintempF'],
                'avg_temp_c': day_data['avgtempC'],
                'avg_temp_f': day_data['avgtempF'],
                'total_snow_cm': day_data['totalSnow_cm'],
                'sun_hour': day_data['sunHour'],
                'uv_index': day_data['uvIndex']
            })
        
        return {
            'success': True,
            'location': location,
            'forecast': forecast_list
        }
        
    except requests.exceptions.Timeout:
        return {
            'success': False,
            'error': 'Request timeout while fetching forecast data'
        }
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'error': f'Failed to fetch forecast data: {str(e)}'
        }
    except (KeyError, IndexError) as e:
        return {
            'success': False,
            'error': f'Invalid response format from weather API: {str(e)}'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Unexpected error: {str(e)}'
        }