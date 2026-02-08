from datetime import datetime
import pytz
from typing import Dict, Any, Optional, List

from Jotty.core.utils.skill_status import SkillStatus

# Status emitter for progress updates
status = SkillStatus("time-converter")



def convert_timezone_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert time from one timezone to another.
    
    Args:
        params: {
            'time': str (ISO format or HH:MM:SS),
            'from_timezone': str (e.g., 'UTC', 'America/New_York'),
            'to_timezone': str (e.g., 'Asia/Tokyo', 'Europe/London'),
            'date': str (optional, defaults to today, format: YYYY-MM-DD)
        }
    
    Returns:
        dict with 'success', 'result' or 'error'
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        time_str = params.get('time')
        from_tz = params.get('from_timezone', 'UTC')
        to_tz = params.get('to_timezone')
        date_str = params.get('date')
        
        if not time_str or not to_tz:
            return {
                'success': False,
                'error': 'Missing required parameters: time and to_timezone'
            }
        
        # Parse date
        if date_str:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
        else:
            date_obj = datetime.now().date()
        
        # Parse time
        if 'T' in time_str or ' ' in time_str:
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        else:
            time_obj = datetime.strptime(time_str, '%H:%M:%S').time()
            dt = datetime.combine(date_obj, time_obj)
        
        # Localize to source timezone
        source_tz = pytz.timezone(from_tz)
        if dt.tzinfo is None:
            dt = source_tz.localize(dt)
        
        # Convert to target timezone
        target_tz = pytz.timezone(to_tz)
        converted_dt = dt.astimezone(target_tz)
        
        return {
            'success': True,
            'result': {
                'original_time': dt.isoformat(),
                'converted_time': converted_dt.isoformat(),
                'from_timezone': from_tz,
                'to_timezone': to_tz,
                'formatted': converted_dt.strftime('%Y-%m-%d %H:%M:%S %Z')
            }
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Time conversion failed: {str(e)}'
        }


def format_time_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format time string to different formats.
    
    Args:
        params: {
            'time': str (ISO format or standard time string),
            'output_format': str (strftime format string, e.g., '%Y-%m-%d %H:%M:%S'),
            'input_format': str (optional, strptime format if not ISO)
        }
    
    Returns:
        dict with 'success', 'result' or 'error'
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        time_str = params.get('time')
        output_format = params.get('output_format', '%Y-%m-%d %H:%M:%S')
        input_format = params.get('input_format')
        
        if not time_str:
            return {
                'success': False,
                'error': 'Missing required parameter: time'
            }
        
        # Parse input time
        if input_format:
            dt = datetime.strptime(time_str, input_format)
        else:
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        
        # Format output
        formatted_time = dt.strftime(output_format)
        
        return {
            'success': True,
            'result': {
                'original': time_str,
                'formatted': formatted_time,
                'format_used': output_format
            }
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Time formatting failed: {str(e)}'
        }


def get_current_time_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get current time in specified timezone.
    
    Args:
        params: {
            'timezone': str (optional, defaults to 'UTC'),
            'format': str (optional, strftime format string)
        }
    
    Returns:
        dict with 'success', 'result' or 'error'
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        timezone_str = params.get('timezone', 'UTC')
        format_str = params.get('format', '%Y-%m-%d %H:%M:%S %Z')
        
        tz = pytz.timezone(timezone_str)
        current_time = datetime.now(tz)
        
        return {
            'success': True,
            'result': {
                'timezone': timezone_str,
                'iso_format': current_time.isoformat(),
                'formatted': current_time.strftime(format_str),
                'timestamp': current_time.timestamp()
            }
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Failed to get current time: {str(e)}'
        }


def list_timezones_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List available timezones, optionally filtered by region.
    
    Args:
        params: {
            'region': str (optional, e.g., 'America', 'Europe', 'Asia')
        }
    
    Returns:
        dict with 'success', 'result' or 'error'
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        region = params.get('region')
        all_timezones = pytz.all_timezones
        
        if region:
            filtered = [tz for tz in all_timezones if tz.startswith(region)]
            return {
                'success': True,
                'result': {
                    'region': region,
                    'timezones': filtered,
                    'count': len(filtered)
                }
            }
        
        return {
            'success': True,
            'result': {
                'timezones': all_timezones,
                'count': len(all_timezones)
            }
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Failed to list timezones: {str(e)}'
        }


def calculate_time_difference_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate difference between two times.
    
    Args:
        params: {
            'time1': str (ISO format or standard time string),
            'time2': str (ISO format or standard time string),
            'unit': str (optional, 'seconds', 'minutes', 'hours', 'days', defaults to 'seconds')
        }
    
    Returns:
        dict with 'success', 'result' or 'error'
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        time1_str = params.get('time1')
        time2_str = params.get('time2')
        unit = params.get('unit', 'seconds')
        
        if not time1_str or not time2_str:
            return {
                'success': False,
                'error': 'Missing required parameters: time1 and time2'
            }
        
        # Parse times
        dt1 = datetime.fromisoformat(time1_str.replace('Z', '+00:00'))
        dt2 = datetime.fromisoformat(time2_str.replace('Z', '+00:00'))
        
        # Calculate difference
        diff = dt2 - dt1
        total_seconds = diff.total_seconds()
        
        # Convert to requested unit
        conversions = {
            'seconds': total_seconds,
            'minutes': total_seconds / 60,
            'hours': total_seconds / 3600,
            'days': total_seconds / 86400
        }
        
        if unit not in conversions:
            unit = 'seconds'
        
        return {
            'success': True,
            'result': {
                'time1': time1_str,
                'time2': time2_str,
                'difference': conversions[unit],
                'unit': unit,
                'total_seconds': total_seconds
            }
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Time difference calculation failed: {str(e)}'
        }