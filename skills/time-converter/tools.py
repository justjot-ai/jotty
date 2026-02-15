from datetime import datetime, timezone
import pytz
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper

@tool_wrapper()
def time_converter_tool(params: dict) -> dict:
    """Convert time between timezones and formats"""
    try:
        time_str = params.get('time')
        from_tz = params.get('from_timezone', 'UTC')
        to_tz = params.get('to_timezone', 'UTC')
        input_format = params.get('input_format', '%Y-%m-%d %H:%M:%S')
        output_format = params.get('output_format', '%Y-%m-%d %H:%M:%S')
        
        if not time_str:
            return {'success': False, 'error': 'time parameter is required'}
        
        try:
            from_timezone = pytz.timezone(from_tz)
        except pytz.exceptions.UnknownTimeZoneError:
            return {'success': False, 'error': f'Unknown timezone: {from_tz}'}
        
        try:
            to_timezone = pytz.timezone(to_tz)
        except pytz.exceptions.UnknownTimeZoneError:
            return {'success': False, 'error': f'Unknown timezone: {to_tz}'}
        
        try:
            dt = datetime.strptime(time_str, input_format)
        except ValueError as e:
            return {'success': False, 'error': f'Invalid time format: {str(e)}'}
        
        dt_localized = from_timezone.localize(dt)
        dt_converted = dt_localized.astimezone(to_timezone)
        converted_time = dt_converted.strftime(output_format)
        
        return {
            'success': True,
            'converted_time': converted_time,
            'from_timezone': from_tz,
            'to_timezone': to_tz,
            'original_time': time_str
        }
    
    except Exception as e:
        return {'success': False, 'error': str(e)}

@tool_wrapper()
def get_current_time_tool(params: dict) -> dict:
    """Get current time in specified timezone"""
    try:
        tz = params.get('timezone', 'UTC')
        output_format = params.get('output_format', '%Y-%m-%d %H:%M:%S')
        
        try:
            target_timezone = pytz.timezone(tz)
        except pytz.exceptions.UnknownTimeZoneError:
            return {'success': False, 'error': f'Unknown timezone: {tz}'}
        
        current_time = datetime.now(target_timezone)
        formatted_time = current_time.strftime(output_format)
        
        return {
            'success': True,
            'current_time': formatted_time,
            'timezone': tz
        }
    
    except Exception as e:
        return {'success': False, 'error': str(e)}

@tool_wrapper()
def list_timezones_tool(params: dict) -> dict:
    """List available timezones"""
    try:
        filter_str = params.get('filter', '')
        all_timezones = pytz.all_timezones
        
        if filter_str:
            filtered = [tz for tz in all_timezones if filter_str.lower() in tz.lower()]
            return {'success': True, 'timezones': filtered, 'count': len(filtered)}
        
        return {'success': True, 'timezones': all_timezones, 'count': len(all_timezones)}
    
    except Exception as e:
        return {'success': False, 'error': str(e)}