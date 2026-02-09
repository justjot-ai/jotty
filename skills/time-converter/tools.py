from datetime import datetime
import pytz

def time_converter_tool(params: dict) -> dict:
    """Convert time between timezones and formats"""
    try:
        time_str = params.get('time')
        from_tz = params.get('from_timezone', 'UTC')
        to_tz = params.get('to_timezone', 'UTC')
        input_format = params.get('input_format', '%Y-%m-%d %H:%M:%S')
        output_format = params.get('output_format', '%Y-%m-%d %H:%M:%S')
        
        if not time_str:
            return {
                'success': False,
                'error': 'time parameter is required'
            }
        
        try:
            from_timezone = pytz.timezone(from_tz)
        except pytz.exceptions.UnknownTimeZoneError:
            return {
                'success': False,
                'error': f'Unknown timezone: {from_tz}'
            }
        
        try:
            to_timezone = pytz.timezone(to_tz)
        except pytz.exceptions.UnknownTimeZoneError:
            return {
                'success': False,
                'error': f'Unknown timezone: {to_tz}'
            }
        
        try:
            dt = datetime.strptime(time_str, input_format)
        except ValueError as e:
            return {
                'success': False,
                'error': f'Invalid time format: {str(e)}'
            }
        
        dt_localized = from_timezone.localize(dt)
        dt_converted = dt_localized.astimezone(to_timezone)
        result_str = dt_converted.strftime(output_format)
        
        return {
            'success': True,
            'original_time': time_str,
            'from_timezone': from_tz,
            'to_timezone': to_tz,
            'converted_time': result_str,
            'utc_offset': str(dt_converted.utcoffset())
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Conversion failed: {str(e)}'
        }

def get_current_time_tool(params: dict) -> dict:
    """Get current time in specified timezone"""
    try:
        timezone = params.get('timezone', 'UTC')
        output_format = params.get('format', '%Y-%m-%d %H:%M:%S')
        
        try:
            tz = pytz.timezone(timezone)
        except pytz.exceptions.UnknownTimeZoneError:
            return {
                'success': False,
                'error': f'Unknown timezone: {timezone}'
            }
        
        now = datetime.now(tz)
        formatted_time = now.strftime(output_format)
        
        return {
            'success': True,
            'timezone': timezone,
            'current_time': formatted_time,
            'utc_offset': str(now.utcoffset()),
            'timestamp': int(now.timestamp())
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Failed to get current time: {str(e)}'
        }

def list_timezones_tool(params: dict) -> dict:
    """List available timezones, optionally filtered by region"""
    try:
        region = params.get('region', '')
        
        all_timezones = pytz.all_timezones
        
        if region:
            filtered = [tz for tz in all_timezones if tz.startswith(region)]
            return {
                'success': True,
                'region': region,
                'timezones': filtered,
                'count': len(filtered)
            }
        
        return {
            'success': True,
            'timezones': list(all_timezones),
            'count': len(all_timezones)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Failed to list timezones: {str(e)}'
        }