from datetime import datetime
import pytz

def time_converter_tool(params: dict) -> dict:
    """Convert time between timezones and formats"""
    try:
        time_str = params.get('time')
        from_timezone = params.get('from_timezone', 'UTC')
        to_timezone = params.get('to_timezone', 'UTC')
        input_format = params.get('input_format', '%Y-%m-%d %H:%M:%S')
        output_format = params.get('output_format', '%Y-%m-%d %H:%M:%S')
        
        if not time_str:
            return {
                'success': False,
                'error': 'time parameter is required'
            }
        
        try:
            from_tz = pytz.timezone(from_timezone)
        except pytz.exceptions.UnknownTimeZoneError:
            return {
                'success': False,
                'error': f'Unknown timezone: {from_timezone}'
            }
        
        try:
            to_tz = pytz.timezone(to_timezone)
        except pytz.exceptions.UnknownTimeZoneError:
            return {
                'success': False,
                'error': f'Unknown timezone: {to_timezone}'
            }
        
        try:
            dt = datetime.strptime(time_str, input_format)
        except ValueError as e:
            return {
                'success': False,
                'error': f'Invalid time format: {str(e)}'
            }
        
        dt_with_tz = from_tz.localize(dt)
        converted_dt = dt_with_tz.astimezone(to_tz)
        result_str = converted_dt.strftime(output_format)
        
        return {
            'success': True,
            'converted_time': result_str,
            'from_timezone': from_timezone,
            'to_timezone': to_timezone,
            'offset': str(converted_dt.utcoffset())
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def get_current_time_tool(params: dict) -> dict:
    """Get current time in specified timezone"""
    try:
        timezone = params.get('timezone', 'UTC')
        output_format = params.get('output_format', '%Y-%m-%d %H:%M:%S')
        
        try:
            tz = pytz.timezone(timezone)
        except pytz.exceptions.UnknownTimeZoneError:
            return {
                'success': False,
                'error': f'Unknown timezone: {timezone}'
            }
        
        current_time = datetime.now(tz)
        formatted_time = current_time.strftime(output_format)
        
        return {
            'success': True,
            'current_time': formatted_time,
            'timezone': timezone,
            'timestamp': current_time.timestamp()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def list_timezones_tool(params: dict) -> dict:
    """List available timezones"""
    try:
        filter_str = params.get('filter', '')
        
        all_timezones = pytz.all_timezones
        
        if filter_str:
            filtered = [tz for tz in all_timezones if filter_str.lower() in tz.lower()]
        else:
            filtered = list(all_timezones)
        
        return {
            'success': True,
            'timezones': filtered,
            'count': len(filtered)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }