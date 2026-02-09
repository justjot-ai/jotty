from datetime import datetime
import pytz

def convert_timezone_tool(params: dict) -> dict:
    """Convert time from one timezone to another"""
    try:
        time_str = params.get('time')
        from_tz = params.get('from_timezone', 'UTC')
        to_tz = params.get('to_timezone', 'UTC')
        time_format = params.get('format', '%Y-%m-%d %H:%M:%S')
        
        if not time_str:
            return {'success': False, 'error': 'time parameter is required'}
        
        try:
            from_timezone = pytz.timezone(from_tz)
            to_timezone = pytz.timezone(to_tz)
        except pytz.exceptions.UnknownTimeZoneError as e:
            return {'success': False, 'error': f'Invalid timezone: {str(e)}'}
        
        dt = datetime.strptime(time_str, time_format)
        dt_from = from_timezone.localize(dt)
        dt_to = dt_from.astimezone(to_timezone)
        
        return {
            'success': True,
            'original_time': time_str,
            'original_timezone': from_tz,
            'converted_time': dt_to.strftime(time_format),
            'converted_timezone': to_tz
        }
    except ValueError as e:
        return {'success': False, 'error': f'Invalid time format: {str(e)}'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def format_time_tool(params: dict) -> dict:
    """Convert time from one format to another"""
    try:
        time_str = params.get('time')
        from_format = params.get('from_format', '%Y-%m-%d %H:%M:%S')
        to_format = params.get('to_format', '%Y-%m-%d %H:%M:%S')
        
        if not time_str:
            return {'success': False, 'error': 'time parameter is required'}
        
        dt = datetime.strptime(time_str, from_format)
        converted = dt.strftime(to_format)
        
        return {
            'success': True,
            'original_time': time_str,
            'original_format': from_format,
            'converted_time': converted,
            'converted_format': to_format
        }
    except ValueError as e:
        return {'success': False, 'error': f'Invalid time or format: {str(e)}'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_current_time_tool(params: dict) -> dict:
    """Get current time in specified timezone and format"""
    try:
        timezone = params.get('timezone', 'UTC')
        time_format = params.get('format', '%Y-%m-%d %H:%M:%S')
        
        try:
            tz = pytz.timezone(timezone)
        except pytz.exceptions.UnknownTimeZoneError as e:
            return {'success': False, 'error': f'Invalid timezone: {str(e)}'}
        
        dt = datetime.now(tz)
        
        return {
            'success': True,
            'current_time': dt.strftime(time_format),
            'timezone': timezone,
            'timestamp': dt.timestamp()
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def timestamp_to_time_tool(params: dict) -> dict:
    """Convert Unix timestamp to formatted time"""
    try:
        timestamp = params.get('timestamp')
        timezone = params.get('timezone', 'UTC')
        time_format = params.get('format', '%Y-%m-%d %H:%M:%S')
        
        if timestamp is None:
            return {'success': False, 'error': 'timestamp parameter is required'}
        
        try:
            tz = pytz.timezone(timezone)
        except pytz.exceptions.UnknownTimeZoneError as e:
            return {'success': False, 'error': f'Invalid timezone: {str(e)}'}
        
        dt = datetime.fromtimestamp(float(timestamp), tz)
        
        return {
            'success': True,
            'timestamp': timestamp,
            'formatted_time': dt.strftime(time_format),
            'timezone': timezone
        }
    except ValueError as e:
        return {'success': False, 'error': f'Invalid timestamp: {str(e)}'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def time_to_timestamp_tool(params: dict) -> dict:
    """Convert formatted time to Unix timestamp"""
    try:
        time_str = params.get('time')
        timezone = params.get('timezone', 'UTC')
        time_format = params.get('format', '%Y-%m-%d %H:%M:%S')
        
        if not time_str:
            return {'success': False, 'error': 'time parameter is required'}
        
        try:
            tz = pytz.timezone(timezone)
        except pytz.exceptions.UnknownTimeZoneError as e:
            return {'success': False, 'error': f'Invalid timezone: {str(e)}'}
        
        dt = datetime.strptime(time_str, time_format)
        dt_tz = tz.localize(dt)
        
        return {
            'success': True,
            'formatted_time': time_str,
            'timestamp': dt_tz.timestamp(),
            'timezone': timezone
        }
    except ValueError as e:
        return {'success': False, 'error': f'Invalid time or format: {str(e)}'}
    except Exception as e:
        return {'success': False, 'error': str(e)}