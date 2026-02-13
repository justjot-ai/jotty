from datetime import datetime
import pytz

def convert_timezone_tool(params):
    """Convert time between timezones"""
    try:
        time_str = params.get('time')
        from_tz = params.get('from_timezone', 'UTC')
        to_tz = params.get('to_timezone', 'UTC')
        time_format = params.get('format', '%Y-%m-%d %H:%M:%S')
        
        if not time_str:
            return {'success': False, 'error': 'Missing time parameter'}
        
        try:
            from_zone = pytz.timezone(from_tz)
        except pytz.exceptions.UnknownTimeZoneError:
            return {'success': False, 'error': f'Unknown timezone: {from_tz}'}
        
        try:
            to_zone = pytz.timezone(to_tz)
        except pytz.exceptions.UnknownTimeZoneError:
            return {'success': False, 'error': f'Unknown timezone: {to_tz}'}
        
        dt = datetime.strptime(time_str, time_format)
        dt_from = from_zone.localize(dt)
        dt_to = dt_from.astimezone(to_zone)
        
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

def format_time_tool(params):
    """Convert time between different formats"""
    try:
        time_str = params.get('time')
        from_format = params.get('from_format', '%Y-%m-%d %H:%M:%S')
        to_format = params.get('to_format', '%Y-%m-%d %H:%M:%S')
        
        if not time_str:
            return {'success': False, 'error': 'Missing time parameter'}
        
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
        return {'success': False, 'error': f'Invalid time format: {str(e)}'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_current_time_tool(params):
    """Get current time in specified timezone and format"""
    try:
        timezone = params.get('timezone', 'UTC')
        time_format = params.get('format', '%Y-%m-%d %H:%M:%S')
        
        try:
            tz = pytz.timezone(timezone)
        except pytz.exceptions.UnknownTimeZoneError:
            return {'success': False, 'error': f'Unknown timezone: {timezone}'}
        
        now = datetime.now(tz)
        
        return {
            'success': True,
            'current_time': now.strftime(time_format),
            'timezone': timezone,
            'format': time_format
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def list_timezones_tool(params):
    """List available timezones, optionally filtered"""
    try:
        filter_str = params.get('filter', '')
        
        all_timezones = pytz.all_timezones
        
        if filter_str:
            filtered = [tz for tz in all_timezones if filter_str.lower() in tz.lower()]
        else:
            filtered = all_timezones
        
        return {
            'success': True,
            'timezones': filtered,
            'count': len(filtered)
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def time_difference_tool(params):
    """Calculate time difference between two times"""
    try:
        time1_str = params.get('time1')
        time2_str = params.get('time2')
        time_format = params.get('format', '%Y-%m-%d %H:%M:%S')
        timezone = params.get('timezone', 'UTC')
        
        if not time1_str or not time2_str:
            return {'success': False, 'error': 'Missing time1 or time2 parameter'}
        
        try:
            tz = pytz.timezone(timezone)
        except pytz.exceptions.UnknownTimeZoneError:
            return {'success': False, 'error': f'Unknown timezone: {timezone}'}
        
        dt1 = datetime.strptime(time1_str, time_format)
        dt2 = datetime.strptime(time2_str, time_format)
        
        dt1 = tz.localize(dt1)
        dt2 = tz.localize(dt2)
        
        diff = dt2 - dt1
        
        days = diff.days
        seconds = diff.seconds
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        return {
            'success': True,
            'time1': time1_str,
            'time2': time2_str,
            'difference_days': days,
            'difference_hours': hours,
            'difference_minutes': minutes,
            'difference_seconds': secs,
            'total_seconds': diff.total_seconds()
        }
    except ValueError as e:
        return {'success': False, 'error': f'Invalid time format: {str(e)}'}
    except Exception as e:
        return {'success': False, 'error': str(e)}