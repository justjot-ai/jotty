from datetime import datetime
import pytz

def time_converter_tool(params: dict) -> dict:
    """
    Convert time between timezones and formats.
    
    params = {
        'time': str (optional, defaults to current time, format: 'YYYY-MM-DD HH:MM:SS' or 'HH:MM:SS'),
        'from_timezone': str (optional, defaults to 'UTC'),
        'to_timezone': str (required),
        'output_format': str (optional, defaults to '%Y-%m-%d %H:%M:%S')
    }
    """
    try:
        to_timezone = params.get('to_timezone')
        if not to_timezone:
            return {
                'success': False,
                'error': 'to_timezone is required'
            }
        
        from_timezone = params.get('from_timezone', 'UTC')
        output_format = params.get('output_format', '%Y-%m-%d %H:%M:%S')
        time_str = params.get('time')
        
        # Validate timezones
        try:
            from_tz = pytz.timezone(from_timezone)
            to_tz = pytz.timezone(to_timezone)
        except pytz.exceptions.UnknownTimeZoneError as e:
            return {
                'success': False,
                'error': f'Invalid timezone: {str(e)}'
            }
        
        # Parse input time
        if time_str:
            try:
                if len(time_str) <= 8:  # HH:MM:SS format
                    dt = datetime.strptime(time_str, '%H:%M:%S')
                    dt = dt.replace(year=datetime.now().year, month=datetime.now().month, day=datetime.now().day)
                else:
                    dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                dt = from_tz.localize(dt)
            except ValueError:
                return {
                    'success': False,
                    'error': 'Invalid time format. Use "YYYY-MM-DD HH:MM:SS" or "HH:MM:SS"'
}
        else:
            dt = datetime.now(from_tz)
        
        # Convert to target timezone
        converted_dt = dt.astimezone(to_tz)
        
        return {
            'success': True,
            'converted_time': converted_dt.strftime(output_format),
            'from_timezone': from_timezone,
            'to_timezone': to_timezone,
            'original_time': dt.strftime(output_format)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def list_timezones_tool(params: dict) -> dict:
    """
    List available timezones.
    
    params = {
        'filter': str (optional, filter timezones by keyword)
    }
    """
    try:
        filter_keyword = params.get('filter', '').lower()
        all_timezones = pytz.all_timezones
        
        if filter_keyword:
            filtered = [tz for tz in all_timezones if filter_keyword in tz.lower()]
            return {
                'success': True,
                'timezones': filtered,
                'count': len(filtered)
            }
        
        return {
            'success': True,
            'timezones': all_timezones,
            'count': len(all_timezones)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def format_time_tool(params: dict) -> dict:
    """
    Format time string to different formats.
    
    params = {
        'time': str (required, format: 'YYYY-MM-DD HH:MM:SS'),
        'input_format': str (optional, defaults to '%Y-%m-%d %H:%M:%S'),
        'output_format': str (required)
    }
    """
    try:
        time_str = params.get('time')
        if not time_str:
            return {
                'success': False,
                'error': 'time is required'
            }
        
        output_format = params.get('output_format')
        if not output_format:
            return {
                'success': False,
                'error': 'output_format is required'
            }
        
        input_format = params.get('input_format', '%Y-%m-%d %H:%M:%S')
        
        try:
            dt = datetime.strptime(time_str, input_format)
            formatted_time = dt.strftime(output_format)
            
            return {
                'success': True,
                'formatted_time': formatted_time,
                'original_time': time_str
            }
        except ValueError as e:
            return {
                'success': False,
                'error': f'Time format error: {str(e)}'
            }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }