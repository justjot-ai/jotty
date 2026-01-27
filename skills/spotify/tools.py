"""
Spotify Skill

Control Spotify playback and manage playlists using the Spotify Web API via requests.
"""
import os
import requests
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    # Try to load .env from Jotty root (parent of skills directory)
    current_file = Path(__file__).resolve()
    jotty_root = current_file.parent.parent.parent  # skills/spotify -> skills -> Jotty
    env_file = jotty_root / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)  # Don't override existing env vars
except ImportError:
    pass  # python-dotenv not available, fall back to os.getenv

logger = logging.getLogger(__name__)

SPOTIFY_API_BASE = "https://api.spotify.com/v1/"


class SpotifyAPIClient:
    """Helper class for Spotify API interactions."""

    def __init__(self, token: Optional[str] = None):
        self.token = token or self._get_token()

    def _get_token(self) -> Optional[str]:
        """Get Spotify token from environment or config file."""
        # Try environment variable first
        token = os.getenv('SPOTIFY_ACCESS_TOKEN')
        if token:
            return token

        # Try config file
        config_path = Path.home() / ".config" / "spotify" / "token"
        if config_path.exists():
            return config_path.read_text().strip()

        return None

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

    def _make_request(self, endpoint: str, method: str = "GET",
                      json_data: Optional[Dict] = None,
                      params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make a request to Spotify API."""
        url = f"{SPOTIFY_API_BASE}{endpoint}"

        try:
            if method == "GET":
                response = requests.get(url, headers=self._get_headers(), params=params, timeout=30)
            elif method == "POST":
                response = requests.post(url, headers=self._get_headers(), json=json_data, params=params, timeout=30)
            elif method == "PUT":
                response = requests.put(url, headers=self._get_headers(), json=json_data, params=params, timeout=30)
            elif method == "DELETE":
                response = requests.delete(url, headers=self._get_headers(), json=json_data, params=params, timeout=30)
            else:
                return {'success': False, 'error': f'Unsupported HTTP method: {method}'}

            # Handle empty responses (204 No Content)
            if response.status_code == 204:
                return {'success': True}

            # Handle error responses
            if not response.ok:
                error_data = response.json() if response.content else {}
                error_msg = error_data.get('error', {}).get('message', response.reason)
                return {
                    'success': False,
                    'error': error_msg,
                    'status_code': response.status_code
                }

            # Handle successful responses with content
            if response.content:
                result = response.json()
                return {'success': True, **result}

            return {'success': True}

        except requests.exceptions.RequestException as e:
            logger.error(f"Spotify API request failed: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}


def search_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search for tracks, albums, artists, or playlists on Spotify.

    Args:
        params: Dictionary containing:
            - query (str, required): Search query string
            - type (str, optional): Type to search for - 'track', 'album', 'artist', 'playlist' (default: 'track')
            - limit (int, optional): Maximum number of results to return (default: 10, max: 50)
            - token (str, optional): Spotify access token (defaults to SPOTIFY_ACCESS_TOKEN env var)

    Returns:
        Dictionary with:
            - success (bool): Whether search succeeded
            - items (list): List of search results
            - total (int): Total number of results
            - error (str, optional): Error message if failed
    """
    try:
        query = params.get('query')
        search_type = params.get('type', 'track')
        limit = min(params.get('limit', 10), 50)

        if not query:
            return {'success': False, 'error': 'query parameter is required'}

        valid_types = ['track', 'album', 'artist', 'playlist']
        if search_type not in valid_types:
            return {'success': False, 'error': f'type must be one of: {", ".join(valid_types)}'}

        client = SpotifyAPIClient(params.get('token'))

        if not client.token:
            return {
                'success': False,
                'error': 'Spotify token required. Set SPOTIFY_ACCESS_TOKEN env var or provide token parameter'
            }

        search_params = {
            'q': query,
            'type': search_type,
            'limit': limit
        }

        logger.info(f"Searching Spotify for {search_type}: {query}")
        result = client._make_request('search', method="GET", params=search_params)

        if result.get('success'):
            # Extract items from the appropriate key (tracks, albums, artists, playlists)
            result_key = f"{search_type}s"
            items_data = result.get(result_key, {})
            items = items_data.get('items', [])

            # Format results based on type
            formatted_items = []
            for item in items:
                formatted_item = {
                    'id': item.get('id'),
                    'name': item.get('name'),
                    'uri': item.get('uri'),
                    'external_url': item.get('external_urls', {}).get('spotify')
                }

                if search_type == 'track':
                    formatted_item['artists'] = [a.get('name') for a in item.get('artists', [])]
                    formatted_item['album'] = item.get('album', {}).get('name')
                    formatted_item['duration_ms'] = item.get('duration_ms')
                elif search_type == 'album':
                    formatted_item['artists'] = [a.get('name') for a in item.get('artists', [])]
                    formatted_item['release_date'] = item.get('release_date')
                    formatted_item['total_tracks'] = item.get('total_tracks')
                elif search_type == 'artist':
                    formatted_item['genres'] = item.get('genres', [])
                    formatted_item['followers'] = item.get('followers', {}).get('total', 0)
                    formatted_item['popularity'] = item.get('popularity')
                elif search_type == 'playlist':
                    formatted_item['owner'] = item.get('owner', {}).get('display_name')
                    formatted_item['tracks_total'] = item.get('tracks', {}).get('total', 0)
                    formatted_item['description'] = item.get('description')

                formatted_items.append(formatted_item)

            return {
                'success': True,
                'type': search_type,
                'items': formatted_items,
                'total': items_data.get('total', len(formatted_items)),
                'query': query
            }

        return result

    except Exception as e:
        logger.error(f"Spotify search error: {e}", exc_info=True)
        return {'success': False, 'error': f'Failed to search Spotify: {str(e)}'}


def get_current_playback_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get information about the user's current playback state.

    Args:
        params: Dictionary containing:
            - token (str, optional): Spotify access token (defaults to SPOTIFY_ACCESS_TOKEN env var)

    Returns:
        Dictionary with:
            - success (bool): Whether request succeeded
            - is_playing (bool): Whether playback is active
            - track (dict): Currently playing track info
            - device (dict): Active device info
            - progress_ms (int): Progress into the track in milliseconds
            - error (str, optional): Error message if failed
    """
    try:
        client = SpotifyAPIClient(params.get('token'))

        if not client.token:
            return {
                'success': False,
                'error': 'Spotify token required. Set SPOTIFY_ACCESS_TOKEN env var or provide token parameter'
            }

        logger.info("Getting current Spotify playback")
        result = client._make_request('me/player', method="GET")

        # No active device/playback
        if not result.get('success') and result.get('status_code') == 204:
            return {
                'success': True,
                'is_playing': False,
                'track': None,
                'device': None,
                'message': 'No active playback'
            }

        if result.get('success'):
            item = result.get('item', {})
            device = result.get('device', {})

            track_info = None
            if item:
                track_info = {
                    'id': item.get('id'),
                    'name': item.get('name'),
                    'uri': item.get('uri'),
                    'artists': [a.get('name') for a in item.get('artists', [])],
                    'album': item.get('album', {}).get('name'),
                    'duration_ms': item.get('duration_ms'),
                    'external_url': item.get('external_urls', {}).get('spotify')
                }

            device_info = None
            if device:
                device_info = {
                    'id': device.get('id'),
                    'name': device.get('name'),
                    'type': device.get('type'),
                    'volume_percent': device.get('volume_percent'),
                    'is_active': device.get('is_active')
                }

            return {
                'success': True,
                'is_playing': result.get('is_playing', False),
                'track': track_info,
                'device': device_info,
                'progress_ms': result.get('progress_ms'),
                'shuffle_state': result.get('shuffle_state'),
                'repeat_state': result.get('repeat_state')
            }

        return result

    except Exception as e:
        logger.error(f"Spotify get playback error: {e}", exc_info=True)
        return {'success': False, 'error': f'Failed to get Spotify playback: {str(e)}'}


def play_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Start or resume playback on Spotify.

    Args:
        params: Dictionary containing:
            - uri (str, optional): Spotify URI to play (track, album, artist, or playlist)
            - device_id (str, optional): Device ID to play on
            - token (str, optional): Spotify access token (defaults to SPOTIFY_ACCESS_TOKEN env var)

    Returns:
        Dictionary with:
            - success (bool): Whether playback started
            - error (str, optional): Error message if failed
    """
    try:
        client = SpotifyAPIClient(params.get('token'))

        if not client.token:
            return {
                'success': False,
                'error': 'Spotify token required. Set SPOTIFY_ACCESS_TOKEN env var or provide token parameter'
            }

        uri = params.get('uri')
        device_id = params.get('device_id')

        query_params = {}
        if device_id:
            query_params['device_id'] = device_id

        json_data = None
        if uri:
            # Determine if it's a track or context (album/playlist/artist)
            if uri.startswith('spotify:track:'):
                json_data = {'uris': [uri]}
            else:
                json_data = {'context_uri': uri}

        logger.info(f"Starting Spotify playback{' for ' + uri if uri else ''}")
        result = client._make_request('me/player/play', method="PUT",
                                      json_data=json_data, params=query_params if query_params else None)

        if result.get('success'):
            return {
                'success': True,
                'message': 'Playback started' if uri else 'Playback resumed',
                'uri': uri
            }

        return result

    except Exception as e:
        logger.error(f"Spotify play error: {e}", exc_info=True)
        return {'success': False, 'error': f'Failed to start Spotify playback: {str(e)}'}


def pause_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pause playback on Spotify.

    Args:
        params: Dictionary containing:
            - device_id (str, optional): Device ID to pause on
            - token (str, optional): Spotify access token (defaults to SPOTIFY_ACCESS_TOKEN env var)

    Returns:
        Dictionary with:
            - success (bool): Whether playback was paused
            - error (str, optional): Error message if failed
    """
    try:
        client = SpotifyAPIClient(params.get('token'))

        if not client.token:
            return {
                'success': False,
                'error': 'Spotify token required. Set SPOTIFY_ACCESS_TOKEN env var or provide token parameter'
            }

        device_id = params.get('device_id')
        query_params = {}
        if device_id:
            query_params['device_id'] = device_id

        logger.info("Pausing Spotify playback")
        result = client._make_request('me/player/pause', method="PUT",
                                      params=query_params if query_params else None)

        if result.get('success'):
            return {
                'success': True,
                'message': 'Playback paused'
            }

        return result

    except Exception as e:
        logger.error(f"Spotify pause error: {e}", exc_info=True)
        return {'success': False, 'error': f'Failed to pause Spotify playback: {str(e)}'}


def next_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Skip to the next track in the queue.

    Args:
        params: Dictionary containing:
            - device_id (str, optional): Device ID
            - token (str, optional): Spotify access token (defaults to SPOTIFY_ACCESS_TOKEN env var)

    Returns:
        Dictionary with:
            - success (bool): Whether skip succeeded
            - error (str, optional): Error message if failed
    """
    try:
        client = SpotifyAPIClient(params.get('token'))

        if not client.token:
            return {
                'success': False,
                'error': 'Spotify token required. Set SPOTIFY_ACCESS_TOKEN env var or provide token parameter'
            }

        device_id = params.get('device_id')
        query_params = {}
        if device_id:
            query_params['device_id'] = device_id

        logger.info("Skipping to next Spotify track")
        result = client._make_request('me/player/next', method="POST",
                                      params=query_params if query_params else None)

        if result.get('success'):
            return {
                'success': True,
                'message': 'Skipped to next track'
            }

        return result

    except Exception as e:
        logger.error(f"Spotify next track error: {e}", exc_info=True)
        return {'success': False, 'error': f'Failed to skip to next track: {str(e)}'}


def previous_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Skip to the previous track.

    Args:
        params: Dictionary containing:
            - device_id (str, optional): Device ID
            - token (str, optional): Spotify access token (defaults to SPOTIFY_ACCESS_TOKEN env var)

    Returns:
        Dictionary with:
            - success (bool): Whether skip succeeded
            - error (str, optional): Error message if failed
    """
    try:
        client = SpotifyAPIClient(params.get('token'))

        if not client.token:
            return {
                'success': False,
                'error': 'Spotify token required. Set SPOTIFY_ACCESS_TOKEN env var or provide token parameter'
            }

        device_id = params.get('device_id')
        query_params = {}
        if device_id:
            query_params['device_id'] = device_id

        logger.info("Skipping to previous Spotify track")
        result = client._make_request('me/player/previous', method="POST",
                                      params=query_params if query_params else None)

        if result.get('success'):
            return {
                'success': True,
                'message': 'Skipped to previous track'
            }

        return result

    except Exception as e:
        logger.error(f"Spotify previous track error: {e}", exc_info=True)
        return {'success': False, 'error': f'Failed to skip to previous track: {str(e)}'}


def get_playlists_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the current user's playlists.

    Args:
        params: Dictionary containing:
            - limit (int, optional): Maximum number of playlists to return (default: 20, max: 50)
            - offset (int, optional): Index of first playlist to return (default: 0)
            - token (str, optional): Spotify access token (defaults to SPOTIFY_ACCESS_TOKEN env var)

    Returns:
        Dictionary with:
            - success (bool): Whether request succeeded
            - playlists (list): List of playlist objects
            - total (int): Total number of playlists
            - error (str, optional): Error message if failed
    """
    try:
        client = SpotifyAPIClient(params.get('token'))

        if not client.token:
            return {
                'success': False,
                'error': 'Spotify token required. Set SPOTIFY_ACCESS_TOKEN env var or provide token parameter'
            }

        limit = min(params.get('limit', 20), 50)
        offset = params.get('offset', 0)

        query_params = {
            'limit': limit,
            'offset': offset
        }

        logger.info("Getting Spotify playlists")
        result = client._make_request('me/playlists', method="GET", params=query_params)

        if result.get('success'):
            items = result.get('items', [])
            playlists = [
                {
                    'id': p.get('id'),
                    'name': p.get('name'),
                    'uri': p.get('uri'),
                    'description': p.get('description'),
                    'owner': p.get('owner', {}).get('display_name'),
                    'tracks_total': p.get('tracks', {}).get('total', 0),
                    'public': p.get('public'),
                    'collaborative': p.get('collaborative'),
                    'external_url': p.get('external_urls', {}).get('spotify')
                }
                for p in items
            ]

            return {
                'success': True,
                'playlists': playlists,
                'total': result.get('total', len(playlists)),
                'offset': offset,
                'limit': limit
            }

        return result

    except Exception as e:
        logger.error(f"Spotify get playlists error: {e}", exc_info=True)
        return {'success': False, 'error': f'Failed to get Spotify playlists: {str(e)}'}


def add_to_playlist_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a track to a playlist.

    Args:
        params: Dictionary containing:
            - playlist_id (str, required): Spotify playlist ID
            - uri (str, required): Spotify track URI to add (e.g., 'spotify:track:4iV5W9uYEdYUVa79Axb7Rh')
            - position (int, optional): Position to insert track (0-indexed, default: end of playlist)
            - token (str, optional): Spotify access token (defaults to SPOTIFY_ACCESS_TOKEN env var)

    Returns:
        Dictionary with:
            - success (bool): Whether track was added
            - snapshot_id (str): Playlist snapshot ID
            - error (str, optional): Error message if failed
    """
    try:
        playlist_id = params.get('playlist_id')
        uri = params.get('uri')

        if not playlist_id:
            return {'success': False, 'error': 'playlist_id parameter is required'}

        if not uri:
            return {'success': False, 'error': 'uri parameter is required'}

        client = SpotifyAPIClient(params.get('token'))

        if not client.token:
            return {
                'success': False,
                'error': 'Spotify token required. Set SPOTIFY_ACCESS_TOKEN env var or provide token parameter'
            }

        json_data = {
            'uris': [uri] if isinstance(uri, str) else uri
        }

        if params.get('position') is not None:
            json_data['position'] = params['position']

        logger.info(f"Adding track to Spotify playlist {playlist_id}")
        result = client._make_request(f'playlists/{playlist_id}/tracks', method="POST", json_data=json_data)

        if result.get('success'):
            return {
                'success': True,
                'message': 'Track added to playlist',
                'snapshot_id': result.get('snapshot_id'),
                'playlist_id': playlist_id,
                'uri': uri
            }

        return result

    except Exception as e:
        logger.error(f"Spotify add to playlist error: {e}", exc_info=True)
        return {'success': False, 'error': f'Failed to add track to playlist: {str(e)}'}


__all__ = [
    'search_tool',
    'get_current_playback_tool',
    'play_tool',
    'pause_tool',
    'next_tool',
    'previous_tool',
    'get_playlists_tool',
    'add_to_playlist_tool'
]
