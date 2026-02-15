"""
Spotify Skill

Control Spotify playback and manage playlists using the Spotify Web API.
Refactored to use Jotty core utilities.
"""

import logging
from typing import Any, Dict, List

from Jotty.core.infrastructure.utils.api_client import BaseAPIClient
from Jotty.core.infrastructure.utils.env_loader import load_jotty_env
from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

# Status emitter for progress updates
status = SkillStatus("spotify")


load_jotty_env()

logger = logging.getLogger(__name__)


class SpotifyAPIClient(BaseAPIClient):
    """Spotify API client using base utilities."""

    BASE_URL = "https://api.spotify.com/v1"
    AUTH_PREFIX = "Bearer"
    TOKEN_ENV_VAR = "SPOTIFY_ACCESS_TOKEN"
    TOKEN_CONFIG_PATH = ".config/spotify/token"


def _get_client(params: Dict[str, Any]) -> tuple:
    """Get Spotify client, returning (client, error) tuple."""
    client = SpotifyAPIClient(params.get("token"))
    if not client.token:
        return None, tool_error(
            "Spotify token required. Set SPOTIFY_ACCESS_TOKEN env var or provide token parameter"
        )
    return client, None


def _format_track(item: Dict) -> Dict:
    """Format track data."""
    return {
        "id": item.get("id"),
        "name": item.get("name"),
        "uri": item.get("uri"),
        "artists": [a.get("name") for a in item.get("artists", [])],
        "album": item.get("album", {}).get("name"),
        "duration_ms": item.get("duration_ms"),
        "external_url": item.get("external_urls", {}).get("spotify"),
    }


@tool_wrapper(required_params=["query"])
def search_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search for tracks, albums, artists, or playlists on Spotify.

    Args:
        params: Dictionary containing:
            - query (str, required): Search query string
            - type (str, optional): 'track', 'album', 'artist', 'playlist' (default: 'track')
            - limit (int, optional): Max results (default: 10, max: 50)
            - token (str, optional): Spotify access token

    Returns:
        Dictionary with success, type, items, total, query
    """
    status.set_callback(params.pop("_status_callback", None))

    search_type = params.get("type", "track")
    valid_types = ["track", "album", "artist", "playlist"]
    if search_type not in valid_types:
        return tool_error(f'type must be one of: {", ".join(valid_types)}')

    client, error = _get_client(params)
    if error:
        return error

    limit = min(params.get("limit", 10), 50)

    logger.info(f"Searching Spotify for {search_type}: {params['query']}")
    result = client._make_request(
        "search", method="GET", params={"q": params["query"], "type": search_type, "limit": limit}
    )

    if not result.get("success"):
        return result

    result_key = f"{search_type}s"
    items_data = result.get(result_key, {})
    items = items_data.get("items", [])

    formatted_items = []
    for item in items:
        formatted = {
            "id": item.get("id"),
            "name": item.get("name"),
            "uri": item.get("uri"),
            "external_url": item.get("external_urls", {}).get("spotify"),
        }

        if search_type == "track":
            formatted["artists"] = [a.get("name") for a in item.get("artists", [])]
            formatted["album"] = item.get("album", {}).get("name")
            formatted["duration_ms"] = item.get("duration_ms")
        elif search_type == "album":
            formatted["artists"] = [a.get("name") for a in item.get("artists", [])]
            formatted["release_date"] = item.get("release_date")
            formatted["total_tracks"] = item.get("total_tracks")
        elif search_type == "artist":
            formatted["genres"] = item.get("genres", [])
            formatted["followers"] = item.get("followers", {}).get("total", 0)
            formatted["popularity"] = item.get("popularity")
        elif search_type == "playlist":
            formatted["owner"] = item.get("owner", {}).get("display_name")
            formatted["tracks_total"] = item.get("tracks", {}).get("total", 0)
            formatted["description"] = item.get("description")

        formatted_items.append(formatted)

    return tool_response(
        type=search_type,
        items=formatted_items,
        total=items_data.get("total", len(formatted_items)),
        query=params["query"],
    )


@tool_wrapper()
def get_current_playback_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get information about the user's current playback state.

    Args:
        params: Dictionary containing:
            - token (str, optional): Spotify access token

    Returns:
        Dictionary with success, is_playing, track, device, progress_ms
    """
    status.set_callback(params.pop("_status_callback", None))

    client, error = _get_client(params)
    if error:
        return error

    logger.info("Getting current Spotify playback")
    result = client._make_request("me/player", method="GET")

    if not result.get("success") and result.get("status_code") == 204:
        return tool_response(
            is_playing=False, track=None, device=None, message="No active playback"
        )

    if not result.get("success"):
        return result

    item = result.get("item", {})
    device = result.get("device", {})

    track_info = _format_track(item) if item else None
    device_info = (
        {
            "id": device.get("id"),
            "name": device.get("name"),
            "type": device.get("type"),
            "volume_percent": device.get("volume_percent"),
            "is_active": device.get("is_active"),
        }
        if device
        else None
    )

    return tool_response(
        is_playing=result.get("is_playing", False),
        track=track_info,
        device=device_info,
        progress_ms=result.get("progress_ms"),
        shuffle_state=result.get("shuffle_state"),
        repeat_state=result.get("repeat_state"),
    )


@tool_wrapper()
def play_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Start or resume playback on Spotify.

    Args:
        params: Dictionary containing:
            - uri (str, optional): Spotify URI to play
            - device_id (str, optional): Device ID to play on
            - token (str, optional): Spotify access token

    Returns:
        Dictionary with success, message, uri
    """
    status.set_callback(params.pop("_status_callback", None))

    client, error = _get_client(params)
    if error:
        return error

    uri = params.get("uri")
    query_params = {"device_id": params["device_id"]} if params.get("device_id") else None

    json_data = None
    if uri:
        if uri.startswith("spotify:track:"):
            json_data = {"uris": [uri]}
        else:
            json_data = {"context_uri": uri}

    logger.info(f"Starting Spotify playback{' for ' + uri if uri else ''}")
    result = client._make_request(
        "me/player/play", method="PUT", json_data=json_data, params=query_params
    )

    if result.get("success"):
        return tool_response(message="Playback started" if uri else "Playback resumed", uri=uri)

    return result


@tool_wrapper()
def pause_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pause playback on Spotify.

    Args:
        params: Dictionary containing:
            - device_id (str, optional): Device ID
            - token (str, optional): Spotify access token

    Returns:
        Dictionary with success, message
    """
    status.set_callback(params.pop("_status_callback", None))

    client, error = _get_client(params)
    if error:
        return error

    query_params = {"device_id": params["device_id"]} if params.get("device_id") else None

    logger.info("Pausing Spotify playback")
    result = client._make_request("me/player/pause", method="PUT", params=query_params)

    if result.get("success"):
        return tool_response(message="Playback paused")

    return result


@tool_wrapper()
def next_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Skip to the next track in the queue.

    Args:
        params: Dictionary containing:
            - device_id (str, optional): Device ID
            - token (str, optional): Spotify access token

    Returns:
        Dictionary with success, message
    """
    status.set_callback(params.pop("_status_callback", None))

    client, error = _get_client(params)
    if error:
        return error

    query_params = {"device_id": params["device_id"]} if params.get("device_id") else None

    logger.info("Skipping to next Spotify track")
    result = client._make_request("me/player/next", method="POST", params=query_params)

    if result.get("success"):
        return tool_response(message="Skipped to next track")

    return result


@tool_wrapper()
def previous_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Skip to the previous track.

    Args:
        params: Dictionary containing:
            - device_id (str, optional): Device ID
            - token (str, optional): Spotify access token

    Returns:
        Dictionary with success, message
    """
    status.set_callback(params.pop("_status_callback", None))

    client, error = _get_client(params)
    if error:
        return error

    query_params = {"device_id": params["device_id"]} if params.get("device_id") else None

    logger.info("Skipping to previous Spotify track")
    result = client._make_request("me/player/previous", method="POST", params=query_params)

    if result.get("success"):
        return tool_response(message="Skipped to previous track")

    return result


@tool_wrapper()
def get_playlists_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the current user's playlists.

    Args:
        params: Dictionary containing:
            - limit (int, optional): Max playlists (default: 20, max: 50)
            - offset (int, optional): Index of first playlist (default: 0)
            - token (str, optional): Spotify access token

    Returns:
        Dictionary with success, playlists, total, offset, limit
    """
    status.set_callback(params.pop("_status_callback", None))

    client, error = _get_client(params)
    if error:
        return error

    limit = min(params.get("limit", 20), 50)
    offset = params.get("offset", 0)

    logger.info("Getting Spotify playlists")
    result = client._make_request(
        "me/playlists", method="GET", params={"limit": limit, "offset": offset}
    )

    if not result.get("success"):
        return result

    playlists = [
        {
            "id": p.get("id"),
            "name": p.get("name"),
            "uri": p.get("uri"),
            "description": p.get("description"),
            "owner": p.get("owner", {}).get("display_name"),
            "tracks_total": p.get("tracks", {}).get("total", 0),
            "public": p.get("public"),
            "collaborative": p.get("collaborative"),
            "external_url": p.get("external_urls", {}).get("spotify"),
        }
        for p in result.get("items", [])
    ]

    return tool_response(
        playlists=playlists, total=result.get("total", len(playlists)), offset=offset, limit=limit
    )


@tool_wrapper(required_params=["playlist_id", "uri"])
def add_to_playlist_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a track to a playlist.

    Args:
        params: Dictionary containing:
            - playlist_id (str, required): Spotify playlist ID
            - uri (str, required): Spotify track URI
            - position (int, optional): Position to insert (0-indexed)
            - token (str, optional): Spotify access token

    Returns:
        Dictionary with success, message, snapshot_id, playlist_id, uri
    """
    status.set_callback(params.pop("_status_callback", None))

    client, error = _get_client(params)
    if error:
        return error

    uri = params["uri"]
    json_data = {"uris": [uri] if isinstance(uri, str) else uri}

    if params.get("position") is not None:
        json_data["position"] = params["position"]

    logger.info(f"Adding track to Spotify playlist {params['playlist_id']}")
    result = client._make_request(
        f"playlists/{params['playlist_id']}/tracks", method="POST", json_data=json_data
    )

    if result.get("success"):
        return tool_response(
            message="Track added to playlist",
            snapshot_id=result.get("snapshot_id"),
            playlist_id=params["playlist_id"],
            uri=uri,
        )

    return result


__all__ = [
    "search_tool",
    "get_current_playback_tool",
    "play_tool",
    "pause_tool",
    "next_tool",
    "previous_tool",
    "get_playlists_tool",
    "add_to_playlist_tool",
]
