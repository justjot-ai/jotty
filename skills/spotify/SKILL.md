---
name: spotify
description: "This skill provides tools to search for music, control playback, and manage playlists using the Spotify Web API via requests. Use when the user wants to music, playlist."
---

# Spotify Skill

Control Spotify playback and manage playlists using the Spotify Web API.

## Description

This skill provides tools to search for music, control playback, and manage playlists using the Spotify Web API via requests.


## Type
base


## Capabilities
- media

## Features

- Search for tracks, albums, artists, and playlists
- Get current playback status
- Control playback (play, pause, next, previous)
- Get user's playlists
- Add tracks to playlists

## Usage

```python
from skills.spotify.tools import (
    search_tool,
    get_current_playback_tool,
    play_tool,
    pause_tool,
    next_tool,
    previous_tool,
    get_playlists_tool,
    add_to_playlist_tool
)

# Search for tracks
result = search_tool({
    'query': 'Bohemian Rhapsody',
    'type': 'track',
    'limit': 5
})

# Get current playback
result = get_current_playback_tool({})

# Play a specific track
result = play_tool({
    'uri': 'spotify:track:4u7EnebtmKWzUH433cf5Qv'
})

# Resume playback
result = play_tool({})

# Pause playback
result = pause_tool({})

# Skip to next track
result = next_tool({})

# Go to previous track
result = previous_tool({})

# Get user's playlists
result = get_playlists_tool({
    'limit': 20
})

# Add track to playlist
result = add_to_playlist_tool({
    'playlist_id': '37i9dQZF1DXcBWIGoYBM5M',
    'uri': 'spotify:track:4u7EnebtmKWzUH433cf5Qv'
})
```

## Tools

### search_tool

Search for tracks, albums, artists, or playlists on Spotify.

**Parameters:**
- `query` (str, required): Search query string
- `type` (str, optional): Type to search for - 'track', 'album', 'artist', 'playlist' (default: 'track')
- `limit` (int, optional): Maximum number of results (default: 10, max: 50)
- `token` (str, optional): Spotify access token

### get_current_playback_tool

Get information about the user's current playback state.

**Parameters:**
- `token` (str, optional): Spotify access token

**Returns:**
- `is_playing` (bool): Whether playback is active
- `track` (dict): Currently playing track info
- `device` (dict): Active device info
- `progress_ms` (int): Progress into the track

### play_tool

Start or resume playback on Spotify.

**Parameters:**
- `uri` (str, optional): Spotify URI to play (track, album, artist, or playlist)
- `device_id` (str, optional): Device ID to play on
- `token` (str, optional): Spotify access token

### pause_tool

Pause playback on Spotify.

**Parameters:**
- `device_id` (str, optional): Device ID to pause on
- `token` (str, optional): Spotify access token

### next_tool

Skip to the next track in the queue.

**Parameters:**
- `device_id` (str, optional): Device ID
- `token` (str, optional): Spotify access token

### previous_tool

Skip to the previous track.

**Parameters:**
- `device_id` (str, optional): Device ID
- `token` (str, optional): Spotify access token

### get_playlists_tool

Get the current user's playlists.

**Parameters:**
- `limit` (int, optional): Maximum playlists to return (default: 20, max: 50)
- `offset` (int, optional): Index of first playlist to return (default: 0)
- `token` (str, optional): Spotify access token

### add_to_playlist_tool

Add a track to a playlist.

**Parameters:**
- `playlist_id` (str, required): Spotify playlist ID
- `uri` (str, required): Spotify track URI to add
- `position` (int, optional): Position to insert track (0-indexed)
- `token` (str, optional): Spotify access token

## Configuration

Set the Spotify access token via:

1. Environment variable: `SPOTIFY_ACCESS_TOKEN`
2. Config file: `~/.config/spotify/token`

### OAuth Authentication

Spotify requires OAuth 2.0 authentication. The access token needs to be refreshed periodically (expires after 1 hour).

To obtain an access token:

1. Register your app at https://developer.spotify.com/dashboard
2. Get your Client ID and Client Secret
3. Implement OAuth 2.0 Authorization Code Flow or use Spotify's PKCE flow
4. Exchange the authorization code for an access token

### Required OAuth Scopes

Ensure your Spotify app requests these OAuth scopes:

- `user-read-playback-state` - Read playback state
- `user-modify-playback-state` - Control playback
- `user-read-currently-playing` - Read currently playing
- `playlist-read-private` - Read private playlists
- `playlist-read-collaborative` - Read collaborative playlists
- `playlist-modify-public` - Modify public playlists
- `playlist-modify-private` - Modify private playlists

## API Reference

Base URL: https://api.spotify.com/v1/

This skill uses the Spotify Web API with the following endpoints:
- `GET /search` - Search for items
- `GET /me/player` - Get playback state
- `PUT /me/player/play` - Start/resume playback
- `PUT /me/player/pause` - Pause playback
- `POST /me/player/next` - Skip to next track
- `POST /me/player/previous` - Skip to previous track
- `GET /me/playlists` - Get user's playlists
- `POST /playlists/{playlist_id}/tracks` - Add tracks to playlist

## Notes

- Spotify requires a Premium account for playback control features
- Access tokens expire after 1 hour and need to be refreshed
- Some endpoints may not work without an active Spotify device

## Triggers
- "spotify"
- "music"
- "playlist"

## Category
workflow-automation
