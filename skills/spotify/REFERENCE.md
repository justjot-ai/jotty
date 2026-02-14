# Spotify Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`search_tool`](#search_tool) | Search for tracks, albums, artists, or playlists on Spotify. |
| [`get_current_playback_tool`](#get_current_playback_tool) | Get information about the user's current playback state. |
| [`play_tool`](#play_tool) | Start or resume playback on Spotify. |
| [`pause_tool`](#pause_tool) | Pause playback on Spotify. |
| [`next_tool`](#next_tool) | Skip to the next track in the queue. |
| [`previous_tool`](#previous_tool) | Skip to the previous track. |
| [`get_playlists_tool`](#get_playlists_tool) | Get the current user's playlists. |
| [`add_to_playlist_tool`](#add_to_playlist_tool) | Add a track to a playlist. |

---

## `search_tool`

Search for tracks, albums, artists, or playlists on Spotify.

**Parameters:**

- **query** (`str, required`): Search query string
- **type** (`str, optional`): 'track', 'album', 'artist', 'playlist' (default: 'track')
- **limit** (`int, optional`): Max results (default: 10, max: 50)
- **token** (`str, optional`): Spotify access token

**Returns:** Dictionary with success, type, items, total, query

---

## `get_current_playback_tool`

Get information about the user's current playback state.

**Parameters:**

- **token** (`str, optional`): Spotify access token

**Returns:** Dictionary with success, is_playing, track, device, progress_ms

---

## `play_tool`

Start or resume playback on Spotify.

**Parameters:**

- **uri** (`str, optional`): Spotify URI to play
- **device_id** (`str, optional`): Device ID to play on
- **token** (`str, optional`): Spotify access token

**Returns:** Dictionary with success, message, uri

---

## `pause_tool`

Pause playback on Spotify.

**Parameters:**

- **device_id** (`str, optional`): Device ID
- **token** (`str, optional`): Spotify access token

**Returns:** Dictionary with success, message

---

## `next_tool`

Skip to the next track in the queue.

**Parameters:**

- **device_id** (`str, optional`): Device ID
- **token** (`str, optional`): Spotify access token

**Returns:** Dictionary with success, message

---

## `previous_tool`

Skip to the previous track.

**Parameters:**

- **device_id** (`str, optional`): Device ID
- **token** (`str, optional`): Spotify access token

**Returns:** Dictionary with success, message

---

## `get_playlists_tool`

Get the current user's playlists.

**Parameters:**

- **limit** (`int, optional`): Max playlists (default: 20, max: 50)
- **offset** (`int, optional`): Index of first playlist (default: 0)
- **token** (`str, optional`): Spotify access token

**Returns:** Dictionary with success, playlists, total, offset, limit

---

## `add_to_playlist_tool`

Add a track to a playlist.

**Parameters:**

- **playlist_id** (`str, required`): Spotify playlist ID
- **uri** (`str, required`): Spotify track URI
- **position** (`int, optional`): Position to insert (0-indexed)
- **token** (`str, optional`): Spotify access token

**Returns:** Dictionary with success, message, snapshot_id, playlist_id, uri
