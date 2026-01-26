# Clawd.bot Skills Analysis & Jotty Bare Minimum

## Overview
Analysis of clawd.bot's skills ecosystem to identify bare minimum essential skills for Jotty.

## Clawd.bot Skills (from GitHub)

### Communication & Messaging
- `discord/` - Discord integration
- `slack/` - Slack integration
- `imsg/` - iMessage integration
- `bluebubbles/` - BlueBubbles integration
- `wacli/` - WhatsApp CLI
- `himalaya/` - Email client

### Notes & Knowledge
- `apple-notes/` - Apple Notes
- `apple-reminders/` - Apple Reminders
- `bear-notes/` - Bear notes
- `obsidian/` - Obsidian vault
- `notion/` - Notion workspace
- `things-mac/` - Things task manager
- `trello/` - Trello boards

### Media & Content
- `openai-image-gen/` - Image generation
- `openai-whisper/` - Speech-to-text
- `openai-whisper-api/` - Whisper API
- `sherpa-onnx-tts/` - Text-to-speech
- `video-frames/` - Video frame extraction
- `songsee/` - Music discovery
- `spotify-player/` - Spotify control
- `sonoscli/` - Sonos control
- `gifgrep/` - GIF search

### Development & Code
- `github/` - GitHub integration
- `coding-agent/` - Code generation
- `tmux/` - Terminal multiplexer
- `blucli/` - Blue CLI

### System & Automation
- `camsnap/` - Camera snapshots
- `canvas/` - Visual canvas
- `peekaboo/` - Screen peek
- `nano-banana-pro/` - Nano banana
- `nano-pdf/` - PDF processing
- `summarize/` - Content summarization
- `session-logs/` - Session logging
- `model-usage/` - Model usage tracking

### External Services
- `weather/` - Weather data
- `goplaces/` - Places search
- `local-places/` - Local places
- `food-order/` - Food ordering
- `ordercli/` - Order CLI
- `openhue/` - Philips Hue
- `eightctl/` - Eight sleep
- `mcporter/` - MC Porter
- `sag/` - SAG service
- `gog/` - GOG integration
- `bird/` - Bird service
- `blogwatcher/` - Blog watching

### Meta Skills
- `clawdhub/` - Skill registry
- `skill-creator/` - Skill generation
- `oracle/` - Oracle service
- `gemini/` - Gemini integration

## Bare Minimum Essential Skills for Jotty

### ✅ Already Implemented
1. **image-generator** - Image generation (open-source models)
2. **time-converter** - Timezone conversion
3. **weather-checker** - Weather data

### 🔴 Missing Essentials

#### 1. File Operations (CRITICAL)
- Read files
- Write files
- List directory contents
- Create directories
- Delete files/directories
- File search
- File metadata (size, modified date)

#### 2. Web Search (CRITICAL)
- Search the web
- Get web page content
- Extract text from URLs

#### 3. Calculator (ESSENTIAL)
- Basic math operations
- Scientific calculations
- Unit conversions

#### 4. Text Utilities (ESSENTIAL)
- Text encoding/decoding
- Text formatting
- String manipulation
- Text extraction from various formats

#### 5. Shell Execution (CRITICAL)
- Execute shell commands
- Run scripts
- Command output capture

#### 6. Process Management (ESSENTIAL)
- List running processes
- Kill processes
- Process status

#### 7. HTTP/API (ESSENTIAL)
- HTTP GET/POST requests
- API calls
- Response parsing

## Priority Order

### Phase 1: Core Functionality (Bare Minimum)
1. **file-operations** - File system access
2. **web-search** - Information retrieval
3. **calculator** - Math operations
4. **shell-exec** - Command execution

### Phase 2: Enhanced Functionality
5. **text-utils** - Text manipulation
6. **process-manager** - Process control
7. **http-client** - HTTP requests

### Phase 3: Advanced Features
8. **email-basic** - Email operations (if needed)
9. **calendar-basic** - Calendar operations (if needed)

## Implementation Notes

- All skills should follow Jotty's skill structure:
  - `SKILL.md` - Skill metadata
  - `tools.py` - Tool implementations
- Use venv for dependencies
- Auto-install dependencies on load
- Support same-run usage after generation

## References
- Clawd.bot GitHub: https://github.com/clawdbot/clawdbot/tree/main/skills
- Clawd.bot Website: https://clawd.bot
- Clawd.bot Docs: https://docs.clawd.bot
