# NotebookLM PDF Generator - Docker/Headless Setup

## Overview

For Docker environments without a display, NotebookLM authentication requires special handling since you can't interactively sign in. This guide covers several methods to authenticate in headless environments.

## Method 1: Pre-authenticated Browser Profile (Recommended)

### Step 1: Authenticate Outside Docker

On a machine with a display, use the standalone authentication script:

```bash
# On your local machine (with display)
cd /var/www/sites/personal/stock_market/Jotty

# Install Playwright if not already installed
pip install playwright
playwright install chromium

# Run authentication script
python3 scripts/auth_notebooklm.py --profile-dir /tmp/notebooklm_profile
```

The script will:
1. Open a browser window
2. Navigate to NotebookLM
3. Wait for you to sign in manually
4. Save the authenticated browser profile

**Alternative**: Use default location (`~/.notebooklm_browser`):
```bash
python3 scripts/auth_notebooklm.py
```

### Step 2: Copy Profile to Docker Host

```bash
# Copy the authenticated profile
cp -r /tmp/notebooklm_auth /path/to/docker/mount/notebooklm_profile
```

### Step 3: Mount Profile in Docker

```yaml
# docker-compose.yml
services:
  jotty:
    volumes:
      - /path/to/notebooklm_profile:/app/.notebooklm_browser
    environment:
      - NOTEBOOKLM_USER_DATA_DIR=/app/.notebooklm_browser
```

Or with docker run:

```bash
docker run -v /path/to/notebooklm_profile:/app/.notebooklm_browser \
  -e NOTEBOOKLM_USER_DATA_DIR=/app/.notebooklm_browser \
  your-image
```

## Method 2: Using Cookies (Alternative)

### Step 1: Extract Cookies from Browser Profile

After authenticating (using `auth_notebooklm.py`), extract cookies:

```bash
# Extract cookies from authenticated profile
python3 scripts/extract_notebooklm_cookies.py --profile-dir /tmp/notebooklm_profile
```

Or use default profile location:
```bash
python3 scripts/extract_notebooklm_cookies.py
```

This will:
1. Load the authenticated browser profile
2. Extract cookies
3. Save to `cookies.json`
4. Show Docker environment variable command

### Step 2: Use Cookies in Docker

```bash
# Set cookies as environment variable
export NOTEBOOKLM_COOKIES=$(cat notebooklm_cookies.json)

# Or in docker-compose.yml
environment:
  - NOTEBOOKLM_COOKIES={"name":"session","value":"...","domain":".google.com"}
```

## Method 3: Docker Build with Pre-authenticated Profile

### Step 1: Create Authentication Script

```bash
# scripts/auth_notebooklm.sh
#!/bin/bash
python3 << 'EOF'
import asyncio
from pathlib import Path
from skills.notebooklm_pdf.tools import notebooklm_pdf_tool

async def auth():
    profile_dir = Path('/app/.notebooklm_browser')
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    result = await notebooklm_pdf_tool({
        'content': '# Auth\n\nTest',
        'title': 'Auth',
        'headless': False,
        'user_data_dir': str(profile_dir)
    })
    print("Auth complete")

asyncio.run(auth())
EOF
```

### Step 2: Build Docker Image with Profile

```dockerfile
# Dockerfile
FROM your-base-image

# Copy authentication script
COPY scripts/auth_notebooklm.sh /app/scripts/

# Run authentication (requires X11 forwarding or VNC)
# This step requires manual intervention
# RUN /app/scripts/auth_notebooklm.sh

# Or copy pre-authenticated profile
COPY notebooklm_profile /app/.notebooklm_browser

ENV NOTEBOOKLM_USER_DATA_DIR=/app/.notebooklm_browser
```

## Method 4: X11 Forwarding (For Development)

If you have X11 forwarding available:

```bash
# Enable X11 forwarding
docker run -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $HOME/.notebooklm_browser:/app/.notebooklm_browser \
  your-image
```

## Method 5: VNC Server (For Remote Authentication)

Set up VNC in Docker:

```dockerfile
# Install VNC
RUN apt-get update && apt-get install -y \
    xvfb x11vnc fluxbox \
    && rm -rf /var/lib/apt/lists/*

# Start VNC
CMD ["x11vnc", "-create", "-forever"]
```

Then connect via VNC client to authenticate.

## Recommended Approach for Production

For production Docker deployments:

1. **Pre-authenticate** on a development machine
2. **Copy browser profile** to a secure location
3. **Mount as read-only volume** in Docker
4. **Set NOTEBOOKLM_USER_DATA_DIR** environment variable

Example docker-compose.yml:

```yaml
version: '3.8'
services:
  jotty:
    image: your-jotty-image
    volumes:
      - ./notebooklm_profile:/app/.notebooklm_browser:ro  # Read-only
    environment:
      - NOTEBOOKLM_USER_DATA_DIR=/app/.notebooklm_browser
      - HEADLESS=true
    # Fallback to Pandoc if NotebookLM unavailable
    # The skill automatically falls back
```

## Fallback Behavior

If authentication fails in Docker, the skill automatically falls back to Pandoc conversion. This ensures PDFs are always generated, though without NotebookLM's AI features.

## Troubleshooting

### "Browser profile not found"

**Solution**: Mount the pre-authenticated profile directory.

### "NotebookLM requires sign-in"

**Solution**: 
1. Authenticate outside Docker
2. Copy browser profile
3. Mount as volume

### Cookies expired

**Solution**: Re-authenticate and update cookies or browser profile.

## Security Notes

- Browser profiles contain authentication tokens - keep them secure
- Use read-only mounts when possible
- Rotate profiles periodically
- Don't commit profiles to git

## Testing Authentication

```python
from skills.notebooklm_pdf.tools import notebooklm_pdf_tool

# Test if authentication works
result = await notebooklm_pdf_tool({
    'content': '# Test\n\nTest content',
    'title': 'Auth Test',
    'headless': True
})

if result.get('success') and result.get('method') == 'browser_automation':
    print("✅ Authentication working!")
else:
    print(f"❌ Auth failed: {result.get('error')}")
    print("Falling back to Pandoc...")
```
