# NotebookLM PDF Generator - Docker/Headless Setup

## Overview

For Docker environments without a display, NotebookLM authentication requires special handling since you can't interactively sign in. This guide covers several methods to authenticate in headless environments.

## Method 1: Pre-authenticated Browser Profile (Recommended)

### Step 1: Authenticate Outside Docker

On a machine with a display, authenticate once:

```bash
# On your local machine (with display)
cd /var/www/sites/personal/stock_market/Jotty
python3 << 'EOF'
import asyncio
from skills.notebooklm_pdf.tools import notebooklm_pdf_tool

async def authenticate():
    result = await notebooklm_pdf_tool({
        'content': '# Test\n\nTest',
        'title': 'Auth Test',
        'headless': False,  # Show browser
        'user_data_dir': '/tmp/notebooklm_auth'  # Custom location
    })
    print("Authentication complete! Profile saved to /tmp/notebooklm_auth")

asyncio.run(authenticate())
EOF
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

### Step 1: Extract Cookies from Browser

On a machine with a display, extract cookies after signing in:

```python
# extract_cookies.py
from playwright.sync_api import sync_playwright
import json

with sync_playwright() as p:
    browser = p.chromium.launch_persistent_context(
        user_data_dir='/tmp/notebooklm_auth',
        headless=False
    )
    page = browser.pages[0]
    page.goto('https://notebooklm.google.com')
    input("Sign in, then press Enter...")
    
    cookies = page.context.cookies()
    print(json.dumps(cookies, indent=2))
    
    # Save to file
    with open('notebooklm_cookies.json', 'w') as f:
        json.dump(cookies, f, indent=2)
    
    browser.close()
```

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
