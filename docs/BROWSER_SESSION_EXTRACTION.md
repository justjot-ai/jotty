# Browser Session/Cookie Extraction Standards

## Overview

This document covers standard methods for extracting browser sessions and cookies, which are essential for headless browser automation in Docker environments.

## Standard Formats

### 1. Browser DevTools Protocol (CDP)

**Standard**: Used by Chrome, Edge, Opera, and Chromium-based browsers

CDP is the **most reliable and standard method** for extracting cookies as it uses the browser's native API.

```python
# Using Playwright (implements CDP)
from playwright.async_api import async_playwright

async with async_playwright() as p:
    browser = await p.chromium.launch_persistent_context(
        user_data_dir=profile_dir,
        headless=True
    )
    page = await browser.new_page()
    await page.goto('https://example.com')
    
    # Get cookies via CDP
    cookies = await page.context.cookies()
```

**Advantages**:
- Standard protocol supported by all Chromium browsers
- Handles encrypted cookies automatically
- Most reliable method

### 2. Netscape Cookie Format

**Standard**: Universal cookie file format

```
# Netscape HTTP Cookie File
domain	flag	path	secure	expiration	name	value
.example.com	TRUE	/	TRUE	1735689600	session	abc123
```

**Usage**: Supported by curl, wget, and most HTTP clients

### 3. JSON Cookie Format

**Standard**: Modern JSON format used by browsers and APIs

```json
[
  {
    "name": "session",
    "value": "abc123",
    "domain": ".example.com",
    "path": "/",
    "expires": 1735689600,
    "secure": true,
    "httpOnly": true,
    "sameSite": "Lax"
  }
]
```

**Usage**: Standard format for REST APIs and modern tools

### 4. Browser Storage Formats

#### Chrome/Chromium (SQLite)
- Location: `{profile}/Default/Cookies`
- Format: SQLite database
- Encryption: Values may be encrypted (need key from Local State)

#### Firefox (JSON/SQLite)
- Location: `{profile}/cookies.json` or `cookies.sqlite`
- Format: JSON (newer) or SQLite (older)
- Encryption: May require master password

## Extraction Methods

### Method 1: CDP (Recommended)

**Best for**: Chrome, Edge, Opera, Playwright profiles

```bash
python3 scripts/extract_browser_session.py \
  --browser chrome \
  --profile-dir ~/.config/google-chrome \
  --method cdp \
  --format json
```

### Method 2: SQLite Direct Access

**Best for**: Chrome/Chromium when CDP unavailable

```bash
python3 scripts/extract_browser_session.py \
  --browser chrome \
  --profile-dir ~/.config/google-chrome \
  --method sqlite \
  --format json
```

### Method 3: Playwright Profile

**Best for**: Playwright-managed profiles (like NotebookLM)

```bash
python3 scripts/extract_browser_session.py \
  --browser playwright \
  --profile-dir ~/.notebooklm_browser \
  --format docker-env
```

## Output Formats

### JSON (Default)
```bash
python3 scripts/extract_browser_session.py \
  --profile-dir ~/.notebooklm_browser \
  --format json \
  --output cookies.json
```

### Netscape (Universal)
```bash
python3 scripts/extract_browser_session.py \
  --profile-dir ~/.notebooklm_browser \
  --format netscape \
  --output cookies.txt
```

### Docker Environment Variable
```bash
python3 scripts/extract_browser_session.py \
  --profile-dir ~/.notebooklm_browser \
  --format docker-env
```

Output:
```bash
export NOTEBOOKLM_COOKIES='[{"name":"session","value":"...","domain":".google.com"}]'
```

### cURL Format
```bash
python3 scripts/extract_browser_session.py \
  --profile-dir ~/.notebooklm_browser \
  --format curl
```

### HAR Format (HTTP Archive)
```bash
python3 scripts/extract_browser_session.py \
  --profile-dir ~/.notebooklm_browser \
  --format har \
  --output cookies.har
```

## Browser-Specific Extraction

### Chrome/Chromium

**Profile Locations**:
- Linux: `~/.config/google-chrome` or `~/.config/chromium`
- macOS: `~/Library/Application Support/Google/Chrome`
- Windows: `%LOCALAPPDATA%\Google\Chrome\User Data`

**Extraction**:
```bash
# Using CDP (recommended)
python3 scripts/extract_browser_session.py \
  --browser chrome \
  --profile-dir ~/.config/google-chrome \
  --method cdp

# Using SQLite (fallback)
python3 scripts/extract_browser_session.py \
  --browser chrome \
  --profile-dir ~/.config/google-chrome \
  --method sqlite
```

### Firefox

**Profile Locations**:
- Linux: `~/.mozilla/firefox/{profile}`
- macOS: `~/Library/Application Support/Firefox/Profiles/{profile}`
- Windows: `%APPDATA%\Mozilla\Firefox\Profiles\{profile}`

**Extraction**:
```bash
python3 scripts/extract_browser_session.py \
  --browser firefox \
  --profile-dir ~/.mozilla/firefox/xxxx.default-release
```

### Edge

**Profile Locations**:
- Linux: `~/.config/microsoft-edge`
- macOS: `~/Library/Application Support/Microsoft Edge`
- Windows: `%LOCALAPPDATA%\Microsoft\Edge\User Data`

**Extraction**:
```bash
python3 scripts/extract_browser_session.py \
  --browser edge \
  --profile-dir ~/.config/microsoft-edge \
  --method cdp
```

## Docker Integration

### Option 1: Mount Browser Profile

```yaml
# docker-compose.yml
volumes:
  - ~/.notebooklm_browser:/app/.notebooklm_browser
environment:
  - NOTEBOOKLM_USER_DATA_DIR=/app/.notebooklm_browser
```

### Option 2: Use Cookies as Environment Variable

```bash
# Extract cookies
python3 scripts/extract_browser_session.py \
  --profile-dir ~/.notebooklm_browser \
  --format docker-env > cookies.env

# In Docker
docker run --env-file cookies.env your-image
```

### Option 3: Use Netscape Format

```bash
# Extract to Netscape format
python3 scripts/extract_browser_session.py \
  --profile-dir ~/.notebooklm_browser \
  --format netscape \
  --output cookies.txt

# Use with curl in Docker
curl --cookie cookies.txt https://notebooklm.google.com
```

## Standards Reference

- **CDP**: [Chrome DevTools Protocol](https://chromedevtools.github.io/devtools-protocol/)
- **Netscape Format**: [RFC 2109](https://tools.ietf.org/html/rfc2109) (deprecated but widely supported)
- **JSON Cookies**: [RFC 6265](https://tools.ietf.org/html/rfc6265) (HTTP State Management)
- **HAR Format**: [W3C HAR Specification](https://w3c.github.io/web-performance/specs/HAR/Overview.html)

## Best Practices

1. **Use CDP when possible** - Most reliable and standard
2. **Prefer JSON format** - Modern, widely supported
3. **Filter by domain** - Extract only needed cookies
4. **Secure storage** - Cookies contain authentication tokens
5. **Rotate regularly** - Cookies expire, refresh periodically

## Troubleshooting

### "No cookies found"
- Check profile directory path
- Ensure browser is closed when extracting
- Try different extraction method (CDP vs SQLite)

### "Encrypted cookies"
- Use CDP method (handles encryption automatically)
- Or decrypt using browser's key from Local State

### "Cookies expired"
- Re-authenticate and extract fresh cookies
- Check expiration timestamps
