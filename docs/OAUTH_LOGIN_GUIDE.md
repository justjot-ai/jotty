# OAuth Login Guide

## Overview

The OAuth automation skill handles Google OAuth login flows, including 2FA with authenticator push notifications.

## Credentials (Private)

- Email: `setia.naveen@gmail.com`
- Password: `N2Setia@india`
- 2FA: Authenticator push notification

**⚠️ These credentials are kept private and never logged or shared.**

## Running OAuth Login

### Option 1: On Machine with Display

```bash
# Set credentials
export OAUTH_EMAIL=setia.naveen@gmail.com
export OAUTH_PASSWORD=N2Setia@india

# Run login (will show browser)
python3 scripts/oauth_login.py \
    --provider google \
    --service-url https://notebooklm.google.com \
    --no-headless
```

### Option 2: Using Test Script

```bash
# Edit scripts/test_oauth_login.sh to set credentials
chmod +x scripts/test_oauth_login.sh
./scripts/test_oauth_login.sh
```

### Option 3: With X11 Forwarding (SSH)

```bash
# Connect with X11 forwarding
ssh -X user@host

# Then run
python3 scripts/oauth_login.py --provider google --no-headless
```

### Option 4: Using xvfb (Virtual Display)

```bash
# Install xvfb
sudo apt-get install xvfb

# Run with virtual display
xvfb-run -a python3 scripts/oauth_login.py \
    --provider google \
    --no-headless
```

## 2FA Handling

The script automatically detects and handles:

1. **Authenticator Push Notifications**
   - Detects push notification prompts
   - Waits up to 60 seconds for approval
   - Shows progress indicators
   - Automatically continues after approval

2. **Manual 2FA Codes**
   - If you have a code, use `--2fa-code` parameter
   - Script will enter the code automatically

## Flow

1. Navigate to NotebookLM
2. Click "Sign in"
3. Enter email: `setia.naveen@gmail.com`
4. Click Next
5. Enter password: `N2Setia@india`
6. Click Next
7. **2FA**: Approve push notification on phone (script waits)
8. Grant permissions if needed
9. Redirect back to NotebookLM
10. Save authenticated profile

## After Authentication

The authenticated browser profile is saved. Use it for:

```bash
# Generate PDFs with authenticated session
python3 scripts/notebooklm_pdf.py \
    --file document.md \
    --profile-dir ~/.oauth_browser \
    --headless
```

## Docker Usage

1. Authenticate outside Docker (with display)
2. Mount profile in Docker:

```yaml
volumes:
  - ~/.oauth_browser:/app/.oauth_browser
environment:
  - OAUTH_PROFILE_DIR=/app/.oauth_browser
```

## Troubleshooting

### "Missing X server or $DISPLAY"

**Solution**: Run on machine with display or use xvfb:
```bash
xvfb-run -a python3 scripts/oauth_login.py --provider google --no-headless
```

### "2FA timeout"

**Solution**: 
- Make sure phone is nearby
- Approve notification quickly
- Or use `--2fa-code` if you have a code

### "Could not find email input"

**Solution**: Google's UI may have changed. Check selectors in code and update if needed.

## Security Notes

- Credentials are never logged
- Use environment variables in production
- Browser profile contains session tokens - keep secure
- Don't commit credentials to git
