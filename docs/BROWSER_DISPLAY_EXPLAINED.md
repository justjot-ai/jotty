# Why Browser Automation Without Headless Doesn't Work

## The Problem

When you try to run a browser in **non-headless mode** (with `headless=False`), the browser needs:

1. **X Server** - A display server that manages graphics
2. **DISPLAY environment variable** - Tells the browser where to show the window
3. **Graphics capabilities** - Hardware or virtual graphics

## Error You See

```
Missing X server or $DISPLAY
ERROR:ui/ozone/platform/x11/ozone_platform_x11.cc:259] Missing X server or $DISPLAY
```

This means: **No display server is available to show the browser window.**

## Why This Happens

### Docker Containers
- **No display by default** - Containers are isolated and don't have access to host display
- **No X server** - No graphics server running inside container
- **No DISPLAY variable** - Not set automatically

### Headless Servers
- **No GUI** - Servers often run without graphical interfaces
- **No X11** - X Window System not installed
- **SSH without X11 forwarding** - Can't forward display

## Solutions

### Solution 1: Use Headless Mode (Recommended for Automation)

```python
# This works everywhere
browser = await p.chromium.launch_persistent_context(
    user_data_dir=profile_dir,
    headless=True  # ✅ No display needed
)
```

**Pros:**
- Works in Docker
- Works on headless servers
- Faster (no rendering)
- Less resources

**Cons:**
- Can't see what's happening
- Harder to debug
- Can't interact manually

### Solution 2: Install X Server in Docker

```dockerfile
# Dockerfile
RUN apt-get update && apt-get install -y \
    xvfb \
    x11vnc \
    fluxbox

# Run with virtual display
CMD ["xvfb-run", "-a", "python", "your_script.py"]
```

### Solution 3: Use X11 Forwarding (SSH)

```bash
# Connect with X11 forwarding
ssh -X user@host

# Set DISPLAY
export DISPLAY=:0

# Now non-headless works
python3 scripts/oauth_login.py --no-headless
```

### Solution 4: Use xvfb (Virtual Frame Buffer)

```bash
# Install xvfb
sudo apt-get install xvfb

# Run with virtual display
xvfb-run -a python3 scripts/oauth_login.py --no-headless
```

This creates a **virtual display** - browser thinks it has a screen, but you can't see it.

### Solution 5: Mount X11 Socket (Docker)

```bash
# Allow X11 access
xhost +local:docker

# Run Docker with X11 socket
docker run -it \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  your-image
```

## For OAuth Login Specifically

### Why You Need Non-Headless for 2FA

When Google sends a **push notification** to your phone:
- You need to **see** the browser to know it's waiting
- You need to **approve** on your phone
- Script needs to **detect** when approval happens

### Best Approach: Hybrid

1. **First time**: Run with `--no-headless` on a machine with display
   - Complete authentication
   - Approve 2FA
   - Save browser profile

2. **Afterwards**: Use saved profile with `--headless`
   - Profile contains authenticated session
   - No need to see browser
   - Works in Docker

```bash
# Step 1: Authenticate (with display)
python3 scripts/oauth_login.py \
    --provider google \
    --no-headless \
    --profile-dir ~/.oauth_profile

# Step 2: Use authenticated profile (headless)
python3 scripts/notebooklm_pdf.py \
    --file doc.md \
    --profile-dir ~/.oauth_profile \
    --headless  # ✅ Works now!
```

## Technical Details

### What is X Server?

X Server (X11) is the display server that:
- Manages windows
- Handles input (mouse, keyboard)
- Renders graphics
- Communicates with applications via DISPLAY

### What is Headless Mode?

Headless mode:
- Browser runs **without** a display
- Uses virtual rendering
- Still executes JavaScript, loads pages
- Just doesn't show a window

### Why Playwright Needs Display for Non-Headless?

Playwright launches real Chromium:
- Chromium expects X11 for non-headless
- Tries to create windows
- Needs graphics context
- Fails if no display available

## Quick Check: Do You Have Display?

```bash
# Check DISPLAY variable
echo $DISPLAY
# Should show something like: :0 or localhost:10.0

# Check if X server running
ps aux | grep X
# Should show X server process

# Test X11
xeyes  # If this works, you have display
```

## Summary

| Mode | Needs Display? | Works in Docker? | Can See Browser? |
|------|---------------|------------------|------------------|
| `headless=True` | ❌ No | ✅ Yes | ❌ No |
| `headless=False` | ✅ Yes | ❌ No (unless configured) | ✅ Yes |

**For OAuth with 2FA:**
- Use `--no-headless` **once** to authenticate
- Save the browser profile
- Use `--headless` afterwards with saved profile
