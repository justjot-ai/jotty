# Jotty Tauri - Desktop & Mobile App

## Overview

The Jotty Tauri app provides a native desktop and mobile experience with:

- **Native Performance**: Rust backend with web frontend
- **System Tray**: Background app with tray icon (desktop)
- **Native Dialogs**: File pickers, notifications, confirmations
- **File System Access**: Read/write local files
- **Offline-First**: Works without internet (with local API)
- **Cross-Platform**: Windows, macOS, Linux, iOS, Android

## Architecture

```
apps/web/
├── src-tauri/
│   ├── src/
│   │   └── main.rs          # Rust backend
│   ├── Cargo.toml           # Rust dependencies
│   ├── tauri.conf.json      # Tauri configuration
│   ├── build.rs             # Build script
│   └── icons/               # App icons
│
├── src/lib/tauri/
│   ├── bridge.ts            # Tauri bridge (350+ lines)
│   └── hooks.ts             # React hooks
│
└── package.json             # Tauri CLI scripts
```

## Setup

### 1. Install Rust

```bash
# Install Rust (required for Tauri)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Verify installation
rustc --version
cargo --version
```

### 2. Install System Dependencies

**macOS:**
```bash
xcode-select --install
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install libwebkit2gtk-4.0-dev \
    build-essential \
    curl \
    wget \
    file \
    libssl-dev \
    libgtk-3-dev \
    libayatana-appindicator3-dev \
    librsvg2-dev
```

**Windows:**
- Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Install [WebView2](https://developer.microsoft.com/en-us/microsoft-edge/webview2/)

### 3. Install Dependencies

```bash
cd apps/web
npm install
```

### 4. Generate Icons

```bash
# Create a 1024x1024 PNG icon
# Then run:
npm run tauri:icon path/to/icon.png

# This generates all required icon sizes:
# - 32x32.png (Windows tray)
# - 128x128.png, 128x128@2x.png (macOS)
# - icon.icns (macOS bundle)
# - icon.ico (Windows bundle)
```

## Development

### Run in Dev Mode

```bash
# Start Tauri dev server (hot reload)
npm run tauri:dev
```

This will:
1. Start Next.js dev server (`npm run dev`)
2. Launch Tauri window pointing to `http://localhost:3000`
3. Enable hot reload for both frontend and Rust changes

### Tauri Bridge API

The `tauriBridge` provides a unified API for native capabilities:

```typescript
import { tauriBridge, isTauri, getPlatformInfo } from '@/lib/tauri/bridge';

// Check if running in Tauri
if (isTauri()) {
  console.log('Running in Tauri!');
}

// Get platform info
const info = await getPlatformInfo();
console.log(info);
// {
//   isTauri: true,
//   platform: 'desktop', // or 'mobile' or 'web'
//   os: 'macos', // or 'windows', 'linux', 'ios', 'android'
//   version: '2.0.0'
// }

// Show native notification
await tauriBridge.showNotification('Hello', 'Notification from Jotty!');

// Open external URL in default browser
await tauriBridge.openURL('https://jotty.ai');

// File dialogs
const file = await tauriBridge.openFileDialog({
  multiple: false,
  filters: [{ name: 'Text', extensions: ['txt', 'md'] }]
});

const savePath = await tauriBridge.saveFileDialog({
  defaultPath: 'document.txt',
  filters: [{ name: 'Text', extensions: ['txt'] }]
});

// Confirm dialog
const confirmed = await tauriBridge.confirm('Are you sure?', 'Confirmation');

// Message dialog
await tauriBridge.showMessage('Operation complete!', 'Success');

// File system (within allowed scope)
const content = await tauriBridge.readTextFile('/path/to/file.txt');
await tauriBridge.writeTextFile('/path/to/file.txt', 'New content');

const exists = await tauriBridge.fileExists('/path/to/file.txt');

// Read/write binary files
const data = await tauriBridge.readBinaryFile('/path/to/image.png');
await tauriBridge.writeBinaryFile('/path/to/output.png', data);

// Minimize to system tray (desktop only)
await tauriBridge.minimizeToTray();
```

### React Hooks

```typescript
import { useTauri, usePlatformInfo, useNewChatEvent } from '@/lib/tauri/hooks';

// Check if running in Tauri
const isTauri = useTauri();

// Get platform info
const { platformInfo, loading } = usePlatformInfo();
if (!loading) {
  console.log(platformInfo.platform); // 'desktop' | 'mobile' | 'web'
}

// Listen for new chat events from system tray
useNewChatEvent(() => {
  console.log('New chat requested from tray');
  // Clear current chat and start fresh
});
```

### Custom Commands

Add Rust commands in `src-tauri/src/main.rs`:

```rust
#[tauri::command]
fn my_custom_command(arg: String) -> Result<String, String> {
    Ok(format!("Hello, {}!", arg))
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            my_custom_command,
            // ... other commands
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

Then call from TypeScript:

```typescript
import { invoke } from '@/lib/tauri/bridge';

const result = await invoke<string>('my_custom_command', { arg: 'World' });
console.log(result); // "Hello, World!"
```

## Building

### Build for Current Platform

```bash
# Production build
npm run tauri:build

# Debug build (faster, larger)
npm run tauri:build:debug
```

Outputs:
- **macOS**: `src-tauri/target/release/bundle/dmg/Jotty AI_2.0.0_x64.dmg`
- **Windows**: `src-tauri/target/release/bundle/msi/Jotty AI_2.0.0_x64_en-US.msi`
- **Linux**: `src-tauri/target/release/bundle/deb/jotty-ai_2.0.0_amd64.deb`

### Cross-Platform Builds

**Build for Windows from macOS/Linux:**
```bash
rustup target add x86_64-pc-windows-msvc
npm run tauri:build -- --target x86_64-pc-windows-msvc
```

**Build for macOS from Linux:**
```bash
# Requires osxcross toolchain
rustup target add x86_64-apple-darwin
npm run tauri:build -- --target x86_64-apple-darwin
```

**Build for Linux from macOS/Windows:**
```bash
# Requires Docker
docker run --rm -v $(pwd):/src -w /src tauri/tauri-linux:latest npm run tauri:build
```

### Mobile Builds

**iOS (requires macOS):**
```bash
# Install Xcode and iOS development tools
rustup target add aarch64-apple-ios

# Initialize iOS project
npm run tauri -- ios init

# Build and run on simulator
npm run tauri -- ios dev

# Build for release
npm run tauri -- ios build
```

**Android:**
```bash
# Install Android Studio and NDK
rustup target add aarch64-linux-android

# Initialize Android project
npm run tauri -- android init

# Build and run on emulator/device
npm run tauri -- android dev

# Build APK
npm run tauri -- android build
```

## System Tray

The app includes a system tray icon (desktop only) with:

- **Left Click**: Show/focus window
- **Menu Items**:
  - Show Jotty
  - New Chat (emits `new_chat` event)
  - Quit

**Listen for tray events:**
```typescript
import { useNewChatEvent } from '@/lib/tauri/hooks';

function ChatPage() {
  useNewChatEvent(() => {
    // Clear messages and start new chat
    clearMessages();
  });

  return <ChatLayout />;
}
```

## Configuration

### tauri.conf.json

Key settings:

```json
{
  "tauri": {
    "allowlist": {
      "shell": { "open": true },
      "dialog": { "all": true },
      "notification": { "all": true },
      "fs": {
        "all": true,
        "scope": ["$APPDATA/*", "$DOWNLOAD/*"]
      },
      "http": {
        "all": true,
        "scope": ["http://localhost:*", "https://**"]
      },
      "window": { "all": true }
    },
    "windows": [{
      "width": 1200,
      "height": 800,
      "minWidth": 800,
      "minHeight": 600,
      "center": true
    }],
    "systemTray": {
      "iconPath": "icons/icon.png"
    }
  }
}
```

### File System Scope

Files can only be accessed within these directories:
- `$APPDATA` - App data directory
- `$APPCONFIG` - App config directory
- `$APPLOCALDATA` - Local data directory
- `$DOWNLOAD` - Downloads directory

To allow additional paths, update `tauri.conf.json`:

```json
"fs": {
  "scope": ["$APPDATA/*", "$HOME/Documents/*", "/custom/path/*"]
}
```

## Security

### Content Security Policy (CSP)

The app has a strict CSP:

```
default-src 'self';
connect-src 'self' http://localhost:* ws://localhost:* https://*;
img-src 'self' data: blob: https:;
style-src 'self' 'unsafe-inline';
script-src 'self' 'unsafe-inline' 'unsafe-eval'
```

### Allowed HTTP Requests

Only requests to these URLs are allowed:
- `http://localhost:*` (local API)
- `https://**` (any HTTPS URL)

Update in `tauri.conf.json` if needed.

## Deployment

### Code Signing

**macOS:**
1. Get an Apple Developer account
2. Create a signing certificate
3. Update `tauri.conf.json`:
```json
"macOS": {
  "signingIdentity": "Developer ID Application: Your Name (TEAM_ID)"
}
```

**Windows:**
1. Get a code signing certificate
2. Update `tauri.conf.json`:
```json
"windows": {
  "certificateThumbprint": "YOUR_CERT_THUMBPRINT"
}
```

### Auto-Updates

Enable in `tauri.conf.json`:

```json
"updater": {
  "active": true,
  "endpoints": ["https://releases.jotty.ai/{{target}}/{{current_version}}"],
  "dialog": true,
  "pubkey": "YOUR_PUBLIC_KEY"
}
```

Generate keys:
```bash
npm run tauri -- signer generate
```

## Testing

### Unit Tests (Rust)

```bash
cd src-tauri
cargo test
```

### Integration Tests

```bash
# Run Tauri in test mode
npm run tauri:dev -- --test
```

### E2E Tests

Use WebDriver with Tauri:

```typescript
import { WebdriverIO } from '@wdio/types';

describe('Jotty Tauri E2E', () => {
  let driver: WebdriverIO.Browser;

  before(async () => {
    driver = await remote({
      capabilities: {
        'tauri:options': {
          application: './src-tauri/target/release/jotty-app'
        }
      }
    });
  });

  it('should open app', async () => {
    const title = await driver.getTitle();
    expect(title).toBe('Jotty AI');
  });
});
```

## Troubleshooting

### Build Errors

**"Cargo not found"**
- Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- Restart terminal

**"webkit2gtk not found" (Linux)**
```bash
sudo apt install libwebkit2gtk-4.0-dev
```

**"MSBuild not found" (Windows)**
- Install Visual Studio Build Tools with C++ support

### Runtime Errors

**"Failed to invoke command"**
- Check that command is registered in `invoke_handler`
- Verify command signature matches Rust function

**"File system access denied"**
- Check that path is within allowed scope in `tauri.conf.json`
- Use `$APPDATA` or other allowed variables

**"WebView crashed"**
- Update WebView2 (Windows)
- Clear app data: `rm -rf ~/Library/Application\ Support/ai.jotty.app` (macOS)

### Debug Mode

Enable Rust debug logging:

```bash
RUST_LOG=debug npm run tauri:dev
```

Open DevTools in Tauri window:
- macOS: `Cmd+Option+I`
- Windows/Linux: `Ctrl+Shift+I`

## Performance

### Bundle Size

Typical sizes:
- macOS DMG: ~15 MB
- Windows MSI: ~10 MB
- Linux DEB: ~12 MB

### Optimization

**Reduce bundle size:**
1. Use `opt-level = "z"` in `Cargo.toml` (already configured)
2. Strip debug symbols: `strip = true` in `Cargo.toml`
3. Enable LTO: `lto = true` in `Cargo.toml`

**Faster build times:**
1. Use `--debug` flag: `npm run tauri:build:debug`
2. Enable incremental compilation (default in dev)

## Resources

- [Tauri Documentation](https://tauri.app/)
- [Tauri API Reference](https://tauri.app/v1/api/js/)
- [Rust Book](https://doc.rust-lang.org/book/)
- [Tauri Discord](https://discord.com/invite/tauri)

## Future Enhancements

- [ ] Sidecar Python process for local Jotty API
- [ ] Encrypted local storage for API keys
- [ ] Auto-update mechanism
- [ ] Keyboard shortcuts (global)
- [ ] Touch Bar support (macOS)
- [ ] Windows taskbar integration
- [ ] Linux desktop notifications
- [ ] iOS/Android builds with mobile UI
