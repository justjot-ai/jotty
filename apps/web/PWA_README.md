# Jotty PWA - Progressive Web App

## Overview

The Jotty PWA provides a native app-like experience for web and mobile users with:

- **Offline Support**: Service worker caching for offline functionality
- **Installable**: Add to home screen on mobile and desktop
- **Voice Input**: Browser-based audio recording and transcription
- **Responsive UI**: Mobile-first design with touch optimizations
- **Auto-Updates**: Background updates with user notifications

## Architecture

```
apps/web/
├── public/
│   ├── manifest.json          # PWA manifest
│   ├── sw.js                  # Service worker
│   └── icons/                 # App icons (72x72 to 512x512)
│
├── src/
│   ├── app/
│   │   ├── layout.tsx         # Root layout with PWA metadata
│   │   ├── globals.css        # Global styles (dark theme)
│   │   └── chat/
│   │       └── page.tsx       # Main chat interface
│   │
│   ├── components/
│   │   ├── chat/
│   │   │   ├── ChatLayout.tsx        # 3-column layout
│   │   │   ├── MessageList.tsx       # Scrolling messages
│   │   │   ├── MessageInput.tsx      # Text input bar
│   │   │   └── VoiceRecorder.tsx     # Voice recording UI
│   │   │
│   │   └── pwa/
│   │       └── PWALifecycle.tsx      # Service worker manager
│   │
│   └── lib/
│       ├── jotty/
│       │   ├── client.ts      # TypeScript SDK client
│       │   ├── types.ts       # Type definitions
│       │   └── hooks.ts       # React hooks
│       │
│       └── pwa/
│           └── serviceWorkerRegistration.ts
│
├── next.config.js             # PWA configuration
├── tailwind.config.js         # Tailwind + custom theme
└── package.json               # Dependencies
```

## Setup

### 1. Install Dependencies

```bash
cd apps/web
npm install
```

Required dependencies:
- `next-pwa` - PWA plugin for Next.js
- `workbox-webpack-plugin` - Service worker generation
- `workbox-window` - Service worker client
- `react-markdown` - Markdown rendering
- `@tailwindcss/typography` - Typography styles

### 2. Generate Icons

Place PNG icons in `public/icons/`:
- 72x72, 96x96, 128x128, 144x144, 152x152, 192x192, 384x384, 512x512

Use a tool like [PWA Asset Generator](https://www.npmjs.com/package/pwa-asset-generator):

```bash
npx pwa-asset-generator logo.png public/icons --icon-only
```

### 3. Configure Backend API

Update `src/lib/jotty/client.ts` with your API URL:

```typescript
this.baseUrl = config.baseUrl || 'http://localhost:8766';
```

Or set it per-instance:

```typescript
const client = new JottyClient({ baseUrl: 'https://api.jotty.ai' });
```

## Development

```bash
# Development mode (PWA disabled for faster iteration)
npm run dev

# Production build
npm run build

# Start production server
npm start
```

Access at `http://localhost:3000/chat`

## React Hooks

### useJotty()

Initialize Jotty client:

```typescript
import { useJotty } from '@/lib/jotty/hooks';

const client = useJotty('https://api.jotty.ai');
```

### useChat()

Chat with message history:

```typescript
import { useChat } from '@/lib/jotty/hooks';

const { messages, loading, error, sendMessage, clearMessages } = useChat('session-123');

// Send message
await sendMessage('Hello, Jotty!');
```

### useVoice()

Voice recording and playback:

```typescript
import { useVoice } from '@/lib/jotty/hooks';

const { recording, transcribing, error, startRecording, stopRecording, speak } = useVoice();

// Record voice
await startRecording();
const transcript = await stopRecording(); // Returns transcribed text

// Speak text
await speak('Hello, world!');
```

### useStream()

Streaming responses:

```typescript
import { useStream } from '@/lib/jotty/hooks';

const { streaming, events, content, startStream } = useStream();

// Start streaming
await startStream('Tell me about AI');
// content updates in real-time as chunks arrive
```

## Service Worker

### Caching Strategy

- **Static Assets** (JS/CSS/images): Cache-first with network fallback
- **API Requests** (/api/*): Network-first with cache fallback
- **HTML Pages**: Network-first with cache fallback to index

### Offline Support

When offline, the app:
1. Serves cached pages and assets
2. Shows offline indicator banner
3. Returns cached API responses if available
4. Queues failed requests for background sync (when implemented)

### Manual Control

```typescript
import {
  registerServiceWorker,
  unregisterServiceWorker,
  getServiceWorkerVersion,
  clearServiceWorkerCache,
} from '@/lib/pwa/serviceWorkerRegistration';

// Register
await registerServiceWorker({
  onSuccess: (reg) => console.log('Registered'),
  onUpdate: (reg) => console.log('Update available'),
  onOffline: () => console.log('Offline'),
  onOnline: () => console.log('Online'),
});

// Get version
const version = await getServiceWorkerVersion(); // 'v1'

// Clear cache
await clearServiceWorkerCache();

// Unregister
await unregisterServiceWorker();
```

## Deployment

### Vercel

```bash
npm install -g vercel
vercel --prod
```

### Docker

```bash
docker build -t jotty-pwa .
docker run -p 3000:3000 jotty-pwa
```

### Nginx

Build static export:

```bash
npm run build
# Output in .next/ folder (standalone mode)
```

Serve with Nginx:

```nginx
server {
  listen 80;
  server_name jotty.example.com;

  location / {
    proxy_pass http://localhost:3000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection 'upgrade';
    proxy_set_header Host $host;
    proxy_cache_bypass $http_upgrade;
  }

  # Service worker must be served with correct headers
  location = /sw.js {
    add_header Cache-Control "no-cache, no-store, must-revalidate";
    add_header Service-Worker-Allowed "/";
    proxy_pass http://localhost:3000/sw.js;
  }
}
```

## Testing

### Manual Testing

1. **Install PWA**: Visit `/chat`, click browser install prompt
2. **Offline Mode**: Toggle network in DevTools, verify app still works
3. **Voice Input**: Click microphone button, grant permissions, record
4. **Updates**: Modify code, rebuild, reload page - should show update notification

### Lighthouse PWA Audit

```bash
# Install Lighthouse
npm install -g lighthouse

# Run audit
lighthouse http://localhost:3000/chat --view
```

Target scores:
- Performance: 90+
- Accessibility: 95+
- Best Practices: 95+
- SEO: 90+
- PWA: 100

### Browser Compatibility

Tested on:
- Chrome/Edge (Desktop & Mobile)
- Safari (iOS & macOS)
- Firefox (Desktop & Android)
- Samsung Internet

### Service Worker DevTools

Chrome DevTools:
1. Application > Service Workers
2. Check "Update on reload" for development
3. Click "Unregister" to reset
4. Click "Skip waiting" to force update

## Troubleshooting

### Service Worker Not Registering

- Check HTTPS (required for PWA, except localhost)
- Verify `/sw.js` is accessible (200 status)
- Check browser console for errors
- Disable "Update on reload" in DevTools

### Voice Not Working

- Requires HTTPS in production
- Check microphone permissions
- Verify API endpoint is accessible
- Check CORS headers on API

### Offline Mode Not Working

- Service worker must be activated (check DevTools)
- Visit pages while online first to cache them
- Check network tab for failed requests
- Clear cache and re-register service worker

### Icons Not Showing

- Verify all icon sizes exist in `/public/icons/`
- Check manifest.json paths
- Clear browser cache
- Re-install PWA

## Future Enhancements

- [ ] IndexedDB for offline message queue
- [ ] Background sync for failed requests
- [ ] Push notifications for new messages
- [ ] Share Target API for sharing to Jotty
- [ ] Shortcuts API for quick actions
- [ ] Badging API for unread counts
- [ ] File System Access API for documents

## Resources

- [Next.js PWA Documentation](https://github.com/shadowwalker/next-pwa)
- [Workbox Guide](https://developers.google.com/web/tools/workbox)
- [Web.dev PWA](https://web.dev/progressive-web-apps/)
- [MDN Service Workers](https://developer.mozilla.org/en-US/docs/Web/API/Service_Worker_API)
