# Jotty Web App - Setup Guide

## What We Built

✅ **Backend** (FastAPI + WebSocket) - `apps/web/backend/server.py`
✅ **Frontend** (React + TypeScript) - `apps/web/frontend/src/App.tsx`
✅ **Shared Components** - Uses `web.tsx` renderer
✅ **36 Commands** - Same CommandRegistry as TUI

---

## Quick Start

### 1. Install Backend Dependencies

```bash
cd /var/www/sites/personal/stock_market/Jotty
pip install fastapi uvicorn websockets
```

### 2. Install Frontend Dependencies

```bash
cd apps/web/frontend
npm install
```

### 3. Start Backend

```bash
python apps/web/backend/server.py
```

**Output:**
```
Starting Jotty Web Server...
WebSocket: ws://localhost:8000/ws
Web UI: http://localhost:8000
```

### 4. Start Frontend (in another terminal)

```bash
cd apps/web/frontend
npm start
```

**Output:**
```
Compiled successfully!
Local: http://localhost:3000
```

### 5. Open Browser

Go to: **http://localhost:3000**

---

## Architecture

```
┌──────────────────────────────────┐
│  Browser (React App)             │
│  ├── App.tsx                     │
│  └── web.tsx (shared renderer)  │
└────────┬─────────────────────────┘
         │ WebSocket
┌────────┴─────────────────────────┐
│  FastAPI Backend                 │
│  ├── WebSocket /ws               │
│  ├── ChatInterface (shared)     │
│  ├── EventProcessor (shared)    │
│  └── CommandRegistry (36 cmds)  │
└────────┬─────────────────────────┘
         │ Jotty SDK
┌────────┴─────────────────────────┐
│  Jotty SDK                       │
└──────────────────────────────────┘
```

---

## Files Created

```
apps/web/
├── backend/
│   └── server.py           # FastAPI + WebSocket server
├── frontend/
│   ├── package.json        # NPM dependencies
│   └── src/
│       └── App.tsx         # React app using web.tsx
├── requirements.txt        # Python dependencies
└── README.md              # Documentation
```

---

## Testing

### Test Backend

```bash
# Health check
curl http://localhost:8000/health

# Should return:
# {"status":"healthy","sessions":0,"version":"1.0.0"}
```

### Test WebSocket

```bash
# Install websocat if needed: brew install websocat

websocat ws://localhost:8000/ws
```

Send:
```json
{"type":"chat","content":"Hello!"}
```

### Test Frontend

Open http://localhost:3000 and try:
- Type: "Hello, who are you?"
- Type: "/help"
- Type: "/status"
- Type: "/skills"

---

## Features

| Platform | Commands | Shared Components | Real-time |
|----------|----------|-------------------|-----------|
| **TUI** | 36 | ✅ | N/A |
| **Telegram** | 36 | ✅ | Polling |
| **Web** | 36 | ✅ | WebSocket ✅ |

**All three platforms now use the same:**
- ChatInterface
- EventProcessor
- CommandRegistry
- Message/Status/Error models

---

## Next Steps

1. ✅ Test basic chat
2. ✅ Test commands
3. ⏭️ Add voice input/output
4. ⏭️ Add file upload
5. ⏭️ Add PWA manifest
6. ⏭️ Deploy to production

---

## Troubleshooting

**Backend won't start:**
```bash
pip install fastapi uvicorn websockets
```

**Frontend won't start:**
```bash
cd apps/web/frontend
rm -rf node_modules package-lock.json
npm install
```

**WebSocket not connecting:**
- Check backend is running on port 8000
- Check no firewall blocking
- Check browser console for errors

---

## Production Deployment

### Build Frontend
```bash
cd apps/web/frontend
npm run build
```

### Serve with Backend
```bash
# Backend will serve built frontend from /
python apps/web/backend/server.py
```

Then open: http://localhost:8000

---

**🎉 Web app ready with full shared component integration!**
