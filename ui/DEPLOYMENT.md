# Jotty UI Deployment Guide

## Overview

Jotty UI is a **standalone Next.js application** located in `/var/www/sites/personal/stock_market/Jotty/ui/`.

**This is separate from JustJot.ai** and has its own deployment process.

---

## Quick Start

### Option 1: Direct Node.js Deployment

```bash
cd /var/www/sites/personal/stock_market/Jotty/ui
npm install
npm run build
npm start
```

### Option 2: Using Deploy Script

```bash
cd /var/www/sites/personal/stock_market/Jotty/ui
./deploy.sh
```

### Option 3: Docker Deployment

```bash
cd /var/www/sites/personal/stock_market/Jotty/ui
docker-compose up -d
```

---

## Environment Variables

Create `.env.local`:

```bash
# MongoDB (shared with JustJot.ai)
MONGODB_URI=mongodb://localhost:27017/justjot

# Clerk Authentication (shared with JustJot.ai)
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_...
CLERK_SECRET_KEY=sk_...

# API Base URL
NEXT_PUBLIC_API_URL=http://localhost:3000
```

---

## Deployment Options

### 1. Standalone Server

**Port:** 3010 (to avoid conflict with JustJot.ai on 3000)

```bash
cd /var/www/sites/personal/stock_market/Jotty/ui
npm install
npm run build
PORT=3010 npm start
```

### 2. Docker Container

```bash
docker-compose up -d
```

Access at: `http://localhost:3010`

### 3. Nginx Reverse Proxy

Add to nginx config:

```nginx
server {
    listen 80;
    server_name jotty.justjot.ai;

    location / {
        proxy_pass http://localhost:3010;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

---

## Integration with JustJot.ai

### Option 1: Separate Deployment (Recommended)

- Jotty UI runs on port 3010
- JustJot.ai runs on port 3000
- Both share MongoDB and Clerk auth
- Access via different URLs/subdomains

### Option 2: Embedded in JustJot.ai

- Copy Jotty UI components to JustJot.ai
- Use same Next.js instance
- Access via `/dashboard/jotty` route

---

## Production Deployment

### Build for Production

```bash
npm run build
```

### Start Production Server

```bash
NODE_ENV=production npm start
```

### With PM2

```bash
pm2 start npm --name "jotty-ui" -- start
pm2 save
pm2 startup
```

---

## Differences from JustJot.ai

| Aspect | JustJot.ai | Jotty UI |
|--------|------------|----------|
| **Location** | `/var/www/sites/personal/stock_market/JustJot.ai/` | `/var/www/sites/personal/stock_market/Jotty/ui/` |
| **Deploy Script** | `./deploy.sh` | `./deploy.sh` (separate) |
| **Port** | 3000 | 3010 |
| **Purpose** | Full JustJot.ai app | Jotty management only |
| **Entities** | Sections, Collections, Ideas, etc. | Agents, Swarms, Workflows, etc. |

---

## Status

✅ **Standalone deployment ready**
✅ **Docker support**
✅ **Separate from JustJot.ai**
