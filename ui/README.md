# Jotty Management UI

**Standalone Next.js application for managing Jotty entities**

---

## Location

This UI is part of the external Jotty package:
```
/var/www/sites/personal/stock_market/Jotty/ui/
```

---

## Structure

```
Jotty/ui/
├── src/
│   ├── app/                    # Next.js App Router
│   │   ├── api/               # API routes
│   │   └── dashboard/         # Dashboard pages
│   ├── components/
│   │   └── management/        # Management components
│   ├── pages/
│   │   └── ManagementPage.tsx # Main page
│   └── types.ts               # Type definitions
├── package.json
├── next.config.js
└── tsconfig.json
```

---

## Installation

```bash
cd /var/www/sites/personal/stock_market/Jotty/ui
npm install
```

## Development

```bash
npm run dev
```

## Build

```bash
npm run build
npm start
```

---

## Integration with JustJot.ai

JustJot.ai imports/re-exports from this external UI:

```typescript
// In JustJot.ai
export { default } from '/var/www/sites/personal/stock_market/Jotty/ui/src/pages/ManagementPage';
```

Or via symlink:
```bash
ln -s /var/www/sites/personal/stock_market/Jotty/ui/src /var/www/sites/personal/stock_market/JustJot.ai/src/jotty-external
```

---

## Standalone Deployment

This UI can be deployed independently:
1. Build: `npm run build`
2. Deploy the `.next` folder
3. Run: `npm start`

---

## Jotty Entities Managed

- Agents, Swarms, Workflows, Tools, MCP Tools
- Presets, Memory Entities, RL Configs

**NO JustJot.ai entities** (sections, collections, etc.)
