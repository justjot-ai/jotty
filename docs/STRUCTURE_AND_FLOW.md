# Jotty Structure, Integration, and Flow (from actual codebase)

This document is derived from **actual paths and imports** in the repo, not from older docs.

---

## 1. Repository layout (actual paths)

```
Jotty/                          # Repo root = package "Jotty"
├── __init__.py                 # Lazy exports via __getattr__ → .jotty, .core.*, .sdk.client
├── jotty.py                    # Jotty class (V3 tiered execution), TierExecutor
├── web.py                      # Standalone entry: JottyCLI + start_gateway (port 8766)
│
├── sdk/                        # Layer 4: Public SDK (apps use this)
│   ├── __init__.py             # Exports Jotty, JottySync, types (ChannelType, SDKEvent, etc.)
│   └── client.py               # Jotty client: chat(), workflow(), stream(); base_url → HTTP or local router
│
├── core/                       # Layers 2–3: Framework
│   ├── interface/               # API surface
│   │   ├── api/                # mode_router.py (ModeRouter, get_mode_router), unified.py (JottyAPI), chat_api, workflow_api
│   │   ├── modalities/         # text, voice (speech_to_text, text_to_speech)
│   │   └── ...
│   ├── capabilities/          # Registry
│   │   └── registry/           # unified_registry, skills_registry (loads skills/), ui_registry
│   ├── intelligence/           # Orchestration, agents, memory, learning
│   │   ├── orchestration/      # Orchestrator, execution (TierExecutor, intent_classifier), swarms, use_cases
│   │   ├── reasoning/         # agents (AutoAgent, etc.), planners, executors
│   │   ├── memory/
│   │   └── learning/
│   └── infrastructure/         # foundation (data_structures, config, LM provider), context, monitoring, integration
│
├── apps/                       # Layer 5: Applications
│   ├── cli/                    # CLI REPL + slash commands
│   │   ├── __main__.py         # Entry: python -m Jotty.apps.cli
│   │   ├── app.py              # JottyCLI (uses Jotty.sdk.Jotty)
│   │   ├── gateway/            # UnifiedGateway, ChannelRouter, start_gateway
│   │   │   ├── __main__.py     # Entry: python -m Jotty.apps.cli.gateway
│   │   │   ├── server.py      # UnifiedGateway, create_app (FastAPI), webhooks, /ws
│   │   │   ├── channels.py    # ChannelRouter → get_mode_router().chat()
│   │   │   ├── responders.py  # ChannelResponderRegistry, get_unified_registry for send
│   │   │   └── sessions.py   # PersistentSessionManager, SDK types
│   │   ├── commands/           # /run, /workflow, /research, /gateway, etc.
│   │   ├── repl/               # REPL engine, session, completer
│   │   └── ui/, config/, plugins/, heartbeat/
│   ├── api/                     # HTTP API server
│   │   ├── __main__.py         # uvicorn.run("Jotty.web.api:app") — may be wrong; app lives in .api
│   │   ├── api.py              # create_app(), app (FastAPI)
│   │   ├── jotty_api.py        # JottyAPI wrapper, ModeRouter, ExecutionContext
│   │   ├── simple_server.py   # get_mode_router(), SDK types
│   │   └── routes/             # chat, voice, documents, tools, system, sessions, sharing
│   ├── web/                     # Next.js frontend + backend
│   │   └── backend/server.py   # Uses Jotty.sdk.Jotty
│   ├── telegram/bot.py         # Uses Jotty.sdk.Jotty
│   ├── whatsapp/client_shared.py  # Uses Jotty.sdk.Jotty
│   └── shared/                  # events, models, renderers (SDK types)
│
└── skills/                     # Skill packages (discovered by SkillsRegistry)
    ├── <name>/                 # One dir per skill
    │   ├── SKILL.md            # Optional metadata
    │   ├── tools.py            # Required (or scripts/ for Claude Code skills)
    │   └── scripts/            # Optional (Claude Code skill)
    └── composite-templates/   # Excluded from registry scan
```

---

## 2. Entry points (how the app is run)

| Entry | Command / path | What runs |
|-------|----------------|-----------|
| CLI interactive | `python -m Jotty.apps.cli` | `apps/cli/__main__.py` → `JottyCLI` (from `apps/cli/app.py`), REPL + slash commands |
| CLI single | `python -m Jotty.apps.cli run "goal"` | Same CLI, `run_once("/run goal")` |
| Gateway (WebSocket + webhooks) | `python -m Jotty.apps.cli.gateway` or `python web.py` | `apps/cli/gateway/__init__.py` → `start_gateway()`; `web.py` imports `JottyCLI` + `start_gateway` from `Jotty.apps.cli.gateway` |
| API server | `python -m Jotty.apps.api` | `apps/api/__main__.py` → uvicorn `"Jotty.web.api:app"` (likely should be `Jotty.apps.api.api:app`); app object is in `apps/api/api.py` |
| Programmatic (in-process) | `from Jotty import Jotty` | `__init__.py` lazy-loads `.jotty` → `jotty.py` (class `Jotty` + `TierExecutor`) |

---

## 3. Integrated vs not integrated

### 3.1 Applications (integrated)

- **apps/cli**: `__main__.py` → `JottyCLI`; all command modules imported via `commands/` registry; gateway used by `web.py` and `/gateway` command.
- **apps/cli/gateway**: `UnifiedGateway`, `ChannelRouter`, `Responders`, `Sessions`; used by `web.py` and `python -m Jotty.apps.cli.gateway`.
- **apps/api**: `jotty_api.py` uses `ModeRouter`, `ExecutionContext`; routes use `get_unified_registry`, `get_skills_registry`, `get_mode_router`, session registry, etc.
- **apps/web/backend/server.py**: Uses `Jotty.sdk.Jotty`.
- **apps/telegram/bot.py**, **apps/whatsapp/client_shared.py**: Use `Jotty.sdk.Jotty`.
- **apps/shared**: Events, models, renderers use SDK types; used by CLI and gateway.

### 3.2 Core (integrated)

- The script `scripts/find_unintegrated_files.py` (run from repo root) reports **all files under `core/` are integrated** (every module is imported by some other file).

### 3.3 Skills (registry integration)

- **SkillsRegistry** (`core/capabilities/registry/skills_registry.py`):
  - **Discovery**: `skills_dir` = `JOTTY_SKILLS_DIR` or repo `skills/` or `~/jotty/skills`; then `self.skills_dir.iterdir()`.
  - **Registered**: Any directory that has **either** `tools.py` **or** `scripts/` (and is not in `excluded_dirs`) gets a lazy `SkillDefinition` via `_register_lazy_skill()`.
  - **Excluded dirs**: `composite-templates`, `__pycache__`, `.git`, `.DS_Store`.

**Integrated (loadable) skills**: Subdirs of `skills/` that contain `tools.py` or `scripts/`, and are not in `excluded_dirs`. Count: **254** (includes `document_tools` and `messaging_tools`; see below).

**Explicitly excluded** (in `SkillsRegistry._scan_skills_metadata()` `excluded_dirs`): `composite-templates`, `_infrastructure`, `_providers`, `_tools`. These are shared libraries, not invokable skills.

**Newly integrated (Feb 2026)**:
- **document_tools**: Root `tools.py` + `SKILL.md` added. Registered as **document-tools** (7 tools: generate_pdf_tool, generate_epub_tool, generate_epub_with_chapters_tool, generate_html_tool, generate_docx_tool, generate_presentation_tool, generate_all_formats_tool).
- **messaging_tools**: Root `tools.py` + `SKILL.md` added. Registered as **messaging-tools** (3 tools: send_to_telegram_tool, send_to_whatsapp_tool, send_to_all_channels_tool).

---

## 4. End-to-end flow (from actual code)

### 4.1 User message → response (e.g. Telegram or WebSocket)

1. **HTTP/WebSocket**
   `UnifiedGateway` (`apps/cli/gateway/server.py`): FastAPI app, webhooks (e.g. `/webhook/telegram`), `/ws`, `/message`.

2. **Channel + session**
   `ChannelRouter.handle_message()` (`apps/cli/gateway/channels.py`): Builds/gets session, builds `ExecutionContext` (mode, channel, session_id, user_id, etc.), adds conversation history to context.

3. **Routing**
   `get_mode_router()` → `ModeRouter.chat(message, context)` (`core/interface/api/mode_router.py`).
   Optional: `ValidationGate` → `DirectChatExecutor` for simple queries; else full executor.

4. **Executor**
   `ModeRouter._get_executor(context)` returns the chat executor (e.g. from `core/intelligence/orchestration/execution/`). Executor uses:
   - `get_unified_registry()` → skills (and UI) from `core/capabilities/registry/`.
   - LLM provider (e.g. `core/infrastructure/foundation/unified_lm_provider`).
   - For workflow mode: `_handle_workflow` → AutoAgent.

5. **Skills**
   Registry comes from `SkillsRegistry.init()`: scans `skills_dir`, `_scan_plugin_skills`, and `~/.claude/skills`; each skill dir with `tools.py` or `scripts/` is registered lazily. Tools are loaded on first use.

6. **Response**
   `ChannelRouter._send_response(ResponseEvent)` → `ChannelResponderRegistry` (`apps/cli/gateway/responders.py`) → format per channel → send (e.g. Telegram via skill or direct API).

### 4.2 SDK client (local vs remote)

- **Jotty** (`sdk/client.py`): `base_url` default `http://localhost:8766`.
  - **Local (no base_url or in-process)**: Uses `get_mode_router()` from `core.interface.api.mode_router` and calls `router.chat()` / `router.workflow()`.
  - **Remote**: HTTP/WebSocket to `base_url` (e.g. gateway on 8766).

So when the gateway is running, SDK clients talk to the same `UnifiedGateway` that uses `ModeRouter` and the rest of the pipeline above.

### 4.3 From `from Jotty import Jotty`

- `__init__.py` `__getattr__("Jotty")` → `import_module(".jotty", "Jotty")` → **jotty.py**.
- `jotty.py`: `Jotty` class uses `TierExecutor` from `core.intelligence.orchestration.execution` (and `TierDetector`, `ExecutionConfig`, `ExecutionResult`, `ExecutionTier`). No direct use of ModeRouter here; TierExecutor implements the tiered execution path (DIRECT → AGENTIC → LEARNING → RESEARCH → AUTONOMOUS).

---

## 5. Import rules (from actual usage)

- **Apps**
  - Use **SDK**: `apps/cli/app.py`, `apps/web/backend/server.py`, `apps/telegram/bot.py`, `apps/whatsapp/client_shared.py` import `Jotty.sdk.Jotty` (and sometimes `Jotty.sdk` types).
  - Use **core** where SDK doesn’t expose enough: e.g. gateway uses `Jotty.core.interface.api.mode_router.get_mode_router`, `Jotty.core.capabilities.registry.unified_registry.get_unified_registry`; api uses `Jotty.core.interface.api.mode_router`, `Jotty.core.capabilities.registry.*`, `Jotty.core.infrastructure.*`, etc.
- **SDK**
  - Imports from **core**: e.g. `sdk/client.py` uses `core.infrastructure.foundation.types.sdk_types`; when running local, it imports `get_mode_router` from `core.interface.api.mode_router`.
- **Core**
  - Cross-module imports use `Jotty.core.*` or relative imports within `core/`.

---

## 6. Summary

| Topic | Result |
|-------|--------|
| **Structure** | Root `Jotty/` (jotty.py, web.py), `sdk/`, `core/` (interface, capabilities, intelligence, infrastructure), `apps/` (cli, api, web, telegram, whatsapp, shared), `skills/`. |
| **Entry points** | `python -m Jotty.apps.cli`, `python -m Jotty.apps.cli.gateway`, `python web.py`, `python -m Jotty.apps.api`, `from Jotty import Jotty`. |
| **Integrated** | All of `core/`; all apps listed above; 252 skill dirs with `tools.py` or `scripts/`. |
| **Not registered as skills** | 3 support packages under `skills/`: `_infrastructure`, `_providers`, `_tools` (excluded by design). `document_tools` and `messaging_tools` are now integrated with root `tools.py` + `SKILL.md`. |
| **Flow** | Gateway/API → ChannelRouter → ModeRouter (chat/workflow) → executor (registry + LLM) → skills → response back via responders. SDK uses same router locally or HTTP to gateway. |
