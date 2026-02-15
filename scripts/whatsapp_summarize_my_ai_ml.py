#!/usr/bin/env python3
"""
Read #my-ai-ml and summarize learnings. Two modes:

1) Live (requires WhatsApp connected and store populated):
   python scripts/whatsapp_summarize_my_ai_ml.py
   Or: python scripts/whatsapp_summarize_my_ai_ml.py --chat-id 120363406441979562@g.us

2) From exported chat file (no bridge needed):
   In WhatsApp: open #my-ai-ml → ⋮ → Export chat (without media). Save as e.g. my-ai-ml.txt
   python scripts/whatsapp_summarize_my_ai_ml.py --file my-ai-ml.txt

Uses ANTHROPIC_API_KEY from .env for summarization (if set).
Run from Jotty root.
"""
import asyncio
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Load .env so ANTHROPIC_API_KEY is available for summarization
_env = repo_root / ".env"
if _env.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env, override=False)
    except ImportError:
        pass


def _ensure_llm_from_env():
    """Configure DSPy with Anthropic API from .env so summarize skill works."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return
    try:
        import dspy
        if dspy.settings.lm is not None:
            return
        from Jotty.core.infrastructure.foundation.direct_anthropic_lm import DirectAnthropicLM
        dspy.configure(lm=DirectAnthropicLM(model="sonnet"))
    except Exception:
        pass


def _get_group_jids_from_session():
    """Find group JIDs from Baileys session sender-key files."""
    session_dir = Path.home() / ".jotty" / "whatsapp_session"
    if not session_dir.exists():
        return []
    jids = set()
    for f in session_dir.iterdir():
        if f.is_file() and "sender-key" in f.name and "@g.us" in f.name:
            # e.g. sender-key-120363406441979562@g.us--...
            part = f.name.split("@g.us")[0].replace("sender-key-", "")
            if part.isdigit():
                jids.add(f"{part}@g.us")
    return sorted(jids)


async def main():
    # --file: summarize exported chat file (no WhatsApp connection needed)
    file_arg = None
    if "--file" in sys.argv:
        i = sys.argv.index("--file")
        if i + 1 < len(sys.argv):
            file_arg = Path(sys.argv[i + 1]).expanduser().resolve()
            if not file_arg.is_file():
                print(f"File not found: {file_arg}")
                return 1
            text = file_arg.read_text(encoding="utf-8", errors="replace")
            _ensure_llm_from_env()
            from Jotty.core.capabilities.registry.skills_registry import get_skills_registry
            reg = get_skills_registry()
            reg.init()
            sum_skill = reg.get_skill("summarize")
            if not sum_skill or not sum_skill.tools:
                print("Summarize skill not found.")
                return 1
            st = sum_skill.tools.get("summarize_text_tool")
            if not st:
                print("summarize_text_tool not found.")
                return 1
            if len(text) > 120000:
                text = text[-120000:] + "\n\n[... truncated ...]"
            out = st({"text": text, "length": "medium", "style": "bullet"})
            if out and out.get("success"):
                print(f"--- Summary (from {file_arg.name}) ---\n")
                print(out.get("summary", ""))
                print("\n--- End ---")
                return 0
            print("Summarize failed:", out.get("error", "unknown"))
            return 1

    from Jotty.cli.channels.whatsapp_web import WhatsAppWebClient, set_global_whatsapp_client

    chat_id_arg = None
    if "--chat-id" in sys.argv:
        i = sys.argv.index("--chat-id")
        if i + 1 < len(sys.argv):
            chat_id_arg = sys.argv[i + 1]

    print("Starting WhatsApp client (using saved credentials)...")
    client = WhatsAppWebClient()
    set_global_whatsapp_client(client)
    ok = await client.start()
    if not ok:
        print("Failed to start WhatsApp client. Is Node.js installed?")
        return 1

    for _ in range(30):
        await asyncio.sleep(1)
        if client.connected:
            print("WhatsApp connected.")
            break
    else:
        print("Timeout waiting for WhatsApp connection.")
        return 1

    print("Syncing: full history (if enabled) + waiting for new messages.")
    print("You can send a message in #my-ai-ml during the wait. Waiting 90s...")
    await asyncio.sleep(90)

    stats = await client.get_store_stats()
    print(f"Store after wait: {stats.get('chat_count', 0)} chats, {len(stats.get('message_jids') or [])} message JIDs")

    _ensure_llm_from_env()
    from Jotty.core.capabilities.registry.skills_registry import get_skills_registry
    reg = get_skills_registry()
    reg.init()
    skill = reg.get_skill("whatsapp-reader")
    if not skill or not skill.tools:
        print("whatsapp-reader skill not found.")
        return 1
    read_tool = skill.tools.get("read_whatsapp_chat_messages_tool")
    summarize_tool = skill.tools.get("summarize_whatsapp_chat_learnings_tool")
    if not read_tool or not summarize_tool:
        print("Tools not found.")
        return 1

    # Resolve chat: by chat_id arg, or by name, or try group JIDs from session
    chat_id_to_use = None
    if chat_id_arg:
        chat_id_to_use = chat_id_arg if "@" in chat_id_arg else f"{chat_id_arg}@g.us"
        print(f"Using chat_id: {chat_id_to_use}")
    else:
        result = await summarize_tool({
            "chat_name": "my-ai-ml",
            "limit": 500,
            "length": "medium",
            "style": "bullet",
        })
        if result.get("success") and (result.get("message_count") or 0) > 0:
            print("Reading messages from #my-ai-ml and summarizing...")
            print(f"\n--- Summary (from {result.get('message_count', 0)} messages) ---\n")
            print(result.get("summary", ""))
            print("\n--- End ---")
            return 0
        if result.get("success") and result.get("message_count") == 0:
            print("Chat 'my-ai-ml' matched but had 0 messages.")
        else:
            print("No chat named 'my-ai-ml' in store. Trying known groups from session...")

        # Prefer known #my-ai-ml group JID if in session (so we fetch the right group's history)
        MY_AI_ML_JID = "120363370069408928@g.us"
        group_jids = _get_group_jids_from_session()
        candidates = []
        for jid in group_jids:
            r = await read_tool({"chat_id": jid, "limit": 10})
            if r.get("success") and (len(r.get("messages") or [])) > 0:
                candidates.append((jid, len(r.get("messages", []))))
        if MY_AI_ML_JID in [c[0] for c in candidates]:
            chat_id_to_use = MY_AI_ML_JID
            print(f"Using #my-ai-ml group: {chat_id_to_use}")
        elif candidates:
            # Pick group with most messages (likely the active one)
            chat_id_to_use = max(candidates, key=lambda x: x[1])[0]
            print(f"Found group with messages: {chat_id_to_use}")
        if not chat_id_to_use:
            print("No group with messages found. Open #my-ai-ml in WhatsApp and send a message, then run again.")
            print("Or run: python scripts/whatsapp_summarize_my_ai_ml.py --chat-id <GROUP_JID>")
            return 1

    # Fetch full history from WhatsApp servers (batch of 50 until up to 500)
    print(f"Fetching full history for {chat_id_to_use} (up to 500 messages)...")
    fetch_result = await client.fetch_chat_history(chat_id_to_use, max_messages=500)
    if fetch_result.get("success"):
        print(f"History fetch: {fetch_result.get('message_count', 0)} messages in store.")
    else:
        print(f"History fetch note: {fetch_result.get('error', 'unknown')} (continuing with current store)")

    # Summarize by chat_id
    result = await read_tool({"chat_id": chat_id_to_use, "limit": 500})
    if not result.get("success"):
        print("Error:", result.get("error", "Unknown"))
        return 1
    messages = result.get("messages") or []
    if not messages:
        print("No messages in this chat.")
        return 0

    # Build text and summarize via summarize skill (uses ANTHROPIC_API_KEY from .env)
    _ensure_llm_from_env()
    from datetime import datetime
    lines = []
    for m in messages:
        body = (m.get("body") or "").strip()
        if not body:
            continue
        ts = m.get("timestamp") or 0
        prefix = "[Me]" if m.get("fromMe") else "[Other]"
        try:
            dt = datetime.fromtimestamp(ts) if ts else ""
            lines.append(f"{prefix} {dt}: {body}")
        except Exception:
            lines.append(f"{prefix}: {body}")
    combined = "\n\n".join(lines)
    if len(combined) > 120000:
        combined = combined[-120000:] + "\n\n[... truncated ...]"

    sum_skill = reg.get_skill("summarize")
    if sum_skill and sum_skill.tools:
        st = sum_skill.tools.get("summarize_text_tool")
        if st:
            out = st({"text": combined, "length": "medium", "style": "bullet"})
            if out and out.get("success"):
                print(f"\n--- Summary (from {len(messages)} messages, chat {chat_id_to_use}) ---\n")
                print(out.get("summary", ""))
                print("\n--- End ---")
                return 0
    print(f"\n--- Raw excerpt (from {len(messages)} messages) ---\n")
    print(combined[:8000] + ("..." if len(combined) > 8000 else ""))
    print("\n--- End ---")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
