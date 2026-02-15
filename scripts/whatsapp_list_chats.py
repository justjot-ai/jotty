#!/usr/bin/env python3
"""List WhatsApp chats from the bridge store. Run from Jotty root."""
import asyncio
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


async def main():
    from Jotty.cli.channels.whatsapp_web import WhatsAppWebClient, set_global_whatsapp_client

    print("Starting WhatsApp client...")
    client = WhatsAppWebClient()
    set_global_whatsapp_client(client)
    ok = await client.start()
    if not ok:
        print("Failed to start client.")
        return 1
    for _ in range(25):
        await asyncio.sleep(1)
        if client.connected:
            break
    else:
        print("Timeout.")
        return 1
    print("Connected. Waiting 15s for chat list to sync...")
    await asyncio.sleep(15)
    print("Fetching chats...")
    chats = await client.get_chats(limit=100)
    print(f"Found {len(chats)} chat(s):")
    for c in chats:
        name = c.get("name") or c.get("id") or "?"
        cid = c.get("id", "")
        group = " [group]" if c.get("is_group") else ""
        print(f"  - {name!r}  id={cid!r}{group}")
    if not chats:
        print(
            "  (No chats in store. Open the #my-ai-ml chat in WhatsApp Web or send a message to populate the store.)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
