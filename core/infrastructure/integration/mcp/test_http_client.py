#!/usr/bin/env python3
"""Minimal MCP HTTP/SSE client: one GET /sse, read endpoint then POST initialize + tools/list; responses arrive on SSE."""
import asyncio
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if _root.name == "stock_market" and str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import httpx

BASE = "http://127.0.0.1:8767"
TOKEN = None


async def main():
    headers = {"Accept": "text/event-stream"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
        headers["X-MCP-Token"] = TOKEN

    session_id = None
    tools_result = []
    got_endpoint = asyncio.Event()
    got_tools = asyncio.Event()

    async with httpx.AsyncClient(timeout=30.0) as client:

        async def read_sse():
            nonlocal session_id
            async with client.stream("GET", f"{BASE}/sse", headers=headers) as resp:
                if resp.status_code != 200:
                    return
                ev = None
                async for line in resp.aiter_lines():
                    if line.startswith("event:"):
                        ev = line[6:].strip()
                        continue
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        if not data:
                            continue
                        if ev == "endpoint" and "session_id=" in data:
                            for part in data.split("?")[-1].split("&"):
                                if part.startswith("session_id="):
                                    session_id = part.split("=", 1)[1].strip()
                                    got_endpoint.set()
                                    break
                            ev = None
                            continue
                        if ev == "message" and data.startswith("{"):
                            try:
                                obj = json.loads(data)
                                if "result" in obj and isinstance(obj.get("result"), dict) and "tools" in obj["result"]:
                                    tools_result.append(obj)
                                    got_tools.set()
                            except Exception:
                                pass
                        ev = None

        reader = asyncio.create_task(read_sse())
        try:
            await asyncio.wait_for(got_endpoint.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            print("Timeout waiting for endpoint event")
            reader.cancel()
            return
        if not session_id:
            print("No session_id")
            reader.cancel()
            return
        print("Session ID:", session_id[:16], "...")

        post_url = f"{BASE}/messages/?session_id={session_id}"
        init_body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0"}},
        }
        print("POST initialize ...")
        post_headers = {k: v for k, v in headers.items() if k.lower() != "accept"}
        post_headers["Content-Type"] = "application/json"
        r = await client.post(post_url, json=init_body, headers=post_headers)
        print("Initialize:", r.status_code)
        await client.post(post_url, json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=post_headers)

        print("POST tools/list ...")
        r = await client.post(post_url, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}, headers=post_headers)
        print("Tools/list POST:", r.status_code)

        try:
            await asyncio.wait_for(got_tools.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            print("Timeout waiting for tools/list response")
        reader.cancel()
        try:
            await reader
        except asyncio.CancelledError:
            pass

        if tools_result:
            tools = tools_result[0].get("result", {}).get("tools", [])
            print("Tools:", [t.get("name") for t in tools])
        else:
            print("No tools result")
        print("OK")


if __name__ == "__main__":
    asyncio.run(main())
