#!/usr/bin/env python3
"""
Test n8n and Activepieces providers against localhost.

Usage:
  cd Jotty && python scripts/test_workflow_providers_localhost.py

Expects:
  - n8n (optional): http://localhost:5678  — set N8N_BASE_URL, N8N_API_KEY if needed
  - Activepieces (optional): http://localhost:8080 — set ACTIVEPIECES_BASE_URL, ACTIVEPIECES_API_KEY

If nothing is running, you'll see empty workflow/flow lists (OK).
"""

import asyncio
import os
import sys

# Ensure Jotty is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    from Jotty.core.capabilities.skills.providers import ContributedSkill, ProviderRegistry

    n8n_url = os.getenv("N8N_BASE_URL", "http://localhost:5678")
    ap_url = os.getenv("ACTIVEPIECES_BASE_URL", "http://localhost:8080")
    print("=== Workflow providers (localhost) ===\n")
    print(f"  N8N_BASE_URL          = {n8n_url}")
    print(f"  ACTIVEPIECES_BASE_URL = {ap_url}\n")

    reg = ProviderRegistry()

    # 1. Registration
    n8n = reg.get_provider("n8n")
    ap = reg.get_provider("activepieces")
    assert n8n and ap, "n8n/activepieces should be registered"
    print("[OK] n8n and activepieces providers registered\n")

    # 2. Initialize (fetch workflows/flows from localhost)
    print("Initializing n8n provider...")
    await n8n.initialize()
    print(f"  n8n available: {n8n.is_available}, skills (workflows): {len(n8n.list_skills())}\n")

    print("Initializing activepieces provider...")
    await ap.initialize()
    print(f"  activepieces available: {ap.is_available}, skills (flows): {len(ap.list_skills())}\n")

    # 3. Unified list
    all_skills = reg.get_all_contributed_skills()
    print(f"get_all_contributed_skills(): {len(all_skills)} total\n")
    for s in all_skills[:15]:
        print(f"  - [{s.provider}] {s.id}  name={s.name!r}")
    if len(all_skills) > 15:
        print(f"  ... and {len(all_skills) - 15} more")

    # 4. Execute (only if we have a workflow and n8n is up)
    if n8n.is_available and n8n.list_skills():
        first = n8n.list_skills()[0]
        wf_id = first.metadata.get("workflow_id")
        print(
            f"\n[Optional] Run first n8n workflow? id={wf_id} (skipping actual run to avoid side effects)"
        )
        # Uncomment to really run:
        # result = await n8n.execute("", {"workflow_id": wf_id, "payload": {}})
        # print(f"  result.success={result.success}, output={str(result.output)[:200]}")
    else:
        print("\nNo n8n workflows found (is n8n running on localhost:5678?). Skip execute test.")

    print("\n=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
