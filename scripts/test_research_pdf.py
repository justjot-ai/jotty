#!/usr/bin/env python3
"""Quick test: ResearchSwarm + PDF generation for RELIANCE.NS."""

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    print("ANTHROPIC_API_KEY not found")
    sys.exit(1)
print(f"API key: {api_key[:20]}...")


async def main():
    from Jotty.core.execution.swarms.research_swarm import ResearchSwarm
    from Jotty.core.execution.swarms.research_swarm.types import ResearchConfig

    config = ResearchConfig(send_telegram=False)
    swarm = ResearchSwarm(config=config)

    start = time.time()
    result = await swarm.research(
        query="Reliance Industries", ticker="RELIANCE.NS", exchange="NSE", send_telegram=False
    )
    elapsed = time.time() - start

    print(f"\n=== RESEARCH SWARM RESULT ===")
    print(f"Success: {getattr(result, 'success', 'N/A')}")
    print(f"Ticker: {getattr(result, 'ticker', 'N/A')}")
    print(f"Company: {getattr(result, 'company_name', 'N/A')}")
    print(f"Rating: {getattr(result, 'rating', 'N/A')}")
    print(f"Price: {getattr(result, 'current_price', 'N/A')}")
    print(f"Target: {getattr(result, 'target_price', 'N/A')}")
    print(f"PDF: {getattr(result, 'pdf_path', 'N/A')}")
    print(f"Execution time: {elapsed:.1f}s")

    # Verify PDF
    pdf_path = getattr(result, "pdf_path", None)
    if pdf_path and Path(pdf_path).exists():
        size = Path(pdf_path).stat().st_size
        print(f"PDF SIZE: {size:,} bytes")
        print("PDF GENERATION: SUCCESS")
    else:
        print(f"PDF GENERATION: FAILED (path={pdf_path})")


if __name__ == "__main__":
    asyncio.run(main())
