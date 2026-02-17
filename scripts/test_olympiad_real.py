#!/usr/bin/env python3
"""
Test OlympiadLearningSwarm with Real LLM
=========================================

Focused test of OlympiadLearningSwarm which is confirmed to work.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add Jotty to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Verify API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    print("❌ ANTHROPIC_API_KEY not found")
    sys.exit(1)
print(f"✅ API key loaded: {api_key[:20]}...")


async def test_olympiad_swarm():
    """Test OlympiadLearningSwarm with real LLM."""
    print("\n" + "=" * 80)
    print("🎓 Testing OlympiadLearningSwarm with REAL LLM")
    print("=" * 80)
    print(f"API Key: {api_key[:20]}...")
    print(f"Provider: Anthropic Claude")

    try:
        from Jotty.core.intelligence.orchestration.swarms.olympiad_learning_swarm import (
            OlympiadLearningSwarm,
        )
        from Jotty.core.intelligence.orchestration.swarms.olympiad_learning_swarm.types import (
            OlympiadLearningConfig,
            Subject,
            LessonDepth,
            DifficultyTier,
        )

        print("\n📝 Creating swarm with config...")
        config = OlympiadLearningConfig(subject=Subject.MATHEMATICS, send_telegram=False)
        swarm = OlympiadLearningSwarm(config=config)

        print("🚀 Starting teaching task...")
        print("   Subject: Mathematics")
        print("   Topic: Basic Addition")
        print("   Depth: Quick")
        print("   Target: Foundation")

        result = await swarm.teach(
            topic="Basic Addition",
            subject=Subject.MATHEMATICS,
            student_name="TestStudent",
            depth=LessonDepth.QUICK,
            target_tier=DifficultyTier.FOUNDATION,
            send_telegram=False,
        )

        print("\n" + "=" * 80)
        print("✅ OlympiadLearningSwarm SUCCESS!")
        print("=" * 80)

        if hasattr(result, "html_content"):
            print(f"📄 HTML content generated: {len(result.html_content):,} characters")
        if hasattr(result, "pdf_path"):
            print(f"📑 PDF path: {result.pdf_path}")
        if hasattr(result, "concepts_count"):
            print(f"🧠 Concepts taught: {result.concepts_count}")
        if hasattr(result, "problems_count"):
            print(f"📊 Practice problems: {result.problems_count}")

        print("\n🎯 Result type:", type(result).__name__)
        print("🎯 Result attributes:", dir(result))

        return {"status": "success", "swarm": "OlympiadLearningSwarm", "result": result}

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ OlympiadLearningSwarm FAILED!")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return {"status": "failed", "swarm": "OlympiadLearningSwarm", "error": str(e)}


if __name__ == "__main__":
    try:
        result = asyncio.run(test_olympiad_swarm())

        if result["status"] == "success":
            print("\n🎉 TEST PASSED!")
            sys.exit(0)
        else:
            print("\n💥 TEST FAILED!")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test crashed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
