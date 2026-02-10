#!/usr/bin/env python3
"""
Generate a world-class learning module using OlympiadLearningSwarm.

Supports any subject + topic. Pre-configures DirectAnthropicLM for API speed.
Includes live cost tracking per LLM call and final cost summary.

Usage:
    # Default: math fractions
    python scripts/generate_5th_grade_fractions.py

    # Science body parts
    python scripts/generate_5th_grade_fractions.py --subject biology --topic "Human Body Parts and Organ Systems"

    # Any subject/topic
    python scripts/generate_5th_grade_fractions.py --subject physics --topic "Forces and Motion" --student "Aria"
"""

import asyncio
import argparse
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Ensure Jotty is on path
repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))

# Load .env so API keys are available
try:
    from Jotty.core.utils.env_loader import load_jotty_env, get_jotty_root
    load_jotty_env()
    anth = get_jotty_root() / ".env.anthropic"
    if anth.exists():
        load_jotty_env(str(anth), override=False)
except Exception:
    for env_file in [repo / ".env", repo / ".env.anthropic"]:
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip('"').strip("'")
                        if v and k not in os.environ:
                            os.environ[k] = v

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate learning module via OlympiadLearningSwarm")
    parser.add_argument("--subject", default="mathematics", help="Subject (mathematics, biology, physics, chemistry, cs, astronomy, general)")
    parser.add_argument("--topic", default="Fractions - meaning of fractions, equivalent fractions", help="Topic to teach")
    parser.add_argument("--student", default="Aria", help="Student name")
    parser.add_argument("--depth", default="standard", choices=["quick", "standard", "deep", "comprehensive"], help="Lesson depth")
    parser.add_argument("--target", default="5th_grader", help="Target level")
    parser.add_argument("--no-telegram", action="store_true", help="Skip sending to Telegram")
    return parser.parse_args()


async def main():
    args = parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not found. Check Jotty/.env")
        return None
    print("API key loaded from .env")

    # Pre-configure DSPy with DirectAnthropicLM (fast API, not CLI)
    from Jotty.core.foundation.direct_anthropic_lm import configure_direct_anthropic
    configure_direct_anthropic(model="haiku")
    print("Pre-configured DSPy with DirectAnthropicLM (API)")

    from Jotty.core.swarms.olympiad_learning_swarm import (
        OlympiadLearningSwarm, OlympiadLearningConfig, Subject, LessonDepth, DifficultyTier,
    )

    # Map subject string to enum
    subject_map = {s.value: s for s in Subject}
    subject_enum = subject_map.get(args.subject, Subject.GENERAL)

    depth_map = {d.value: d for d in LessonDepth}
    depth_enum = depth_map.get(args.depth, LessonDepth.STANDARD)

    config = OlympiadLearningConfig(
        subject=subject_enum,
        student_name=args.student,
        depth=depth_enum,
        target_tier=DifficultyTier.OLYMPIAD,
        send_telegram=not args.no_telegram,
        generate_pdf=True,
        generate_html=True,
    )

    swarm = OlympiadLearningSwarm(config)

    print()
    print("=" * 60)
    print(f"  Subject: {args.subject}")
    print(f"  Topic: {args.topic}")
    print(f"  Student: {args.student}")
    print(f"  Target: {args.target}")
    print(f"  Depth: {args.depth}")
    print(f"  Telegram: {'No' if args.no_telegram else 'Yes'}")
    print("=" * 60)
    print()

    start = datetime.now()
    result = await swarm.teach(
        topic=args.topic,
        student_name=args.student,
        send_telegram=not args.no_telegram,
    )
    elapsed = (datetime.now() - start).total_seconds()

    print()
    print("=" * 60)
    if result.success:
        content = result.content
        print(f"SUCCESS: {len(content.core_concepts) if content else 0} concepts, "
              f"{result.problems_generated} problems")
        print(f"Breakthroughs: {result.breakthrough_moments}")
        print(f"Words: {content.total_words if content else 0}")
        print(f"Rank tips: {len(content.rank_tips) if content else 0}")
        if result.pdf_path:
            print(f"PDF: {result.pdf_path}")
        if result.html_path:
            print(f"HTML: {result.html_path}")

        # Show first 3 rank tips
        if content and content.rank_tips:
            print(f"\nFirst 3 rank tips:")
            for tip in content.rank_tips[:3]:
                print(f"  {tip[:100]}")
    else:
        print(f"FAILED: {result.error}")

    # Final cost summary
    try:
        from Jotty.core.foundation.direct_anthropic_lm import get_cost_tracker
        tracker = get_cost_tracker()
        metrics = tracker.get_metrics()
        print(f"\n{'─' * 40}")
        print(f"COST REPORT")
        print(f"  Total cost:    ${metrics.total_cost:.4f}")
        print(f"  Total calls:   {metrics.total_calls}")
        print(f"  Input tokens:  {metrics.total_input_tokens:,}")
        print(f"  Output tokens: {metrics.total_output_tokens:,}")
        print(f"  Avg cost/call: ${metrics.avg_cost_per_call:.4f}")
        print(f"  Time:          {elapsed:.1f}s")
        for model, cost in metrics.cost_by_model.items():
            calls = metrics.calls_by_model.get(model, 0)
            print(f"  {model}: ${cost:.4f} ({calls} calls)")
        print(f"{'─' * 40}")
    except Exception:
        pass

    print("=" * 60)
    return result


if __name__ == "__main__":
    asyncio.run(main())
