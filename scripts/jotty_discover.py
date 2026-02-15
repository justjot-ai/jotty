#!/usr/bin/env python3
"""
Jotty Discover - Find the right swarm/skill for your task
==========================================================

Helps users quickly find which swarm or skill to use.

Usage:
    python scripts/jotty_discover.py "create learning materials"
    python scripts/jotty_discover.py "write code"
    python scripts/jotty_discover.py "analyze stock"
"""

import sys
from typing import List, Tuple

# Task → Swarm mapping
SWARM_MAPPING = {
    # Learning & Education
    ("learning", "education", "teach", "study", "lesson", "tutorial", "k-12", "student"): {
        "swarm": "olympiad_learning_swarm",
        "description": "Generate educational content with PDF/HTML output",
        "example": 'await learn_topic("math", "Calculus Basics", "Student")'
    },
    ("arxiv", "paper", "research paper", "academic", "publication"): {
        "swarm": "arxiv_learning_swarm",
        "description": "Transform research papers into learning materials",
        "example": 'await learn_paper("2301.00001")'
    },

    # Development
    ("code", "program", "develop", "implement", "build", "api", "function"): {
        "swarm": "coding_swarm",
        "description": "Generate, edit, and review code with tests",
        "example": 'swarm.execute("Create REST API for users")'
    },
    ("test", "testing", "qa", "quality", "unit test"): {
        "swarm": "testing_swarm",
        "description": "Generate comprehensive test suites",
        "example": 'swarm.execute(code_path="app.py")'
    },
    ("review", "code review", "audit", "quality check"): {
        "swarm": "review_swarm",
        "description": "Code review and quality analysis",
        "example": 'swarm.execute(code_path="src/")'
    },

    # Research & Analysis
    ("research", "investigate", "analyze", "report"): {
        "swarm": "research_swarm",
        "description": "Deep research with web search and PDF reports",
        "example": 'swarm.execute(ticker="AAPL") or research_topic("AI trends")'
    },
    ("stock", "ticker", "company", "fundamental", "financial"): {
        "swarm": "fundamental_swarm",
        "description": "Financial statement and fundamental analysis",
        "example": 'swarm.execute(ticker="MSFT")'
    },
    ("data", "dataset", "statistics", "visualization", "chart"): {
        "swarm": "data_analysis_swarm",
        "description": "Data analysis and visualization",
        "example": 'swarm.execute(data_path="sales.csv")'
    },

    # DevOps & Infrastructure
    ("docker", "kubernetes", "deploy", "ci/cd", "devops", "infrastructure"): {
        "swarm": "devops_swarm",
        "description": "Infrastructure and deployment automation",
        "example": 'swarm.execute("Create Dockerfile for FastAPI")'
    },

    # Writing & Content
    ("write", "blog", "article", "content", "copy", "marketing"): {
        "swarm": "idea_writer_swarm",
        "description": "Creative writing and content generation",
        "example": 'swarm.execute(topic="AI in Healthcare", content_type="blog_post")'
    },
}


def find_swarm(query: str) -> List[Tuple[str, dict, int]]:
    """Find matching swarms for a query."""
    query_lower = query.lower()
    matches = []

    for keywords, swarm_info in SWARM_MAPPING.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            matches.append((swarm_info["swarm"], swarm_info, score))

    # Sort by relevance score
    matches.sort(key=lambda x: x[2], reverse=True)
    return matches


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/jotty_discover.py <task description>")
        print("\nExamples:")
        print('  jotty_discover "create learning materials for math"')
        print('  jotty_discover "write code for REST API"')
        print('  jotty_discover "analyze stock fundamentals"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"🔍 Searching for: {query}\n")

    matches = find_swarm(query)

    if not matches:
        print("❌ No matching swarms found.")
        print("\nAvailable categories:")
        print("  - Learning & Education (olympiad_learning_swarm, arxiv_learning_swarm)")
        print("  - Development (coding_swarm, testing_swarm, review_swarm)")
        print("  - Research (research_swarm, fundamental_swarm, data_analysis_swarm)")
        print("  - DevOps (devops_swarm)")
        print("  - Writing (idea_writer_swarm)")
        sys.exit(0)

    print(f"✅ Found {len(matches)} matching swarm(s):\n")

    for i, (swarm_name, info, score) in enumerate(matches[:5], 1):
        relevance = "🔥" * min(score, 3)
        print(f"{i}. {swarm_name} {relevance}")
        print(f"   {info['description']}")
        print(f"   Example: {info['example']}")
        print()

    # Show import path for top match
    top_swarm = matches[0][0]
    print("📦 Quick Start:")
    if "olympiad" in top_swarm or "arxiv" in top_swarm or "research" in top_swarm:
        module = f"Jotty.core.swarms.{top_swarm}"
        print(f"   from {module} import *")
    else:
        print(f"   from Jotty.core.intelligence.swarms import {top_swarm.replace('_swarm', '').title()}Swarm")


if __name__ == "__main__":
    main()
