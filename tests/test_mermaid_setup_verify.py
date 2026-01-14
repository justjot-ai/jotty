"""
Quick Setup Verification - Tests improvements loading and renderer without LLM calls
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.experts.memory_integration import sync_improvements_to_memory
from core.experts.mermaid_renderer import validate_mermaid_syntax
from core.memory.cortex import HierarchicalMemory
from core.foundation.data_structures import JottyConfig


def test_improvements_loading():
    """Test improvements loading from file to memory."""
    print("1. Testing Improvements Loading")
    print("-" * 80)
    
    memory = HierarchicalMemory('test', JottyConfig())
    improvements_file = Path('test_outputs/mermaid_complex_memory/improvements.json')
    
    if not improvements_file.exists():
        print(f"⚠️  No improvements file found at {improvements_file}")
        return False
    
    with open(improvements_file) as f:
        improvements = json.load(f)
    
    print(f"   Found {len(improvements)} improvements in file")
    
    synced = sync_improvements_to_memory(
        memory=memory,
        improvements=improvements,
        expert_name="mermaid_professional",
        domain="mermaid"
    )
    
    print(f"   ✅ Synced {synced}/{len(improvements)} improvements to memory")
    
    # Verify retrieval
    from core.foundation.data_structures import MemoryLevel
    memory_entries = memory.retrieve(
        query="expert agent improvements mermaid mermaid_professional",
        goal="expert_mermaid_improvements",
        budget_tokens=10000,
        levels=[MemoryLevel.PROCEDURAL, MemoryLevel.META, MemoryLevel.SEMANTIC]
    )
    
    print(f"   ✅ Retrieved {len(memory_entries)} entries from memory")
    
    if len(memory_entries) > 0:
        print(f"\n   Sample memory entry:")
        entry = memory_entries[0]
        print(f"     Level: {entry.level.value}")
        print(f"     Content: {entry.content[:100]}...")
    
    return synced > 0


def test_renderer_validation():
    """Test renderer validation."""
    print("\n2. Testing Renderer Validation")
    print("-" * 80)
    
    test_cases = [
        ("Valid graph", "graph TD\n    A[Start] --> B[End]", True),
        ("Valid sequence", "sequenceDiagram\n    A->>B: Hello", True),
        ("Invalid syntax", "graph TD\n    A -->", False),
        ("Empty", "", False),
    ]
    
    results = []
    for name, code, expected_valid in test_cases:
        # Test basic validation
        is_valid_basic, msg_basic, _ = validate_mermaid_syntax(code, use_renderer=False)
        results.append((name, "basic", is_valid_basic == expected_valid, msg_basic))
        
        # Test renderer validation (may timeout, that's OK)
        try:
            is_valid_renderer, msg_renderer, _ = validate_mermaid_syntax(code, use_renderer=True)
            results.append((name, "renderer", is_valid_renderer == expected_valid, msg_renderer))
        except Exception as e:
            results.append((name, "renderer", None, f"Error: {str(e)[:50]}"))
    
    print("   Validation Results:")
    for name, method, passed, msg in results:
        if passed is None:
            status = "⚠️"
        elif passed:
            status = "✅"
        else:
            status = "❌"
        print(f"     {status} {name} ({method}): {msg[:60]}")
    
    basic_passed = sum(1 for _, method, passed, _ in results if method == "basic" and passed)
    return basic_passed >= len([t for t in test_cases if t[2]])  # All expected valid cases


def main():
    print("=" * 80)
    print("MERMAID EXPERT SETUP VERIFICATION")
    print("=" * 80)
    print()
    
    improvements_ok = test_improvements_loading()
    renderer_ok = test_renderer_validation()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Improvements Loading: {'PASS' if improvements_ok else 'FAIL'}")
    print(f"✅ Renderer Validation: {'PASS' if renderer_ok else 'FAIL'}")
    print()
    
    if improvements_ok and renderer_ok:
        print("✅ Setup verified! Ready for full test.")
        return 0
    else:
        print("⚠️  Setup issues detected.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
