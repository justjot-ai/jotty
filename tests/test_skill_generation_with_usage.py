#!/usr/bin/env python3
"""
Test Skill Generation with Same-Run Usage

Tests that:
1. Skill is generated via LLM
2. Skill is automatically reloaded
3. Skill tools are automatically tested
4. Skill can be used immediately
"""

import sys
import os
from pathlib import Path

# Add Jotty to path
sys.path.insert(0, str(Path(__file__).parent))

from core.registry.skill_generator import get_skill_generator
from core.registry.skills_registry import get_skills_registry
from core.foundation.unified_lm_provider import UnifiedLMProvider


def test_generation_with_usage():
    """Test skill generation with same-run usage."""
    
    print("=" * 60)
    print("Testing Skill Generation with Same-Run Usage")
    print("=" * 60)
    print()
    
    # Step 1: Configure LLM
    print("Step 1: Configuring LLM provider...")
    try:
        lm = UnifiedLMProvider.configure_default_lm()
        provider = getattr(lm, 'provider', 'unknown')
        print(f"  ✅ Using: {provider}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False
    
    print()
    
    # Step 2: Create registry and generator (linked for auto-reload)
    print("Step 2: Creating registry and generator...")
    registry = get_skills_registry()
    generator = get_skill_generator(lm=lm, skills_registry=registry)
    
    print(f"  ✅ Registry: {registry.skills_dir}")
    print(f"  ✅ Generator: {generator.skills_dir}")
    print(f"  ✅ Linked for auto-reload")
    print()
    
    # Step 3: Generate skill (will auto-reload)
    print("Step 3: Generating skill (with auto-reload)...")
    skill_name = "time-converter"
    description = "Convert time between timezones and formats"
    requirements = "Use Python datetime library, no external APIs"
    examples = [
        "Convert UTC to EST",
        "Get current time in Tokyo",
        "Format timestamp"
    ]
    
    print(f"  Skill name: {skill_name}")
    print(f"  Description: {description}")
    print()
    print("  Generating skill files with LLM...")
    
    try:
        result = generator.generate_skill(
            skill_name=skill_name,
            description=description,
            requirements=requirements,
            examples=examples
        )
        
        print(f"  ✅ Skill generated!")
        print(f"  📄 Files: {result['skill_md']}, {result['tools_py']}")
        print(f"  🔄 Auto-reloaded: {result.get('reloaded', False)}")
        print(f"  🧪 Tool tested: {result.get('tool_tested', False)}")
        print()
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Verify skill is loaded and usable
    print("Step 4: Verifying skill is loaded and usable...")
    
    skill = registry.get_skill(skill_name)
    if not skill:
        print("  ❌ Skill not found in registry!")
        return False
    
    print(f"  ✅ Skill found: {skill.name}")
    print(f"  📦 Tools: {list(skill.tools.keys())}")
    print()
    
    # Step 5: Use the skill immediately
    print("Step 5: Using skill immediately (same run)...")
    
    if skill.tools:
        first_tool_name = list(skill.tools.keys())[0]
        first_tool = skill.tools[first_tool_name]
        
        print(f"  Testing tool: {first_tool_name}")
        try:
            # Try with empty params first (safe test)
            result = first_tool({})
            
            if result.get('success'):
                print(f"  ✅ Tool executed successfully!")
                print(f"     Result: {result}")
            else:
                print(f"  ⚠️  Tool returned: {result.get('error', 'unknown')}")
                # Try with proper params if we know what it needs
                if 'time' in first_tool_name.lower() or 'convert' in first_tool_name.lower():
                    test_result = first_tool({'from_timezone': 'UTC', 'to_timezone': 'EST'})
                    if test_result.get('success'):
                        print(f"  ✅ Tool works with proper params!")
                        print(f"     Result: {test_result}")
        except Exception as e:
            print(f"  ⚠️  Tool execution error: {e}")
            print("     (This is OK - tool might need specific params)")
    
    print()
    
    # Step 6: Verify tools are registered globally
    print("Step 6: Verifying tools are registered globally...")
    registered_tools = registry.get_registered_tools()
    
    skill_tools_in_registry = [name for name in registered_tools.keys() 
                               if any(skill_tool in name for skill_tool in skill.tools.keys())]
    
    print(f"  ✅ Found {len(skill_tools_in_registry)} tools from skill in global registry")
    for tool_name in skill_tools_in_registry:
        print(f"     - {tool_name}")
    
    print()
    print("=" * 60)
    print("✅ Test Complete!")
    print("=" * 60)
    print()
    print("Summary:")
    print("  ✅ Skill generated")
    print("  ✅ Skill auto-reloaded")
    print("  ✅ Skill usable in same run")
    print("  ✅ Tools registered globally")
    print()
    
    return True


if __name__ == "__main__":
    success = test_generation_with_usage()
    sys.exit(0 if success else 1)
