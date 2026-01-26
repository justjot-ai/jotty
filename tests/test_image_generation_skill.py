#!/usr/bin/env python3
"""
Test Image Generation Skill Creation

Creates a skill that uses open-source image generation models
(no API key required).
"""

import sys
import os
from pathlib import Path

# Add Jotty to path
sys.path.insert(0, str(Path(__file__).parent))

from core.registry.skill_generator import get_skill_generator
from core.registry.skills_registry import get_skills_registry
from core.foundation.unified_lm_provider import UnifiedLMProvider


def test_image_generation_skill():
    """Test creating image generation skill."""
    
    print("=" * 60)
    print("Testing Image Generation Skill Creation")
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
    
    # Step 2: Create registry and generator
    print("Step 2: Creating registry and generator...")
    registry = get_skills_registry()
    generator = get_skill_generator(lm=lm, skills_registry=registry)
    
    print(f"  ✅ Registry: {registry.skills_dir}")
    print(f"  ✅ Generator: {generator.skills_dir}")
    print()
    
    # Step 3: Generate image generation skill
    print("Step 3: Generating image generation skill...")
    skill_name = "image-generator"
    description = "Generate images using open-source models like Stable Diffusion, Flux, or SDXL. No API key required."
    requirements = "Use open-source models via Hugging Face transformers or diffusers library. Support models like: stabilityai/stable-diffusion-xl-base-1.0, black-forest-labs/FLUX.1-dev, runwayml/stable-diffusion-v1-5"
    examples = [
        "Generate image of a sunset over mountains",
        "Create a picture of a cat wearing sunglasses",
        "Generate abstract art with blue and green colors"
    ]
    
    print(f"  Skill name: {skill_name}")
    print(f"  Description: {description}")
    print()
    print("  Generating skill files with LLM...")
    print("  (This may take a moment - generating code for image generation)")
    
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
    
    # Step 4: Verify skill is loaded
    print("Step 4: Verifying skill is loaded...")
    
    skill = registry.get_skill(skill_name)
    if not skill:
        print("  ❌ Skill not found in registry!")
        return False
    
    print(f"  ✅ Skill found: {skill.name}")
    print(f"  📦 Tools: {list(skill.tools.keys())}")
    print()
    
    # Step 5: Show generated code
    print("Step 5: Generated code preview...")
    tools_py_path = Path(result['tools_py'])
    if tools_py_path.exists():
        content = tools_py_path.read_text()
        lines = content.split('\n')
        print(f"  📄 tools.py ({len(lines)} lines)")
        print("  First 30 lines:")
        print("-" * 60)
        for i, line in enumerate(lines[:30], 1):
            print(f"  {i:3}: {line}")
        if len(lines) > 30:
            print(f"  ... ({len(lines) - 30} more lines)")
        print("-" * 60)
        print()
    
    print("=" * 60)
    print("✅ Test Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Install required dependencies (diffusers, torch, etc.)")
    print("  2. Test image generation")
    print("  3. Use skill in agents")
    print()
    
    return True


if __name__ == "__main__":
    success = test_image_generation_skill()
    sys.exit(0 if success else 1)
