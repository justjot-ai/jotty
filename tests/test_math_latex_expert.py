"""
Test Math LaTeX Expert Agent

Tests the generic expert agent architecture with Math LaTeX domain.
"""

import asyncio
import sys
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.experts import MathLaTeXExpertAgent, ExpertAgentConfig
from core.experts.math_latex_renderer import validate_math_latex_syntax


# Test Cases for Math LaTeX
TEST_CASES = [
    {
        "name": "1. Quadratic Formula",
        "type": "display",
        "description": "Generate the quadratic formula: x = (-b ± √(b² - 4ac)) / 2a",
        "required_elements": ["frac", "sqrt", "pm", "^"],
        "gold_standard": "$$\\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$"
    },
    {
        "name": "2. Pythagorean Theorem",
        "type": "display",
        "description": "Pythagorean theorem: a² + b² = c²",
        "required_elements": ["^", "="],
        "gold_standard": "$$a^2 + b^2 = c^2$$"
    },
    {
        "name": "3. Euler's Identity",
        "type": "display",
        "description": "Euler's identity: e^(iπ) + 1 = 0",
        "required_elements": ["e", "pi", "^", "="],
        "gold_standard": "$$e^{i\\pi} + 1 = 0$$"
    },
    {
        "name": "4. Integral Formula",
        "type": "display",
        "description": "Definite integral from a to b of f(x)",
        "required_elements": ["int", "dx"],
        "gold_standard": "$$\\int_a^b f(x) \\, dx$$"
    },
    {
        "name": "5. Sum Formula",
        "type": "display",
        "description": "Sum from i=1 to n: Σi = n(n+1)/2",
        "required_elements": ["sum", "frac", "="],
        "gold_standard": "$$\\sum_{i=1}^n i = \\frac{n(n+1)}{2}$$"
    },
    {
        "name": "6. Complex Expression (Large - 414 test)",
        "type": "display",
        "description": "Complex mathematical expression with multiple fractions, roots, and integrals",
        "required_elements": ["frac", "sqrt", "int", "sum"],
        "gold_standard": "$$\\frac{\\sum_{i=1}^n \\sqrt{\\int_0^1 x^i \\, dx}}{\\prod_{j=1}^m \\frac{j}{j+1}}$$"
    }
]


async def test_math_latex_expert():
    """Test Math LaTeX expert agent."""
    print("=" * 80)
    print("MATH LATEX EXPERT TEST")
    print("Testing Generic Expert Agent Architecture")
    print("=" * 80)
    print()
    
    # Step 1: Create expert
    print("Step 1: Creating Math LaTeX Expert Agent")
    print("-" * 80)
    
    try:
        from examples.claude_cli_wrapper import ClaudeCLILM
        import dspy
        
        # Initialize Claude CLI and configure DSPy
        lm = ClaudeCLILM(model="sonnet")
        dspy.configure(lm=lm)
        print("✅ Claude CLI initialized and DSPy configured")
        
        config = ExpertAgentConfig(
            name="Math LaTeX Expert",
            description="Expert for generating Math LaTeX expressions",
            domain="math_latex"
        )
        
        expert = MathLaTeXExpertAgent(config=config)
        print("✅ Expert agent created")
        print()
    except Exception as e:
        print(f"❌ Failed to create expert: {e}")
        print("   Claude CLI may not be installed")
        print("   Install from: https://github.com/anthropics/claude-code")
        return
    
    # Step 2: Quick training
    print("Step 2: Quick Training")
    print("-" * 80)
    
    try:
        default_gold_standards = expert._get_default_training_cases()
        print(f"Training with {len(default_gold_standards)} default cases...")
        
        training_result = await expert.train(
            gold_standards=default_gold_standards,
            enable_pre_training=True,
            training_mode="pattern_extraction",  # Quick pre-training only
            force_retrain=False
        )
        
        print("✅ Training completed")
        print(f"   Patterns learned: {training_result.get('patterns_learned', 0)}")
        print(f"   Expert trained: {expert.trained}")
        print()
    except Exception as e:
        print(f"⚠️  Training error: {e}")
        print("   Will attempt generation anyway")
        print()
    
    # Step 3: Test generation
    print("Step 3: Testing Generation")
    print("=" * 80)
    print()
    
    results = []
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"Test {i}/{len(TEST_CASES)}: {test_case['name']}")
        print("-" * 80)
        print(f"Type: {test_case['type']}")
        print(f"Description: {test_case['description']}")
        print()
        
        try:
            # Generate LaTeX expression
            print("Generating Math LaTeX expression...")
            output = await expert.generate_math_latex(
                description=test_case['description'],
                expression_type=test_case['type'],
                required_elements=test_case['required_elements']
            )
            
            if not output:
                print("❌ No output generated")
                results.append({
                    "case": test_case['name'],
                    "status": "failed",
                    "error": "No output"
                })
                continue
            
            # Check output size
            output_length = len(output)
            print(f"✅ Generated output: {output_length} characters")
            print(f"   Preview: {output[:100]}...")
            print()
            
            # Validate syntax
            print("Validating syntax...")
            is_valid, error_msg, metadata = validate_math_latex_syntax(
                output,
                use_renderer=True
            )
            
            validation_method = metadata.get("validation_method", metadata.get("method", "renderer"))
            status_code = metadata.get("status_code", "N/A")
            
            if is_valid:
                print(f"✅ Valid: True, Method: {validation_method}, Status: {status_code}")
            else:
                print(f"❌ Valid: False, Method: {validation_method}, Status: {status_code}")
                print(f"   Error: {error_msg}")
            
            # Check for 414 error
            has_414 = "414" in str(status_code) or "414" in error_msg
            if has_414:
                print("⚠️  HTTP 414 detected - Large expression (expected for case 6)")
                print("   Using fallback validation (structure-based)")
            
            # Check expected elements
            found_elements = []
            for element in test_case['required_elements']:
                if element.lower() in output.lower() or f"\\{element}" in output:
                    found_elements.append(element)
            
            element_coverage = len(found_elements) / len(test_case['required_elements']) * 100
            print(f"📊 Elements: {len(found_elements)}/{len(test_case['required_elements'])} ({element_coverage:.0f}%)")
            if found_elements:
                print(f"   Found: {', '.join(found_elements)}")
            
            # Check type
            output_lower = output.lower()
            has_display = '$$' in output or '\\[' in output or '\\begin{equation}' in output_lower
            has_inline = '$' in output and not has_display
            type_match = (
                (test_case['type'] == "display" and has_display) or
                (test_case['type'] == "inline" and has_inline)
            )
            
            print(f"📋 Type: {test_case['type']} ({'✓' if type_match else '✗'})")
            print(f"📋 Delimiters: Display ({'✓' if has_display else '✗'}), Inline ({'✓' if has_inline else '✗'})")
            
            # Store result
            results.append({
                "case": test_case['name'],
                "status": "success" if is_valid else "validation_failed",
                "output_length": output_length,
                "is_valid": is_valid,
                "validation_method": validation_method,
                "status_code": status_code,
                "has_414": has_414,
                "element_coverage": element_coverage,
                "type_match": type_match,
                "has_delimiters": has_display or has_inline,
                "error": error_msg if not is_valid else None
            })
            
        except RuntimeError as e:
            if "must be trained" in str(e):
                print("❌ Expert not trained - skipping")
                results.append({
                    "case": test_case['name'],
                    "status": "error",
                    "error": "Expert not trained"
                })
            else:
                print(f"❌ Error: {e}")
                results.append({
                    "case": test_case['name'],
                    "status": "error",
                    "error": str(e)
                })
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "case": test_case['name'],
                "status": "error",
                "error": str(e)
            })
        
        print()
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()
    
    total = len(results)
    successful = sum(1 for r in results if r.get("is_valid", False))
    has_414_cases = sum(1 for r in results if r.get("has_414", False))
    
    print(f"Total Cases: {total}")
    print(f"✅ Valid: {successful}/{total}")
    print(f"⚠️  HTTP 414 Cases: {has_414_cases}/{total}")
    print()
    
    print("Detailed Results:")
    for i, result in enumerate(results, 1):
        status_icon = "✅" if result.get("is_valid") else "❌"
        print(f"{status_icon} Case {i}: {result['case']}")
        print(f"   Status: {result['status']}")
        if result.get("output_length"):
            print(f"   Size: {result['output_length']} chars")
        if result.get("has_414"):
            print(f"   ⚠️  HTTP 414: Handled with fallback validation")
        if result.get("element_coverage"):
            print(f"   Elements: {result['element_coverage']:.0f}%")
        if result.get("error"):
            print(f"   Error: {result['error']}")
        print()
    
    # Save results
    results_file = Path("./test_outputs/math_latex_expert_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            "test_date": datetime.now().isoformat(),
            "total_cases": total,
            "successful": successful,
            "has_414_cases": has_414_cases,
            "results": results
        }, f, indent=2)
    
    print(f"✅ Results saved to: {results_file}")
    print()
    
    # Final verdict
    print("=" * 80)
    if successful == total:
        print("✅ ALL TESTS PASSED!")
        print("✅ Generic Expert Agent Architecture Verified!")
    elif successful >= total * 0.8:
        print("⚠️  MOSTLY PASSED (80%+)")
        print("✅ Generic architecture working!")
    else:
        print("❌ SOME TESTS FAILED")
        print("⚠️  May need more training or fixes")
    print("=" * 80)


if __name__ == "__main__":
    print("Testing Math LaTeX Expert - Verifying Generic Architecture")
    print()
    asyncio.run(test_math_latex_expert())
