"""
Train PlantUML Expert from GitHub Examples

1. Downloads examples from GitHub (or uses cached JSON)
2. Trains expert with those examples
3. Saves training results
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.experts import ExpertAgentConfig, PlantUMLExpertAgent
from core.experts.plantuml_expert import PlantUMLExpertAgent as PlantUMLExpert


async def download_and_train():
    """Download GitHub examples and train expert."""
    print("=" * 80)
    print("PLANTUML EXPERT TRAINING FROM GITHUB EXAMPLES")
    print("=" * 80)
    print()

    # Step 1: Check/Download GitHub examples
    print("Step 1: Loading Gold Standards from GitHub")
    print("-" * 80)

    json_file = Path("./expert_data/plantuml_expert/github_training_examples.json")

    # Check if JSON exists and has data
    gold_standards = []
    if json_file.exists():
        print(f"✅ Found existing JSON file: {json_file}")
        with open(json_file) as f:
            data = json.load(f)
            gold_standards = data.get("gold_standards", [])

        if gold_standards:
            print(f"✅ Loaded {len(gold_standards)} gold standards from JSON cache")
            print(f"   Source: {data.get('source', 'unknown')}")
            print(f"   Loaded at: {data.get('loaded_at', 'unknown')}")
        else:
            print("⚠️  JSON file exists but has no gold_standards")
            print("   Downloading from GitHub...")
            gold_standards = await PlantUMLExpert.load_training_examples_from_github(
                repo_url="https://github.com/joelparkerhenderson/plantuml-examples",
                max_examples=50,  # Download 50 examples
                save_to_file=True,
            )
    else:
        print("⚠️  JSON file not found")
        print("   Downloading from GitHub...")
        gold_standards = await PlantUMLExpert.load_training_examples_from_github(
            repo_url="https://github.com/joelparkerhenderson/plantuml-examples",
            max_examples=50,  # Download 50 examples
            save_to_file=True,
        )

    if not gold_standards:
        print("❌ No gold_standards loaded!")
        print("   This may be due to:")
        print("   - GitHub API rate limits")
        print("   - Network issues")
        print("   - Repository structure changes")
        print()
        print("   Falling back to default training cases...")
        # Use default cases as fallback
        expert = PlantUMLExpertAgent()
        gold_standards = expert._get_default_training_cases()
        print(f"✅ Using {len(gold_standards)} default training cases")
    else:
        print(f"✅ Ready to train with {len(gold_standards)} gold standards")

    print()

    # Step 2: Create expert
    print("Step 2: Creating PlantUML Expert Agent")
    print("-" * 80)

    try:
        import dspy
        from examples.claude_cli_wrapper import ClaudeCLILM

        # Initialize Claude CLI and configure DSPy
        lm = ClaudeCLILM(model="sonnet")
        dspy.configure(lm=lm)
        print("✅ Claude CLI initialized and DSPy configured")

        config = ExpertAgentConfig(
            name="PlantUML Expert",
            description="Expert for generating PlantUML diagrams",
            domain="plantuml",
        )

        expert = PlantUMLExpertAgent(config=config)
        print("✅ Expert agent created")
        print()
    except Exception as e:
        print(f"❌ Failed to create expert: {e}")
        print("   Claude CLI may not be installed")
        print("   Install from: https://github.com/anthropics/claude-code")
        return

    # Step 3: Train expert
    print("Step 3: Training Expert")
    print("-" * 80)
    print(f"Training with {len(gold_standards)} gold standards...")
    print()
    print("Training process:")
    print("  1. Pre-training: Extract patterns from gold_standards")
    print("  2. Iterative learning: Learn from mistakes via teacher")
    print("  3. Store improvements in memory")
    print()

    try:
        # Configure optimization pipeline for training
        # The train() method will create the pipeline with these settings
        training_result = await expert.train(
            gold_standards=gold_standards,
            enable_pre_training=True,  # Extract patterns first
            training_mode="both",  # Both pre-training and iterative learning
            force_retrain=False,  # Don't retrain if already trained
        )

        print("✅ Training completed!")
        print()

        # Show training results
        print("Training Results:")
        print("-" * 80)
        print(f"   Iterations: {training_result.get('iterations', 0)}")
        print(f"   Final score: {training_result.get('final_score', 0):.2f}")
        print(f"   Improvements learned: {len(training_result.get('improvements', []))}")
        print(f"   Patterns extracted: {training_result.get('patterns_learned', 0)}")
        print()

        # Show sample improvements
        improvements = training_result.get("improvements", [])
        if improvements:
            print("Sample Learned Improvements:")
            for i, imp in enumerate(improvements[:5], 1):
                pattern = imp.get("learned_pattern", "N/A")
                print(f"   {i}. {pattern[:80]}...")
            print()

        # Check if expert is marked as trained
        if expert.trained:
            print("✅ Expert is now marked as TRAINED")
            print("   Can now be used for generation")
        else:
            print("⚠️  Expert not marked as trained (may need more training)")

        # Save training results
        results_file = Path("./expert_data/plantuml_expert/training_results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, "w") as f:
            json.dump(
                {
                    "training_date": datetime.now().isoformat(),
                    "gold_standards_count": len(gold_standards),
                    "training_result": training_result,
                },
                f,
                indent=2,
                default=str,
            )

        print(f"✅ Training results saved to: {results_file}")
        print()

    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback

        traceback.print_exc()
        return

    # Step 4: Verify training
    print("Step 4: Verifying Training")
    print("-" * 80)

    try:
        # Try a simple generation to verify expert is trained
        test_output = await expert.generate_plantuml(
            description="Simple sequence diagram: User -> System: Hello", diagram_type="sequence"
        )

        if test_output:
            print("✅ Generation test successful!")
            print(f"   Output preview: {test_output[:100]}...")
        else:
            print("⚠️  Generation returned empty output")

    except RuntimeError as e:
        if "must be trained" in str(e):
            print("❌ Expert still requires training")
            print("   Training may not have completed successfully")
        else:
            print(f"❌ Generation error: {e}")
    except Exception as e:
        print(f"⚠️  Generation test error: {e}")

    print()

    # Summary
    print("=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print()
    print(f"✅ Gold Standards: {len(gold_standards)} loaded")
    print(f"✅ Expert: Created and configured")
    print(f"✅ Training: {'Completed' if expert.trained else 'May need more'}")
    print(f"✅ Improvements: {len(improvements)} learned")
    print()

    if expert.trained:
        print("🎉 Expert is ready to use!")
        print("   You can now run: python tests/test_plantuml_5_complex_cases.py")
    else:
        print("⚠️  Expert may need additional training")
        print("   Try running training again or check for errors")
    print()


if __name__ == "__main__":
    print("Training PlantUML Expert from GitHub Examples")
    print()
    asyncio.run(download_and_train())
