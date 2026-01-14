"""
Interactive Mermaid Learning Demo

Shows how agent learns to generate perfect Mermaid diagrams.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.orchestration.optimization_pipeline import create_optimization_pipeline
from core.foundation.agent_config import AgentConfig


class MermaidAgent:
    def __init__(self):
        self.attempts = 0
        self.learned = None
    
    def forward(self, task: str = None, teacher_output: str = None, **kwargs) -> str:
        self.attempts += 1
        result = type('Result', (), {})()
        
        if teacher_output:
            self.learned = teacher_output
            result._store = {"output": teacher_output}
            return result
        
        if self.learned:
            result._store = {"output": self.learned}
            return result
        
        # Wrong outputs
        if self.attempts == 1:
            result._store = {"output": "graph A --> B"}  # Missing nodes
        else:
            result._store = {"output": "graph TD\n    A[Start]\n    B[End]"}  # Missing arrow
        
        return result


class Teacher:
    def forward(self, **kwargs) -> str:
        result = type('Result', (), {})()
        result._store = {"output": kwargs.get('gold_standard', '')}
        return result


async def main():
    print("=" * 80)
    print("MERMAID DIAGRAM LEARNING DEMONSTRATION")
    print("=" * 80)
    print()
    
    agents = [
        AgentConfig(name="mermaid", agent=MermaidAgent(), outputs=["output"]),
        AgentConfig(name="teacher", agent=Teacher(), metadata={"is_teacher": True})
    ]
    
    async def evaluate(output, gold_standard, task, context):
        out = str(output).strip()
        gold_str = str(gold_standard).strip()
        valid = "graph" in out.lower() and "[" in out and "-->" in out
        matches = out == gold_str
        score = 1.0 if matches else (0.5 if valid else 0.0)
        return {
            "score": score,
            "status": "CORRECT" if score == 1.0 else ("PARTIAL" if score == 0.5 else "INCORRECT")
        }
    
    pipeline = create_optimization_pipeline(
        agents=agents,
        max_iterations=3,
        required_pass_count=1,
        enable_teacher_model=True,
        save_improvements=True,
        output_path="./test_outputs/mermaid_demo"
    )
    pipeline.config.evaluation_function = evaluate
    
    gold = """graph TD
    A[Start]
    B[End]
    A --> B"""
    
    print("🎯 Target Mermaid Diagram:")
    print(gold)
    print()
    print("Starting optimization...")
    print()
    
    result = await pipeline.optimize(
        task="Generate diagram",
        context={},
        gold_standard=gold
    )
    
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    
    for it in result['iterations']:
        print(f"Iteration {it['iteration']}:")
        print(f"  Score: {it['evaluation_score']:.2f}")
        print(f"  Status: {it['evaluation_status']}")
        print(f"  Teacher Used: {it.get('has_teacher_output', False)}")
        print()
    
    print(f"Final Output:")
    print(result['final_result']['output'])
    print()
    print(f"✅ Success: {result['optimization_complete']}")
    
    # Show improvements
    imp_file = Path("./test_outputs/mermaid_demo/improvements.json")
    if imp_file.exists():
        import json
        with open(imp_file, 'r') as f:
            imps = json.load(f)
        print()
        print(f"📚 Learned {len(imps)} improvement(s)")
        for imp in imps:
            print(f"  Pattern: {imp['learned_pattern'][:80]}...")


if __name__ == "__main__":
    asyncio.run(main())
