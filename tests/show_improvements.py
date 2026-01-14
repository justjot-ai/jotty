"""
Show detailed improvements made by OptimizationPipeline.

This script demonstrates the step-by-step improvement process.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.orchestration.optimization_pipeline import create_optimization_pipeline
from core.foundation.agent_config import AgentConfig


class ImprovingAgent:
    """Agent that starts wrong and improves."""
    def __init__(self):
        self.attempts = 0
        self.learned_from_teacher = None
    
    def forward(self, task: str = None, teacher_output: str = None, **kwargs) -> str:
        self.attempts += 1
        result = type('Result', (), {})()
        
        # If teacher output is available, use it!
        if teacher_output:
            self.learned_from_teacher = teacher_output
            result._store = {"output": teacher_output}
            return result
        
        # Otherwise, use learned value if available
        if self.learned_from_teacher:
            result._store = {"output": self.learned_from_teacher}
            return result
        
        # First attempt: Wrong output
        if self.attempts == 1:
            result._store = {"output": "Wrong answer"}
        else:
            result._store = {"output": "Still wrong"}
        
        return result


class TeacherAgent:
    """Teacher that provides correct answer."""
    def forward(self, **kwargs) -> str:
        gold_standard = kwargs.get('gold_standard', '')
        result = type('Result', (), {})()
        result._store = {"output": gold_standard}
        return result


async def show_improvements():
    """Show detailed improvement process."""
    print("=" * 80)
    print("OPTIMIZATION PIPELINE - IMPROVEMENT DEMONSTRATION")
    print("=" * 80)
    print()
    
    improving_agent = ImprovingAgent()
    
    agents = [
        AgentConfig(
            name="main_agent",
            agent=improving_agent,
            outputs=["output"]
        ),
        AgentConfig(
            name="teacher",
            agent=TeacherAgent(),
            metadata={"is_teacher": True}
        )
    ]
    
    async def evaluate(output, gold_standard, task, context):
        output_str = str(output).strip()
        gold_str = str(gold_standard).strip()
        score = 1.0 if output_str == gold_str else 0.0
        return {
            "score": score,
            "status": "CORRECT" if score == 1.0 else "INCORRECT",
            "difference": f"Expected '{gold_str}', got '{output_str}'" if score == 0.0 else None
        }
    
    pipeline = create_optimization_pipeline(
        agents=agents,
        max_iterations=5,
        required_pass_count=1,
        enable_teacher_model=True,
        output_path="./test_outputs/show_improvements"
    )
    pipeline.config.evaluation_function = evaluate
    
    gold_standard = "Correct answer"
    
    print("📋 TASK: Generate correct answer")
    print(f"🎯 GOLD STANDARD: '{gold_standard}'")
    print()
    print("=" * 80)
    print("ITERATION-BY-ITERATION IMPROVEMENT PROCESS")
    print("=" * 80)
    print()
    
    result = await pipeline.optimize(
        task="Generate correct answer",
        context={},
        gold_standard=gold_standard
    )
    
    print()
    print("=" * 80)
    print("IMPROVEMENT SUMMARY")
    print("=" * 80)
    print()
    
    for i, iteration in enumerate(result['iterations'], 1):
        iter_num = iteration['iteration']
        # Get actual output from metadata if available
        output = iteration.get('output')
        if not output and iteration.get('metadata'):
            pipeline_result = iteration['metadata'].get('pipeline_result', {})
            output = pipeline_result.get('output', 'N/A')
        
        print(f"📊 ITERATION {iter_num}:")
        print(f"   Agent Output: {repr(str(output)[:50]) if output else 'N/A'}")
        print(f"   Score: {iteration['evaluation_score']:.2f} / 1.0")
        print(f"   Status: {iteration['evaluation_status']}")
        print(f"   Teacher Used: {'✓ Yes' if iteration.get('has_teacher_output') else '✗ No'}")
        print(f"   Success: {'✓ YES' if iteration['success'] else '✗ NO'}")
        
        # Show teacher output if available
        if iteration.get('has_teacher_output') and iteration.get('metadata'):
            eval_result = iteration['metadata'].get('evaluation_result', {})
            teacher_eval = eval_result.get('teacher_evaluation', {})
            if teacher_eval:
                print(f"   🎓 Teacher Output Score: {teacher_eval.get('score', 0):.2f}")
        
        if i < len(result['iterations']):
            next_iter = result['iterations'][i]
            if next_iter['evaluation_score'] > iteration['evaluation_score']:
                improvement = next_iter['evaluation_score'] - iteration['evaluation_score']
                print(f"   📈 → Next iteration improves by +{improvement:.2f}")
        print()
    
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print()
    print(f"✅ Optimization Complete: {result['optimization_complete']}")
    print(f"📊 Total Iterations: {result['total_iterations']}")
    print(f"🎯 Consecutive Passes: {result['consecutive_passes']}")
    print()
    
    if result.get('final_result'):
        final = result['final_result']
        print(f"🏆 FINAL OUTPUT: {repr(final.get('output'))}")
        print(f"🏆 FINAL SCORE: {final.get('evaluation_score')} / 1.0")
        print(f"🏆 FINAL STATUS: {final.get('evaluation_status')}")
        print()
        
        # Show improvement
        if result['iterations']:
            first_output = result['iterations'][0].get('output', '')
            final_output = final.get('output', '')
            first_score = result['iterations'][0]['evaluation_score']
            final_score = final.get('evaluation_score', 0)
            
            print("=" * 80)
            print("BEFORE → AFTER COMPARISON")
            print("=" * 80)
            print()
            print(f"BEFORE (Iteration 1):")
            print(f"   Output: {repr(first_output)}")
            print(f"   Score: {first_score:.2f} / 1.0")
            print(f"   Status: {result['iterations'][0]['evaluation_status']}")
            print()
            print(f"AFTER (Final):")
            print(f"   Output: {repr(final_output)}")
            print(f"   Score: {final_score:.2f} / 1.0")
            print(f"   Status: {final.get('evaluation_status')}")
            print()
            print(f"📈 IMPROVEMENT: Score improved from {first_score:.2f} to {final_score:.2f} (+{final_score - first_score:.2f})")
            if first_output != final_output:
                print(f"📝 OUTPUT CHANGED: '{first_output}' → '{final_output}'")
    
    print()
    print("=" * 80)
    print("THINKING LOG (showing optimization process)")
    print("=" * 80)
    print()
    
    thinking_log_path = Path("./test_outputs/show_improvements/thinking.log")
    if thinking_log_path.exists():
        with open(thinking_log_path, 'r') as f:
            log_content = f.read()
            # Show key moments
            lines = log_content.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in [
                    'iteration', 'evaluation', 'teacher', 'correct', 'wrong', 
                    'passed', 'failed', 'optimization complete'
                ]):
                    print(line)
    else:
        print("Thinking log not found")
    
    return result


if __name__ == "__main__":
    asyncio.run(show_improvements())
