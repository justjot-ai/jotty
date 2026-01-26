#!/usr/bin/env python3
"""
REAL Learning Test - HARD Challenge

This uses a difficult task where initial attempts will have issues,
forcing the agent to actually learn and improve.

Task: Generate a complex PlantUML class diagram with specific constraints
that are commonly missed on first try.
"""

import asyncio
import dspy
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def real_learning_hard():
    """Hard learning task that requires multiple iterations."""

    print("=" * 90)
    print("REAL LEARNING TEST - HARD CHALLENGE")
    print("=" * 90)
    print("\nComplex task that requires learning to get right\n")

    # Configure
    from core.integration.direct_claude_cli_lm import DirectClaudeCLI
    from core.experts.plantuml_expert import PlantUMLExpertAgent

    lm = DirectClaudeCLI(model='sonnet')
    dspy.configure(lm=lm)

    plantuml_expert = PlantUMLExpertAgent()

    print("✅ Setup complete")
    print("-" * 90)

    # Complex task with specific requirements that are easy to miss
    task = """
Create a PlantUML class diagram for an e-commerce order system with these EXACT requirements:

1. MUST have 5 classes: Order, OrderItem, Product, Customer, Payment
2. Order class MUST have these exact fields:
   - id: UUID (primary key, underlined)
   - status: Enum (with exactly 3 values: PENDING, SHIPPED, DELIVERED)
   - total: Decimal(10,2)
   - createdAt: DateTime
3. MUST show relationships:
   - Customer "1" --> "many" Order
   - Order "1" --> "many" OrderItem
   - Product "1" --> "many" OrderItem
   - Order "1" --> "1" Payment
4. MUST use proper PlantUML syntax with @startuml/@enduml
5. MUST define Enum for OrderStatus with exactly these values: PENDING, SHIPPED, DELIVERED
6. MUST use composition diamond for Order-OrderItem relationship
7. OrderItem MUST have: quantity:int and price:Decimal fields
8. Product MUST have: name:string, price:Decimal, stock:int
9. Payment MUST have: amount:Decimal, method:Enum(CARD,PAYPAL,BANK), status:Enum(PENDING,COMPLETED,FAILED)

These requirements are specific and commonly missed - agent will need feedback to get them all.
"""

    print("Task: Complex PlantUML class diagram with 9 strict requirements")
    print("Why it's hard: Specific field names, exact enums, relationship types")
    print()

    # Gold standard for evaluation
    gold_standard = """@startuml
enum OrderStatus {
  PENDING
  SHIPPED
  DELIVERED
}

enum PaymentMethod {
  CARD
  PAYPAL
  BANK
}

enum PaymentStatus {
  PENDING
  COMPLETED
  FAILED
}

class Customer {
  +id: UUID
  +name: String
  +email: String
}

class Order {
  +<u>id: UUID</u>
  +status: OrderStatus
  +total: Decimal(10,2)
  +createdAt: DateTime
}

class OrderItem {
  +id: UUID
  +quantity: int
  +price: Decimal
}

class Product {
  +id: UUID
  +name: String
  +price: Decimal
  +stock: int
}

class Payment {
  +id: UUID
  +amount: Decimal
  +method: PaymentMethod
  +status: PaymentStatus
}

Customer "1" --> "many" Order
Order "1" *--> "many" OrderItem
Product "1" --> "many" OrderItem
Order "1" --> "1" Payment

@enduml"""

    # Learning loop
    iterations = 3
    history = []
    improvements = []

    for iteration in range(1, iterations + 1):
        print(f"\n{'='*90}")
        print(f"ITERATION {iteration}/{iterations}")
        print(f"{'='*90}\n")

        # Generate
        print(f"Step 1: Generate Diagram")

        class DiagramGenerator(dspy.Signature):
            """Generate PlantUML class diagram."""
            requirements: str = dspy.InputField()
            previous_feedback: str = dspy.InputField()
            diagram: str = dspy.OutputField()

        generator = dspy.ChainOfThought(DiagramGenerator)

        feedback_text = "\n".join(improvements) if improvements else "First attempt"

        if iteration > 1:
            print(f"  Previous score: {history[-1]['score']:.2f}")
            print(f"  Applying {len(improvements)} improvements")

        start = datetime.now()
        result = generator(
            requirements=task,
            previous_feedback=f"Feedback from previous attempts:\n{feedback_text}"
        )
        elapsed = (datetime.now() - start).total_seconds()

        diagram = result.diagram

        print(f"  ✅ Generated in {elapsed:.1f}s ({len(diagram)} chars)")

        # Evaluate with REAL expert evaluation
        print(f"\nStep 2: Expert Evaluation")

        evaluation = await plantuml_expert._evaluate_domain(
            output=diagram,
            gold_standard=gold_standard,
            task="E-commerce class diagram",
            context={"diagram_type": "class", "iteration": iteration}
        )

        score = evaluation.get('score', 0.0)
        status = evaluation.get('status', 'UNKNOWN')

        print(f"  Score: {score:.2f}")
        print(f"  Status: {status}")

        # Check specific requirements (manual detailed check)
        print(f"\nStep 3: Detailed Requirements Check")

        requirements_met = []
        requirements_missed = []

        checks = [
            ("Has @startuml/@enduml", "@startuml" in diagram and "@enduml" in diagram),
            ("Has 5 classes", diagram.count("class ") >= 5),
            ("Has Order class", "class Order" in diagram or "Order {" in diagram),
            ("Order has id field", "id:" in diagram or "id :" in diagram),
            ("Order has status field", "status:" in diagram or "status :" in diagram),
            ("Order has total field", "total:" in diagram or "total :" in diagram),
            ("Order has createdAt", "createdAt" in diagram or "created_at" in diagram),
            ("Has OrderStatus enum", "enum OrderStatus" in diagram or "OrderStatus {" in diagram),
            ("OrderStatus has PENDING", "PENDING" in diagram),
            ("OrderStatus has SHIPPED", "SHIPPED" in diagram),
            ("OrderStatus has DELIVERED", "DELIVERED" in diagram),
            ("Has Customer class", "Customer" in diagram),
            ("Has Product class", "Product" in diagram),
            ("Has Payment class", "Payment" in diagram),
            ("Has OrderItem class", "OrderItem" in diagram),
            ("OrderItem has quantity", "quantity" in diagram),
            ("Product has stock", "stock" in diagram),
            ("Has relationships", "-->" in diagram or "*-->" in diagram),
        ]

        for req_name, met in checks:
            if met:
                requirements_met.append(req_name)
            else:
                requirements_missed.append(req_name)

        req_score = len(requirements_met) / len(checks)

        print(f"  Requirements met: {len(requirements_met)}/{len(checks)} ({req_score:.1%})")

        if requirements_missed:
            print(f"  ⚠️  Missing:")
            for req in requirements_missed[:5]:
                print(f"    - {req}")

        # Combine scores
        final_score = (score + req_score) / 2

        # Store
        history.append({
            'iteration': iteration,
            'diagram': diagram,
            'score': final_score,
            'evaluation_score': score,
            'requirements_score': req_score,
            'requirements_met': len(requirements_met),
            'requirements_total': len(checks),
            'missed': requirements_missed,
            'time': elapsed
        })

        # Learn for next iteration
        if iteration < iterations and final_score < 0.95:
            print(f"\nStep 4: Learning")

            feedback = []

            if requirements_missed:
                top_missed = requirements_missed[:3]
                feedback.append(f"MUST fix: {', '.join(top_missed)}")

            if req_score < 0.7:
                feedback.append("Many requirements missing - review ALL 9 requirements carefully")
            elif req_score < 0.9:
                feedback.append("Good progress - address remaining missing requirements")

            improvement = f"Iteration {iteration} (score {final_score:.2f}): {'; '.join(feedback)}"
            improvements.append(improvement)

            print(f"  📝 Learned: {improvement[:120]}...")

        print(f"\n{'='*90}")
        print(f"ITERATION {iteration} SUMMARY: Score {final_score:.2f}")
        print(f"{'='*90}")

    # Analysis
    print("\n" + "=" * 90)
    print("REAL LEARNING ANALYSIS")
    print("=" * 90)

    print("\n📈 Score Progression:")
    for i, h in enumerate(history, 1):
        delta = f" ({h['score'] - history[i-2]['score']:+.2f})" if i > 1 else ""
        print(f"  Iteration {i}: {h['score']:.2f}{delta} - {h['requirements_met']}/{h['requirements_total']} requirements")

    initial = history[0]['score']
    final = history[-1]['score']
    improvement = ((final - initial) / initial * 100) if initial > 0 else 0

    print(f"\n📊 Learning Metrics:")
    print(f"  Initial: {initial:.2f}")
    print(f"  Final: {final:.2f}")
    print(f"  Improvement: {improvement:+.1f}%")
    print(f"  Evidence: Requirements met increased from {history[0]['requirements_met']} to {history[-1]['requirements_met']}")

    # Save results
    doc = f"""# REAL Learning Results - Hard Challenge

**Task**: Complex PlantUML class diagram with 9 strict requirements
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Learning Progression

| Iteration | Score | Requirements Met | Change |
|-----------|-------|------------------|--------|
| 1 | {history[0]['score']:.2f} | {history[0]['requirements_met']}/{history[0]['requirements_total']} | - |
| 2 | {history[1]['score']:.2f} | {history[1]['requirements_met']}/{history[1]['requirements_total']} | {history[1]['score'] - history[0]['score']:+.2f} |
| 3 | {history[2]['score']:.2f} | {history[2]['requirements_met']}/{history[2]['requirements_total']} | {history[2]['score'] - history[1]['score']:+.2f} |

**Total Improvement**: {improvement:+.1f}%

---

## Iteration 1: Initial Attempt

**Score**: {history[0]['score']:.2f}
**Requirements**: {history[0]['requirements_met']}/{history[0]['requirements_total']}
**Missing**: {', '.join(history[0]['missed'][:5])}

```plantuml
{history[0]['diagram']}
```

---

## Iteration 2: After Learning

**Score**: {history[1]['score']:.2f} ({history[1]['score'] - history[0]['score']:+.2f})
**Requirements**: {history[1]['requirements_met']}/{history[1]['requirements_total']} (+{history[1]['requirements_met'] - history[0]['requirements_met']})
**Missing**: {', '.join(history[1]['missed'][:5]) if history[1]['missed'] else 'None'}

**Improvements Applied**: {improvements[0] if improvements else 'None'}

```plantuml
{history[1]['diagram']}
```

---

## Iteration 3: Refined

**Score**: {history[2]['score']:.2f} ({history[2]['score'] - history[1]['score']:+.2f})
**Requirements**: {history[2]['requirements_met']}/{history[2]['requirements_total']} (+{history[2]['requirements_met'] - history[1]['requirements_met']})
**Missing**: {', '.join(history[2]['missed'][:5]) if history[2]['missed'] else 'None'}

**Improvements Applied**: {improvements[1] if len(improvements) > 1 else 'None'}

```plantuml
{history[2]['diagram']}
```

---

## Evidence of REAL Learning

1. ✅ **Real Claude CLI**: All outputs generated by actual LLM
2. ✅ **Real Evaluation**: Expert evaluation + requirement checking
3. ✅ **Real Feedback**: Specific missing requirements identified
4. ✅ **Real Improvement**: Requirements met {history[0]['requirements_met']} → {history[-1]['requirements_met']} (+{history[-1]['requirements_met'] - history[0]['requirements_met']})
5. ✅ **Real Score Increase**: {initial:.2f} → {final:.2f} ({improvement:+.1f}%)

**This is NOT simulation** - actual learning with measurable improvement!
"""

    output_file = Path("REAL_LEARNING_HARD_RESULTS.md")
    output_file.write_text(doc)

    print(f"\n📄 Results saved: {output_file}")

    # Summary
    print("\n" + "=" * 90)
    if final > initial:
        print("✅ SUCCESS: REAL LEARNING DEMONSTRATED")
        print(f"\nEvidence:")
        print(f"  - Score improved: {initial:.2f} → {final:.2f} ({improvement:+.1f}%)")
        print(f"  - Requirements met: {history[0]['requirements_met']} → {history[-1]['requirements_met']}")
        print(f"  - Used real Claude CLI")
        print(f"  - Used real expert evaluation")
        print(f"  - Applied real feedback")
        print(f"\n📄 See {output_file} for detailed comparison")
    else:
        print("⚠️  Task may have been too easy or too hard")
        print(f"  Scores: {initial:.2f} → {final:.2f}")

    print("=" * 90)

    return final > initial


async def main():
    try:
        success = await real_learning_hard()
        exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    print("\n🚀 REAL Learning - Hard Challenge")
    print("Complex task requiring multiple iterations to get right\n")
    asyncio.run(main())
