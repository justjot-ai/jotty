"""
DSPy PlantUML Optimization — Few-shot optimization from web research.

Flow:
1. Extract gold PlantUML examples from web research docs + teacher-crafted examples
2. Build a DSPy module (PlantUMLGenerator) with a proper signature
3. Run BootstrapFewShot to optimize the prompt with best examples
4. Save the optimized module for use by the crystallized agent
5. A/B test: Haiku raw vs Haiku+DSPy-optimized vs Sonnet
"""

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.infrastructure.foundation.dspy_init import load_api_keys

load_api_keys()

import anthropic
import dspy

HAIKU = "claude-3-haiku-20240307"
SONNET = "claude-sonnet-4-20250514"
SAVE_DIR = Path.home() / "jotty" / "learning" / "dspy_optimized"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ─── Step 1: Gold examples ──────────────────────────────────────────────────


def extract_examples_from_research_docs() -> list[dict]:
    """Parse research docs to extract (description, plantuml_code) pairs."""
    examples = []
    research_files = [
        "/tmp/plantuml_deployment_reference.md",
        "/tmp/plantuml_class_reference.md",
        "/tmp/plantuml_sequence_reference.md",
        "/tmp/plantuml_activity_reference.md",
    ]

    for filepath in research_files:
        if not os.path.exists(filepath):
            continue
        with open(filepath) as f:
            content = f.read()

        # Extract all @startuml...@enduml blocks with surrounding context
        blocks = re.findall(
            r"(?:```(?:puml|plantuml)?\s*\n)(@startuml.*?@enduml)(?:\s*```)?",
            content,
            re.DOTALL,
        )

        for block in blocks:
            block = block.strip()
            if len(block) < 50:
                continue

            # Infer diagram type from content
            if "sequenceDiagram" in block or "->" in block and "participant" in block.lower():
                dtype = "sequence"
            elif "class " in block or "interface " in block:
                dtype = "class"
            elif "node " in block or "component " in block or "database " in block:
                dtype = "deployment"
            elif "|" in block and ":" in block and ";" in block:
                dtype = "activity"
            else:
                dtype = "general"

            # Create a description from the code
            desc = f"Generate a PlantUML {dtype} diagram"
            examples.append(
                {
                    "description": desc,
                    "diagram_type": dtype,
                    "plantuml_code": block,
                    "source": os.path.basename(filepath),
                }
            )

    return examples


def get_teacher_examples() -> list[dict]:
    """High-quality hand-crafted examples for each diagram type."""
    return [
        {
            "description": "Generate a PlantUML sequence diagram for a REST API authentication flow with JWT tokens, showing login, token validation, refresh, and error handling",
            "diagram_type": "sequence",
            "plantuml_code": """@startuml
title REST API JWT Authentication Flow

participant "Client" as C
participant "API Gateway" as GW
participant "Auth Service" as Auth
participant "User DB" as DB
participant "Resource API" as API

== Login Flow ==
C -> GW: POST /login {email, password}
activate GW
GW -> Auth: validateCredentials(email, password)
activate Auth
Auth -> DB: findUser(email)
activate DB
DB --> Auth: User record
deactivate DB

alt Valid credentials
    Auth --> GW: JWT token + refresh token
    GW --> C: 200 {access_token, refresh_token, expires_in}
else Invalid credentials
    Auth --> GW: AuthError
    GW --> C: 401 Unauthorized
end
deactivate Auth
deactivate GW

== API Request with Token ==
C -> GW: GET /api/data\\nAuthorization: Bearer <token>
activate GW
GW -> Auth: validateToken(jwt)
activate Auth

alt Token valid
    Auth --> GW: {user_id, roles}
    deactivate Auth
    GW -> API: GET /data (user_id=123)
    activate API
    API --> GW: 200 {data}
    deactivate API
    GW --> C: 200 {data}
else Token expired
    Auth --> GW: TokenExpired
    deactivate Auth
    GW --> C: 401 Token expired
    note right of C: Client uses refresh token
end
deactivate GW

== Token Refresh ==
C -> GW: POST /refresh {refresh_token}
activate GW
GW -> Auth: refreshToken(refresh_token)
activate Auth
Auth -> DB: validateRefreshToken
activate DB
DB --> Auth: Valid
deactivate DB
Auth --> GW: New JWT + new refresh token
deactivate Auth
GW --> C: 200 {access_token, refresh_token}
deactivate GW

@enduml""",
            "source": "teacher",
        },
        {
            "description": "Generate a PlantUML class diagram for an Observer design pattern with abstract Subject, ConcreteSubject, Observer interface, and multiple ConcreteObservers",
            "diagram_type": "class",
            "plantuml_code": """@startuml
title Observer Design Pattern

abstract class Subject {
    - observers: List<Observer>
    + {abstract} getState(): Object
    + attach(observer: Observer): void
    + detach(observer: Observer): void
    + notify(): void
}

class ConcreteSubject {
    - state: String
    + getState(): String
    + setState(state: String): void
}

interface Observer <<interface>> {
    + update(subject: Subject): void
}

class ConcreteObserverA {
    - observerState: String
    + update(subject: Subject): void
    + display(): void
}

class ConcreteObserverB {
    - log: List<String>
    + update(subject: Subject): void
    + getLog(): List<String>
}

Subject <|-- ConcreteSubject
Observer <|.. ConcreteObserverA
Observer <|.. ConcreteObserverB
Subject "1" o-- "0..*" Observer : notifies >

note right of Subject::notify
  for each observer:
    observer.update(this)
end note

note bottom of ConcreteSubject
  When state changes,
  calls notify()
end note

@enduml""",
            "source": "teacher",
        },
        {
            "description": "Generate a PlantUML deployment diagram for a cloud-native microservices architecture on AWS with EKS, RDS, ElastiCache, and external APIs",
            "diagram_type": "deployment",
            "plantuml_code": """@startuml
title Cloud-Native Microservices Architecture

cloud "AWS Cloud" {
    node "VPC" {
        node "EKS Cluster" {
            node "Node Group 1" {
                component [API Gateway] <<Kong>>
                component [Auth Service] <<Go>>
                component [Order Service] <<Java>>
            }
            node "Node Group 2" {
                component [Payment Service] <<Node.js>>
                component [Notification Service] <<Python>>
                component [Worker Queue] <<Celery>>
            }
        }

        database "RDS PostgreSQL" as db <<Primary>> {
            [Orders DB]
            [Users DB]
        }

        database "ElastiCache" as cache <<Redis>> {
            [Session Store]
            [Rate Limiter]
        }

        storage "S3" as s3 {
            [Static Assets]
            [File Uploads]
        }
    }

    node "CloudFront CDN" as cdn
}

cloud "External Services" {
    component [Stripe API] <<Payment>>
    component [SendGrid] <<Email>>
    component [Twilio] <<SMS>>
}

actor "User" as user

user --> cdn : HTTPS
cdn --> [API Gateway] : HTTPS
[API Gateway] --> [Auth Service] : gRPC
[API Gateway] --> [Order Service] : gRPC
[Order Service] --> [Payment Service] : gRPC
[Payment Service] --> [Stripe API] : HTTPS
[Order Service] --> db : TCP/5432
[Auth Service] --> cache : TCP/6379
[Auth Service] --> [Users DB] : TCP/5432
[Notification Service] --> [SendGrid] : HTTPS
[Notification Service] --> [Twilio] : HTTPS
[Worker Queue] --> [Notification Service]
[API Gateway] --> s3 : HTTPS

@enduml""",
            "source": "teacher",
        },
        {
            "description": "Generate a PlantUML activity diagram with swimlanes for a CI/CD pipeline involving Developer, CI Server, QA, and Production",
            "diagram_type": "activity",
            "plantuml_code": """@startuml
title CI/CD Pipeline

|Developer|
start
:Push code to Git;
:Create Pull Request;

|CI Server|
:Trigger build pipeline;
fork
    :Run unit tests;
fork again
    :Run linter & static analysis;
fork again
    :Security scan (SAST);
end fork

if (All checks pass?) then (yes)
    :Build Docker image;
    :Push to container registry;
else (no)
    |Developer|
    :Fix issues;
    :Push updated code;
    |CI Server|
    :Re-trigger pipeline;
    stop
endif

|QA|
:Deploy to staging;
:Run integration tests;
:Run E2E tests;

if (QA approved?) then (yes)
    :Approve for production;
else (no)
    |Developer|
    :Fix bugs;
    stop
endif

|Production|
:Blue/green deployment;
:Run smoke tests;

if (Smoke tests pass?) then (yes)
    :Switch traffic to new version;
    :Monitor metrics (15 min);
    :Deployment complete;
else (no)
    :Rollback to previous version;
    :Alert on-call team;
endif

stop

@enduml""",
            "source": "teacher",
        },
    ]


# ─── Step 2: DSPy Module ────────────────────────────────────────────────────


class PlantUMLSignature(dspy.Signature):
    """Generate syntactically valid PlantUML diagram code from a description."""

    description: str = dspy.InputField(desc="What diagram to generate")
    diagram_type: str = dspy.InputField(desc="Type: sequence, class, deployment, activity")
    plantuml_code: str = dspy.OutputField(
        desc="Complete PlantUML code wrapped in @startuml/@enduml"
    )


class PlantUMLGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(PlantUMLSignature)

    def forward(self, description: str, diagram_type: str):
        return self.generate(description=description, diagram_type=diagram_type)


# ─── Step 3: Metric for optimization ────────────────────────────────────────


def plantuml_quality_metric(example, prediction, trace=None) -> float:
    """Score PlantUML output quality."""
    code = prediction.plantuml_code if hasattr(prediction, "plantuml_code") else ""
    if not code:
        return 0.0

    score = 0.0
    has_start = "@startuml" in code
    has_end = "@enduml" in code
    if has_start and has_end:
        score += 0.3

    gold = example.plantuml_code if hasattr(example, "plantuml_code") else ""
    dtype = example.diagram_type if hasattr(example, "diagram_type") else ""

    # Structural similarity based on diagram type
    if dtype == "sequence":
        features = ["participant", "activate", "deactivate", "alt", "->", "-->"]
    elif dtype == "class":
        features = ["class ", "interface", "<|--", "..|>", "+", "-", "#"]
    elif dtype == "deployment":
        features = ["node ", "component", "database", "cloud", "-->"]
    elif dtype == "activity":
        features = ["|", ":", ";", "if ", "fork", "start", "stop"]
    else:
        features = ["@startuml", "@enduml"]

    gold_features = sum(1 for f in features if f in gold)
    pred_features = sum(1 for f in features if f in code)
    if gold_features > 0:
        score += 0.4 * min(pred_features / gold_features, 1.0)

    # Length similarity (not too short, not too long)
    if len(gold) > 0:
        ratio = len(code) / len(gold)
        if 0.5 <= ratio <= 2.0:
            score += 0.2
        elif 0.3 <= ratio <= 3.0:
            score += 0.1

    # Non-empty content
    lines = [l for l in code.split("\n") if l.strip()]
    if len(lines) > 5:
        score += 0.1

    return min(score, 1.0)


# ─── Step 4: Optimize ───────────────────────────────────────────────────────


def optimize_plantuml_module():
    """Run BootstrapFewShot optimization."""
    # Gather all examples
    research_examples = extract_examples_from_research_docs()
    teacher_examples = get_teacher_examples()

    print(f"Examples from web research: {len(research_examples)}")
    print(f"Examples from teacher: {len(teacher_examples)}")

    all_data = teacher_examples + research_examples

    # Convert to DSPy Examples
    trainset = []
    for ex in all_data:
        dspy_ex = dspy.Example(
            description=ex["description"],
            diagram_type=ex["diagram_type"],
            plantuml_code=ex["plantuml_code"],
        ).with_inputs("description", "diagram_type")
        trainset.append(dspy_ex)

    print(f"Total training examples: {len(trainset)}")

    if len(trainset) < 3:
        print("ERROR: Too few examples for optimization")
        return None

    # Configure DSPy with the teacher model (Sonnet) for bootstrapping
    from core.infrastructure.foundation.unified_lm_provider import UnifiedLMProvider

    teacher_lm = UnifiedLMProvider.create_lm(provider="anthropic", model="sonnet")
    student_lm = UnifiedLMProvider.create_lm(provider="anthropic", model="haiku")

    # Create unoptimized module
    module = PlantUMLGenerator()

    # Run BootstrapFewShot
    print("\nRunning BootstrapFewShot optimization...")
    print("  Teacher: Sonnet (generates bootstrapped demos)")
    print("  Student: Haiku (will use the optimized prompts)")
    t0 = time.time()

    optimizer = dspy.BootstrapFewShot(
        metric=plantuml_quality_metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=min(len(trainset), 6),
    )

    with dspy.context(lm=teacher_lm):
        optimized = optimizer.compile(module, trainset=trainset[:8])

    elapsed = time.time() - t0
    print(f"  Optimization complete in {elapsed:.1f}s")

    # Save the optimized module
    save_path = str(SAVE_DIR / "plantuml_generator.json")
    optimized.save(save_path)
    print(f"  Saved to {save_path}")

    return optimized, student_lm


# ─── Step 5: A/B Test ────────────────────────────────────────────────────────

TASKS = [
    {
        "name": "Deployment (K8s)",
        "description": "Generate a PlantUML deployment diagram for a Kubernetes microservices architecture with API Gateway, Auth, Order, Payment services inside an EKS cluster, plus RDS PostgreSQL, ElastiCache Redis, and external Stripe API. Show connections with protocols.",
        "diagram_type": "deployment",
    },
    {
        "name": "Sequence (Saga)",
        "description": "Generate a PlantUML sequence diagram for a Saga pattern distributed transaction: OrderService -> PaymentService -> InventoryService -> ShippingService. Show happy path, then alt block where InventoryService fails triggering compensating transactions. Use activate/deactivate, group blocks, return arrows with HTTP codes, and notes.",
        "diagram_type": "sequence",
    },
    {
        "name": "Class (Strategy+Factory)",
        "description": "Generate a PlantUML class diagram for an e-commerce system with Strategy+Factory patterns: PaymentStrategy interface with pay() method, CreditCardPayment/PayPalPayment/CryptoPayment implementing it, PaymentFactory, Order with items and strategy, OrderItem with product/quantity/price, Product with id/name/price. Show visibility modifiers, multiplicities, and implements (..|>) relationships.",
        "diagram_type": "class",
    },
    {
        "name": "Activity (Swimlanes)",
        "description": "Generate a PlantUML activity diagram with swimlanes |HR|, |IT|, |Manager|, |Employee| for an employee onboarding process: HR creates account, IT provisions laptop AND email in parallel (fork), Manager assigns buddy, Employee completes orientation, then if/else for background check pass/fail.",
        "diagram_type": "activity",
    },
]


def analyze_plantuml(code: str, dtype: str) -> dict:
    lines = [l for l in code.split("\n") if l.strip()]
    result = {
        "lines": len(lines),
        "chars": len(code),
        "valid": "@startuml" in code and "@enduml" in code,
    }

    if dtype == "deployment":
        result["nodes"] = len(re.findall(r"\bnode\b", code))
        result["components"] = len(re.findall(r"\bcomponent\b|\[.*?\]", code))
        result["databases"] = len(re.findall(r"\bdatabase\b", code))
        result["connections"] = len(re.findall(r"-->|->", code))
        result["protocols"] = len(re.findall(r"HTTPS|gRPC|TCP", code))
    elif dtype == "sequence":
        result["participants"] = len(re.findall(r"\bparticipant\b|\bactor\b", code))
        result["activates"] = len(re.findall(r"\bactivate\b", code))
        result["alt_blocks"] = len(re.findall(r"\balt\b", code))
        result["notes"] = len(re.findall(r"\bnote\b", code, re.I))
        result["http_codes"] = len(re.findall(r"\b[2345]\d{2}\b", code))
    elif dtype == "class":
        result["classes"] = len(re.findall(r"\bclass\b", code))
        result["interfaces"] = len(re.findall(r"\binterface\b|<<interface>>", code))
        result["implements"] = len(re.findall(r"\.\.\|>", code))
        result["visibility"] = len(re.findall(r"[+\-#~]\s*\w", code))
        result["methods"] = len(re.findall(r"[+\-#~]\s*\w+\(", code))
    elif dtype == "activity":
        result["swimlanes"] = len(set(re.findall(r"\|(\w+)\|", code)))
        result["actions"] = len(re.findall(r":.*?;", code))
        result["forks"] = len(re.findall(r"\bfork\b", code))
        result["ifs"] = len(re.findall(r"\bif\b", code))

    return result


async def run_ab_test(optimized_module, student_lm):
    """Run the A/B/C comparison."""
    client = anthropic.Anthropic()

    COSTS = {
        HAIKU: {"input": 0.25, "output": 1.25},
        SONNET: {"input": 3.0, "output": 15.0},
    }

    print("\n" + "=" * 70)
    print("A/B/C TEST: Haiku vs Haiku+DSPy vs Sonnet")
    print("=" * 70)

    for task in TASKS:
        print(f"\n{'─' * 70}")
        print(f"TASK: {task['name']}")
        print(f"{'─' * 70}")

        prompt = (
            f"{task['description']} "
            f"Use proper PlantUML syntax with @startuml/@enduml. "
            f"Output ONLY the PlantUML code."
        )

        # A: Haiku baseline
        print("  [A] Haiku baseline...")
        t0 = time.time()
        resp_a = client.messages.create(
            model=HAIKU,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        time_a = time.time() - t0
        raw_a = resp_a.content[0].text
        cost_a = (
            resp_a.usage.input_tokens * COSTS[HAIKU]["input"]
            + resp_a.usage.output_tokens * COSTS[HAIKU]["output"]
        ) / 1e6
        match_a = re.search(r"@startuml.*?@enduml", raw_a, re.DOTALL)
        code_a = match_a.group(0) if match_a else raw_a
        analysis_a = analyze_plantuml(code_a, task["diagram_type"])
        print(f"      {time_a:.1f}s | ${cost_a:.5f} | {analysis_a['lines']} lines")

        # B: Haiku + DSPy optimized
        print("  [B] Haiku + DSPy optimized (teacher=Sonnet, student=Haiku)...")
        t0 = time.time()
        with dspy.context(lm=student_lm):
            result_b = optimized_module(
                description=task["description"],
                diagram_type=task["diagram_type"],
            )
        time_b = time.time() - t0
        code_b = result_b.plantuml_code if hasattr(result_b, "plantuml_code") else ""
        # Clean up code fences if present
        if "```" in code_b:
            code_b = re.sub(r"```(?:plantuml|puml)?\s*", "", code_b).strip().rstrip("`")
        match_b = re.search(r"@startuml.*?@enduml", code_b, re.DOTALL)
        if match_b:
            code_b = match_b.group(0)
        analysis_b = analyze_plantuml(code_b, task["diagram_type"])
        # Approximate DSPy cost (input includes few-shot demos)
        cost_b = cost_a * 2.5  # rough estimate: more input tokens from demos
        print(f"      {time_b:.1f}s | ~${cost_b:.5f} | {analysis_b['lines']} lines")

        # C: Sonnet
        print("  [C] Sonnet (gold standard)...")
        t0 = time.time()
        resp_c = client.messages.create(
            model=SONNET,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        time_c = time.time() - t0
        raw_c = resp_c.content[0].text
        cost_c = (
            resp_c.usage.input_tokens * COSTS[SONNET]["input"]
            + resp_c.usage.output_tokens * COSTS[SONNET]["output"]
        ) / 1e6
        match_c = re.search(r"@startuml.*?@enduml", raw_c, re.DOTALL)
        code_c = match_c.group(0) if match_c else raw_c
        analysis_c = analyze_plantuml(code_c, task["diagram_type"])
        print(f"      {time_c:.1f}s | ${cost_c:.5f} | {analysis_c['lines']} lines")

        # Comparison
        print(f"\n  {'Metric':<20} {'Haiku':>8} {'Haiku+DSPy':>11} {'Sonnet':>8}")
        print(f"  {'─' * 47}")
        for key in analysis_a:
            va = analysis_a[key]
            vb = analysis_b.get(key, "—")
            vc = analysis_c.get(key, "—")
            if isinstance(va, bool):
                va, vb, vc = "✓" if va else "✗", "✓" if vb else "✗", "✓" if vc else "✗"
            print(f"  {key:<20} {str(va):>8} {str(vb):>11} {str(vc):>8}")

        # Save outputs
        slug = task["name"].lower().replace(" ", "_").replace("(", "").replace(")", "")
        for variant, code in [("haiku", code_a), ("dspy", code_b), ("sonnet", code_c)]:
            with open(f"/tmp/plantuml_dspy_{slug}_{variant}.puml", "w") as f:
                f.write(code)


async def main():
    print("=" * 70)
    print("DSPy PlantUML Optimization")
    print("=" * 70)

    result = optimize_plantuml_module()
    if result is None:
        print("Optimization failed — not enough examples")
        return

    optimized_module, student_lm = result
    await run_ab_test(optimized_module, student_lm)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
