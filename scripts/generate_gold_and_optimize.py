"""Generate gold PlantUML examples from Sonnet, re-optimize, and A/B test."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.infrastructure.foundation.dspy_init import load_api_keys

load_api_keys()

from core.intelligence.learning.advanced_learning import DomainDSPyOptimizer

# Task descriptions for Sonnet to generate gold examples.
# Focus on weak spots (class diagrams) + cover all types.
PLANTUML_TASKS = [
    # Class diagrams (biggest gap)
    "Generate a PlantUML class diagram for the Strategy pattern: PaymentStrategy interface with pay(amount: Money): Receipt method, CreditCardPayment and PayPalPayment implementing it via ..|>, Order class using PaymentStrategy, show visibility modifiers (+/-/#), multiplicities, and return types.",
    "Generate a PlantUML class diagram for a Repository pattern: generic Repository<T> interface with find/save/delete methods, UserRepository and ProductRepository implementing it, User and Product entity classes with typed fields, show inheritance and composition relationships.",
    "Generate a PlantUML class diagram for an event-driven system: EventBus with publish/subscribe, Event abstract class, OrderCreatedEvent and PaymentReceivedEvent extending it, EventHandler interface with handle(event), three concrete handlers. Use ..|> for implements, <|-- for extends.",
    # Sequence diagrams
    "Generate a PlantUML sequence diagram for OAuth2 PKCE flow: Mobile App, Authorization Server, Resource Server. Show code_verifier generation, authorization request with code_challenge, token exchange, and API call. Use activate/deactivate, alt for error cases, notes for security details.",
    "Generate a PlantUML sequence diagram for a distributed cache read-through pattern: Client, API, Cache (Redis), Database. Show cache hit path (short), cache miss path with DB fetch and cache populate, TTL expiry. Use alt blocks, return arrows, activate/deactivate.",
    # Deployment diagrams
    "Generate a PlantUML deployment diagram for a data pipeline on GCP: Cloud Pub/Sub ingestion, Dataflow processing, BigQuery warehouse, Cloud Functions for alerts, GCS for raw storage. Show node nesting, component stereotypes, and connection protocols.",
    # Activity diagrams
    "Generate a PlantUML activity diagram with swimlanes for a code review process: |Developer|, |Reviewer|, |CI|, |Lead|. Developer submits PR, CI runs tests in parallel (fork: unit tests, integration tests, lint), Reviewer reviews, if approved merge else Developer fixes. Use proper PlantUML activity syntax.",
    # Mixed complexity
    "Generate a PlantUML class diagram for a clean architecture layers: Controller, UseCase interface, UseCaseImpl, Repository interface, RepositoryImpl, Entity. Show dependency inversion with ..|> arrows pointing inward, package grouping for each layer.",
]


def main():
    optimizer = DomainDSPyOptimizer.get_instance()

    print("Generating gold examples from Sonnet...")
    added = optimizer.generate_gold_from_llm("plantuml", PLANTUML_TASKS)
    print(f"  Added {added}/{len(PLANTUML_TASKS)} valid gold examples\n")

    print("Re-optimizing with expanded gold data...")
    import time

    t0 = time.time()
    optimizer.optimize("plantuml", num_candidate_programs=6)
    print(f"  Done in {time.time() - t0:.0f}s\n")

    print("Running A/B test...")
    from scripts.ab_dspy_test import main as ab_main

    ab_main()


if __name__ == "__main__":
    main()
