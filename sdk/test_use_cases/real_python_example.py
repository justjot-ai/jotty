"""
Real Use Case Example using Generated Python SDK

This example shows how to use the generated SDK in a real application.
"""

from jotty_api_client import Client
from jotty_api_client.api.chat import chat_execute, chat_stream
from jotty_api_client.api.workflow import workflow_execute
from jotty_api_client.models import ChatExecuteRequest, ChatMessage, WorkflowExecuteRequest


def example_chat():
    """Example: Chat with Jotty."""
    # Initialize client
    client = Client(base_url="http://localhost:8080", timeout=30.0)

    # Create request
    request = ChatExecuteRequest(message="What is the weather today?", history=[])

    # Execute chat
    result = chat_execute.sync(client=client, body=request)

    print(f"Response: {result.final_output}")
    return result


def example_chat_with_history():
    """Example: Chat with conversation history."""
    client = Client(base_url="http://localhost:8080")

    request = ChatExecuteRequest(
        messages=[
            ChatMessage(role="user", content="Hello!"),
            ChatMessage(role="assistant", content="Hi! How can I help?"),
            ChatMessage(role="user", content="What's the weather?"),
        ]
    )

    result = chat_execute.sync(client=client, body=request)
    return result


def example_workflow():
    """Example: Execute a workflow."""
    client = Client(base_url="http://localhost:8080")

    request = WorkflowExecuteRequest(
        goal="Analyze sales data and generate report",
        context={"date_range": "2024-01-01 to 2024-12-31", "department": "sales"},
        mode="dynamic",
    )

    result = workflow_execute.sync(client=client, body=request)
    print(f"Workflow result: {result.final_output}")
    return result


def example_streaming():
    """Example: Stream chat response."""
    client = Client(base_url="http://localhost:8080")

    request = ChatExecuteRequest(message="Tell me a story")

    # Stream response
    for event in chat_stream.sync(client=client, body=request):
        print(f"Event: {event}")
        # Process streaming events


if __name__ == "__main__":
    # Run examples (requires server to be running)
    print("Running examples...")

    try:
        result = example_chat()
        print("✅ Chat example completed")
    except Exception as e:
        print(f"⚠️  Chat example failed (server may not be running): {e}")

    try:
        result = example_workflow()
        print("✅ Workflow example completed")
    except Exception as e:
        print(f"⚠️  Workflow example failed (server may not be running): {e}")
