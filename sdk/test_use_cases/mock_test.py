"""
Mock test for Jotty SDK - simulates real usage
"""
import json
from typing import Dict, Any

# Simulated SDK client (would be imported from generated SDK)
class MockJottyClient:
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url
        self.api_key = api_key
    
    def chat_execute(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate chat execution."""
        # In real SDK, this would make HTTP request
        return {
            "success": True,
            "final_output": "Hello! I can help you with various tasks.",
            "agent_id": "default",
            "metadata": {}
        }
    
    def workflow_execute(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate workflow execution."""
        return {
            "success": True,
            "final_output": "Workflow completed successfully",
            "steps": [],
            "metadata": {}
        }


def test_chat_use_case():
    """Test chat use case."""
    client = MockJottyClient("http://localhost:8080", "test-key")
    
    result = client.chat_execute({
        "message": "Hello, how can you help?",
        "history": []
    })
    
    assert result["success"] == True
    assert "final_output" in result
    print("✅ Chat use case test passed")
    return result


def test_workflow_use_case():
    """Test workflow use case."""
    client = MockJottyClient("http://localhost:8080", "test-key")
    
    result = client.workflow_execute({
        "goal": "Analyze data and generate report",
        "context": {"department": "sales"}
    })
    
    assert result["success"] == True
    assert "final_output" in result
    print("✅ Workflow use case test passed")
    return result


if __name__ == "__main__":
    print("Running mock SDK tests...")
    test_chat_use_case()
    test_workflow_use_case()
    print("\n✅ All mock tests passed!")
