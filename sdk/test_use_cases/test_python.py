"""
Test use case for Jotty Python SDK
"""
from jotty_sdk import Configuration, ChatApi

def test_chat():
    """Test chat execution."""
    config = Configuration(
        host="http://localhost:8080",
        access_token="your-api-key"
    )
    
    chat_api = ChatApi(config)
    
    result = chat_api.chat_execute(
        body={
            "message": "Hello, how can you help?",
            "history": []
        }
    )
    
    print(f"Response: {result.final_output}")
    return result

if __name__ == "__main__":
    test_chat()
