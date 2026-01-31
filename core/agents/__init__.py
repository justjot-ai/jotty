"""DSPy-based agents with MCP tool support"""

from .chat_assistant import ChatAssistant, create_chat_assistant
from .auto_agent import AutoAgent, run_task, TaskType, ExecutionResult
from .model_chat_agent import ModelChatAgent

__all__ = [
    'ChatAssistant',
    'create_chat_assistant',
    'AutoAgent',
    'run_task',
    'TaskType',
    'ExecutionResult',
    'ModelChatAgent',
]
