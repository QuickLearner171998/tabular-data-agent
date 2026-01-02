"""Conversation memory management for the agent."""

from typing import Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver


class ConversationMemory:
    """Manages conversation history and context for the agent."""
    
    def __init__(self, max_messages: int = 50):
        self.max_messages = max_messages
        self.messages: list[BaseMessage] = []
        self.checkpointer = MemorySaver()
        self.analysis_context: dict[str, Any] = {}
    
    def add_human_message(self, content: str) -> None:
        """Add a human message to the conversation."""
        self.messages.append(HumanMessage(content=content))
        self._trim_messages()
    
    def add_ai_message(self, content: str) -> None:
        """Add an AI message to the conversation."""
        self.messages.append(AIMessage(content=content))
        self._trim_messages()
    
    def _trim_messages(self) -> None:
        """Trim messages to max_messages limit, keeping most recent."""
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_messages(self) -> list[BaseMessage]:
        """Get all messages in the conversation."""
        return self.messages.copy()
    
    def clear(self) -> None:
        """Clear all messages and context."""
        self.messages = []
        self.analysis_context = {}
    
    def set_context(self, key: str, value: Any) -> None:
        """Set a context value for the conversation."""
        self.analysis_context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context value from the conversation."""
        return self.analysis_context.get(key, default)
    
    def get_checkpointer(self) -> MemorySaver:
        """Get the LangGraph checkpointer for state persistence."""
        return self.checkpointer
    
    def get_conversation_summary(self) -> str:
        """Get a summary of recent conversation for context."""
        if not self.messages:
            return "No conversation history."
        
        recent = self.messages[-10:]
        summary_parts = []
        for msg in recent:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            summary_parts.append(f"{role}: {content}")
        
        return "\n".join(summary_parts)
