from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    """Format chat history for inclusion in the prompt."""
    try:
        if not chat_history:
            return ""

        formatted_history = []
        for msg in chat_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                formatted_history.append(f"User: {content}")
            elif role == "assistant":
                # Only include the main content, not the thought process
                formatted_history.append(f"Assistant: {content}")

        return "\n".join(formatted_history)
    except Exception as e:
        logger.error(f"Error formatting chat history: {e}")
        return ""
