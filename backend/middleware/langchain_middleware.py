from typing import Dict
import logging

logger = logging.getLogger(__name__)


def check_context_presence(input_dict: Dict) -> Dict:
    """Adds a system flag if the guardrail blocked all documents."""
    context = input_dict.get("context", "")
    # Check if our specific content delimiter is present
    if not context or "--------- CONTENT ---------" not in context:
        input_dict["context"] = (
            "[SYSTEM NOTE: NO RELEVANT DATA FOUND IN KNOWLEDGE GRAPH]"
        )
        logger.info("Context fallback applied: No relevant data found.")
    return input_dict


def process_with_topic_analysis(input_dict: Dict) -> Dict:
    """Enriches input with topic continuity analysis before processing."""
    try:
        from utils.topic_manager import TopicManager

        question = input_dict.get("question", "")
        session_topic = input_dict.get("session_topic", "")
        session_id = input_dict.get("session_id", "")

        # Calculate similarity
        similarity_data = TopicManager.calculate_topic_similarity(
            question, session_topic
        )

        # Get relevant previous context
        relevant_context = []
        if session_id:
            relevant_context = TopicManager.get_relevant_context_for_continuation(
                session_id, question, max_messages=3
            )

        # Format relevant context
        relevant_context_str = ""
        if relevant_context:
            for msg in relevant_context:
                role = "User" if msg["role"] == "user" else "Assistant"
                relevant_context_str += f"{role}: {msg['content']}\n\n"

        # Return enriched input
        return {
            **input_dict,
            "session_topic": session_topic or "General Discussion",
            "topic_similarity_score": f"{similarity_data['similarity_score']:.2f}",
            "topic_confidence": similarity_data["confidence_level"],
            "topic_status": similarity_data["recommendation"],
            "relevant_context": relevant_context_str
            or "[No previous context available]",
            "continuity_instruction": similarity_data["recommendation"],
        }
    except Exception as e:
        logger.error(f"Error in topic analysis: {e}")
        return {
            **input_dict,
            "session_topic": input_dict.get("session_topic", "General Discussion"),
            "topic_similarity_score": "0.50",
            "topic_confidence": "low",
            "topic_status": "Unable to determine topic status",
            "relevant_context": "[Error retrieving relevant context]",
            "continuity_instruction": "Proceed with caution due to analysis error",
        }
