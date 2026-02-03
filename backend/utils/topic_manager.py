import logging
import numpy as np
from setup.init_config import embedding_model
from utils.memory import get_chat_history, get_graph_instance

logger = logging.getLogger(__name__)


class TopicManager:
    """
    Manages topic continuity, similarity calculations, and context retrieval.
    """

    @staticmethod
    def get_session_topic(session_id: str) -> str:
        """
        Retrieves the topic for a specific session.
        Returns empty string if no topic is set.
        """
        try:
            graph = get_graph_instance()
            query = """
            MATCH (s:Session {id: $session_id})
            RETURN s.topic AS topic
            LIMIT 1
            """
            result = graph.query(query, params={"session_id": session_id})
            if result and result[0].get("topic"):
                return result[0]["topic"]
            return ""
        except Exception as e:
            logger.error(f"Error getting session topic: {e}")
            return ""

    @staticmethod
    def calculate_topic_similarity(question: str, session_topic: str) -> dict:
        """
        Calculates semantic similarity between current question and session topic.
        Returns a dict with similarity_score, is_continuation, and recommendation.
        """
        try:
            # FORCE CONTINUITY: Always treat as continuation of the session topic
            # This logic overrides any semantic similarity check.

            return {
                "similarity_score": 1.0,
                "is_continuation": True,
                "confidence_level": "high",
                "recommendation": "CONTINUATION: Forced session continuity enabled. Treating as follow-up.",
            }
        except Exception as e:
            logger.error(f"Error calculating topic similarity: {e}")
            return {
                "similarity_score": 1.0,
                "is_continuation": True,
                "confidence_level": "high",
                "recommendation": "Error handling forced to continuation.",
            }

    @staticmethod
    def get_relevant_context_for_continuation(
        session_id: str, question: str, max_messages: int = 3
    ) -> list:
        """
        Retrieves only the most relevant previous messages for topic continuation.
        Filters messages based on semantic relevance to the current question.
        """
        try:
            # Get all messages
            history = get_chat_history(session_id)
            all_messages = history.messages if hasattr(history, "messages") else []

            if len(all_messages) <= 2:  # Only system message + first user message
                return []

            # Embed the question
            question_embedding = np.array(embedding_model().embed_query(question))

            # Score all messages based on relevance to current question
            scored_messages = []
            for msg in all_messages[1:]:  # Skip first message
                msg_content = getattr(msg, "content", "")
                if not msg_content:
                    continue

                msg_embedding = np.array(embedding_model().embed_query(msg_content))
                similarity = float(
                    np.dot(question_embedding, msg_embedding)
                    / (
                        np.linalg.norm(question_embedding)
                        * np.linalg.norm(msg_embedding)
                        + 1e-10
                    )
                )

                scored_messages.append(
                    {
                        "similarity": similarity,
                        "role": getattr(msg, "type", "unknown"),
                        "content": msg_content,
                    }
                )

            # Sort by similarity and return top N
            scored_messages.sort(key=lambda x: x["similarity"], reverse=True)
            relevant = scored_messages[:max_messages]

            logger.info(f"Retrieved {len(relevant)} relevant context messages")
            return relevant
        except Exception as e:
            logger.error(f"Error getting relevant context: {e}")
            return []

    @staticmethod
    def update_session_topic_if_changed(
        session_id: str, new_question: str, similarity_data: dict
    ) -> bool:
        """
        Updates the session topic if the user has clearly switched to a new topic.

        DISABLED: We now enforce strict session continuity based on the first question.
        """
        # We explicitly do NOT want to update the topic, ever.
        # The session topic should remain fixed to the first question.
        return False
