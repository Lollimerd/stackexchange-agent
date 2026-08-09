import logging
import numpy as np
from datetime import datetime
from setup.init import EMBEDDINGS
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
            if not session_topic:
                return {
                    "similarity_score": 1.0,
                    "is_continuation": True,
                    "confidence_level": "high",
                    "recommendation": "First message in session",
                }

            # Embed both question and topic
            question_embedding = EMBEDDINGS.embed_query(question)
            topic_embedding = EMBEDDINGS.embed_query(session_topic)

            # Calculate cosine similarity
            q_vec = np.array(question_embedding)
            t_vec = np.array(topic_embedding)

            similarity_score = float(
                np.dot(q_vec, t_vec)
                / (np.linalg.norm(q_vec) * np.linalg.norm(t_vec) + 1e-10)
            )

            # Determine confidence level
            if similarity_score > 0.75:
                confidence_level = "high"
                recommendation = "CONTINUATION: User is asking a follow-up on the same topic. Build upon previous context."
                is_continuation = True
            elif similarity_score > 0.55:
                confidence_level = "medium"
                recommendation = "POSSIBLE_CONTINUATION: User may be asking a tangential question. Acknowledge the current topic but allow for context shift."
                is_continuation = True
            else:
                confidence_level = "low"
                recommendation = "NEW_TOPIC: User appears to be switching to a new topic. You can acknowledge this shift gracefully."
                is_continuation = False

            logger.info(
                f"Topic similarity score: {similarity_score:.2f} ({confidence_level})"
            )

            return {
                "similarity_score": similarity_score,
                "is_continuation": is_continuation,
                "confidence_level": confidence_level,
                "recommendation": recommendation,
            }
        except Exception as e:
            logger.error(f"Error calculating topic similarity: {e}")
            return {
                "similarity_score": 0.5,
                "is_continuation": True,
                "confidence_level": "low",
                "recommendation": "Unable to determine similarity. Proceed with caution.",
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
            question_embedding = np.array(EMBEDDINGS.embed_query(question))

            # Score all messages based on relevance to current question
            scored_messages = []
            for msg in all_messages[1:]:  # Skip first message
                msg_content = getattr(msg, "content", "")
                if not msg_content:
                    continue

                msg_embedding = np.array(EMBEDDINGS.embed_query(msg_content))
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
        Only updates if similarity is very low (< 0.4).
        """
        try:
            graph = get_graph_instance()

            # Only update if very different from current topic
            if similarity_data["similarity_score"] < 0.4:
                query = """
                MATCH (s:Session {id: $session_id})
                SET s.topic = $new_topic,
                    s.topic_changed_at = $timestamp,
                    s.previous_topic = coalesce(s.topic, '')
                """
                graph.query(
                    query,
                    params={
                        "session_id": session_id,
                        "new_topic": new_question,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
                logger.info(
                    f"Session {session_id} topic updated to: {new_question[:50]}..."
                )
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating session topic: {e}")
            return False
