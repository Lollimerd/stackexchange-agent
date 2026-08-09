from langchain_neo4j import Neo4jChatMessageHistory, Neo4jGraph
from setup.init import NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# FIX: Create module-level graph instance for connection pooling
# Instead of creating new connections on every call
_graph_instance = None
_chat_history_cache = {}

def get_graph_instance() -> Neo4jGraph:
    """Get or create a reusable Neo4j graph instance (connection pooling)."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = Neo4jGraph(
            url=NEO4J_URL,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )
    return _graph_instance

def get_chat_history(session_id: str):
    """
    Returns a chat message history object stored in Neo4j.
    It creates a node for the session and links messages to it.
    """
    try:
        return Neo4jChatMessageHistory(
            session_id=session_id,
            url=NEO4J_URL,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )
    except Exception as e:
        logger.error(f"Error getting chat history for session {session_id}: {e}")
        # Return an empty history object that won't crash
        class EmptyHistory:
            messages = []
        return EmptyHistory()

def add_user_message_to_session(session_id: str, content: str):
    """
    Adds a user message to the session and explicitly creates a HAS_MESSAGE relationship.
    """
    try:
        # 1. Add message via LangChain (creates node + linked list + LAST_MESSAGE)
        history = get_chat_history(session_id)
        history.add_user_message(content)

        # 2. Enforce HAS_MESSAGE relationship using the LAST_MESSAGE pointer
        # FIX: Also set created_at timestamp and use pooled connection
        graph = get_graph_instance()
        # Match the session and the message marked as LAST_MESSAGE (which is the one just added)
        # Then create HAS_MESSAGE and set created_at
        query = """
        MATCH (s:Session {id: $session_id})-[:LAST_MESSAGE]->(m:Message)
        SET m.created_at = $timestamp
        MERGE (s)-[:HAS_MESSAGE]->(m)
        """
        graph.query(query, params={"session_id": session_id, "timestamp": datetime.now().isoformat()})
        logger.debug(f"User message added to session {session_id}")
    except Exception as e:
        logger.error(f"Error adding user message to session {session_id}: {e}")

def add_ai_message_to_session(session_id: str, content: str, thought: str = None):
    """
    Adds an AI message to the session and explicitly creates a HAS_MESSAGE relationship.
    Also stores the reasoning/thought process if provided.
    """
    try:
        history = get_chat_history(session_id)
        history.add_ai_message(content)

        # FIX: Use pooled connection
        graph = get_graph_instance()
        # Match the session and the message marked as LAST_MESSAGE
        # Set the thought property, created_at, and create HAS_MESSAGE
        query = """
        MATCH (s:Session {id: $session_id})-[:LAST_MESSAGE]->(m:Message)
        SET m.thought = $thought, m.created_at = $timestamp
        MERGE (s)-[:HAS_MESSAGE]->(m)
        """
        graph.query(query, params={
            "session_id": session_id, 
            "thought": thought, 
            "timestamp": datetime.now().isoformat()
        })
        logger.debug(f"AI message added to session {session_id}")
    except Exception as e:
        logger.error(f"Error adding AI message to session {session_id}: {e}")

def get_all_sessions():
    """
    Retrieves all chat sessions stored in Neo4j.
    Returns a list of dicts with session_id and last_message details.
    """
    try:
        # FIX: Use pooled connection
        graph = get_graph_instance()

        # FIX: Optimize query - avoid unbounded traversal [*0..], limit results
        query = """
        MATCH (s:Session)
        OPTIONAL MATCH (s)-[:HAS_MESSAGE]->(m:Message)
        WITH s, m ORDER BY coalesce(m.created_at, m.timestamp, elementId(m)) DESC
        WITH s, head(collect(m)) AS last_msg
        RETURN s.id AS session_id, last_msg.content AS last_message
        LIMIT 100
        """

        return graph.query(query)
    except Exception as e:
        logger.error(f"Error getting all sessions: {e}")
        return []

def link_session_to_user(session_id: str, user_id: str, topic: str = None):
    """
    Links a session to an AppUser. Creates the user if not exists.
    Optionally stores the conversation topic for the session.
    """
    if not user_id:
        return

    try:
        # FIX: Use pooled connection
        graph = get_graph_instance()

        # Set topic on the session if provided (typically on first message)
        if topic:
            query = """
            MERGE (u:AppUser {id: $user_id})
            MERGE (s:Session {id: $session_id, topic: $topic})
            ON CREATE SET s.topic = $topic
            MERGE (u)-[:HAS_SESSION]->(s)
            """
            graph.query(query, params={"user_id": user_id, "session_id": session_id, "topic": topic})
        else:
            query = """
            MERGE (u:AppUser {id: $user_id})
            MERGE (s:Session {id: $session_id})
            MERGE (u)-[:HAS_SESSION]->(s)
            """
            graph.query(query, params={"user_id": user_id, "session_id": session_id})
        
        logger.debug(f"Linked session {session_id} to user {user_id}" + (f" with topic: {topic}" if topic else ""))
    except Exception as e:
        logger.error(f"Error linking session to user: {e}")

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

def get_user_sessions(user_id: str):
    """
    Retrieves chat sessions for a specific user.
    Returns a list of dicts with session_id and last_message details.
    """
    try:
        # FIX: Use pooled connection
        graph = get_graph_instance()

        # FIX: Optimize query - avoid unbounded traversal, limit results, use correct order
        query = """
        MATCH (u:AppUser {id: $user_id})-[:HAS_SESSION]->(s:Session)
        OPTIONAL MATCH (s)-[:HAS_MESSAGE]->(m:Message)
        WITH s, m ORDER BY coalesce(m.created_at, m.timestamp, elementId(m)) DESC
        WITH s, head(collect(m)) AS last_msg
        RETURN s.id AS session_id, last_msg.content AS last_message
        LIMIT 100
        """

        return graph.query(query, params={"user_id": user_id})
    except Exception as e:
        logger.error(f"Error getting sessions for user {user_id}: {e}")
        return []

def get_all_users():
    """
    Retrieves a list of all existing AppUser IDs.
    """
    try:
        # FIX: Use pooled connection
        graph = get_graph_instance()
        query = "MATCH (u:AppUser) RETURN u.id as user_id LIMIT 1000"
        result = graph.query(query)
        return [record['user_id'] for record in result]
    except Exception as e:
        logger.error(f"Error getting all users: {e}")
        return []

def delete_session(session_id: str):
    """
    Deletes a session and its messages from the database.
    Uses a broad match to ensure all connected messages (linked list or star) are removed.
    """
    try:
        # FIX: Use pooled connection and optimize query
        graph = get_graph_instance()
        query = """
        MATCH (s:Session {id: $session_id})
        OPTIONAL MATCH (s)-[:HAS_MESSAGE]->(m:Message)
        DETACH DELETE m, s
        """
        graph.query(query, params={"session_id": session_id})
        logger.info(f"Session {session_id} deleted")
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")

def delete_user(user_id: str):
    """
    Deletes a user and all their sessions/messages.
    """
    try:
        # FIX: Use pooled connection and optimize query
        graph = get_graph_instance()
        query = """
        MATCH (u:AppUser {id: $user_id})
        OPTIONAL MATCH (u)-[:HAS_SESSION]->(s:Session)
        OPTIONAL MATCH (s)-[:HAS_MESSAGE]->(m:Message)
        DETACH DELETE m, s, u
        """
        graph.query(query, params={"user_id": user_id})
        logger.info(f"User {user_id} and all their data deleted")
    except Exception as e:
        logger.error(f"Error deleting user {user_id}: {e}")

def calculate_topic_similarity(question: str, session_topic: str) -> dict:
    """
    Calculates semantic similarity between current question and session topic.
    Returns a dict with similarity_score, is_continuation, and recommendation.
    
    Args:
        question: The current user question
        session_topic: The original topic of the session
        
    Returns:
        dict with keys:
            - similarity_score: float (0-1)
            - is_continuation: bool (True if score > 0.6)
            - confidence_level: str ('high', 'medium', 'low')
            - recommendation: str (instructions for the LLM)
    """
    try:
        from setup.init import EMBEDDINGS
        import numpy as np
        
        if not session_topic:
            return {
                "similarity_score": 1.0,
                "is_continuation": True,
                "confidence_level": "high",
                "recommendation": "First message in session"
            }
        
        # Embed both question and topic
        question_embedding = EMBEDDINGS.embed_query(question)
        topic_embedding = EMBEDDINGS.embed_query(session_topic)
        
        # Calculate cosine similarity
        q_vec = np.array(question_embedding)
        t_vec = np.array(topic_embedding)
        
        similarity_score = float(np.dot(q_vec, t_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(t_vec) + 1e-10))
        
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
        
        logger.info(f"Topic similarity score: {similarity_score:.2f} ({confidence_level})")
        
        return {
            "similarity_score": similarity_score,
            "is_continuation": is_continuation,
            "confidence_level": confidence_level,
            "recommendation": recommendation
        }
    except Exception as e:
        logger.error(f"Error calculating topic similarity: {e}")
        return {
            "similarity_score": 0.5,
            "is_continuation": True,
            "confidence_level": "low",
            "recommendation": "Unable to determine similarity. Proceed with caution."
        }

def get_relevant_context_for_continuation(session_id: str, question: str, max_messages: int = 3) -> list:
    """
    Retrieves only the most relevant previous messages for topic continuation.
    Filters messages based on semantic relevance to the current question.
    
    Args:
        session_id: The session ID
        question: The current question
        max_messages: Maximum number of previous messages to retrieve
        
    Returns:
        List of relevant message dicts with role and content
    """
    try:
        from setup.init import EMBEDDINGS
        import numpy as np
        
        # Get all messages
        history = get_chat_history(session_id)
        all_messages = history.messages if hasattr(history, 'messages') else []
        
        if len(all_messages) <= 2:  # Only system message + first user message
            return []
        
        # Embed the question
        question_embedding = np.array(EMBEDDINGS.embed_query(question))
        
        # Score all messages based on relevance to current question
        scored_messages = []
        for msg in all_messages[1:]:  # Skip first message
            msg_content = getattr(msg, 'content', '')
            if not msg_content:
                continue
                
            msg_embedding = np.array(EMBEDDINGS.embed_query(msg_content))
            similarity = float(np.dot(question_embedding, msg_embedding) / 
                             (np.linalg.norm(question_embedding) * np.linalg.norm(msg_embedding) + 1e-10))
            
            scored_messages.append({
                "similarity": similarity,
                "role": getattr(msg, 'type', 'unknown'),
                "content": msg_content
            })
        
        # Sort by similarity and return top N
        scored_messages.sort(key=lambda x: x['similarity'], reverse=True)
        relevant = scored_messages[:max_messages]
        
        logger.info(f"Retrieved {len(relevant)} relevant context messages")
        return relevant
    except Exception as e:
        logger.error(f"Error getting relevant context: {e}")
        return []

def update_session_topic_if_changed(session_id: str, new_question: str, similarity_data: dict) -> bool:
    """
    Updates the session topic if the user has clearly switched to a new topic.
    Only updates if similarity is very low (< 0.4).
    
    Args:
        session_id: The session ID
        new_question: The new question asked
        similarity_data: The dict returned from calculate_topic_similarity()
        
    Returns:
        bool: True if topic was updated, False otherwise
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
            graph.query(query, params={
                "session_id": session_id,
                "new_topic": new_question,
                "timestamp": datetime.now().isoformat()
            })
            logger.info(f"Session {session_id} topic updated to: {new_question[:50]}...")
            return True
        return False
    except Exception as e:
        logger.error(f"Error updating session topic: {e}")
        return False
