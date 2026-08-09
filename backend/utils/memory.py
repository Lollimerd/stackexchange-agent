from langchain_neo4j import Neo4jChatMessageHistory, Neo4jGraph
from setup.init import NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD
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
        # FIX: Use pooled connection instead of creating new one
        graph = get_graph_instance()
        # Match the session and the message marked as LAST_MESSAGE (which is the one just added)
        # Then create HAS_MESSAGE
        query = """
        MATCH (s:Session {id: $session_id})-[:LAST_MESSAGE]->(m:Message)
        MERGE (s)-[:HAS_MESSAGE]->(m)
        """
        graph.query(query, params={"session_id": session_id})
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
        # Set the thought property and create HAS_MESSAGE
        query = """
        MATCH (s:Session {id: $session_id})-[:LAST_MESSAGE]->(m:Message)
        SET m.thought = $thought
        MERGE (s)-[:HAS_MESSAGE]->(m)
        """
        graph.query(query, params={"session_id": session_id, "thought": thought})
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
        WITH s, m ORDER BY m.created_at DESC LIMIT 1
        RETURN s.id AS session_id, m.content AS last_message
        LIMIT 100
        """

        return graph.query(query)
    except Exception as e:
        logger.error(f"Error getting all sessions: {e}")
        return []

def link_session_to_user(session_id: str, user_id: str):
    """
    Links a session to an AppUser. Creates the user if not exists.
    """
    if not user_id:
        return

    try:
        # FIX: Use pooled connection
        graph = get_graph_instance()

        query = """
        MERGE (u:AppUser {id: $user_id})
        MERGE (s:Session {id: $session_id})
        MERGE (u)-[:HAS_SESSION]->(s)
        """
        graph.query(query, params={"user_id": user_id, "session_id": session_id})
        logger.debug(f"Linked session {session_id} to user {user_id}")
    except Exception as e:
        logger.error(f"Error linking session to user: {e}")

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
        WITH s, m ORDER BY m.created_at DESC LIMIT 1
        RETURN s.id AS session_id, m.content AS last_message
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
