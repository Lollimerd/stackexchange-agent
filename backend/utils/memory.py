from langchain_neo4j import Neo4jChatMessageHistory, Neo4jGraph
from setup.init_config import (
    NEO4J_URL,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    get_graph_instance,
)
from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)


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
            password=NEO4J_PASSWORD,
        )
    except Exception as e:
        logger.error(f"Error getting chat history for session {session_id}: {e}")

        # Return an empty history object that won't crash
        class EmptyHistory:
            messages = []

        return EmptyHistory()


def _ensure_has_message_relationship(
    session_id: str, message_content: str, message_type: str, additional_params: dict = None
):
    """
    Ensures that a message has the HAS_MESSAGE relationship with the Session.
    This is a helper function that creates the relationship atomically.
    
    Args:
        session_id: The session ID
        message_content: The message content
        message_type: 'user' or 'assistant'
        additional_params: Additional parameters to set on the message (e.g., thought)
    """
    graph = get_graph_instance()
    timestamp = datetime.now().isoformat()
    
    # First, ensure the Session node exists
    graph.query(
        """
        MERGE (s:Session {id: $session_id})
        """,
        params={"session_id": session_id},
    )
    
    # Create the Message node with HAS_MESSAGE relationship in a single operation
    # This avoids relying on LAST_MESSAGE which might not be set yet
    query = """
    MATCH (s:Session {id: $session_id})
    CREATE (m:Message {
        content: $content,
        type: $type,
        created_at: $timestamp
    })
    CREATE (s)-[:HAS_MESSAGE]->(m)
    CREATE (s)-[:LAST_MESSAGE]->(m)
    RETURN m
    """
    
    params = {
        "session_id": session_id,
        "content": message_content,
        "type": message_type,
        "timestamp": timestamp,
    }
    
    if additional_params:
        # Add any additional properties to the message
        for key, value in additional_params.items():
            if value is not None:
                params[key] = value
                # We'll need to SET these after creation
                if key == "thought":
                    query = query.replace(
                        "CREATE (m:Message {",
                        "CREATE (m:Message {"
                    )
    
    # Better approach: Create with all properties at once
    query = """
    MATCH (s:Session {id: $session_id})
    CREATE (m:Message {
        content: $content,
        type: $type,
        created_at: $timestamp
    """
    
    # Add optional properties
    if additional_params and "thought" in additional_params:
        query += ",\n        thought: $thought"
    
    query += """
    })
    CREATE (s)-[:HAS_MESSAGE]->(m)
    CREATE (s)-[:LAST_MESSAGE]->(m)
    RETURN elementId(m) AS message_id
    """
    
    graph.query(query, params=params)
    logger.debug(f"Created message with HAS_MESSAGE relationship for session {session_id}")


def add_user_message_to_session(session_id: str, content: str):
    """
    Adds a user message to the session atomically.
    Creates the Message node and HAS_MESSAGE relationship in a single operation.
    """
    try:
        graph = get_graph_instance()
        timestamp = datetime.now().isoformat()
        
        # Ensure Session node exists
        graph.query(
            """
            MERGE (s:Session {id: $session_id})
            """,
            params={"session_id": session_id},
        )
        
        # Create Message with HAS_MESSAGE relationship atomically
        # Also update LAST_MESSAGE to point to this new message
        query = """
        MATCH (s:Session {id: $session_id})
        OPTIONAL MATCH (s)-[old_rel:LAST_MESSAGE]->(old_msg:Message)
        DELETE old_rel
        WITH s
        CREATE (m:Message {
            content: $content,
            type: 'user',
            created_at: $timestamp
        })
        CREATE (s)-[:LAST_MESSAGE]->(m)
        CREATE (s)-[:HAS_MESSAGE]->(m)
        RETURN elementId(m) AS message_id
        """
        
        graph.query(
            query,
            params={
                "session_id": session_id,
                "content": content,
                "timestamp": timestamp,
            },
        )
        logger.info(f"Successfully added user message to session {session_id}")
        
    except Exception as e:
        logger.error(f"Error adding user message to session {session_id}: {e}")
        # Don't silently fail - attempt to add via LangChain as fallback
        try:
            logger.warning(f"Attempting fallback via LangChain for session {session_id}")
            history = get_chat_history(session_id)
            history.add_user_message(content)
            
            # Retry creating HAS_MESSAGE relationship
            graph = get_graph_instance()
            query = """
            MATCH (s:Session {id: $session_id})-[:LAST_MESSAGE]->(m:Message)
            WHERE NOT EXISTS((s)-[:HAS_MESSAGE]->(m))
            SET m.created_at = $timestamp, m.type = 'user', m.content = $content
            MERGE (s)-[:HAS_MESSAGE]->(m)
            """
            graph.query(
                query,
                params={
                    "session_id": session_id,
                    "content": content,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            logger.info(f"Fallback succeeded for session {session_id}")
        except Exception as fallback_err:
            logger.error(f"Fallback also failed for session {session_id}: {fallback_err}")


def add_ai_message_to_session(session_id: str, content: str, thought: str):
    """
    Adds an AI message to the session atomically.
    Creates the Message node with thought and HAS_MESSAGE relationship in a single operation.
    Also stores the reasoning/thought process if provided.
    """
    try:
        graph = get_graph_instance()
        timestamp = datetime.now().isoformat()
        
        # Ensure Session node exists
        graph.query(
            """
            MERGE (s:Session {id: $session_id})
            """,
            params={"session_id": session_id},
        )
        
        # Create Message with HAS_MESSAGE relationship atomically
        # Also update LAST_MESSAGE to point to this new message
        query = """
        MATCH (s:Session {id: $session_id})
        OPTIONAL MATCH (s)-[old_rel:LAST_MESSAGE]->(old_msg:Message)
        DELETE old_rel
        WITH s
        CREATE (m:Message {
            content: $content,
            type: 'assistant',
            thought: $thought,
            created_at: $timestamp
        })
        CREATE (s)-[:LAST_MESSAGE]->(m)
        CREATE (s)-[:HAS_MESSAGE]->(m)
        RETURN elementId(m) AS message_id
        """
        
        graph.query(
            query,
            params={
                "session_id": session_id,
                "content": content,
                "thought": thought if thought else None,
                "timestamp": timestamp,
            },
        )
        logger.info(
            f"Successfully added AI message to session {session_id} "
            f"(thought length: {len(thought) if thought else 0})"
        )
        
    except Exception as e:
        logger.error(f"Error adding AI message to session {session_id}: {e}")
        # Don't silently fail - attempt to add via LangChain as fallback
        try:
            logger.warning(f"Attempting fallback via LangChain for session {session_id}")
            history = get_chat_history(session_id)
            history.add_ai_message(content)
            
            # Retry creating HAS_MESSAGE relationship with thought
            graph = get_graph_instance()
            query = """
            MATCH (s:Session {id: $session_id})-[:LAST_MESSAGE]->(m:Message)
            WHERE NOT EXISTS((s)-[:HAS_MESSAGE]->(m))
            SET m.thought = $thought, 
                m.created_at = $timestamp, 
                m.type = 'assistant', 
                m.content = $content
            MERGE (s)-[:HAS_MESSAGE]->(m)
            """
            graph.query(
                query,
                params={
                    "session_id": session_id,
                    "content": content,
                    "thought": thought if thought else None,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            logger.info(f"Fallback succeeded for session {session_id}")
        except Exception as fallback_err:
            logger.error(f"Fallback also failed for session {session_id}: {fallback_err}")


def repair_missing_has_message_relationships():
    """
    Repairs sessions that have messages with LAST_MESSAGE but missing HAS_MESSAGE relationships.
    This fixes the data inconsistency issue where messages exist but are invisible to queries.
    
    Returns:
        int: Number of relationships repaired
    """
    try:
        graph = get_graph_instance()
        
        # Find all Message nodes that have LAST_MESSAGE but not HAS_MESSAGE
        # and create the missing HAS_MESSAGE relationships
        repair_query = """
        MATCH (s:Session)-[:LAST_MESSAGE]->(m:Message)
        WHERE NOT EXISTS((s)-[:HAS_MESSAGE]->(m))
        MERGE (s)-[:HAS_MESSAGE]->(m)
        RETURN count(m) AS repaired_count
        """
        
        result = graph.query(repair_query)
        repaired = result[0]["repaired_count"] if result else 0
        
        if repaired > 0:
            logger.info(f"✅ Repaired {repaired} missing HAS_MESSAGE relationships")
        else:
            logger.debug("No missing HAS_MESSAGE relationships found")
        
        return repaired
        
    except Exception as e:
        logger.error(f"Error repairing HAS_MESSAGE relationships: {e}")
        return 0


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


def link_session_to_user(session_id: str, user_id: str, topic: str):
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
            MERGE (s:Session {id: $session_id})
            ON CREATE SET s.topic = $topic
            MERGE (u)-[:HAS_SESSION]->(s)
            """
            graph.query(
                query,
                params={"user_id": user_id, "session_id": session_id, "topic": topic},
            )
        else:
            query = """
            MERGE (u:AppUser {id: $user_id})
            MERGE (s:Session {id: $session_id})
            MERGE (u)-[:HAS_SESSION]->(s)
            """
            graph.query(query, params={"user_id": user_id, "session_id": session_id})

        logger.debug(
            f"Linked session {session_id} to user {user_id}"
            + (f" with topic: {topic}" if topic else "")
        )
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
        return [record["user_id"] for record in result]
    except Exception as e:
        logger.error(f"Error getting all users: {e}")
        return []


def delete_session(session_id: str):
    """
    Deletes a session and its messages from the database.
    Uses a broad traversal to ensure all connected messages are removed regardless
    of which relationship type links them (HAS_MESSAGE, LAST_MESSAGE, NEXT, etc.).
    """
    try:
        graph = get_graph_instance()
        query = """
        MATCH (s:Session {id: $session_id})
        OPTIONAL MATCH (s)-[*1..50]-(m:Message)
        DETACH DELETE m, s
        """
        graph.query(query, params={"session_id": session_id})
        logger.info(f"Session {session_id} deleted")
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")


def delete_user(user_id: str):
    """
    Deletes a user and all their sessions/messages.
    Uses a broad traversal from each session to catch all linked Message nodes
    regardless of relationship type (HAS_MESSAGE, LAST_MESSAGE, NEXT, etc.).
    """
    try:
        graph = get_graph_instance()
        query = """
        MATCH (u:AppUser {id: $user_id})
        OPTIONAL MATCH (u)-[:HAS_SESSION]->(s:Session)
        OPTIONAL MATCH (s)-[*1..50]-(m:Message)
        DETACH DELETE m, s, u
        """
        graph.query(query, params={"user_id": user_id})
        logger.info(f"User {user_id} and all their data deleted")
    except Exception as e:
        logger.error(f"Error deleting user {user_id}: {e}")
