import streamlit as st
import re, hashlib, requests, uuid, json, os
from streamlit_mermaid import st_mermaid
from typing import List
from datetime import datetime

# --- API Configuration ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
FASTAPI_URL = f"{BACKEND_URL}/stream-ask"
CHATS_URL = f"{BACKEND_URL}/api/v1/user"  # /{user_id}/chats
CHAT_HISTORY_URL = f"{BACKEND_URL}/api/v1/chat"
USERS_URL = f"{BACKEND_URL}/api/v1/users"

# --- API Helper Functions with Error Handling ---
def fetch_all_users(retry_count=2):
    """Fetch all users with retry logic."""
    for attempt in range(retry_count):
        try:
            response = requests.get(USERS_URL, timeout=5)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success":
                return data.get("users", [])
            return []
        except requests.exceptions.Timeout:
            if attempt < retry_count - 1:
                st.warning("Connection timeout, retrying...")
                continue
            logger.error(f"Timeout fetching users after {retry_count} attempts")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching users: {e}")
            if attempt < retry_count - 1:
                continue
            return []
    return []

def delete_chat_api(session_id):
    """Delete a chat session."""
    try:
        requests.delete(f"{CHAT_HISTORY_URL}/{session_id}", timeout=5)
        logger.info(f"Chat {session_id} deleted")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error deleting chat {session_id}: {e}")
        st.warning(f"Could not delete chat: {str(e)[:50]}")

def delete_user_api(user_id):
    """Delete a user and all their data."""
    try:
        requests.delete(f"{BACKEND_URL}/api/v1/user/{user_id}", timeout=5)
        logger.info(f"User {user_id} deleted")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error deleting user {user_id}: {e}")
        st.warning(f"Could not delete user: {str(e)[:50]}")

def fetch_user_chats(user_id, retry_count=2):
    """Fetch user's chat sessions with retry logic."""
    for attempt in range(retry_count):
        try:
            response = requests.get(f"{CHATS_URL}/{user_id}/chats", timeout=5)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success":
                return data.get("chats", [])
            return []
        except requests.exceptions.Timeout:
            if attempt < retry_count - 1:
                continue
            logger.error(f"Timeout fetching chats for user {user_id}")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching chats for user {user_id}: {e}")
            if attempt < retry_count - 1:
                continue
            return []
    return []

def fetch_chat_history(session_id, retry_count=2):
    """Fetch chat history with retry logic."""
    for attempt in range(retry_count):
        try:
            response = requests.get(f"{CHAT_HISTORY_URL}/{session_id}", timeout=5)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success":
                messages = data.get("messages", [])
                # Validate messages have required fields
                validated = []
                for msg in messages:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        validated.append(msg)
                return validated
            return []
        except requests.exceptions.Timeout:
            if attempt < retry_count - 1:
                continue
            logger.error(f"Timeout fetching history for {session_id}")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching history for {session_id}: {e}")
            if attempt < retry_count - 1:
                continue
            return []
    return []

def extract_title_and_question(input_string):
    lines = input_string.strip().split("\n")
    title = ""
    question = ""
    is_question = False  # flag to know if we are inside a "Question" block

    for line in lines:
        if line.startswith("Title:"):
            title = line.split("Title: ", 1)[1].strip()
        elif line.startswith("Question:"):
            question = line.split("Question: ", 1)[1].strip()
            is_question = (
                True  # set the flag to True once we encounter a "Question:" line
            )
        elif is_question:
            # if the line does not start with "Question:" but we are inside a "Question" block,
            # then it is a continuation of the question
            question += "\n" + line.strip()

    return title, question

def create_vector_index(driver) -> None:
    index_query = "CREATE VECTOR INDEX stackoverflow IF NOT EXISTS FOR (m:Question) ON m.embedding"
    try:
        driver.query(index_query)
    except:  # Already exists
        pass
    index_query = (
        "CREATE VECTOR INDEX top_answers IF NOT EXISTS FOR (m:Answer) ON m.embedding"
    )
    try:
        driver.query(index_query)
    except:  # Already exists
        pass

def create_constraints(driver):
    driver.query(
        "CREATE CONSTRAINT question_id IF NOT EXISTS FOR (q:Question) REQUIRE (q.id) IS UNIQUE"
    )
    driver.query(
        "CREATE CONSTRAINT answer_id IF NOT EXISTS FOR (a:Answer) REQUIRE (a.id) IS UNIQUE"
    )
    driver.query(
        "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE (u.id) IS UNIQUE"
    )
    driver.query(
        "CREATE CONSTRAINT tag_name IF NOT EXISTS FOR (t:Tag) REQUIRE (t.name) IS UNIQUE"
    )
    driver.query(
        "CREATE CONSTRAINT importlog_id IF NOT EXISTS FOR (i:ImportLog) REQUIRE (i.id) IS UNIQUE"
    )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# This is a placeholder for LangChain's Document class
class Document:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata

def format_docs_with_metadata(docs: List[Document]) -> str:
    """
    Formats a list of Documents into a single string, where each
    document's page_content is followed by its corresponding metadata.
    """
    # Create a list of formatted strings, one for each document
    formatted_blocks = []
    for doc in docs:
        # Format the metadata as a pretty JSON string
        metadata_str = json.dumps(doc.metadata, indent=2)

        # Create a combined block for the document's content and its metadata
        block = (
            f"Content: \n{doc.page_content}\n"
            f"--- METADATA ---\n"
            f"{metadata_str}"
        )
        formatted_blocks.append(block)

    # Join all the individual document blocks with a clear separator
    return "\n\n======================================================\n\n".join(formatted_blocks)

# --- Mermaid Rendering Function ---
def render_message_with_mermaid(content):
    """Parses a message and renders Markdown and Mermaid blocks separately."""
    # Use re.split to keep the text and the diagrams in order
    # The pattern captures the mermaid block, and split keeps the delimiters
    parts = re.split(r"(```mermaid\n.*?\n```)", content, flags=re.DOTALL)

    for part in parts:
        # This is a mermaid block
        if part.strip().startswith("```mermaid"):
            # Extract the code by removing the fences
            mermaid_code = part.strip().replace("```mermaid", "").replace("```", "").strip()
            if mermaid_code:
                # Generate a unique key based on the mermaid code content only
                # This ensures the same diagram always gets the same key
                key = hashlib.sha256(mermaid_code.encode()).hexdigest()
                try:
                    st_mermaid(mermaid_code, key=key)
                except Exception as e:
                    st.error(f"Failed to render Mermaid diagram: {e}")
                    st.code(mermaid_code, language="mermaid") # Show the raw code on failure
        else:
            # This is a regular markdown block
            if part.strip():
                st.markdown(part)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
CONFIG_URL = f"{BACKEND_URL}/api/v1/config" # New API endpoint

# --- 🆕 Function to fetch and display container name ---
def display_container_name():
    """Fetches and displays the Neo4j container name in the sidebar."""
    try:
        with st.sidebar:
            with st.spinner("Connecting to DB..."):
                response = requests.get(CONFIG_URL)
                response.raise_for_status()
                data = response.json()
                container_name = data.get("container_name", "N/A")
                st.success(f"DB Connected: **{container_name}**", icon="🐳")
    except requests.exceptions.RequestException:
        st.sidebar.error("**DB Status:** Connection failed.")

# --- Config Func ---
def get_system_config():
    """Fetches configuration from the backend API."""
    try:
        response = requests.get(CONFIG_URL)
        response.raise_for_status() # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Could not fetch config: {e}")
        return None # Return None on failure
    
import_query = """
    UNWIND $data AS q
    MERGE (question:Question {id:q.question_id}) 
    ON CREATE SET question.title = q.title, question.link = q.link, question.score = q.score,
        question.favorite_count = q.favorite_count, question.creation_date = datetime({epochSeconds: q.creation_date}),
        question.body = q.body_markdown, question.embedding = q.embedding
    FOREACH (tagName IN q.tags | 
        MERGE (tag:Tag {name:tagName}) 
        MERGE (question)-[:TAGGED]->(tag)
    )
    FOREACH (a IN q.answers |
        MERGE (question)<-[:ANSWERS]-(answer:Answer {id:a.answer_id})
        SET answer.is_accepted = a.is_accepted,
            answer.score = a.score,
            answer.creation_date = datetime({epochSeconds:a.creation_date}),
            answer.body = a.body_markdown,
            answer.embedding = a.embedding
        MERGE (answerer:User {id:coalesce(a.owner.user_id, "deleted")}) 
        ON CREATE SET answerer.display_name = a.owner.display_name,
                      answerer.reputation= a.owner.reputation
        MERGE (answer)<-[:PROVIDED]-(answerer)
    )
    WITH * WHERE NOT q.owner.user_id IS NULL
    MERGE (owner:User {id:q.owner.user_id})
    ON CREATE SET owner.display_name = q.owner.display_name,
                  owner.reputation = q.owner.reputation
    MERGE (owner)-[:ASKED]->(question)
    """

def record_import_session(driver, total_questions: int, tags_list: list, total_pages: int):
    """Record an import session in Neo4j as an ImportLog node."""
    import_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    query = """
    CREATE (log:ImportLog {
        id: $import_id,
        timestamp: datetime($timestamp),
        total_questions: $total_questions,
        total_tags: $total_tags,
        total_pages: $total_pages,
        tags_list: $tags_list
    })
    """
    
    driver.query(query, {
        "import_id": import_id,
        "timestamp": timestamp,
        "total_questions": total_questions,
        "total_tags": len(tags_list),
        "total_pages": total_pages,
        "tags_list": tags_list
    })
    
    return import_id

def get_database_summary(driver):
    """Get summary statistics from the database."""
    summary_query = """
    MATCH (q:Question)
    WITH count(q) as total_questions
    MATCH (t:Tag)
    WITH total_questions, count(t) as total_tags
    MATCH (a:Answer)
    WITH total_questions, total_tags, count(a) as total_answers
    MATCH (u:User)
    WITH total_questions, total_tags, total_answers, count(u) as total_users
    MATCH (log:ImportLog)
    WITH total_questions, total_tags, total_answers, total_users, count(log) as total_imports
    MATCH (log:ImportLog)
    WITH total_questions, total_tags, total_answers, total_users, total_imports, 
         max(log.timestamp) as last_import
    RETURN total_questions, total_tags, total_answers, total_users, total_imports, last_import
    """
    
    result = driver.query(summary_query)
    if result and len(result) > 0:
        return result[0]
    return {
        "total_questions": 0,
        "total_tags": 0, 
        "total_answers": 0,
        "total_users": 0,
        "total_imports": 0,
        "last_import": None
    }

def get_import_history(driver, limit: int = 20):
    """Get recent import history from ImportLog nodes."""
    history_query = """
    MATCH (log:ImportLog)
    RETURN log.id as id, log.timestamp as timestamp, log.total_questions as questions,
           log.total_tags as tags, log.total_pages as pages, log.tags_list as tags_list
    ORDER BY log.timestamp DESC
    LIMIT $limit
    """
    
    result = driver.query(history_query, {"limit": limit})
    return result if result else []


def get_entity_counts(driver):
    """Get counts for all entity types and relationships in the database."""
    # Node counts query
    node_counts_query = """
    CALL {
        MATCH (q:Question) RETURN 'Question' as label, count(q) as count
        UNION ALL
        MATCH (a:Answer) RETURN 'Answer' as label, count(a) as count
        UNION ALL
        MATCH (t:Tag) RETURN 'Tag' as label, count(t) as count
        UNION ALL
        MATCH (u:User) RETURN 'User' as label, count(u) as count
        UNION ALL
        MATCH (i:ImportLog) RETURN 'ImportLog' as label, count(i) as count
    }
    RETURN label, count
    """
    
    # Relationship counts query
    rel_counts_query = """
    CALL {
        MATCH ()-[r:TAGGED]->() RETURN 'TAGGED' as type, count(r) as count
        UNION ALL
        MATCH ()-[r:ANSWERS]->() RETURN 'ANSWERS' as type, count(r) as count
        UNION ALL
        MATCH ()-[r:PROVIDED]->() RETURN 'PROVIDED' as type, count(r) as count
        UNION ALL
        MATCH ()-[r:ASKED]->() RETURN 'ASKED' as type, count(r) as count
    }
    RETURN type, count
    """
    
    node_results = driver.query(node_counts_query)
    rel_results = driver.query(rel_counts_query)
    
    nodes = {r['label']: r['count'] for r in node_results} if node_results else {}
    relationships = {r['type']: r['count'] for r in rel_results} if rel_results else {}
    
    return {"nodes": nodes, "relationships": relationships}


def search_nodes(driver, search_term: str, limit: int = 10):
    """
    Search for nodes by title, name, or display_name.
    """
    query = """
    MATCH (n)
    WHERE n.title CONTAINS $term OR n.name CONTAINS $term OR n.display_name CONTAINS $term
    RETURN elementId(n) as id, labels(n)[0] as type, 
           COALESCE(n.title, n.name, n.display_name) as label
    LIMIT $limit
    """
    result = driver.query(query, {"term": search_term, "limit": limit})
    return result if result else []

def get_graph_sample(driver, node_types: list = None, rel_types: list = None, limit: int = 50, focus_node_id: str = None):
    """
    Fetch a sample of nodes and relationships for visualization.
    
    Args:
        driver: Neo4j graph driver
        node_types: List of node labels to include (default: all)
        rel_types: List of relationship types to include (default: all)
        limit: Maximum number of nodes to return
        focus_node_id: Optional elementId of a node to focus on
    
    Returns:
        dict with 'nodes' and 'edges' lists for visualization
    """
    all_node_types = ['Question', 'Answer', 'Tag', 'User']
    all_rel_types = ['TAGGED', 'ANSWERS', 'PROVIDED', 'ASKED']
    
    if not node_types:
        node_types = all_node_types
    if not rel_types:
        rel_types = all_rel_types
    
    # Build dynamic query based on selected types
    nodes = []
    edges = []
    node_ids = set()
    
    # Base query structure depends on whether we are focusing on a node or sampling
    if focus_node_id is not None:
        # Query for neighbors of the focused node (up to 2 hops for better context)
        # We find paths starting from the focused node
        query = """
        MATCH path = (root)-[r*1..2]-(m)
        WHERE elementId(root) = $focus_node_id
        AND ALL(n IN nodes(path) WHERE labels(n)[0] IN $node_types)
        And ALL(rel IN relationships(path) WHERE type(rel) IN $rel_types)
        WITH relationships(path) as rels
        UNWIND rels as r
        WITH startNode(r) as n, r, endNode(r) as m
        LIMIT $limit
        RETURN 
            elementId(n) as source_id, 
            labels(n)[0] as source_label,
            properties(n) as source_props,
            CASE labels(n)[0]
                WHEN 'Question' THEN COALESCE(n.title, 'Question ' + elementId(n))
                WHEN 'Answer' THEN 'Answer ' + elementId(n)
                WHEN 'Tag' THEN n.name
                WHEN 'User' THEN COALESCE(n.display_name, 'User ' + elementId(n))
                ELSE elementId(n)
            END as source_name,
            elementId(m) as target_id,
            labels(m)[0] as target_label,
            properties(m) as target_props,
            CASE labels(m)[0]
                WHEN 'Question' THEN COALESCE(m.title, 'Question ' + elementId(m))
                WHEN 'Answer' THEN 'Answer ' + elementId(m)
                WHEN 'Tag' THEN n.name
                WHEN 'User' THEN COALESCE(n.display_name, 'User ' + elementId(n))
                ELSE elementId(m)
            END as target_name,
            type(r) as rel_type
        """
        params = {
            "node_types": node_types,
            "rel_types": rel_types,
            "limit": limit * 3, # Allow more edges for focused view
            "focus_node_id": str(focus_node_id)
        }
    else:
        # Standard sampling query
        query = """
        MATCH (n)-[r]->(m)
        WHERE (labels(n)[0] IN $node_types OR labels(m)[0] IN $node_types)
          AND type(r) IN $rel_types
        WITH n, r, m
        LIMIT $limit
        RETURN 
            elementId(n) as source_id, 
            labels(n)[0] as source_label,
            properties(n) as source_props,
            CASE labels(n)[0]
                WHEN 'Question' THEN COALESCE(n.title, 'Question ' + elementId(n))
                WHEN 'Answer' THEN 'Answer ' + elementId(n)
                WHEN 'Tag' THEN n.name
                WHEN 'User' THEN COALESCE(n.display_name, 'User ' + elementId(n))
                ELSE elementId(n)
            END as source_name,
            elementId(m) as target_id,
            labels(m)[0] as target_label,
            properties(m) as target_props,
            CASE labels(m)[0]
                WHEN 'Question' THEN COALESCE(m.title, 'Question ' + elementId(m))
                WHEN 'Answer' THEN 'Answer ' + elementId(m)
                WHEN 'Tag' THEN m.name
                WHEN 'User' THEN COALESCE(m.display_name, 'User ' + elementId(m))
                ELSE elementId(m)
            END as target_name,
            type(r) as rel_type
        """
        params = {
            "node_types": node_types,
            "rel_types": rel_types,
            "limit": limit * 2
        }
    
    results = driver.query(query, params)
    
    if results:
        for r in results:
            # Add source node
            if r['source_id'] not in node_ids:
                nodes.append({
                    'id': r['source_id'],
                    'label': r['source_name'][:30] if r['source_name'] else str(r['source_id']),
                    'type': r['source_label'],
                    'title': r['source_name'],  # Will be improved in frontend
                    'properties': r.get('source_props', {})
                })
                node_ids.add(r['source_id'])
            
            # Add target node
            if r['target_id'] not in node_ids:
                nodes.append({
                    'id': r['target_id'],
                    'label': r['target_name'][:30] if r['target_name'] else str(r['target_id']),
                    'type': r['target_label'],
                    'title': r['target_name'],  # Will be improved in frontend
                    'properties': r.get('target_props', {})
                })
                node_ids.add(r['target_id'])
            
            # Add edge
            edges.append({
                'from': r['source_id'],
                'to': r['target_id'],
                'label': r['rel_type'],
                'title': r['rel_type']
            })
            
            # Limit total nodes (soft limit to ensure connectedness)
            if len(nodes) >= limit * 1.5:
                break
    
    return {"nodes": nodes, "edges": edges}
