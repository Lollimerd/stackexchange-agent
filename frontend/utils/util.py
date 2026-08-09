import streamlit as st
import re, hashlib, requests, uuid, json, os
from streamlit_mermaid import st_mermaid
from typing import List
from datetime import datetime

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

    for i, part in enumerate(parts):
        # This is a mermaid block
        if part.strip().startswith("```mermaid"):
            # Extract the code by removing the fences
            mermaid_code = part.strip().replace("```mermaid", "").replace("```", "")
            key = hashlib.sha256(mermaid_code.encode()).hexdigest()
            try:
                st_mermaid(mermaid_code, key=f"{key}_{i}")
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
@st.cache_data(ttl=3600) # Cache the data for 1 hour
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
