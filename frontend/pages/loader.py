import os, time, requests
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
import streamlit as st
from streamlit.logger import get_logger
from langchain_ollama import OllamaEmbeddings
from utils.util import create_constraints, create_vector_index, import_query, record_import_session
# from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# load credential details
load_dotenv()
url = os.getenv("NEO4J_URL")
username = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASS")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")

logger = get_logger(__name__)

so_api_base_url = "https://api.stackexchange.com/2.3/search/advanced"

embeddings = OllamaEmbeddings(
    model="jina/jina-embeddings-v2-base-en:latest", 
    base_url=ollama_base_url, 
    num_ctx=8192, # 8k context
)

embedding_sub = OllamaEmbeddings(
    model="embeddinggemma:300m", 
    base_url=ollama_base_url, 
    num_ctx=2048, # 2k context
)

# if Neo4j is local, you can go to http://localhost:7474/ to browse the database
neo4j_graph = Neo4jGraph(
    url=url, 
    username=username, 
    password=password, 
)

def load_so_data(tag: str, page: int, site: str) -> dict:
    """
    Load Stack Overflow data and handle potential errors gracefully.
    This function is now designed to run in a background thread and should NOT call any st.* functions.
    It returns a dictionary indicating the result.
    """
    try:
        api_key = os.getenv("STACKEXCHANGE_API_KEY") 
        key_param = f"&key={api_key}" if api_key else ""
        site = "stackoverflow"
        parameters = (f"""?pagesize=100&page={page}&order=desc&sort=creation&answers=1&tagged={tag}&site={site}&filter=!*236eb_eL9rai)MOSNZ-6D3Q6ZKb0buI*IVotWaTb{key_param}""")
        
        # Wrap the network request in its own try-except block
        response = requests.get(so_api_base_url + parameters)
        response.raise_for_status()  # This will raise an exception for HTTP errors (4xx or 5xx)
        data = response.json()

        if "items" in data and data["items"]:
            # Handle API backoff requests
            if "backoff" in data:
                time.sleep(data["backoff"])
            elif "error_name" in data:
                backoff_time = min(300, 2 ** (page % 8))  # Max 300 seconds
                time.sleep(backoff_time)
            insert_so_data(data)
            return {"status": "success", "tag": tag, "page": page, "count": len(data["items"])}
        else:
            return {"status": "empty", "tag": tag, "page": page}
            
    except requests.exceptions.RequestException as e:
        return {"status": "error", "tag": tag, "page": page, "error": f"Network error: {e}"}
    except Exception as e:
        return {"status": "error", "tag": tag, "page": page, "error": f"An unexpected error occurred: {e}"}

def load_high_score_so_data(site: str) -> None:
    """load stackoverflow data with a high score"""
    parameters = (f"""?fromdate=1664150400&order=desc&sort=votes&site={site}&filter=!.DK56VBPooplF.)bWW5iOX32Fh1lcCkw1b_Y6Zkb7YD8.ZMhrR5.FRRsR6Z1uK8*Z5wPaONvyII""")
    data = requests.get(so_api_base_url + parameters).json()
    if "items" in data and data["items"]:
        if "error_name" in data:
            # backoff_time = min(300, 2 ** (page % 8))  # Max 300 seconds
            backoff_time = 10  # Fixed backoff time for high score data
            st.warning(f"API requested a backoff of {backoff_time} seconds.")
            time.sleep(backoff_time)
        insert_so_data(data)
    else:
        st.warning("No highly ranked items found. Skipping.")

def insert_so_data(data: dict) -> None:
    """Insert StackOverflow data into Neo4j."""
    # Calculate embedding values for questions and answers
    for q in data["items"]:
        question_text = q["title"] + "\n" + q["body_markdown"]
        q["embedding"] = embeddings.embed_query(question_text)
        time.sleep(0.5)  # to avoid hitting rate limits
        for a in q["answers"]:
            a["embedding"] = embeddings.embed_query(
                question_text + "\n" + a["body_markdown"]
            )
            time.sleep(0.5)  # to avoid hitting rate limits

    neo4j_graph.query(import_query, {"data": data["items"]})


# --- Streamlit ---
def get_tags() -> list[str]:
    """Gets a comma-separated string of tags and returns a clean list."""
    input_text = st.text_input(
        "Enter tags separated by commas", value="neo4j, cypher, python"
    )
    return [tag.strip() for tag in input_text.split(",") if tag.strip()]

def get_site() -> str:
    """Gets the Stack Exchange site to import from."""
    site = st.text_input(
        "Enter Stack Exchange site", value="stackoverflow"
    )
    return site.strip()

def get_pages():
    col1, col2 = st.columns(2)
    with col1:
        num_pages = st.number_input(
            "Number of pages (100 questions per page)", step=1, min_value=1
        )
    with col2:
        start_page = st.number_input("Start page", step=1, min_value=1)
    st.caption("Only questions with answers will be imported.")
    return (int(num_pages), int(start_page))

# --- Main Page Rendering (Modified Logic) ---
def render_page():
    st.header("StackOverflow Loader")
    st.subheader("Choose StackOverflow tags to load into Neo4j")
    st.caption("Go to http://localhost:7473/ to explore the graph.")

    site = get_site()
    tags_to_import = get_tags()
    num_pages, start_page = get_pages()

    if st.button("Import", type="primary"):
        with st.spinner("Loading... This might take a minute or two."):
            info_placeholder = st.empty()
            error_placeholder = st.container() # A container to log errors
            
            tasks_to_complete = len(tags_to_import) * num_pages
            completed_tasks = 0
            total_imported_count = 0
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(load_so_data, tag, start_page + i, site)
                    for tag in tags_to_import
                    for i in range(num_pages)
                ]

                for future in as_completed(futures):
                    time.sleep(0.5)
                    completed_tasks += 1
                    result = future.result()
                    
                    progress = (completed_tasks / tasks_to_complete) * 100
                    
                    with info_placeholder:
                        if result["status"] == "success":
                            total_imported_count += result["count"]
                            st.info(f"({progress:.2f}%) ✅ Success: Imported page {result['page']} for tag '{result['tag']}' ({result['count']} items per page).")
                        elif result["status"] == "empty":
                            st.info(f"({progress:.2f}%) 🟡 Skipped: No items on page {result['page']} for tag '{result['tag']}'.")
                    with error_placeholder:
                        if result["status"] == "error":
                            # Log the error to the UI without stopping
                            st.error(f"({progress:.2f}%) ❌ Failed: Page {result['page']} for tag '{result['tag']}'. Reason: {result['error']}")

            st.success(f"Import complete! Successfully imported {total_imported_count} questions.", icon="✅")
            
            # Record the import session in Neo4j
            try:
                record_import_session(
                    neo4j_graph, 
                    total_imported_count, 
                    tags_to_import, 
                    num_pages
                )
                st.info("📊 Import session recorded in dashboard")
            except Exception as e:
                st.warning(f"Could not record import session: {e}")
            
            st.caption("Go to http://localhost:7473/ to interact with the database")

render_page()