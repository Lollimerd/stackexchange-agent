# app.py
import json, requests, datetime, uuid, httpx, os, logging
from httpx_sse import connect_sse
import streamlit as st
# from st_pages import add_page_title, get_nav_from_toml
import streamlit.components.v1 as components
# from streamlit_timeline import timeline
from utils.util import render_message_with_mermaid, display_container_name, get_system_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(page_title="Custom GPT", page_icon="🧠", layout="wide", initial_sidebar_state="expanded",
    menu_items={
    'Get Help': 'https://www.extremelycoolapp.com/help',
    'Report a bug': "https://www.extremelycoolapp.com/bug",
    'About': "# This is a header. This is an *extremely* cool app!"})

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

# --- Initialize Session State AND Sync with Backend ---
if "chats" not in st.session_state:
    st.session_state.chats = {}  # Initialize empty first

# --- sidebar init ---
with st.sidebar:
    try:
        display_container_name()
    except Exception as e:
        logger.error(f"Error displaying container info: {e}")
        st.warning("Could not connect to backend for system info")

    st.sidebar.title("⚙️ Settings", help="config settings here")
    
    # --- General Settings ---
    # Fetch existing users
    existing_users = fetch_all_users()

    # Add option for new user
    user_options = existing_users + ["+ Create New User"]

    # Determine default index
    default_index = 0
    if "user_name" in st.session_state and st.session_state.user_name in existing_users:
        default_index = existing_users.index(st.session_state.user_name)

    selected_option = st.sidebar.selectbox("Select User", user_options, index=default_index)

    if selected_option == "+ Create New User":
        name = st.sidebar.text_input("Enter New Username", key="new_user_input")
        if not name:
            name = "test"  # default fallback
    else:
        name = selected_option

    # Store name in session state to detect changes
    if "user_name" not in st.session_state:
        st.session_state.user_name = name

    # Detect name change to reload chats
    if st.session_state.user_name != name:
        st.session_state.user_name = name
        # Clear current chats to force reload logic below
        st.session_state.chats = {}
        st.session_state.active_chat_id = None
        # We need to rerun to let the logic below fetch new chats
        st.rerun()

    # --- User Deletion UI ---
    if selected_option != "+ Create New User" and name:
        if st.sidebar.button("🗑️ Delete User", help="Permanently delete this user and all data"):
            delete_user_api(name)
            # Clear state and refresh
            st.session_state.chats = {}
            st.session_state.active_chat_id = None
            st.rerun()

# Logic to load chats if empty (initial load or name change)
if not st.session_state.chats and st.session_state.get("user_name"):
    backend_chats = fetch_user_chats(st.session_state.user_name)
    for chat in backend_chats:
        s_id = chat.get('session_id')
        last_msg = chat.get('last_message', 'New Chat')
        title = (last_msg[:30] + "...") if last_msg else "New Chat"

        if s_id:  # Only add if we have a valid session_id
            st.session_state.chats[s_id] = {
                "title": title,
                "messages": [],
                "loaded": False
            }

if "active_chat_id" not in st.session_state or st.session_state.active_chat_id is None:
    # If we have chats from DB, pick the first one (most recent)
    if st.session_state.chats:
        st.session_state.active_chat_id = list(st.session_state.chats.keys())[0]
    else:
        # Create a default first chat if DB is empty
        first_chat_id = str(uuid.uuid4())
        st.session_state.active_chat_id = first_chat_id
        st.session_state.chats[first_chat_id] = {
            "title": "New Chat",
            "messages": [],
            "thoughts": [],
            "loaded": True
        }

# Helper function to get the active chat object
def get_active_chat():
    """Get the active chat data, creating it if necessary."""
    chat_id = st.session_state.active_chat_id
    if chat_id not in st.session_state.chats:
        # Fallback if somehow ID is missing
        return None

    chat = st.session_state.chats[chat_id]

    # Lazy load history if not loaded yet
    if not chat.get("loaded", False):
        msgs = fetch_chat_history(chat_id)
        chat["messages"] = msgs
        chat["loaded"] = True

    return chat

# --- Sidebar UI elements (System Info & Chat List) ---
with st.sidebar:
    # --- System Info ---
    with st.expander("System Info & DB Details", expanded=False):
        try:
            config_data = get_system_config()
            if config_data and isinstance(config_data, dict):
                st.markdown(f"**Ollama Model:** `{config_data.get('ollama_model', 'N/A')}`")
                st.markdown(f"**Neo4j URL:** `{config_data.get('neo4j_url', 'N/A')}`")
                st.markdown(f"**DB connected:** `{config_data.get('container_name', 'N/A')}`")
                st.markdown(f"**Neo4j User:** `{config_data.get('neo4j_user', 'N/A')}`")
                if config_data.get("status") != "success":
                    st.warning("⚠️ Some services may not be connected")
            else:
                st.error("Could not retrieve system info - backend may be offline")
        except Exception as e:
            logger.error(f"Error in system info: {e}")
            st.error(f"Error: {str(e)[:100]}")

    # --- Chat Management UI ---
    st.subheader("Chats", help="navigate your chats here")

    col_new, col_refresh = st.columns([0.8, 0.2])
    with col_new:
        if st.button("➕ New Chat", use_container_width=True):
            new_chat_id = str(uuid.uuid4())
            st.session_state.chats[new_chat_id] = {
                "title": "New Chat",
                "messages": [],
                "thoughts": [],
                "loaded": True
            }
            st.session_state.active_chat_id = new_chat_id
            st.rerun()

    with col_refresh:
        if st.button("🔄", help="Refresh Chat List"):
            # Clear local state to force reload from backend
            st.session_state.chats = {}
            st.rerun()

    # --- Display chats as individual buttons ---
    # Create a reversed list of chat IDs to display the newest chats first
    chat_ids = list(st.session_state.chats.keys())

    # Create a container with fixed height inside the sidebar (scrollable pane)
    chat_container = st.container(height=300, border=True)

    for chat_id in reversed(chat_ids):
        chat_data = st.session_state.chats[chat_id]

        # Use columns to place chat button and delete button on the same line
        col1, col2 = chat_container.columns([0.2, 0.8])

        with col1:
            # Button to delete the chat
            if st.button("🗑️", key=f"delete_chat_{chat_id}", use_container_width=True):
                # Call API to delete from DB
                delete_chat_api(chat_id)

                # Remove the chat from the dictionary
                del st.session_state.chats[chat_id]

                # If the deleted chat was the active one, select a new active chat
                if st.session_state.active_chat_id == chat_id:
                    remaining_chats = list(st.session_state.chats.keys())
                    st.session_state.active_chat_id = remaining_chats[0] if remaining_chats else None
                st.rerun()

        with col2:
            # Button to select the chat
            if st.button(
                chat_data['title'],
                key=f"chat_button_{chat_id}",
                use_container_width=True,
                disabled=(chat_id == st.session_state.active_chat_id)
            ):
                st.session_state.active_chat_id = chat_id
                st.rerun()

    # --- Clear history for the ACTIVE chat ---
    if st.button("Clear Active Chat History", use_container_width=True):
        active_chat = get_active_chat()
        if active_chat:
            active_chat["thoughts"] = []
            active_chat["messages"] = []
            st.rerun()
            
    st.write("OPSEC ©LOLLIMERD 2025")

active_chat = get_active_chat()

# --- Main Content Area ---
if active_chat is None:
    st.error("No active chat available")
else:
    # --- UI Elements ---
    st.title(f"🧠 Lollimerd's AI")
    st.header("""**:violet-badge[:material/star: OSU GPT]** **:blue-badge[:material/star: Ollama]** **:green-badge[:material/Verified: Mixture of Experts (MOE) model -> Qwen3]** **:blue-badge[:material/component_exchange: GraphRAG]**""")
    st.markdown("""Ask a question to get a real-time analysis from the knowledge graph. Feel free to ask the bot whatever your queries may be.
                Be specific in what you are asking, create table, generate graph of asking for data within a specified duration of time.
                Inferences, analysis and predictions are supported too :)
    """)
    st.subheader(body=f"Welcome back, {st.session_state.get('user_name', 'Guest')}")

    # When displaying past messages from the ACTIVE chat:
    for message in active_chat["messages"]:
        role = message.get("role", "user")
        if role == "user":
            author_name = st.session_state.get("user_name", "User")
        elif role == "assistant":
            author_name = "Assistant"
        else:
            author_name = role.capitalize() if isinstance(role, str) else "Author"

        with st.chat_message(name=author_name):
            if role == "assistant":
                if message.get("thought"):
                    with st.expander("Show Agent Thoughts"):
                        render_message_with_mermaid(message["thought"])
                render_message_with_mermaid(message.get("content", ""))
            else:
                st.markdown(message.get("content", ""))

    # --- Main Interaction Logic (operates on the active chat) ---
    if prompt := st.chat_input("Ask your question..."):
        # Get the active chat
        active_chat = get_active_chat()

        if active_chat:
            # Append to the active chat's message list
            active_chat["messages"].append({"role": "user", "content": prompt})

            # Set a title for new chats based on the first message
            if active_chat["title"] == "New Chat" or active_chat["title"].startswith("Chat "):
                active_chat["title"] = prompt[:15] + "..."  # Truncate for display

            with st.chat_message(name=st.session_state.get("user_name", "User")):
                st.markdown(prompt)

            with st.chat_message(name="Assistant"):
                with st.spinner("Analyzing graph data..."):
                    thought_container = st.expander("Show Agent Thoughts", expanded=True)
                    thought_placeholder = thought_container.empty()
                    answer_placeholder = st.empty()
                    thought_content = ""
                    answer_content = ""

                    try:
                        timeout = httpx.Timeout(300, read=300)
                        with httpx.Client(timeout=timeout) as client:
                            # Send request with session_id (history is now handled by backend)
                            with connect_sse(
                                client, 
                                "POST", 
                                FASTAPI_URL, 
                                json={
                                    "question": prompt,
                                    "session_id": st.session_state.active_chat_id,
                                    "user_id": st.session_state.get("user_name", "")
                                }
                            ) as event_source:
                                for sse in event_source.iter_sse():
                                    if sse.data:
                                        try:
                                            # Parse the JSON payload from the SSE event
                                            data = json.loads(sse.data)

                                            # Append chunks to full strings
                                            chunk_content = data.get("content", "")
                                            chunk_thought = data.get("reasoning_content", "")

                                            answer_content += chunk_content
                                            thought_content += chunk_thought

                                            # Display the current accumulated output in the UI
                                            if answer_content:
                                                answer_placeholder.markdown(answer_content + "▌")
                                            if thought_content:
                                                thought_placeholder.markdown(thought_content + "▌")

                                        except json.JSONDecodeError as e:
                                            logger.error(f"Error decoding JSON: {sse.data}")
                                            st.error(f"Decoding error: {str(e)[:100]}")
                                            continue

                        # --- Final Processing and Rendering ---
                        # Remove type cursors and render final markdown/mermaid
                        answer_placeholder.empty()
                        thought_placeholder.empty()

                        with thought_container:
                            if thought_content:
                                render_message_with_mermaid(thought_content)
                            else:
                                st.info("No agent thoughts captured")
                        
                        if answer_content:
                            render_message_with_mermaid(answer_content)
                        else:
                            st.warning("No response content received")

                        # Append the final response to the active chat's message list
                        active_chat["messages"].append(
                            {
                                "role": "assistant",
                                "thought": thought_content,
                                "content": answer_content,
                            }
                        )
                        st.rerun()  # Rerun to update the chat list in the sidebar if the title changed

                    except httpx.TimeoutException as e:
                        logger.error(f"Request timeout: {e}")
                        st.error("The request timed out. The server is taking too long to respond. Please try again later.")
                    except requests.exceptions.ConnectionError as e:
                        logger.error(f"Connection error: {e}")
                        st.error(f"Could not connect to the API at {BACKEND_URL}. Is the backend running?")
                    except requests.exceptions.RequestException as e:
                        logger.error(f"Request exception: {e}")
                        st.error(f"API Error: {str(e)[:200]}")
                    except Exception as e:
                        logger.error(f"Unexpected error: {e}")
                        st.error(f"Unexpected error: {str(e)[:200]}")

    # --- Auto-Scroll to Bottom ---
    components.html(
        """
        <script>
            var scrollingElement = (document.scrollingElement || document.body);
            scrollingElement.scrollTop = scrollingElement.scrollHeight;
        </script>
        """,
        height=0,
        width=0,
    )

