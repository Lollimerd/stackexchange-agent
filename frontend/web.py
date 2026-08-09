# app.py
import json, requests, datetime, uuid, httpx, os
from httpx_sse import connect_sse
import streamlit as st
# from st_pages import add_page_title, get_nav_from_toml
# import streamlit.components.v1 as components
# from streamlit_timeline import timeline 
from utils.util import render_message_with_mermaid, display_container_name, get_system_config

# --- Page Configuration ---
st.set_page_config(page_title="Custom GPT", page_icon="🧠", layout="wide", initial_sidebar_state="expanded", 
    menu_items={
    'Get Help': 'https://www.extremelycoolapp.com/help',
    'Report a bug': "https://www.extremelycoolapp.com/bug",
    'About': "# This is a header. This is an *extremely* cool app!"})
 
# --- API Configuration ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
FASTAPI_URL = f"{BACKEND_URL}/stream-ask"

# --- Initialize Session State for Multi-Chat ---
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "active_chat_id" not in st.session_state:
    # Create a default first chat on the first run
    first_chat_id = str(uuid.uuid4())
    st.session_state.active_chat_id = first_chat_id
    st.session_state.chats[first_chat_id] = {
        "title": "New Chat",
        "messages": [],
        "thoughts": [] # Retaining thoughts if you wish to use them per chat
    }

# Helper function to get the active chat object
def get_active_chat():
    return st.session_state.chats[st.session_state.active_chat_id]

# --- sidebar init ---
with st.sidebar:
    display_container_name()
    st.sidebar.title("⚙️ Settings", help="config settings here")
    # --- General Settings ---
    name = st.sidebar.text_input("Your Name", "test")

    # --- ADD THIS EXPANDER FOR SYSTEM DETAILS ---
    with st.expander("System Info & DB Details", expanded=False):
        config_data = get_system_config()
        if config_data:
            st.markdown(f"**Ollama Model:** `{config_data.get('ollama_model', 'N/A')}`")
            st.markdown(f"**Neo4j URL:** `{config_data.get('neo4j_url', 'N/A')}`")
            st.markdown(f"**DB connected:** `{config_data.get('container_name', 'N/A')}`")
            st.markdown(f"**Neo4j User:** `{config_data.get('neo4j_user', 'N/A')}`")
        else:
            st.error("Could not retrieve system info.")

    # --- Chat Management UI ---
    st.sidebar.subheader("Chats", help="navigate your chats here")
    if st.sidebar.button("➕ New Chat", use_container_width=True):
        new_chat_id = str(uuid.uuid4())
        st.session_state.chats[new_chat_id] = {
            "title": f"Chat {len(st.session_state.chats) + 1}",
            "messages": [],
            "thoughts": []
        }
        st.session_state.active_chat_id = new_chat_id
        st.rerun()

    # --- Display chats as individual buttons ---
    # Create a reversed list of chat IDs to display the newest chats first
    chat_ids = list(st.session_state.chats.keys())
    
    # Create a container with a fixed height inside the sidebar
    chat_container = st.container(height=150) # Adjust height as needed
    
    for chat_id in reversed(chat_ids):
        chat_data = st.session_state.chats[chat_id]
        
        # Use columns to place chat button and delete button on the same line
        col1, col2 = chat_container.columns([0.2, 0.8])
        
        with col1:
            # Button to delete the chat
            if st.button("🗑️", key=f"delete_chat_{chat_id}", use_container_width=True):
                # Remove the chat from the dictionary
                del st.session_state.chats[chat_id]
                
                # If the deleted chat was the active one, select a new active chat
                if st.session_state.active_chat_id == chat_id:
                    # Set to the first available chat, or None if no chats are left
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
        active_chat["thoughts"] = []
        active_chat["messages"] = []
        st.rerun()    
    st.write("OPSEC ©LOLLIMERD 2025")
    # st.sidebar.markdown("---") # Visual separator

# Create tabs
# tab1, tab2, tab3, tab4= st.tabs(["💬OPSEC CHATBOT", "stackoverflow loader", "🗃📊 DEV LOGS", "📈💰📊Dashboard"])

active_chat = get_active_chat()
# f"{active_chat['title']}" # chat no
# --- UI Elements ---
st.title(f"🧠 Lollimerd's AI ")
st.header("""**:violet-badge[:material/star: OSU GPT]** **:blue-badge[:material/star: Ollama]** **:green-badge[:material/Verified: Mixture of Experts (MOE) model -> Qwen3]** **:blue-badge[:material/component_exchange: GraphRAG]**
""")
st.markdown("""Ask a question to get a real-time analysis from the knowledge graph. Feel free to ask the bot whatever your queries may be.
            Be specific in what you are asking, create table, generate graph of asking for data within a specified duration of time. 
            Inferences, analysis and predictions are supported too :)
""")
st.subheader(body=f"Welcome back, {name} ⎛⎝ ≽  >  ⩊   < ≼ ⎠⎞")

# When displaying past messages from the ACTIVE chat:
for message in active_chat["messages"]:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            if "thought" in message and message["thought"]:
                with st.expander("Show Agent Thoughts"):
                    render_message_with_mermaid(message["thought"])
            render_message_with_mermaid(message["content"])
        else:
            st.markdown(message["content"])

# --- Main Interaction Logic (operates on the active chat) ---
if prompt := st.chat_input("Ask your question..."):
    # Get the active chat
    active_chat = get_active_chat()
    
    # Append to the active chat's message list
    active_chat["messages"].append({"role": "user", "content": prompt})

    # Set a title for new chats based on the first message
    if active_chat["title"] == "New Chat" or active_chat["title"].startswith("Chat "):
        active_chat["title"] = prompt[:15] + "..." # Truncate for display

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing graph data..."):
            thought_placeholder = st.expander("Show Agent Thoughts").empty()
            answer_placeholder = st.empty()
            thought_content = ""
            answer_content = ""

            try:
                timeout = httpx.Timeout(300, read=300)
                with httpx.Client(timeout=timeout) as client:
                    # Prepare chat history for the request
                    chat_history = []
                    for msg in active_chat["messages"][:-1]:  # Exclude the current message
                        chat_history.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                    
                    # Send request with chat history
                    with connect_sse(client, "POST", FASTAPI_URL, json={
                        "question": prompt,
                        "chat_history": chat_history
                    }) as event_source:
                        for sse in event_source.iter_sse():
                            if sse.data:
                                try:
                                    # Parse the JSON payload from the SSE event
                                    data = json.loads(sse.data)

                                    # Append chunks to full strings
                                    answer_content += data.get("content", "")
                                    thought_content += data.get("reasoning_content", "")

                                    # Display the current accumulated output in the UI with typing cursor effect
                                    answer_placeholder.markdown(answer_content + "▌")
                                    if thought_content:
                                        thought_placeholder.code(thought_content, language="markdown")

                                except json.JSONDecodeError:
                                    st.error(f"Error decoding JSON: {sse.data}")
                                    continue

                # --- Final Processing and Rendering ---
                render_message_with_mermaid(thought_content)
                render_message_with_mermaid(answer_content)

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
                st.error(f"The request timed out. The server is taking too long to respond. Please try again later.")
                # It's also good practice to log the error for debugging
                print(f"ERROR: httpx.TimeoutException - {e}")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the API. Error: {e}")