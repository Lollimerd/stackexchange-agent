import streamlit as st
import requests
import uuid
import logging
from utils.util import AGENT_URL, display_container_name

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Agent Test", page_icon="🤖", layout="wide")

st.title("🤖 LangChain Agent Test")
st.markdown(
    "This page interacts purely with the `/agent/ask` endpoint (AgentExecutor)."
)

# Sidebar for context (optional but good for consistency)
with st.sidebar:
    try:
        display_container_name()
    except:
        pass
    st.info(
        "This interface bypasses the streaming endpoint and uses the AgentExecutor directly."
    )

# Session State for simple chat history in this test page
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

# Display history
for msg in st.session_state.agent_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
if prompt := st.chat_input("Ask the agent something..."):
    # User message
    st.session_state.agent_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Agent Response
    with st.chat_message("assistant"):
        status_box = st.status("🚀 Agent starting...", expanded=True)
        response_placeholder = st.empty()
        full_response = ""

        try:
            # Prepare payload
            session_id = st.session_state.get("active_chat_id", str(uuid.uuid4()))
            user_id = st.session_state.get("user_name", "test_user")

            payload = {"question": prompt, "session_id": session_id, "user_id": user_id}

            import httpx
            from httpx_sse import connect_sse
            import json

            # SSE Connection
            timeout = httpx.Timeout(300.0, read=300.0)
            with httpx.Client(timeout=timeout) as client:
                with connect_sse(
                    client, "POST", AGENT_URL, json=payload
                ) as event_source:
                    for sse in event_source.iter_sse():
                        if sse.data:
                            try:
                                data = json.loads(sse.data)
                                msg_type = data.get("type")

                                if msg_type == "status":
                                    stage = data.get("stage")
                                    message = data.get("message")
                                    status_state = data.get("status")

                                    status_box.update(label=message, state="running")
                                    if status_state == "complete":
                                        status_box.write(f"✔️ {message}")

                                elif msg_type == "token":
                                    content = data.get("content", "")
                                    full_response += content
                                    response_placeholder.markdown(full_response + "▌")

                                    # Update status to complete if we start getting tokens
                                    status_box.update(
                                        label="🤖 Generating Answer...",
                                        state="running",
                                        expanded=False,
                                    )

                                elif msg_type == "error":
                                    st.error(data.get("content"))
                                    status_box.update(label="❌ Error", state="error")

                            except Exception as e:
                                logger.error(f"Error parse: {e}")

            # Final render
            response_placeholder.markdown(full_response)
            status_box.update(label="✅ Complete", state="complete", expanded=False)
            st.session_state.agent_messages.append(
                {"role": "assistant", "content": full_response}
            )

        except Exception as e:
            st.error(f"Connection Error: {e}")
