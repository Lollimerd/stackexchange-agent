# main.py
import asyncio, os, json, uvicorn, logging
from datetime import datetime
from typing import AsyncGenerator, List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from urllib.parse import urlparse
from dotenv import load_dotenv
from utils.util import find_container_by_port
from setup.init import ANSWER_LLM, NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD
from tools.custom_tool import graph_rag_chain
from utils.memory import (
    get_chat_history, get_user_sessions, link_session_to_user, 
    get_all_users, delete_session, delete_user, 
    add_user_message_to_session, add_ai_message_to_session
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================================================================================================================================================
# FastAPI Backend Server
# ===========================================================================================================================================================

# initialise fastapi
app = FastAPI(title="GraphRAG API", version="1.2.0")

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    """configure question template for answer LLM"""
    question: str
    session_id: str
    user_id: str = ""  # Default for backward compatibility or simple testing

@app.get('/')
def index():
    return {"status": "online", "message": "Welcome to the GraphRAG API"}

@app.get('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# --- Add config endpoint ---
@app.get("/api/v1/config")
def get_configuration():
    """Provides frontend with configuration details for display."""
    try:
        parsed_url = urlparse(NEO4J_URL)
        neo4j_port = parsed_url.port
        neo4j_host = parsed_url.hostname
        discovered_name = find_container_by_port(neo4j_port)

        if "not mounted" in discovered_name or "Error" in discovered_name or "Invalid" in discovered_name:
            container_name = f"{neo4j_host} (Configured Host)"
        else:
            container_name = discovered_name

        return {
            "ollama_model": ANSWER_LLM.model,
            "neo4j_url": NEO4J_URL,
            "container_name": container_name,
            "neo4j_user": NEO4J_USERNAME,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error in get_configuration: {e}")
        return {
            "status": "error",
            "message": str(e),
            "ollama_model": "unknown",
            "neo4j_url": NEO4J_URL,
            "container_name": "unknown"
        }

@app.get("/api/v1/users")
def get_users():
    """Returns a list of all application users."""
    try:
        users = get_all_users()
        return {"users": users, "status": "success"}
    except Exception as e:
        logger.error(f"Error fetching users: {e}")
        return {"users": [], "status": "error", "message": str(e)}

@app.get("/api/v1/user/{user_id}/chats")
def get_user_chats(user_id: str):
    """Returns a list of chat sessions for a specific user."""
    try:
        sessions = get_user_sessions(user_id)
        return {"chats": sessions, "status": "success"}
    except Exception as e:
        logger.error(f"Error fetching chats for user {user_id}: {e}")
        return {"chats": [], "status": "error", "message": str(e)}

@app.get("/api/v1/chat/{session_id}")
def get_chat_messages(session_id: str):
    """Returns the message history for a specific session, including thoughts for AI messages."""
    try:
        from langchain_neo4j import Neo4jGraph

        graph = Neo4jGraph(
            url=NEO4J_URL,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )

        query = """
        MATCH (s:Session {id: $session_id})-[:HAS_MESSAGE]->(m:Message)
        RETURN m.type AS role, m.content AS content, m.thought AS thought
        ORDER BY m.created_at ASC, elementId(m) ASC
        LIMIT 1000
        """
        results = graph.query(query, params={"session_id": session_id})

        if not results:
            results = []

        # Map 'ai' role to 'assistant' for frontend compatibility
        for res in results:
            if res.get('role') == 'ai':
                res['role'] = 'assistant'
            # Ensure thought is present (might be None)
            if 'thought' not in res:
                res['thought'] = None

        return {"messages": results, "status": "success"}
    except Exception as e:
        logger.error(f"Error fetching chat history for {session_id}: {e}")
        return {"messages": [], "status": "error", "message": str(e)}

@app.delete("/api/v1/chat/{session_id}")
def delete_user_session(session_id: str):
    """Deletes a specific chat session."""
    try:
        delete_session(session_id)
        return {"status": "success", "message": f"Session {session_id} deleted"}
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        return {"status": "error", "message": str(e)}

@app.delete("/api/v1/user/{user_id}")
def delete_app_user(user_id: str):
    """Deletes a user and all their data."""
    try:
        delete_user(user_id)
        return {"status": "success", "message": f"User {user_id} deleted"}
    except Exception as e:
        logger.error(f"Error deleting user {user_id}: {e}")
        return {"status": "error", "message": str(e)}

# --- ✨ REFACTORED: Streaming Endpoint with Thinking Handler ---
# FIX: Use asyncio.to_thread to prevent Neo4j blocking from blocking event loop

@app.post("/stream-ask")
async def stream_ask_question(request: QueryRequest) -> StreamingResponse:
    """This endpoint now includes chat history for context-aware responses."""
    async def stream_generator() -> AsyncGenerator[str]:
        logger.info(f"Incoming question: '{request.question[:50]}...' from user {request.user_id}")

        # FIX: Run blocking Neo4j operations in thread pool to avoid blocking event loop
        try:
            # 0. Link Session to User (non-blocking)
            await asyncio.to_thread(
                link_session_to_user,
                request.session_id,
                request.user_id
            )

            # 1. Retrieve History from Neo4j (non-blocking)
            chat_history_obj = await asyncio.to_thread(
                get_chat_history,
                request.session_id
            )
            stored_messages = chat_history_obj.messages if chat_history_obj else []
        except Exception as e:
            logger.warning(f"Error during DB setup: {e}, continuing with empty history")
            # Continue with empty history if DB fails
            stored_messages = []

        # Convert to the format your chain expects
        formatted_history = [
            {"role": msg.type, "content": msg.content}
            for msg in stored_messages
        ]

        logger.info(f"Chat history loaded: {len(formatted_history)} messages")

        # 2. Add User Message to DB (non-blocking)
        try:
            await asyncio.to_thread(
                add_user_message_to_session,
                request.session_id,
                request.question
            )
        except Exception as e:
            logger.warning(f"Error saving user message: {e}")

        # 3. Accumulated output for saving later
        response_chunks = []
        thought_chunks = []

        try:
            # Pass chat history to the chain
            async for chunk in graph_rag_chain.astream(
                {
                    "question": request.question,
                    "chat_history": formatted_history
                },
            ):
                # Extract content and reasoning
                content_chunk = chunk.content if hasattr(chunk, 'content') else str(chunk)
                reasoning_chunk = chunk.additional_kwargs.get("reasoning_content", "") if hasattr(chunk, 'additional_kwargs') else ""

                # FIX: Avoid O(n²) string concatenation - collect in list
                response_chunks.append(content_chunk)
                thought_chunks.append(reasoning_chunk)

                # Create a dictionary for this stream chunk
                event_data = {
                    "content": content_chunk,
                    "reasoning_content": reasoning_chunk,
                }

                # Format as an SSE data payload
                yield f"data: {json.dumps(event_data)}\n\n"

        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            error_event = {
                "content": f"[Error processing response: {str(e)}]",
                "reasoning_content": ""
            }
            yield f"data: {json.dumps(error_event)}\n\n"
            raise

        # 4. Join accumulated chunks and save AI Response to DB (non-blocking, after stream)
        try:
            full_response = "".join(response_chunks)
            full_thought = "".join(thought_chunks)
            await asyncio.to_thread(
                add_ai_message_to_session,
                request.session_id,
                full_response,
                full_thought
            )
            logger.info(f"Response saved to DB: {len(full_response)} chars")
        except Exception as e:
            logger.warning(f"Error saving AI response: {e}")

    # FIX: Add proper headers for SSE and disable buffering
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

# uvicorn main:app --reload
if __name__ == "__main__":
    # Run the app with Uvicorn, specifying host and port here
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
