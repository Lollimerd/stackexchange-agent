import asyncio
import json
import logging

from datetime import datetime
from typing import AsyncGenerator, Dict, List
from urllib.parse import urlparse
import uuid
import uvicorn

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from setup.init_config import (
    answer_LLM,
    embedding_model,
    get_graph_instance,
    NEO4J_URL,
    NEO4J_USERNAME,
)
from tools.graph_rag_tool import graph_rag_tool
from agent.agent import agent_executor
from utils.util import find_container_by_port
from utils.memory import (
    add_ai_message_to_session,
    add_user_message_to_session,
    delete_session,
    delete_user,
    get_all_users,
    get_chat_history,
    get_user_sessions,
    link_session_to_user,
)
from utils.topic_manager import TopicManager

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
    user_id: str = ""


class IngestRequest(BaseModel):
    data: List[Dict]


class ImportRecordRequest(BaseModel):
    total_questions: int
    tags_list: List[str]
    total_pages: int


@app.get("/")
def index():
    return {"status": "online", "message": "Welcome to the GraphRAG API"}


@app.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# --- Add config endpoint ---
@app.get("/api/v1/config")
def get_configuration():
    """Provides frontend with configuration details for display."""
    try:
        parsed_url = urlparse(NEO4J_URL)
        neo4j_port = parsed_url.port or 7687
        neo4j_host = parsed_url.hostname
        discovered_name = find_container_by_port(neo4j_port)

        if (
            "not mounted" in discovered_name
            or "Error" in discovered_name
            or "Invalid" in discovered_name
        ):
            container_name = f"{neo4j_host} (Configured Host)"
        else:
            container_name = discovered_name

        return {
            "ollama_model": answer_LLM().model,
            "neo4j_url": NEO4J_URL,
            "container_name": container_name,
            "neo4j_user": NEO4J_USERNAME,
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error in get_configuration: {e}")
        return {
            "status": "error",
            "message": str(e),
            "ollama_model": "unknown",
            "neo4j_url": NEO4J_URL,
            "container_name": "unknown",
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
        query = """
        MATCH (s:Session {id: $session_id})-[:HAS_MESSAGE]->(m:Message)
        RETURN m.type AS role, m.content AS content, m.thought AS thought
        ORDER BY coalesce(m.created_at, m.timestamp, elementId(m)) ASC
        LIMIT 1000
        """
        results = get_graph_instance().query(query, params={"session_id": session_id})

        if not results:
            results = []

        # Map 'ai' role to 'assistant' for frontend compatibility
        for res in results:
            if res.get("role") == "ai":
                res["role"] = "assistant"
            # Ensure thought is present (might be None)
            if "thought" not in res:
                res["thought"] = None

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


# ===========================================================================================================================================================
# Ingestion Endpoints
# ===========================================================================================================================================================


@app.post("/api/v1/ingest")
async def ingest_stackoverflow_data(request: IngestRequest):
    """Ingest StackOverflow data: compute embeddings and insert into Neo4j."""
    try:
        from tools.graph_rag_tool import import_query

        data_items = request.data
        if not data_items:
            return {"status": "skipped", "message": "No data items to ingest."}

        # Use a separate thread for the heavy lifting (embeddings + DB)
        def process_ingestion(items):
            # 1. Prepare texts for batch embedding
            texts_to_embed = []
            map_to_object = []  # List of tuples/dicts to map back: (type, parent_idx, answer_idx)

            for q_idx, q in enumerate(items):
                # Question text
                q_text = q.get("title", "") + "\n" + q.get("body_markdown", "")
                texts_to_embed.append(q_text)
                map_to_object.append(("question", q_idx, -1))

                # Answer texts
                # Note: Original code used question_text + answer body for answer embedding
                for a_idx, a in enumerate(q.get("answers", [])):
                    a_text = q_text + "\n" + a.get("body_markdown", "")
                    texts_to_embed.append(a_text)
                    map_to_object.append(("answer", q_idx, a_idx))

            # 2. Compute embeddings in batch
            if texts_to_embed:
                embeddings = embedding_model().embed_documents(texts_to_embed)

                # 3. Assign embeddings back
                for i, embedding in enumerate(embeddings):
                    obj_type, q_idx, a_idx = map_to_object[i]
                    if obj_type == "question":
                        items[q_idx]["embedding"] = embedding
                    elif obj_type == "answer":
                        items[q_idx]["answers"][a_idx]["embedding"] = embedding

            # 4. Insert into Neo4j
            get_graph_instance().query(import_query, {"data": items})
            return len(items)

        count = await asyncio.to_thread(process_ingestion, data_items)

        return {"status": "success", "count": count}

    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/api/v1/ingest/record")
async def record_import_session(request: ImportRecordRequest):
    """Record an import session in Neo4j."""
    try:
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

        params = {
            "import_id": import_id,
            "timestamp": timestamp,
            "total_questions": request.total_questions,
            "total_tags": len(request.tags_list),
            "total_pages": request.total_pages,
            "tags_list": request.tags_list,
        }

        # Run query in thread
        await asyncio.to_thread(get_graph_instance().query, query, params)

        return {"status": "success", "import_id": import_id}
    except Exception as e:
        logger.error(f"Error recording import session: {e}")
        return {"status": "error", "message": str(e)}


@app.put("/api/v1/ingest/record/{import_id}")
async def update_import_session(import_id: str, request: ImportRecordRequest):
    """Update an existing import session in Neo4j."""
    try:
        query = """
        MATCH (log:ImportLog {id: $import_id})
        SET log.total_questions = $total_questions,
            log.total_tags = $total_tags,
            log.total_pages = $total_pages,
            log.tags_list = $tags_list
        RETURN log
        """

        params = {
            "import_id": import_id,
            "total_questions": request.total_questions,
            "total_tags": len(request.tags_list),
            "total_pages": request.total_pages,
            "tags_list": request.tags_list,
        }

        # Run query in thread
        await asyncio.to_thread(get_graph_instance().query, query, params)

        return {"status": "success", "message": f"Import session {import_id} updated"}
    except Exception as e:
        logger.error(f"Error updating import session: {e}")
        return {"status": "error", "message": str(e)}


@app.delete("/api/v1/ingest/record/{import_id}")
async def delete_import_session(import_id: str):
    """Delete an import session from Neo4j."""
    try:
        query = """
        MATCH (log:ImportLog {id: $import_id})
        DETACH DELETE log
        """

        # Run query in thread
        await asyncio.to_thread(
            get_graph_instance().query, query, {"import_id": import_id}
        )

        return {"status": "success", "message": f"Import session {import_id} deleted"}
    except Exception as e:
        logger.error(f"Error deleting import session: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/stream-ask")
async def stream_ask_question(request: QueryRequest) -> StreamingResponse:
    """This endpoint includes topic continuity analysis for context-aware responses."""

    async def stream_generator() -> AsyncGenerator[str]:
        logger.info(
            f"Incoming question: '{request.question[:50]}...' from user {request.user_id}"
        )

        # FIX: Run blocking Neo4j operations in thread pool to avoid blocking event loop
        try:
            # Check if first message and get current topic
            existing_history = await asyncio.to_thread(
                get_chat_history, request.session_id
            )
            is_first_message = not existing_history or not existing_history.messages

            # On first message, use the question as the topic
            topic = request.question if is_first_message else None

            await asyncio.to_thread(
                link_session_to_user, request.session_id, request.user_id, topic or ""
            )

            # Get chat history and current topic
            chat_history_obj = await asyncio.to_thread(
                get_chat_history, request.session_id
            )
            stored_messages = chat_history_obj.messages if chat_history_obj else []

            session_topic = await asyncio.to_thread(
                TopicManager.get_session_topic, request.session_id
            )

            # Analyze topic continuity
            similarity_data = await asyncio.to_thread(
                TopicManager.calculate_topic_similarity, request.question, session_topic
            )

            logger.info(f"Topic continuity: {similarity_data['recommendation']}")

            # Update topic if significant shift detected
            if not is_first_message:
                await asyncio.to_thread(
                    TopicManager.update_session_topic_if_changed,
                    request.session_id,
                    request.question,
                    similarity_data,
                )

        except Exception as e:
            logger.warning(f"Error during DB setup: {e}, continuing with empty history")
            stored_messages = []
            session_topic = ""
            similarity_data = {
                "similarity_score": 1.0,
                "is_continuation": True,
                "confidence_level": "high",
                "recommendation": "Error in analysis, proceeding normally",
            }

        # Convert to the format your chain expects
        formatted_history = [
            {"role": msg.type, "content": msg.content} for msg in stored_messages
        ]

        logger.info(
            f"Chat history: {len(formatted_history)} messages, topic: {session_topic}"
        )

        # Add user message to DB
        try:
            await asyncio.to_thread(
                add_user_message_to_session, request.session_id, request.question
            )
        except Exception as e:
            logger.warning(f"Error saving user message: {e}")

        # Accumulated output for saving later
        response_chunks = []
        thought_chunks = []

        try:
            # Pass both session_id and topic to the chain
            async for event in graph_rag_tool.astream_events(
                {
                    "question": request.question,
                    "chat_history": formatted_history,
                    "session_topic": session_topic,
                    "session_id": request.session_id,
                },
                version="v2",
            ):
                event_type = event["event"]
                event_name = event["name"]

                # --- Status Updates ---
                if event_type == "on_chain_start":
                    if event_name == "GraphTraversal":
                        yield f"data: {
                            json.dumps(
                                {
                                    'type': 'status',
                                    'stage': 'graph_traversal',
                                    'status': 'running',
                                    'message': '🕷️ Traversing Knowledge Graph...',
                                }
                            )
                        }\n\n"
                    elif event_name == "Reranking":
                        yield f"data: {
                            json.dumps(
                                {
                                    'type': 'status',
                                    'stage': 'reranking',
                                    'status': 'running',
                                    'message': '⚖️ Reranking Documents...',
                                }
                            )
                        }\n\n"

                elif event_type == "on_chain_end":
                    if event_name == "GraphTraversal":
                        output = event["data"].get("output", [])
                        count = len(output) if isinstance(output, list) else 0
                        yield f"data: {
                            json.dumps(
                                {
                                    'type': 'status',
                                    'stage': 'graph_traversal',
                                    'status': 'complete',
                                    'message': f'✅ Found {count} documents',
                                    'count': count,
                                }
                            )
                        }\n\n"
                    elif event_name == "Reranking":
                        output = event["data"].get("output", [])
                        count = len(output) if isinstance(output, list) else 0
                        yield f"data: {
                            json.dumps(
                                {
                                    'type': 'status',
                                    'stage': 'reranking',
                                    'status': 'complete',
                                    'message': f'✅ Top {count} documents selected',
                                    'count': count,
                                }
                            )
                        }\n\n"

                elif event_type == "on_chat_model_start":
                    yield f"data: {
                        json.dumps(
                            {
                                'type': 'status',
                                'stage': 'thinking',
                                'status': 'running',
                                'message': '🤔 Thinking...',
                            }
                        )
                    }\n\n"

                # --- Token Streaming ---
                elif event_type == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    if not chunk:
                        continue

                    # Extract content and reasoning
                    content_chunk = (
                        chunk.content if hasattr(chunk, "content") else str(chunk)
                    )
                    reasoning_chunk = (
                        chunk.additional_kwargs.get("reasoning_content", "")
                        if hasattr(chunk, "additional_kwargs")
                        else ""
                    )

                    # Accumulate for DB save
                    response_chunks.append(content_chunk)
                    thought_chunks.append(reasoning_chunk)

                    # Create a dictionary for this stream chunk
                    event_data = {
                        "type": "token",
                        "content": content_chunk,
                        "reasoning_content": reasoning_chunk,
                    }

                    # Format as an SSE data payload
                    yield f"data: {json.dumps(event_data)}\n\n"

        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            error_event = {
                "type": "error",
                "content": f"[Error processing response: {str(e)}]",
                "reasoning_content": "",
            }
            yield f"data: {json.dumps(error_event)}\n\n"
            # Don't raise here to allow DB save of partial response if any

        # Save AI response
        try:
            full_response = "".join(response_chunks)
            full_thought = "".join(thought_chunks)

            # Only save if we got something
            if full_response or full_thought:
                await asyncio.to_thread(
                    add_ai_message_to_session,
                    request.session_id,
                    full_response,
                    full_thought,
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
        },
    )


@app.post("/agent/ask")
async def agent_ask(request: QueryRequest) -> StreamingResponse:
    """
    Endpoint to query the new LangChain Agent with SSE streaming.
    """

    async def agent_stream_generator() -> AsyncGenerator[str]:
        logger.info(
            f"Agent request: '{request.question[:50]}...' from user {request.user_id}"
        )

        try:
            # 1. Prepare Input
            # Retrieve history
            chat_history_obj = await asyncio.to_thread(
                get_chat_history, request.session_id
            )
            messages = chat_history_obj.messages if chat_history_obj else []

            input_data = {"input": request.question, "chat_history": messages}

            # 2. Stream Events from Agent Executor
            # version="v1" for langchain < 0.2, "v2" for >= 0.2
            # checking installed version or trying v2 is safer for new setups
            async for event in agent_executor.astream_events(
                input_data,
                version="v2"
            ):
                event_type = event["event"]
                event_name = event["name"]
                
                # --- A. Status Updates (Tools) ---
                if event_type == "on_tool_start":
                    # Notify frontend that a tool is running
                    yield f"data: {json.dumps({
                        'type': 'status',
                        'stage': 'tool_start',
                        'status': 'running',
                        'message': f'🛠️ Using tool: {event_name}...'
                    })}\n\n"
                    
                elif event_type == "on_tool_end":
                    yield f"data: {json.dumps({
                        'type': 'status',
                        'stage': 'tool_end',
                        'status': 'complete',
                        'message': f'✅ Tool {event_name} completed'
                    })}\n\n"

                # --- B. Internal Tool Steps (GraphTraversal & Reranking) ---
                # These events happen *inside* the tool execution
                elif event_type == "on_chain_start":
                    if event_name == "GraphTraversal":
                        yield f"data: {json.dumps({
                            'type': 'status',
                            'stage': 'graph_traversal',
                            'status': 'running',
                            'message': '🕷️ Traversing Knowledge Graph...'
                        })}\n\n"
                    elif event_name == "Reranking":
                        yield f"data: {json.dumps({
                            'type': 'status',
                            'stage': 'reranking',
                            'status': 'running',
                            'message': '⚖️ Reranking Documents...'
                        })}\n\n"

                elif event_type == "on_chain_end":
                    if event_name == "GraphTraversal":
                        # Attempt to extract count if possible, though 'output' structure varies
                        output = event["data"].get("output", [])
                        count = len(output) if isinstance(output, list) else 0
                        yield f"data: {json.dumps({
                            'type': 'status',
                            'stage': 'graph_traversal',
                            'status': 'complete',
                            'message': f'✅ Found {count} documents',
                            'count': count
                        })}\n\n"
                    elif event_name == "Reranking":
                        output = event["data"].get("output", [])
                        count = len(output) if isinstance(output, list) else 0
                        yield f"data: {json.dumps({
                            'type': 'status',
                            'stage': 'reranking',
                            'status': 'complete',
                            'message': f'✅ Top {count} documents selected',
                            'count': count
                        })}\n\n"

                # --- C. Token Streaming (LLM) ---
                # We only want tokens from the final chat model in the agent, not internal steps if possible.
                # Usually, 'on_chat_model_stream' works for the final response generation.
                elif event_type == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    if chunk:
                        content = chunk.content if hasattr(chunk, "content") else str(chunk)
                        if content:
                            yield f"data: {json.dumps({
                                'type': 'token',
                                'content': content
                            })}\n\n"

                # --- D. Final Output ---
                # 'on_chain_end' for the main executor might contain the final output,
                # but valid streaming builds the answer token-by-token.

        except Exception as e:
            logger.error(f"Error in agent stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        agent_stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


# uvicorn main:app --reload
if __name__ == "__main__":
    # Run the app with Uvicorn, specifying host and port here
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
