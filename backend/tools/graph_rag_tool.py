from setup.init_config import (
    get_graph_instance,
    embedding_model,
    create_vector_stores,
    reranker_model,
)
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import (
    CrossEncoderReranker,
)
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from typing import List, Dict, Optional, Type, Any
from langchain_core.documents import Document
from tools.graph_rag_prompt import analyst_prompt, retrieval_query
from utils.util import format_docs_with_metadata, escape_lucene_chars
from langchain_core.tools import BaseTool
from middleware.langchain_middleware import (
    process_with_topic_analysis,
)
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


def format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    """Format chat history for inclusion in the prompt."""
    try:
        if not chat_history:
            return ""

        formatted_history = []
        for msg in chat_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                formatted_history.append(f"User: {content}")
            elif role == "assistant":
                # Only include the main content, not the thought process
                formatted_history.append(f"Assistant: {content}")

        return "\n".join(formatted_history)
    except Exception as e:
        logger.error(f"Error formatting chat history: {e}")
        return ""


# Create vector stores with error handling
try:
    stores = create_vector_stores(
        get_graph_instance(), embedding_model(), retrieval_query
    )
    tagstore = stores.get("tagstore")
    userstore = stores.get("userstore")
    questionstore = stores.get("questionstore")
    answerstore = stores.get("answerstore")

    # Verify all stores were created
    if not all([tagstore, userstore, questionstore, answerstore]):
        logger.warning("Some vector stores were not created successfully")
except Exception as e:
    logger.error(f"Error creating vector stores: {e}")
    raise

# create compressor
try:
    compressor = CrossEncoderReranker(
        model=reranker_model(),
        top_n=10,  # This will return the top n most relevant documents.
    )
except Exception as e:
    logger.error(f"Error creating compressor: {e}")
    raise

# ===========================================================================================================================================================
# Setting Up Retrievers from vectorstores for EnsembleRetriever
# ===========================================================================================================================================================


# Split retrieval into steps for observability
def retrieve_raw_docs(question: str) -> List[Document]:
    """Step 1: Graph Traversal & Ensemble Retrieval"""
    try:
        # Define the common search arguments once
        common_search_kwargs = {
            "k": 50,  # Increased initial pool: wider net across all entity types
            "score_threshold": 0.9,  # Slightly lowered to ensure we catch cross-domain links
            "fetch_k": 10000,  # Number of candidates for the initial vector search
            "lambda_mult": 0.5,  # Balanced weight between Vector and Full-text
            "params": {
                "embedding": embedding_model().embed_query(question),
                "keyword_query": escape_lucene_chars(question),
            },
        }

        # Use a list of vectorstores, filtering out any that failed to initialize
        vectorstores = [
            s
            for s in [tagstore, userstore, questionstore, answerstore]
            if s is not None
        ]

        if not vectorstores:
            logger.warning("No vector stores available for retrieval")
            return []

        # Create the retrievers using a list comprehension
        retrievers = [
            store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs=common_search_kwargs,
            )
            for store in vectorstores
        ]

        # Calculate equal weights dynamically
        num_retrievers = len(retrievers)
        weights = [1.0 / num_retrievers] * num_retrievers

        # init ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=retrievers,
            weights=weights,
        )

        logger.info(f"--- 🌐 GLOBAL RETRIEVAL: {question} ---")
        docs = ensemble_retriever.invoke(question)
        logger.info(f"Graph Traversal Complete. Found {len(docs)} documents.")
        return docs
    except Exception as e:
        logger.error(f"Error in retrieve_raw_docs: {e}")
        return []


def rerank_docs(inputs: Dict) -> List[Document]:
    """Step 2: Reranking"""
    try:
        docs = inputs.get("docs", [])
        question = inputs.get("question", "")

        RELEVANCY_THRESHOLD = 0.95

        if not docs:
            return []

        logger.info(f"Reranking {len(docs)} documents...")
        reranked_docs = compressor.compress_documents(documents=docs, query=question)

        # ✨ RELEVANCE GUARDRAIL: Filter by score
        high_quality_docs = [
            doc
            for doc in reranked_docs
            if doc.metadata.get("relevance_score", 1.0) >= RELEVANCY_THRESHOLD
        ]

        # Handle Low-Confidence Situations
        if not high_quality_docs:
            logger.warning(
                f"⚠️ GUARDRAIL TRIGGERED: No docs met threshold {RELEVANCY_THRESHOLD}"
            )
            return []  # Returns empty context to trigger LLM fallback

        logger.info(f"✅ {len(high_quality_docs)} docs passed relevancy guardrail.")
        return high_quality_docs
    except Exception as e:
        logger.error(f"Error in rerank_docs: {e}")
        return []


# ===========================================================================================================================================================
# Chain Assembly
# ===========================================================================================================================================================

# 1. Retrieval Sequence: Fetch -> Rerank
retrieval_chain = RunnablePassthrough.assign(
    docs=lambda x: RunnableLambda(retrieve_raw_docs)
    .with_config(run_name="GraphTraversal")
    .invoke(x["question"])
) | RunnableLambda(rerank_docs).with_config(run_name="Reranking")

# 2. Main GraphRAG Chain
# Flow: Input -> Context/History Prep -> Topic Analysis -> LLM Generation
try:
    # Prepare inputs for the LLM (Context + History)
    input_preparation = RunnablePassthrough.assign(
        context=retrieval_chain | format_docs_with_metadata,
        chat_history_formatted=lambda x: format_chat_history(x.get("chat_history", [])),
    )

    graph_rag_chain = (
        input_preparation
        # Topic Analysis Middleware: Inspects context/history to maintain session topic
        | process_with_topic_analysis
        # inject system prompt
        | analyst_prompt
    )

    logger.info("GraphRAG chain with topic analysis initialized successfully")
except Exception as e:
    logger.error(f"Error initializing GraphRAG chain: {e}")
    raise


# define state for application
class GraphRAGInput(BaseModel):
    """Input for the GraphRAG tool."""

    question: str = Field(description="The question to ask the knowledge base")
    # Make other fields optional and loose to prevent LLM validation errors
    # The LLM likely won't provide these anyway, so defaults work.
    chat_history: Optional[Any] = Field(
        description="Previous chat history context (optional)", default=[]
    )
    session_topic: Optional[str] = Field(
        description="Current session topic for continuity (optional)", default=""
    )
    session_id: Optional[str] = Field(
        description="Unique session identifier (optional)", default=""
    )


class GraphRAGTool(BaseTool):
    """Tool that queries the GraphRAG knowledge base."""

    name: str = "graph_rag_tool"
    description: str = "ALWAYS use this tool for ANY technical question, code snippet, or knowledge base query. Do not answer from memory."
    args_schema: Type[BaseModel] = GraphRAGInput

    # synchronous execution
    def _run(
        self,
        question: str,
        chat_history: Optional[Any] = [],
        session_topic: str = "",
        session_id: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the tool synchronously."""
        # Ensure chat_history is a list if it comes as None or something else
        if chat_history is None:
            chat_history = []
        if not isinstance(chat_history, list):
            # Fallback if LLM sends a string or other type
            chat_history = []
        result = graph_rag_chain.invoke(
            {
                "question": question,
                "chat_history": chat_history,
                "session_topic": session_topic,
                "session_id": session_id,
            },
            config={"callbacks": run_manager.get_child() if run_manager else None},
        )
        return result.to_string()

    # asynchronous execution
    async def _arun(
        self,
        question: str,
        chat_history: Optional[Any] = [],
        session_topic: str = "",
        session_id: str = "",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the tool asynchronously."""
        # Ensure chat_history is a list
        if chat_history is None:
            chat_history = []
        if not isinstance(chat_history, list):
            chat_history = []
        result = await graph_rag_chain.ainvoke(
            {
                "question": question,
                "chat_history": chat_history,
                "session_topic": session_topic,
                "session_id": session_id,
            },
            config={"callbacks": run_manager.get_child() if run_manager else None},
        )
        return result.to_string()


# Initialize the tool
graph_rag_tool = GraphRAGTool()
