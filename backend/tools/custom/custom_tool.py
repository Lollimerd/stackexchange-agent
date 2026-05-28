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
from tools.custom.cypher_src import retrieval_query
from utils.util import format_docs_with_metadata, escape_lucene_chars
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

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
        top_n=20,  # This will return the top n most relevant documents.
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
            "k": 100,  # Increased initial pool: wider net across all entity types
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

        if not docs:
            return []

        logger.info(f"Reranking {len(docs)} documents...")
        reranked_docs = compressor.compress_documents(documents=docs, query=question)

        # ⚠️ BAAI/bge-reranker outputs logits (often negative or < 0.95). 
        # A strict 0.95 probability threshold will drop almost all valid context!
        high_quality_docs = reranked_docs

        # Handle Low-Confidence Situations
        if not high_quality_docs:
            logger.warning("⚠️ No docs returned from reranker.")
            return []  # Returns empty context to trigger LLM fallback

        # final_docs = list(high_quality_docs)[:50]
        final_docs = list(high_quality_docs)

        logger.info(f"✅ {len(final_docs)} docs passed reranking.")
        return final_docs
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
# Flow: Input -> Context Prep
try:
    # Prepare inputs (Context only)
    graph_rag_chain = RunnablePassthrough.assign(
        context=retrieval_chain | format_docs_with_metadata,
    )

    logger.info("GraphRAG chain initialized successfully")
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


class CustomRAGTool(BaseTool):
    """Tool that queries the GraphRAG knowledge base."""

    name: str = "custom_rag_tool"
    description: str = "Search the knowledge base for technical context. Call this tool ONCE to retrieve data, then immediately use that data to answer the user's question. DO NOT call this tool repeatedly for the same question."
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
        result = graph_rag_chain.invoke(
            {
                "question": question,
            },
            config={"callbacks": run_manager.get_child() if run_manager else None},
        )
        context = result.get("context", "")
        return context if context.strip() else "No relevant context found in the knowledge graph. Do not try again."

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
        result = await graph_rag_chain.ainvoke(
            {
                "question": question,
            },
            config={"callbacks": run_manager.get_child() if run_manager else None},
        )
        context = result.get("context", "")
        return context if context.strip() else "No relevant context found in the knowledge graph. Do not try again."


# Initialize the tool
custom_rag_tool = CustomRAGTool()
